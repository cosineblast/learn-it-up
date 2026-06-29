
from math import floor
from collections import namedtuple 

import itertools
import numpy as np
import torch

import random

import loading

import audio_util

CONTEXT_RADIUS = loading.CONTEXT_RADIUS
FRAMES_PER_SECOND = 100 

class PPC_CNNOnsetDataset(torch.utils.data.Dataset):
    """
    The dataset for the CNN onset model. 

    Yields
    - X: (15 x 80 x 3) np tensor containing the input frames
    - Difficulty: (25,) np tensor containing the one-hot difficulty
    - y: True or False depending on whether that frame is a step or not
    """
    def __init__(self, stepfiles, all_features, transform=(lambda x:x)):
        self.inner = PPC_LSTMOnsetDataset(stepfiles, all_features, 1)
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, target_index):
        frames, difficulty, is_step = self.inner[target_index]

        return self.transform((frames[0], difficulty, is_step[0]))

_ChartBlockStats = namedtuple('ChartBlockStats', ['len_blocks', 'first_frame_index', 'last_frame_index', 'stepfile_index'])

class PPC_LSTMOnsetDataset(torch.utils.data.Dataset):
    """
    The dataset for the C-LSTM onset model. 

    Parameters:
    - unroll_length: The ideal/maximum size of the returned sequences

    Yields:
    - X: (N x 15 x 80 x 3) np tensor containing the input frames, where 1 <= N <= UnrollLength
    - Difficulty: (25,) np tensor containing the one-hot difficulty for the returned frames
    - y: (N,) bool np tensor determining whether the frames are step or not
    """
    def __init__(self, stepfiles, all_features, unroll_length, transform=(lambda x:x)):
        # For more information on how data is stored in this class, see the LSTM selection model class

        def get_chart_stats(chart, stepfile_index):
            # these are inclusive
            first_frame_index = floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)
            last_frame_index = floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

            len_frames = last_frame_index - first_frame_index + 1

            assert len(chart.steps) >= 2
            assert first_frame_index >= 0
            assert last_frame_index < all_features[stepfile_index].len
            assert first_frame_index < last_frame_index

            return _ChartBlockStats(len_frames // unroll_length + (len_frames % unroll_length != 0),
                                    first_frame_index,
                                    last_frame_index,
                                    stepfile_index)

        charts = [(chart, stepfile_index)
            for stepfile_index, stepfile in enumerate(stepfiles) for chart in stepfile.charts]

        chart_stats = [get_chart_stats(chart, index) for chart,index in charts]
        block_counts = list(stat.len_blocks for stat in chart_stats)

        self.charts_and_stepfiles = charts
        self.chart_stats = chart_stats
        self.len_blocks = sum(stat.len_blocks for stat in chart_stats) 
        self.all_features = all_features 
        self.stepfiles = stepfiles
        self.chart_lens = block_counts
        self.chart_len_sums = list(itertools.accumulate(block_counts))
        self.transform = transform
        self.unroll_length = unroll_length

    def __len__(self):
        return self.len_blocks

    def _get_target_chart_index(self, target_index):
        return _binary_search_index(self.chart_len_sums, target_index)

    def _get_frame_context(self, view, first_frame, last_frame):
        array, start, length = view

        result = []
        for feature_index in range(first_frame, last_frame+1):
            a = start+feature_index-CONTEXT_RADIUS
            b = start+feature_index+CONTEXT_RADIUS+1
            result.append(array[a:b])

        return np.stack(result)

    def _get_next_step_index(self, steps, target_index):
        if steps[0].time_in_seconds * FRAMES_PER_SECOND >= target_index:
            return 0

        if steps[-1].time_in_seconds * FRAMES_PER_SECOND < target_index:
            return None

        l = 0
        r = len(steps)-1

        assert not (steps[l].time_in_seconds * FRAMES_PER_SECOND >= target_index)
        assert steps[r].time_in_seconds * FRAMES_PER_SECOND >= target_index

        while l+1!=r:
            m = (l + r) // 2

            if steps[m].time_in_seconds * FRAMES_PER_SECOND >= target_index:
                r = m
            else:
                l = m
                
        assert not (steps[l].time_in_seconds * FRAMES_PER_SECOND >= target_index)
        assert steps[r].time_in_seconds * FRAMES_PER_SECOND >= target_index

        return r


    def __getitem__(self, target_index):
        assert isinstance(target_index, int)
        assert target_index < len(self)

        chart_index, total_blocks_before_chart = self._get_target_chart_index(target_index)

        stats = self.chart_stats[chart_index]
        chart, stepfile_index = self.charts_and_stepfiles[chart_index]
        file_features = self.all_features[stepfile_index]

        block_index = target_index - total_blocks_before_chart

        block_first_frame = block_index * self.unroll_length + stats.first_frame_index

        # inclusive
        block_last_frame = block_first_frame + self.unroll_length-1
        block_last_frame = min(block_last_frame, stats.last_frame_index)

        frames = self._get_frame_context(file_features, block_first_frame, block_last_frame)

        block_length = block_last_frame - block_first_frame + 1
        difficulty = np.zeros(25)
        difficulty[chart.difficulty-1] = 1.0

        first_next_step = self._get_next_step_index(chart.steps, block_first_frame)
        last_next_step = self._get_next_step_index(chart.steps, block_last_frame)

        # since first_next_step and last_next_step are limited by the last step in the file,
        # there must always be a step >= it
        assert first_next_step is not None
        assert last_next_step is not None, f"at index {target_index}, chart={chart_index}: expected to have next step"

        step_indices = np.array([int(step.time_in_seconds * FRAMES_PER_SECOND) - block_first_frame
                                for step in chart.steps[first_next_step:last_next_step+1]],
                                dtype=int)

        step_indices_ok = (0 <= step_indices) & (step_indices < block_length)

        is_step = np.zeros(block_length, dtype=bool)
        is_step[step_indices[step_indices_ok]] = True
        
        return self.transform((frames, difficulty, is_step))

class PPC_AlignedOnsetDataset(torch.utils.data.Dataset):
    """
    The dataset for the aligned C-LSTM onset model. 

    Parameters:
    - unroll_length: The ideal/maximum size of the returned sequences

    Yields:
    - X: (N x 5 x 32 x 80 x 3) numpy tensor containing the input beat frames, where 1 <= N <= UnrollLength
    - NPS: target nps of the song, normalized to zero mean, 1 variance over all the stepfiles
    - BPMS: (N x 48) the bpms of each batch segment, normalized to zero mean 1 variance over all the stepfiles
    - y: (N x 48) bool np tensor with the onsets for the given beats
    """
    def __init__(self, stepfiles, all_features, unroll_length, transform=(lambda x:x)):
        def get_chart_block_count(chart):
            first_beat_index = 0
            last_beat_index = floor(chart.steps[-1].time_in_beats)

            len_beats = last_beat_index - first_beat_index + 1

            assert len(chart.steps) >= 2
            assert first_beat_index <= last_beat_index

            return len_beats // unroll_length + (len_beats % unroll_length != 0)

        charts = [(chart, stepfile_index)
            for stepfile_index, stepfile in enumerate(stepfiles) for chart in stepfile.charts]

        block_counts = [get_chart_block_count(chart) for chart,_ in charts]

        self.charts_and_stepfiles = charts
        self.len_blocks = sum(block_counts) 
        self.all_features = all_features 
        self.stepfiles = stepfiles
        self.chart_len_sums = list(itertools.accumulate(block_counts))
        self.transform = transform
        self.unroll_length = unroll_length
        self.bpm_mean, self.bpm_std, self.nps_mean, self.nps_std = _find_average_bpm_nps(stepfiles)

    def __len__(self):
        return self.len_blocks

    def _get_target_chart_index(self, target_index):
        return _binary_search_index(self.chart_len_sums, target_index)

    def _get_beat_context(self, chart, view, first_beat, last_beat):
        array, start, length = view

        slice = array[start:start+length]
        resampled = audio_util.resample_features(slice, chart.beat_start_end_times)

        default_value = np.min(slice, axis=0)
        padding = np.tile(default_value.reshape((1, 1, 80, 3)), (2, 32, 1, 1))
        padded = np.concat([padding, resampled, padding])
        offset = 2
       
        result = []
        for beat_index in range(first_beat, last_beat+1):
            a = beat_index - 2
            b = beat_index + 2 +1
            result.append(padded[a+offset:b+offset])

        return np.stack(result)

    def __getitem__(self, target_index):
        assert isinstance(target_index, int)
        assert target_index < len(self)

        chart_index, total_blocks_before_chart = self._get_target_chart_index(target_index)

        chart, stepfile_index = self.charts_and_stepfiles[chart_index]
        features = self.all_features[stepfile_index]

        chart_first_beat = 0
        chart_last_beat = floor(chart.steps[-1].time_in_beats)

        block_index = target_index - total_blocks_before_chart

        block_first_beat = block_index * self.unroll_length + chart_first_beat
        block_last_beat = block_first_beat + self.unroll_length-1
        block_last_beat = min(block_last_beat, chart_last_beat)

        beat_frames = self._get_beat_context(chart, features, block_first_beat, block_last_beat)

        block_length = block_last_beat - block_first_beat + 1

        nps = chart.nps
        nps = (nps - self.nps_mean) / self.nps_std

        bpms = np.array(chart.beat_bpms[block_first_beat:block_last_beat+1])
        bpms = (bpms - self.bpm_mean) / self.bpm_std

        onsets = np.array(chart.beat_onset_vectors[block_first_beat:block_last_beat+1], dtype=bool)
        
        return self.transform((beat_frames, nps, bpms, onsets))

def _find_average_bpm_nps(stepfiles):
    bpms = [chart.avg_bpm for file in stepfiles for chart in file.charts]
    bpm_mean = np.mean(bpms)
    bpm_std = np.std(bpms)
    bpm_std = 1 if bpm_std == 0 else bpm_std

    npss = [chart.nps for file in stepfiles for chart in file.charts]
    nps_mean = np.mean(npss)
    nps_std = np.std(npss)
    nps_std = 1 if nps_std == 0 else nps_std

    return bpm_mean, bpm_std, nps_mean, nps_std

class PPC_SelectionLSTMDataset(torch.utils.data.Dataset):
    """
    The dataset for the LSTM selection model. 

    Parameters:
    - unroll_length: The ideal/maximum size of the returned sequences

    Yields:
    - X: (N x 5 x 4) np tensor containing the input steps
    - Delta: (N x 3) np tensor containing step time related info
    - y: (N,) integer np tensor containing the one-hot index of the next steps of the sequence
    """

    def __init__(self, stepfiles, unroll_length, transform=(lambda x:x)):
        # this class divides all the input chart steps into blocks of size UnrollLength and allows
        # access to them by index.
        # charts may have a number of steps that is not divisble by the unroll length, so the returned blocks
        # may be shorter. that's when MaskAndPaddingTransform comes in handy
        # this way, we solve a todo in the original ddc:
        # "first sequence incredibly unlikely to appear, balance this"

        # in order to locate the stepfile, chart and steps of the given block index, we store
        # the prefix sum array of the block count of each file, and run a binary search to locate
        # the right chart

        charts = [chart for stepfile in stepfiles for chart in stepfile.charts]
        chart_step_counts = [len(chart.steps) for chart in charts]
        chart_block_counts = [length // unroll_length + int(length % unroll_length != 0) for length in chart_step_counts]

        len_blocks = sum(chart_block_counts) 

        self.unroll_length = unroll_length
        self.len_blocks = len_blocks
        self.charts = charts
        self.stepfiles = stepfiles
        self.chart_block_counts = chart_block_counts
        self.chart_block_counts_sum = list(itertools.accumulate(chart_block_counts))
        self.transform = transform

    def __len__(self):
        return self.len_blocks

    def _get_target_chart_index(self, target_index):
        return _binary_search_index(self.chart_block_counts_sum, target_index)

    def __getitem__(self, target_index):
        assert isinstance(target_index, int)
        assert target_index < len(self)

        chart_index, total_before_chart = self._get_target_chart_index(target_index)

        chart = self.charts[chart_index]

        block_index = target_index - total_before_chart

        start_index = block_index * self.unroll_length
        end_index =  (block_index + 1) * self.unroll_length
        block = chart.steps[start_index : end_index]

        # block length may be shorter than unroll_length
        end_index = start_index + len(block)

        return self.transform(steps_to_model_input(chart.steps, start_index, end_index))

def _binary_search_index(sums, target_index):
    """
    Given the prefix sum of an array A of items, returns the index of the
    first item such that target < prefix_sum item.
    It also returns the sum of the items before the returned one, so prefix_sum (item-1)
    """
    if target_index < sums[0]:
        return 0, 0

    l = 0
    r = len(sums)-1

    assert not (target_index < sums[l])
    assert target_index < sums[r]

    while l+1 != r:
        m = (l + r) // 2

        if target_index < sums[m]:
            r = m
        else:
            l = m

    assert not (target_index < sums[l])
    assert target_index < sums[r]

    return r, sums[l]

    

def steps_to_model_input(steps, start_index, end_index):
    """
    Given a list of StepInfo values, this function  returns
    the selection model bag-of-arrows np tensor input for the steps in the given range.

    This function receives the start and end index and the whole step list, instead of just receiving
    a slice list of the steps, because the returned model input contains information about next and previous steps,
    so it needs to know whether it is dealing with the start of the chart or not.
    """
    assert len(steps) >= 2
    x = np.stack([
        np.zeros((5, 4)) if i == 0 else
        stepcode_to_bag_tensor(steps[i-1].stepcode)
        for i in range(start_index, end_index)
    ])

    def deltaof(i):
        time_before = 0.0 if i == 0 else steps[i].time_in_seconds - steps[i-1].time_in_seconds
        time_after = 0.0 if i == len(steps)-1 else steps[i+1].time_in_seconds - steps[i].time_in_seconds
        is_first = float(i == 0)
        return np.array([time_before, time_after, is_first])

    delta = np.stack([deltaof(i) for i in range(start_index, end_index)])

    y = np.stack([
        stepcode_to_index(steps[i].stepcode)
        for i in range(start_index, end_index)
    ])

    return x, delta, y

_bag_of_arrows_cache = {}
def stepcode_to_bag_tensor(stepcode):
    if stepcode in _bag_of_arrows_cache:
        return _bag_of_arrows_cache[stepcode]

    result = np.zeros((5, 4))

    for i, char in enumerate(stepcode):
        result[i, int(char)] = 1.0

    _bag_of_arrows_cache[stepcode] = result
    return result

_index_cache = {}
def stepcode_to_index(_stepcode):
    if _stepcode in _index_cache:
        return _index_cache[_stepcode]

    stepcode = list(map(int, _stepcode))

    index = stepcode[0] + 4*stepcode[1] + 16* stepcode[2] + 64 * stepcode[3] + 256* stepcode[4]

    _index_cache[_stepcode] = index

    return index

def index_to_stepcode(index):
    stepcode = [0, 0, 0, 0, 0]
    stepcode[0] = index % 4
    stepcode[1] = (index // 4) % 4
    stepcode[2] = (index // 16) % 4
    stepcode[3] = (index // 64) % 4
    stepcode[4] = (index // 256) % 4
    return ''.join(map(str, stepcode))

