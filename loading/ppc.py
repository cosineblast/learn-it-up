
from math import floor
from collections import namedtuple 

import itertools
import numpy as np
import torch

import random

import loading
from loading import FeatureView

import audio_util

from ssc_util import RefinedStepFile, RefinedChart, StepInfo

CONTEXT_RADIUS = loading.CONTEXT_RADIUS
FRAMES_PER_SECOND = 100 

class PerChartDataset(torch.utils.data.Dataset):
    def __init__(self, *, chartlen, getitem, stepfiles: list[RefinedStepFile], features: list[FeatureView]):
        charts_and_frames = [(chart, frames) for stepfile, frames in zip(stepfiles, features) for chart in stepfile.charts]
        lengths = [chartlen(chart) for chart, _ in charts_and_frames]
        lengths_sums = list(itertools.accumulate(lengths))
        total_length = sum(lengths)

        self.chartlen, self.getitem = chartlen, getitem
        self.features, self.stepfiles = features, stepfiles
        self.charts_and_frames = charts_and_frames
        self.lengths = lengths
        self.lengths_sums = lengths_sums
        self.total_length = total_length

    def __getitem__(self, index):
        chart_index, before_chart = _binary_search_index(self.lengths_sums, index)
        return self.getitem(*self.charts_and_frames[chart_index], index - before_chart)

    def __len__(self):
        return self.total_length

class PPC_CNNOnsetDataset(torch.utils.data.Dataset):
    """
    The dataset for the CNN onset model. 

    Yields
    - X: (15 x 80 x 3) np tensor containing the input frames
    - Difficulty: (25,) np tensor containing the one-hot difficulty
    - y: True or False depending on whether that frame is a step or not
    """
    def __init__(self, stepfiles: list[RefinedStepFile], all_features: list[FeatureView], transform=(lambda x:x)):
        self.inner = PPC_LSTMOnsetDataset(stepfiles, all_features, 1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        target_index = index
        frames, difficulty, is_step = self.inner[target_index]

        return self.transform((frames[0], difficulty, is_step[0]))

ChartBlockStats = namedtuple('ChartBlockStats', ['len_blocks', 'first_frame_index', 'last_frame_index', 'stepfile_index'])

class PPC_LSTMOnsetDataset(PerChartDataset):
    """
    The dataset for the C-LSTM onset model. 

    Parameters:
    - unroll_length: The ideal/maximum size of the returned sequences

    Yields:
    - X: (N x 15 x 80 x 3) np tensor containing the input frames, where 1 <= N <= UnrollLength
    - Difficulty: (25,) np tensor containing the one-hot difficulty for the returned frames
    - y: (N,) bool np tensor determining whether the frames are step or not
    """
    def __init__(self, stepfiles: list[RefinedStepFile], features: list[FeatureView], unroll_length: int, transform=(lambda x:x)):
        super().__init__(
            chartlen=lambda chart: lstm_onset_chart_len(chart, unroll_length),
            getitem=lambda chart, features, index: transform(lstm_onset_getitem(chart, features, index, unroll_length)),
            stepfiles=stepfiles,
            features=features
        )

def chart_first_frame(chart: RefinedChart) -> int:
    return floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)

def chart_last_frame(chart: RefinedChart) -> int:
    return floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

def lstm_onset_chart_len(chart: RefinedChart, unroll_length: int = 1) -> int:
    frame_count = chart_last_frame(chart) - chart_first_frame(chart) + 1

    return frame_count // unroll_length + (frame_count % unroll_length != 0)

def lstm_onset_getitem(chart, music_frames, index, unroll_length):
    assert index < lstm_onset_chart_len(chart, unroll_length)

    block_first_frame = index * unroll_length + chart_first_frame(chart)

    # inclusive
    block_last_frame = block_first_frame + unroll_length-1
    block_last_frame = min(block_last_frame, chart_last_frame(chart))

    frames = _lstm_onset_frame_context(music_frames, block_first_frame, block_last_frame)

    block_length = block_last_frame - block_first_frame + 1
    difficulty = np.zeros(25)
    difficulty[chart.difficulty-1] = 1.0

    first_next_step = _lstm_onset_next_step_index(chart.steps, block_first_frame)
    last_next_step = _lstm_onset_next_step_index(chart.steps, block_last_frame)

    # since first_next_step and last_next_step are limited by the last step in the file,
    # there must always be a step >= it
    assert first_next_step is not None
    assert last_next_step is not None, f"at index {index}, chart={chart}: expected to have next step"

    step_indices = np.array([int(step.time_in_seconds * FRAMES_PER_SECOND) - block_first_frame
                            for step in chart.steps[first_next_step:last_next_step+1]],
                            dtype=int)

    step_indices_ok = (0 <= step_indices) & (step_indices < block_length)

    is_step = np.zeros(block_length, dtype=bool)
    is_step[step_indices[step_indices_ok]] = True
    
    return (frames, difficulty, is_step)

def _lstm_onset_frame_context(view: FeatureView, first_frame: int, last_frame: int) -> np.ndarray:
    array, start, length = view

    result = []
    for feature_index in range(first_frame, last_frame+1):
        a = start+feature_index-CONTEXT_RADIUS
        b = start+feature_index+CONTEXT_RADIUS+1
        result.append(array[a:b])

    return np.stack(result)

def _lstm_onset_next_step_index(steps: list[StepInfo], target_index: int) -> int | None:
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



from dataclasses import dataclass

class PPC_AlignedOnsetDataset(PerChartDataset):
        
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
    def __init__(self, stepfiles: list[RefinedStepFile], all_features: list[FeatureView], unroll_length: int, transform=(lambda x:x)):
        super().__init__(
            chartlen=lambda chart: aligned_onset_chart_len(chart, unroll_length),
            getitem=lambda chart, features, index: transform(aligned_onset_getitem(chart, features, index, unroll_length)),
            stepfiles=stepfiles,
            features=all_features,
        )

def aligned_onset_chart_len(chart: RefinedChart, unroll_length: int) -> int:
    first_beat_index = 0
    last_beat_index = floor(chart.steps[-1].time_in_beats)

    len_beats = last_beat_index - first_beat_index + 1

    assert len(chart.steps) >= 2
    assert first_beat_index <= last_beat_index

    return len_beats // unroll_length + (len_beats % unroll_length != 0)

def aligned_onset_getitem(chart, features, block_index, unroll_length):
    chart_first_beat = 0
    chart_last_beat = floor(chart.steps[-1].time_in_beats)

    block_first_beat = block_index * unroll_length + chart_first_beat
    block_last_beat = block_first_beat + unroll_length-1
    block_last_beat = min(block_last_beat, chart_last_beat)

    beat_frames = _get_aligned_beat_context(chart, features, block_first_beat, block_last_beat)

    block_length = block_last_beat - block_first_beat + 1

    nps = chart.nps
    bpms = np.array(chart.beat_bpms[block_first_beat:block_last_beat+1])

    onsets = np.array(chart.beat_onset_vectors[block_first_beat:block_last_beat+1], dtype=bool)
    
    return (beat_frames, nps, bpms, onsets)

def _get_aligned_beat_context(chart: RefinedChart, view: FeatureView, first_beat: int, last_beat: int) -> np.ndarray:
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

@dataclass
class AlignedOnsetDatasetStats:
    bpm_mean: float
    bpm_std: float
    nps_mean: float
    nps_std: float

    @staticmethod
    def from_stepfiles(stepfiles: list[RefinedStepFile]):
        bpms = [chart.avg_bpm for file in stepfiles for chart in file.charts]
        bpm_mean = float(np.mean(bpms))
        bpm_std = float(np.std(bpms))
        bpm_std = 1 if bpm_std == 0 else bpm_std

        npss = [chart.nps for file in stepfiles for chart in file.charts]
        nps_mean = float(np.mean(npss))
        nps_std = float(np.std(npss))
        nps_std = 1 if nps_std == 0 else nps_std

        return AlignedOnsetDatasetStats(bpm_mean=bpm_mean, bpm_std=bpm_std, nps_mean=nps_mean, nps_std=nps_std)

def AlignedNormalizedStatsTransform(stats: AlignedOnsetDatasetStats):

    def run(stuff):
        x, nps, bpms, y = stuff

        nps = (nps - stats.nps_mean) / stats.nps_std
        bpms = (bpms - stats.bpm_mean) / stats.bpm_std

        return x, nps, bpms, y

    return run


def _find_average_bpm_nps(stepfiles: list[RefinedStepFile]) -> AlignedOnsetDatasetStats:
    bpms = [chart.avg_bpm for file in stepfiles for chart in file.charts]
    bpm_mean = float(np.mean(bpms))
    bpm_std = float(np.std(bpms))
    bpm_std = 1 if bpm_std == 0 else bpm_std

    npss = [chart.nps for file in stepfiles for chart in file.charts]
    nps_mean = float(np.mean(npss))
    nps_std = float(np.std(npss))
    nps_std = 1 if nps_std == 0 else nps_std

    return PPC_AlignedOnsetDataset.Stats(bpm_mean=bpm_mean, bpm_std=bpm_std, nps_mean=nps_mean, nps_std=nps_std)

class PPC_SelectionLSTMDataset(PerChartDataset):
    """
    The dataset for the LSTM selection model. 

    Parameters:
    - unroll_length: The ideal/maximum size of the returned sequences

    Yields:
    - X: (N x 5 x 4) np tensor containing the input steps
    - Delta: (N x 3) np tensor containing step time related info
    - y: (N,) integer np tensor containing the one-hot index of the next steps of the sequence
    """

    def __init__(self, stepfiles: list[RefinedStepFile], unroll_length: int, transform=(lambda x:x)):
        # this class divides all the input chart steps into blocks of size UnrollLength and allows
        # access to them by index.
        # charts may have a number of steps that is not divisble by the unroll length, so the returned blocks
        # may be shorter. that's when MaskAndPaddingTransform comes in handy
        # this way, we solve a todo in the original ddc:
        # "first sequence incredibly unlikely to appear, balance this"

        super().__init__(
            getitem=lambda chart, features, index: transform(lstm_selection_getitem(chart, index,unroll_length)),
            chartlen=lambda chart: lstm_selection_chart_len(chart, unroll_length),
            stepfiles=stepfiles,
            features=[() for _ in stepfiles]
        )

def lstm_selection_chart_len(chart, unroll_length):
    return (len(chart.steps) // unroll_length) + bool(len(chart.steps) % unroll_length)

def lstm_selection_getitem(chart, index, unroll_length):
    block_index = index

    start_index = block_index * unroll_length
    end_index =  (block_index + 1) * unroll_length
    block = chart.steps[start_index : end_index]

    # block length may be shorter than unroll_length
    end_index = start_index + len(block)

    return steps_to_model_input(chart.steps, start_index, end_index)

def _binary_search_index(sums, target_index: int) -> tuple[int, int]:
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

    

def steps_to_model_input(steps: list[StepInfo], start_index: int, end_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def stepcode_to_bag_tensor(stepcode: str) -> np.ndarray:
    if stepcode in _bag_of_arrows_cache:
        return _bag_of_arrows_cache[stepcode]

    result = np.zeros((5, 4))

    for i, char in enumerate(stepcode):
        result[i, int(char)] = 1.0

    _bag_of_arrows_cache[stepcode] = result
    return result

_index_cache = {}
def stepcode_to_index(_stepcode: str) -> int:
    if _stepcode in _index_cache:
        return _index_cache[_stepcode]

    stepcode = list(map(int, _stepcode))

    index = stepcode[0] + 4*stepcode[1] + 16* stepcode[2] + 64 * stepcode[3] + 256* stepcode[4]

    _index_cache[_stepcode] = index

    return index

def index_to_stepcode(index: int) -> str:
    stepcode = [0, 0, 0, 0, 0]
    stepcode[0] = index % 4
    stepcode[1] = (index // 4) % 4
    stepcode[2] = (index // 16) % 4
    stepcode[3] = (index // 64) % 4
    stepcode[4] = (index // 256) % 4
    return ''.join(map(str, stepcode))

