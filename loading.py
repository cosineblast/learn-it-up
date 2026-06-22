FRAMES_PER_SECOND = 100 

from math import floor
from collections import namedtuple 

import itertools
import numpy as np
import torch
import pickle

import random

from pathlib import Path

FeatureView = namedtuple('FeatureView', ['array', 'start', 'len'])


CONTEXT_RADIUS=7

class LoadFeaturesCached():
    """
    Loads features from disk applying padding and caching on paths.
    """
    def __init__(self, normalize_features=True):
        self.cache = {}
        self.normalize_features = normalize_features

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]

        with open(path, 'rb') as f:
            features = pickle.load(f)

            view = prepare_features(features, normalize=self.normalize_features)

            self.cache[path] = view

            return view

def prepare_features(features, normalize=True):
    """Adds 7 frames of padding before and after the feature, padded with minimum values of features,
       and normalizes features to mean 0 and standard deviation 1.
       Returns a feature view of the non-padded features. 
       """

    if normalize:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    default_value = np.min(features, axis=0)
    padding = np.tile(default_value.reshape((1, 80, 3)), (CONTEXT_RADIUS, 1, 1))
    padded = np.concat([padding, features, padding])

    return FeatureView(padded, CONTEXT_RADIUS, features.shape[0])

def get_all_song_context_features(features_view, first_frame, last_frame, upshape=False):
    """Returns a list with all the 15-frame context windows of the given feature view,
       within the given inclusive frame range"""

    features, start, length = features_view

    frame_indices = start + np.arange(first_frame,last_frame+1)
    total_frames = frame_indices.shape[0]

    context_indices = np.tile(np.arange(-7, 8), (total_frames, 1)).transpose((1,0)) + frame_indices
    context_indices = context_indices.transpose((1, 0))
    frame_features = features[context_indices]

    return frame_features[None, :] if upshape else frame_features


class PumpItUpConvolutionCNNOnsetDataset(torch.utils.data.Dataset):
    def __init__(self, stepfiles, all_features, transform=(lambda x:x)):
        self.inner = PumpItUpConvolutionLSTMOnsetDataset(stepfiles, all_features, 1)
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, target_index):
        frames, difficulty, is_step = self.inner[target_index]

        return self.transform((frames[0], difficulty, is_step[0]))

ChartBlockStats = namedtuple('ChartBlockStats', ['len_blocks', 'first_frame_index', 'last_frame_index', 'stepfile_index'])

class PumpItUpConvolutionLSTMOnsetDataset(torch.utils.data.Dataset):

    def __init__(self, stepfiles, all_features, unroll_length, transform=(lambda x:x)):
        def get_chart_stats(chart, stepfile_index):
            # these are inclusive
            first_frame_index = floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)
            last_frame_index = floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

            len_frames = last_frame_index - first_frame_index + 1

            assert len(chart.steps) >= 2
            assert first_frame_index >= 0
            assert last_frame_index < all_features[stepfile_index].len
            assert first_frame_index < last_frame_index

            return ChartBlockStats(len_frames // unroll_length + (len_frames % unroll_length != 0), first_frame_index, last_frame_index, stepfile_index)

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
        if target_index < self.chart_len_sums[0]:
            return 0, 0

        l = 0
        r = len(self.chart_len_sums)-1

        assert not (target_index < self.chart_len_sums[l])
        assert target_index < self.chart_len_sums[r]

        while l+1 != r:
            m = (l + r) // 2

            if target_index < self.chart_len_sums[m]:
                r = m
            else:
                l = m

        assert not (target_index < self.chart_len_sums[l])
        assert target_index < self.chart_len_sums[r]

        return r, self.chart_len_sums[l]

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

class PumpItUpConvolutionSelectionLSTMDataset(torch.utils.data.Dataset):

    def __init__(self, stepfiles, unroll_length, transform=(lambda x:x)):
        # original ddc todo: "first sequence incredibly unlikely to appear, balance this"
        # we solve this by sampling all blocks in an epoch. although this makes so that blocks
        # will always be aligned by 100 steps to the start of the song, this means every block
        # gets accessed once per epoch

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
        if target_index < self.chart_block_counts_sum[0]:
            return 0, 0

        l = 0
        r = len(self.chart_block_counts_sum)-1

        assert not (target_index < self.chart_block_counts_sum[l])
        assert target_index < self.chart_block_counts_sum[r]

        while l+1 != r:
            m = (l + r) // 2

            if target_index < self.chart_block_counts_sum[m]:
                r = m
            else:
                l = m

        assert not (target_index < self.chart_block_counts_sum[l])
        assert target_index < self.chart_block_counts_sum[r]

        return r, self.chart_block_counts_sum[l]

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

def steps_to_model_input(steps, start_index, end_index):
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

def MaskAndPaddingTransform(unroll_length, skip=None):
    """
    This transform function takes a tuple containing multiple tensors X1, ..., Xn of
    possibly different shapes, but all matching the size of their their main axis as `k` (X1.shape[0] = ... = Xn.shape[0] = k).
    If k is less than the unroll length n, then all those vectors are padded with zeroes in their main axis to size n.
    Additionally, an additional element `mask` is added to the returned tuple, containing a vector of shape (n,) with ones
    in the first k positions, and zero in the last n-k positions.
    """
    def maskAndPad(stuff):
        size    = stuff[0].shape[0]

        new_stuff = tuple(
            np.pad(x,[(0, unroll_length - x.shape[0])] + [(0, 0)] * (len(x.shape)-1))
            if skip is None or i != skip else x
            for i, x in enumerate(stuff))

        mask = np.pad(np.ones(size), (0, unroll_length - stuff[0].shape[0]))

        return *new_stuff, mask

    return maskAndPad  
