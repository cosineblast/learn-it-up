FRAMES_PER_SECOND = 100 

from math import floor
from collections import namedtuple 

import itertools
import numpy as np
import torch
import pickle

import random

from pathlib import Path

ChartStats = namedtuple('ChartStats', ['len_frames', 'first_frame_index', 'last_frame_index'])
StepfileStats = namedtuple('StepfileStats', ['len_frames', 'chart_stats'])
FeatureView =namedtuple('FeatureView', ['array', 'start', 'len'])


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

def get_all_song_context_features(features_view, first_frame, last_frame):
    """Returns a list with all the 15-frame context windows of the given feature view,
       within the given inclusive frame range"""

    features, start, length = features_view

    frame_indices = start + np.arange(first_frame,last_frame+1)
    total_frames = frame_indices.shape[0]

    context_indices = np.tile(np.arange(-7, 8), (total_frames, 1)).transpose((1,0)) + frame_indices
    context_indices = context_indices.transpose((1, 0))
    frame_features = features[context_indices]

    return frame_features


class PumpItUpConvolutionCNNOnsetDataset(torch.utils.data.Dataset):

    def __len__(self):
        return self.len_frames

    def __init__(self, stepfiles, all_features, transform_x=(lambda x:x), transform_y=(lambda y:y)):
        def get_chart_stats(chart, features):
            first_frame_index = floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)
            last_frame_index = floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

            len_frames = last_frame_index - first_frame_index + 1

            assert len(chart.steps) >= 2
            assert first_frame_index >= 0
            assert last_frame_index < features.len
            assert first_frame_index < last_frame_index

            # inclusive
            return ChartStats(len_frames, first_frame_index, last_frame_index)

        def get_stepfile_stats(stepfile, features):
            assert len(stepfile.charts) > 0

            chart_stats = [get_chart_stats(chart, features) for chart in stepfile.charts]
            len_frames = sum(stat.len_frames for stat in chart_stats)

            return StepfileStats(len_frames, chart_stats)


        stepfile_stats = [get_stepfile_stats(stepfile, features) 
                          for features, stepfile in zip(all_features, stepfiles)]
        len_frames = sum(stat.len_frames for stat in stepfile_stats) 

        self.stepfile_stats = stepfile_stats
        self.len_frames = len_frames
        self.all_features = all_features 
        self.stepfiles = stepfiles

        self.stepfile_len_sums = list(itertools.accumulate(stats.len_frames for stats in stepfile_stats))
        self.stepfile_lens = list((stats.len_frames for stats in stepfile_stats))
        self.transform_x = transform_x
        self.transform_y = transform_y


    def _get_target_chart(self, file, file_stats, target_index, total):
        target_chart = None
        for chart, chart_stats in zip(file.charts, file_stats.chart_stats):
            if target_index < total + chart_stats.len_frames:
                target_chart = (chart, chart_stats)
                break
            total += chart_stats.len_frames

        assert target_chart is not None
        return target_chart, total


    def _get_target_stepfile(self, target_index):
        if target_index < self.stepfile_len_sums[0]:
            return (self.stepfiles[0], self.stepfile_stats[0], self.all_features[0]), 0

        l = 0
        r = len(self.stepfile_len_sums)-1

        assert not (target_index < self.stepfile_len_sums[l])
        assert target_index < self.stepfile_len_sums[r]

        while l+1 != r:
            m = (l + r) // 2

            if target_index < self.stepfile_len_sums[m]:
                r = m
            else:
                l = m

        assert not (target_index < self.stepfile_len_sums[l])
        assert target_index < self.stepfile_len_sums[r]

        return (self.stepfiles[r], self.stepfile_stats[r], self.all_features[r]), self.stepfile_len_sums[l]

    def _get_frame_context(self, view, feature_index):
        array, start, length = view
        return array[start+feature_index-CONTEXT_RADIUS:start+feature_index+CONTEXT_RADIUS+1]

    def _get_next_step(self, steps, target_time):
        if steps[0].time_in_seconds >= target_time:
            return steps[0]

        if steps[-1].time_in_seconds < target_time:
            return None

        l = 0
        r = len(steps)-1

        assert not (steps[l].time_in_seconds >= target_time)
        assert steps[r].time_in_seconds >= target_time

        while l+1!=r:
            m = (l + r) // 2

            if steps[m].time_in_seconds >= target_time:
                r = m
            else:
                l = m
                
        assert not (steps[l].time_in_seconds >= target_time)
        assert steps[r].time_in_seconds >= target_time

        return steps[r]


    def __getitem__(self, target_index):
        # TODO: precompute (target_features, target_index, target_difficuly, is_frame) for each frame
        # if this becomes too slow

        assert isinstance(target_index, int)
        assert target_index < len(self)

        (file, file_stats, file_features), total_before_file = self._get_target_stepfile(target_index)
        # we now know what stepfile the target index is, now we need to pick the right chart

        (chart, chart_stats), total_before_chart = self._get_target_chart(file, file_stats, target_index, total_before_file)
        # we now know what chart the target index is, now we need to pick the right frame

        feature_index = target_index - total_before_chart + chart_stats.first_frame_index

        frames = self._get_frame_context(file_features, feature_index)

        next_step = self._get_next_step(chart.steps, feature_index / FRAMES_PER_SECOND)

        difficulty = np.zeros(25)
        difficulty[chart.difficulty-1] = 1.0
        
        is_step = (False if next_step is None else 
                   floor(next_step.time_in_seconds * FRAMES_PER_SECOND) == feature_index)

        return self.transform_x((frames, difficulty)), self.transform_y(is_step)

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
    x = np.stack([
        np.zeros((5, 4)) if i == 0 else
        stepcode_to_bag_tensor(steps[i-1].stepcode)
        for i in range(start_index, end_index)
    ])

    delta = np.stack([
        np.array([0.0, 1.0]) if i == 0 else
        (steps[i].time_in_seconds - steps[i-1].time_in_seconds) * np.array([1.0, 0.0])
        for i in range(start_index, end_index)
    ])

    y = np.stack([
        stepcode_to_onehot_tensor(steps[i].stepcode)
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

_onehot_cache = {}
def stepcode_to_onehot_tensor(_stepcode):
    if _stepcode in _onehot_cache:
        return _onehot_cache[_stepcode]

    stepcode = list(map(int, _stepcode))

    index = stepcode[0] + 4*stepcode[1] + 16* stepcode[2] + 64 * stepcode[3] + 256* stepcode[4]

    _onehot_cache[_stepcode] = index

    return index

def index_to_stepcode(index):
    stepcode = [0, 0, 0, 0, 0]
    stepcode[0] = index % 4
    stepcode[1] = (index // 4) % 4
    stepcode[2] = (index // 16) % 4
    stepcode[3] = (index // 64) % 4
    stepcode[4] = (index // 256) % 4
    return ''.join(map(str, stepcode))

def MaskAndPaddingTransform(unroll_length):
    def maskAndPad(data):
        x, delta, y = data

        size    = x.shape[0]
        padding = unroll_length - size

        x       = np.pad(x, [(0, padding), (0, 0), (0, 0)])
        delta   = np.pad(delta, [(0, padding), (0, 0)])
        y       = np.pad(y, [(0, padding)])

        mask = np.pad(np.ones(size), (0, padding))

        return x, delta, y, mask

    return maskAndPad  
