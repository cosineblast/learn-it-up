FRAMES_PER_SECOND = 100 

from math import floor
from collections import namedtuple 

import itertools
import numpy as np
import torch
import pickle


from pathlib import Path

ChartStats = namedtuple('ChartStats', ['len_frames', 'first_frame_index', 'last_frame_index'])
StepfileStats = namedtuple('StepfileStats', ['len_frames', 'chart_stats'])


# TODO: use default value as minimum value from features for each file, since
# files have varying silences.
DEFAULT_VALUE = np.log(1e-16)

class LoadFeaturesCached():
    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]

        with open(path, 'rb') as f:
            features = pickle.load(f)
            self.cache[path] = features
            return features

class PumpItUpConvolutionCNNOnsetDataset(torch.utils.data.Dataset):

    def __len__(self):
        return self.len_frames

    def __init__(self, stepfiles, paths, loader):
        all_features = [loader(get_feature_path_for(path)) for path in paths]

        def get_chart_stats(chart, features):
            first_frame_index = floor(chart.steps[0].time_in_seconds * FRAMES_PER_SECOND)
            last_frame_index = floor(chart.steps[-1].time_in_seconds * FRAMES_PER_SECOND)

            len_frames = last_frame_index - first_frame_index + 1

            assert len(chart.steps) >= 2
            assert first_frame_index >= 0
            assert last_frame_index < features.shape[0]
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
        self.loader = loader
        self.all_features = all_features 
        self.stepfiles = stepfiles

        self.stepfile_len_sums = list(itertools.accumulate(stats.len_frames for stats in stepfile_stats))
        self.stepfile_lens = list((stats.len_frames for stats in stepfile_stats))


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

    def _get_frame_context(self, file_features, feature_index):
        context_radius = 7
        # +1 because it is inclusive
        indices = np.arange(feature_index-context_radius, feature_index+context_radius+1).astype(int)
        indices_bad = (indices < 0) | (indices > file_features.shape[0])
        indices_clipped = np.clip(indices, 0, file_features.shape[0])
        result = file_features[indices_clipped]
        result[indices_bad] = DEFAULT_VALUE
        return result

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

        return (frames, difficulty), is_step

def get_feature_path_for(refined_stepfile_path):
    assert str(refined_stepfile_path).endswith(".ssc.bin")
    return Path("data/features") / (Path(Path(refined_stepfile_path).stem).stem + ".feat.bin")
