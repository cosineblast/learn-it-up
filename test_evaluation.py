
import unittest
import loading
import numpy as np
import torch

import evaluation
from ssc_util import RefinedStepFile, RefinedChart, StepInfo
from math import floor

DEFAULT_VALUE = -1

class TestEvaluationWorks(unittest.TestCase):

    def test_works(self):
        pass


    def setUp(self):
        def chart_of(steps, diff=16):
            return RefinedChart(
                steps=steps,
                offset=0.0,
                description=f'S{diff}',
                difficulty=diff,
                credit='UNKNOWN',
                avg_bpm=120.0,
                nps=5.0,

                bpms=[],
                beat_start_end_times=[],
                beat_bpms=[],
                beat_onset_vectors=[],
            )

        def step_at(time):
            return StepInfo(time_in_beats=0.0, time_in_seconds=time, stepcode='10000')

        charts = [
            chart_of([step_at(1.0), step_at(1.5), step_at(1.75), step_at(1.832), step_at(1.948), step_at(2.0)], diff=2),
            chart_of([step_at(1.5), step_at(2.5), step_at(2.75), step_at(3.5)], diff=8),
            chart_of([step_at(0.01), step_at(0.5), step_at(4.91)], diff=12),
        ]

        # 5 seconds of audio features
        audio = np.zeros((3, 80, 500)) + np.arange(500)
        audio = audio.transpose((2, 1, 0))

        self.audio = audio
        audio_len = audio.shape[0]

        padding = np.ones((7, 80, 3)) * DEFAULT_VALUE
        padded_audio = np.concat([padding, audio, padding])

        audio_view = loading.FeatureView(padded_audio, 7, audio_len) 
        self.audio_view = audio_view

        def perfect_model_single(features, difficulty):
            for chart in charts:
                if difficulty[chart.difficulty-1] == 1.0:
                    for step in chart.steps:
                        if np.sum(np.abs(self.context_around(frame_of(step)) - features)) < 0.01:
                            return 10

            return -10

        def perfect_model(features_batch, difficulty_batch):
            features_batch = torch.flatten(features_batch, start_dim=0, end_dim=1).numpy()
            difficulty_batch = difficulty_batch.numpy()

            result = np.zeros(features_batch.shape[0]) - 10

            for i in range(features_batch.shape[0]):
                result[i] = perfect_model_single(features_batch[i], difficulty_batch[0])

            return torch.tensor(result)[None, :]


        # almost perfect, just misaligned by 1 frame.
        # misses one frame if first and second frames are steps
        def near_perfect_model(features_batch, difficulty_batch):
            features_batch = torch.flatten(features_batch, start_dim=0, end_dim=1).numpy()
            difficulty_batch = difficulty_batch.numpy()

            result = np.zeros(features_batch.shape[0]) - 10

            for i in range(1, features_batch.shape[0]):
                result[i-1] = perfect_model_single(features_batch[i], difficulty_batch[0])

            result[0] = perfect_model_single(features_batch[0], difficulty_batch[0])

            return torch.tensor(result)[None, :]

        def off_by_one_model(features_batch, difficulty_batch):
            features_batch = torch.flatten(features_batch, start_dim=0, end_dim=1).numpy()
            difficulty_batch = difficulty_batch.numpy()

            result = np.zeros(features_batch.shape[0]) - 10

            skipped = False
            for i in range(0, features_batch.shape[0]):
                output = perfect_model_single(features_batch[i], difficulty_batch[0])
                if output > 0 and not skipped:
                    output = -10
                    skipped = True
                result[i] = output

            return torch.tensor(result)[None, :]

        # misaligned by 3 frames.
        def very_misaligned_model(features_batch, difficulty_batch):
            features_batch = torch.flatten(features_batch, start_dim=0, end_dim=1).numpy()
            difficulty_batch = difficulty_batch.numpy()

            result = np.zeros(features_batch.shape[0]) - 10

            for i in range(3, features_batch.shape[0]):
                result[i-3] = perfect_model_single(features_batch[i], difficulty_batch[0])

            return torch.tensor(result)[None, :]
            
        def inverse_perfect_model(features_batch, difficulty_batch):
            return perfect_model(features_batch, difficulty_batch) * (-1)

        self.charts = charts
        self.perfect_model = perfect_model
        self.inverse_perfect_model = inverse_perfect_model
        self.near_perfect_model = near_perfect_model
        self.off_by_one_model = off_by_one_model
        self.very_misaligned_model = very_misaligned_model

            
    def context_around(self, i):
        result = np.zeros((15, 80, 3))

        k = i-7
        j = 0

        while k <= i+7:
            if k < 0 or k >= self.audio.shape[0]:
                result[j] = DEFAULT_VALUE
            else:
                result[j] = self.audio[k]
            k += 1
            j += 1

        return result


    def test_perfect_model_has_perfect_metrics(self):
        for i, chart in enumerate(self.charts):
            result = evaluation.measure_onset_performance(self.perfect_model, self.charts[i], self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

            self.assertGreater(result.accuracy, 0.99, f'Accuracy for chart {i} must be near 1')
            self.assertGreater(result.recall, 0.99, f'Recall for chart {i} must be near 1')
            self.assertGreater(result.precision, 0.99, f'Precision for chart {i} must be near 1')

    def test_imperfect_model_has_low_metrics(self):
        for i, chart in enumerate(self.charts):
            result = evaluation.measure_onset_performance(self.inverse_perfect_model, self.charts[i], self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

            # measure_onset_performance does alignment and pick the best possible thresholds, so recall may be high,
            # and precision and accuracy may not be zero.

            self.assertLess(result.accuracy, 0.1, f'Accuracy for chart {i} must be near 0')
            self.assertLess(result.precision, 0.1, f'Precision for chart {i} must be near 0')

    def test_near_perfect_model_has_perfect_metrics(self):
        for i, chart in enumerate(self.charts):
            result = evaluation.measure_onset_performance(self.near_perfect_model, self.charts[i], self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

            self.assertGreater(result.accuracy, 0.99, f'Accuracy for chart {i} must be near 1')
            self.assertGreater(result.recall, 0.99, f'Recall for chart {i} must be near 1')
            self.assertGreater(result.precision, 0.99, f'Precision for chart {i} must be near 1')

    def test_off_by_one_model_has_expected_metrics(self):
        for i, chart in enumerate(self.charts):
            result = evaluation.measure_onset_performance(self.off_by_one_model, self.charts[i], self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

            self.assertGreater(result.precision, 0.99, f'Precision for chart {i} must be near 1')

            expected_recall = (len(chart.steps) - 1) / len(chart.steps)

            self.assertAlmostEqual(result.recall,expected_recall, f'Recall for chart {i} must match expected', 0.05)

    def test_very_misaligned_model_has_expected_metrics(self):
        for i, chart in enumerate(self.charts):
            result = evaluation.measure_onset_performance(self.very_misaligned_model, self.charts[i], self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')
            self.assertLess(result.precision, 0.1, f'Precision for chart {i} must be near 0')




        
FRAMES_PER_SECOND = 100

def frame_of(step):
    return floor(step.time_in_seconds * FRAMES_PER_SECOND)


if __name__ == '__main__':
    unittest.main()


