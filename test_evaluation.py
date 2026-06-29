
import unittest
import loading
import numpy as np
import torch

import evaluation
from ssc_util import RefinedStepFile, RefinedChart, StepInfo
import ssc_util
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



class TestAlignedEvaluation(unittest.TestCase):

    def setUp(self):
        audio = np.zeros((3, 80, 450)) + np.arange(450)
        audio = audio.transpose((2, 1, 0))
        audio_len = audio.shape[0]

        padding = np.ones((64, 80, 3)) * DEFAULT_VALUE
        padded_audio = np.concat([padding, audio, padding])

        self.audio_view = loading.FeatureView(padded_audio, 64, audio_len) 

        raw_chart = ssc_util.Chart(
            NOTES = [
                [ '10000', '00100', '01000', '000100' ],
                [ '00100', '00001', '10000', '000001' ],
            ],
            OFFSET = 0.0,
            BPMS = [(0.0, 120.0)],
            DESCRIPTION = 'S11',
            CREDIT = 'test',
            TIMESIGNATURES = [(0.0, 4.0, 4.0)],
            STOPS = [],
            DELAYS = [],
            WARPS = [],
            FAKES = [],
        )

        self.chart = ssc_util.refine_chart(raw_chart)

    def test_perfect_model_has_perfect_metrics(self):
        @under_numpy
        def perfect_model(x, nps, bpms):
            assert x.shape[0] == 1
            unroll = x.shape[1]

            result = np.array(self.chart.beat_onset_vectors[:unroll], dtype=float)

            result = -5 + result * 10
            return result[None, :]


        metrics = evaluation.measure_aligned_onset_performance(perfect_model, self.chart, self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

        self.assertGreater(metrics.precision, 0.99)
        self.assertGreater(metrics.recall, 0.99)
        self.assertGreater(metrics.fscore, 0.99)
        self.assertGreater(metrics.raw_auc_score, 0.99)
        self.assertGreater(metrics.aligned_auc_score, 0.99)
        self.assertGreater(metrics.accuracy, 0.99)

    def test_near_perfect_model_has_perfect_metrics(self):

        @under_numpy
        def near_perfect_model(x, nps, bpms):
            assert x.shape[0] == 1
            unroll = x.shape[1]

            original = np.array(self.chart.beat_onset_vectors[:unroll], dtype=float)

            result = []

            for onset in original:
                if np.sum(onset) > 0:
                    new_onset = np.zeros(48, dtype=float)
                    new_onset[min(np.argmax(onset)+1, 47)] = 1.0
                else:
                    new_onset = onset

                result.append(new_onset)

            result = np.array(result)

            assert np.sum(result) == np.sum(original)

            result = -5 + 10 * result + np.random.standard_normal(result.shape) / 2
            
            return result[None, :]

        metrics = evaluation.measure_aligned_onset_performance(near_perfect_model, self.chart, self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

        self.assertGreater(metrics.precision, 0.99)
        self.assertGreater(metrics.recall, 0.99)
        self.assertGreater(metrics.fscore, 0.99)
        self.assertGreater(metrics.aligned_auc_score, 0.99)
        self.assertGreater(metrics.accuracy, 0.99)

    def test_real_model_does_not_crash(self):

        import models.ppc
        
        model = models.ppc.PumpPumpConvolutionAlignedOnset()

        metrics = evaluation.measure_aligned_onset_performance(model, self.chart, self.audio_view, torch.nn.BCEWithLogitsLoss(), 'cpu')

        self.assertGreater(metrics.precision, 0.0)
        self.assertGreater(metrics.recall, 0.0)
        self.assertGreater(metrics.fscore, 0.0)
        self.assertGreater(metrics.aligned_auc_score, 0.0)
        self.assertGreater(metrics.raw_auc_score, 0.0)
        self.assertGreater(metrics.accuracy, 0.0)


def under_numpy(f):
    def np_f(*stuff):
        np_stuff = [thing.numpy() for thing in stuff]
        y = f(*np_stuff)
        if isinstance(y, tuple):
            torch_y = tuple(torch.tensor(thing) for thing in y)
        else:
            torch_y = torch.tensor(y)
        return torch_y

    return np_f
    

        
FRAMES_PER_SECOND = 100

def frame_of(step):
    return floor(step.time_in_seconds * FRAMES_PER_SECOND)


if __name__ == '__main__':
    unittest.main()


