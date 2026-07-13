
import unittest
import numpy as np
import loading
import loading.ppc
import fractions
from ssc_util import RefinedStepFile, RefinedChart, StepInfo

DEFAULT_VALUE = -1

# These loaders should not care about any beat-related info,
# so we just don't fill any 

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

def stepfile_of(charts, title='some file'):
    return RefinedStepFile(
        info={},
        music='idk.mp3',
        title=title,
        charts=charts
    )

def step_at(time):
    return StepInfo(time_in_beats=fractions.Fraction(0), time_in_seconds=time, stepcode='10000')

def sample_input():

    stepfiles = [
        stepfile_of([
            # from frame 0 to frame 100
            chart_of([step_at(1.0), step_at(1.5), step_at(1.75), step_at(1.832), step_at(1.948), step_at(2.0)], diff=2),

            # from frame 101 to frame 301
            chart_of([step_at(1.5), step_at(2.5), step_at(2.75), step_at(3.5)], diff=8),
        ]),

        stepfile_of([
                chart_of([step_at(0.01), step_at(0.5), step_at(4.91)], diff=12),
                # from frame 302 to frame 792
        ]),
        ]

    # 5 seconds of audio features
    audio = np.zeros((3, 500)) + np.arange(500)
    audio = audio.transpose(1, 0)

    audio_len = audio.shape[0]

    padding = np.ones((7, 3)) * DEFAULT_VALUE
    padded_audio = np.concat([padding, audio, padding])

    padded_audio_view = loading.FeatureView(padded_audio, 7, audio_len)

    return stepfiles, [padded_audio_view, padded_audio_view], audio

class TestCNNDataset(unittest.TestCase):

    def test_all_ok(self):
        stepfiles, audios, _ = sample_input()
        self.dataset = loading.ppc.PPC_CNNOnsetDataset(stepfiles, audios) 
        self.dataset1 = loading.ppc.PPC_LSTMOnsetDataset(stepfiles, audios, 1)
        
        assert len(self.dataset) == len(self.dataset1)

        for i in range(len(self.dataset)):

            frames1, difficulty1, is_step1 = self.dataset1[i]
            frames, difficulty, is_step = self.dataset[i]

            np.testing.assert_array_equal(frames1[0], frames)
            np.testing.assert_array_equal(difficulty1, difficulty)
            np.testing.assert_array_equal(is_step1[0], is_step)
        
class TestLSTMOnsetInputLoader(unittest.TestCase):
    def setUp(self):
        self.stepfiles, self.audios, self.audio = sample_input()

        charts = [chart for stepfile in self.stepfiles for chart in stepfile.charts]

        self.charts = charts
        self.subject = lambda i, chart: loading.ppc.lstm_onset_getitem(charts[chart], self.audios[0], i, 1)

    def test_len_charts_is_ok(self):
        chartlen = lambda chart: loading.ppc.lstm_onset_chart_len(chart, 1)

        self.assertEqual(chartlen(self.charts[0]), 101)
        self.assertEqual(chartlen(self.charts[1]), 201)
        self.assertEqual(chartlen(self.charts[2]), 491)
   
    def test_difficulty_first_chart_ok(self):
        for i in irange(0, 100):
            _, difficulty, _, = self.subject(i, chart=0)

            self.assertEqual(difficulty[1], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_first_chart_ok(self):
        step_indices = set()

        for i in irange(0, 100):
            _, _, y = self.subject(i, chart=0)

            self.assertEqual(y.shape, (1,))

            if y[0]:
                step_indices.add(i)

        self.assertEqual(step_indices, {0, 50, 75, 83, 94, 100})

    def test_frames_first_chart_ok(self):
        for i in irange(0, 100):
            frames, _, _ = self.subject(i, chart=0)
            f = self.context_around(i+100)
            np.testing.assert_array_equal(frames, f)

    def context_around(self, i):
        result = np.zeros((15, 3))

        k = i-7
        j = 0

        while k <= i+7:
            if k < 0 or k >= self.audio.shape[0]:
                result[j] = DEFAULT_VALUE
            else:
                result[j] = self.audio[k]
            k += 1
            j += 1

        return result.reshape((1, 15, 3))


    def test_difficulty_second_chart_ok(self):
        for i in irange(0,200):
            _, difficulty, _ = self.subject(i, chart=1)
            self.assertEqual(difficulty[7], 1)
            self.assertEqual(np.sum(difficulty), 1)


    def test_frames_second_chart_ok(self):
        for i in irange(0,200):
            frames, _, _ = self.subject(i, chart=1)
            f = self.context_around(i+150)
            np.testing.assert_array_equal(frames, f)

    def test_steps_second_chart_ok(self):
        step_indices = set()

        for i in irange(0,200):
            _, _, y = self.subject(i, chart=1)
            if y:
                step_indices.add(i)

        self.assertEqual(len(step_indices), 4)
        self.assertEqual(step_indices, {0, 100, 125, 200})

    ## third chart

    def test_difficulty_third_chart_ok(self):
        for i in irange(0, 490):
            _, difficulty, _ = self.subject(i, chart=2)
            self.assertEqual(difficulty[11], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_third_chart_ok(self):
        step_indices = set()

        for i in irange(0, 490):
            _, _, y = self.subject(i, chart=2)

            if y:
                step_indices.add(i)

        self.assertEqual(step_indices, {0, 49, 490})

    def test_frames_third_chart_ok(self):
        for i in irange(0, 490):
            frames, _, _, = self.subject(i, chart=2)
            f = self.context_around(i+1)
            np.testing.assert_array_equal(frames, f)

class TestPerChartDataset(unittest.TestCase):
    def setUp(self):
        stepfiles, audios, _ = sample_input()

        self.charts = [chart for stepfile in stepfiles for chart in stepfile.charts]
        self.features = audios[0]
        self.dataset = loading.ppc.PPC_LSTMOnsetDataset(stepfiles, audios, 1)

    def test_len_ok(self):
        self.assertEqual(len(self.dataset), 793)

    def test_first_chart_ok(self):
        for i in irange(0, 100):
            x1, difficulty1, y1 = self.dataset[i]
            x2, difficulty2, y2 = loading.ppc.lstm_onset_getitem(self.charts[0], self.features, i, 1)

            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(difficulty1, difficulty2)
            np.testing.assert_array_equal(y1, y2)

    def test_second_chart_ok(self):
        for i in irange(0, 200):
            x1, difficulty1, y1 = self.dataset[i+101]
            x2, difficulty2, y2 = loading.ppc.lstm_onset_getitem(self.charts[1], self.features, i, 1)

            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(difficulty1, difficulty2)
            np.testing.assert_array_equal(y1, y2)

    def test_difficulty_third_chart_ok(self):
        for i in irange(0, 490):
            x1, difficulty1, y1 = self.dataset[i+302]
            x2, difficulty2, y2 = loading.ppc.lstm_onset_getitem(self.charts[2], self.features, i, 1)

            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(difficulty1, difficulty2)
            np.testing.assert_array_equal(y1, y2)

class TestLSTMOnsetDatasetUnrollN(unittest.TestCase):

    def test_all_ok(self):
        stepfiles, audios, _ = sample_input()
        
        dataset1 = loading.ppc.PPC_LSTMOnsetDataset(stepfiles, audios, 1)
        dataset_block = loading.ppc.PPC_LSTMOnsetDataset(stepfiles, audios, 20)

        base = 0
        for block_index in range(len(dataset_block)):

            frames_block, difficulty_block, is_step_block = dataset_block[block_index]

            n = frames_block.shape[0]

            self.assertEqual(frames_block.shape, (n, 15, 3))
            self.assertEqual(is_step_block.shape, (n,))

            for i in range(frames_block.shape[0]):
                frames, difficulty, is_step = dataset1[base+i]

                np.testing.assert_array_equal(difficulty_block, difficulty)
                np.testing.assert_array_equal(is_step_block[i].reshape((1,)), is_step)
                np.testing.assert_array_equal(frames_block[i].reshape((1, 15, 3)), frames)

            base += frames_block.shape[0]

from hypothesis import assume, given, strategies as st

class TestMaskAndPaddingTransformWorks(unittest.TestCase):

    @given(st.integers(1, 100), st.integers(1, 100))
    def test_works_with_predefined_shape(self, unroll, size):
        assume(unroll >= size)
       
        x = np.random.rand(size, 5, 4)
        delta = np.random.rand(size, 3)
        y = np.random.rand(size)

        new_x, new_delta, new_y, mask = loading.MaskAndPaddingTransform(unroll)((x, delta, y))

        self.assertEqual(new_x.shape, (unroll, 5, 4))
        self.assertEqual(new_delta.shape, (unroll, 3))
        self.assertEqual(new_y.shape,     (unroll,))
        self.assertEqual(mask.shape,     (unroll,))

        np.testing.assert_array_equal(new_x[0:size], x)
        np.testing.assert_array_equal(delta[0:size], delta)
        np.testing.assert_array_equal(y[0:size], y)

        self.assertEqual(np.sum(mask[size:unroll]), 0)
        self.assertEqual(np.sum(mask[0:size]), size)

    def test_lstm_works_with_transform(self):
        stepfiles, audios, _ = sample_input()

        dataset = loading.ppc.PPC_LSTMOnsetDataset(stepfiles, audios, 20, loading.MaskAndPaddingTransform(20, skip=1))

        for i in range(len(dataset)):
            frames, diff, y, mask = dataset[i]

            self.assertEqual(frames.shape, (20, 15, 3))
            self.assertEqual(diff.shape, (25,))
            self.assertEqual(y.shape, (20,))
            self.assertEqual(mask.shape, (20,))

class TestStepSelectionDataset(unittest.TestCase):
    def setUp(self):
        raw_chart = ssc_util.Chart(
            NOTES = [
                [ '10000', '00100', '01000', '00100' ],
                [ '10000', '10000',
                  '00100', '00100',
                  '01000', '01000',
                  '00100', '00100'
              ],
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

    def test_len_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        dataset = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 1)

        self.assertEqual(len(dataset), len(self.chart.steps))

        dataset2 = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 2)

        self.assertEqual(len(dataset2), len(self.chart.steps) // 2)

    def test_first_input_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        dataset = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 1)

        x, delta, y = dataset[0]

        self.assertEqual(np.sum(x), 0)
        self.assertEqual(delta[0][0], 0.0)
        self.assertEqual(delta[0][2], 1.0)

    def test_x_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        dataset = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 1)

        for i in range(1, len(dataset)):
            x, delta, y = dataset[i]
            np.testing.assert_array_equal(x[0], loading.ppc.stepcode_to_bag_tensor(self.chart.steps[i-1].stepcode))

    def test_delta_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        dataset = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 1)

        for i in range(1, len(dataset)):
            x, delta, y = dataset[i]
            self.assertEqual(delta[0][2], 0.0)

    def test_y_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        dataset = loading.ppc.PPC_SelectionLSTMDataset(stepfiles, 1)

        for i in range(len(dataset)):
            x, delta, y = dataset[i]
            self.assertEqual(y, loading.ppc.stepcode_to_index(self.chart.steps[i].stepcode))






import ssc_util
import models.ppc
import torch

class TestAlignedPPCDataset(unittest.TestCase):


    def setUp(self):
        audio = np.zeros((3, 80, 450)) + np.arange(450)
        audio = audio.transpose((2, 1, 0))
        audio_len = audio.shape[0]

        padding = np.ones((64, 80, 3)) * DEFAULT_VALUE
        padded_audio = np.concat([padding, audio, padding])

        self.audio = audio
        self.audio_view = loading.FeatureView(padded_audio, 64, audio_len) 

        raw_chart = ssc_util.Chart(
            NOTES = [
                [ '10000', '00100', '01000', '00100' ],
                [ '10000', '10000',
                  '00100', '00100',
                  '01000', '01000',
                  '00100', '00100'
              ],
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

    def test_len_works(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]
        dataset = loading.ppc.PPC_AlignedOnsetDataset(stepfiles, features, 2)

        self.assertEqual(len(dataset), 4)

    def test_stats_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]

        stats = loading.ppc.AlignedOnsetDatasetStats.from_stepfiles(stepfiles)

        self.assertAlmostEqual(self.chart.avg_bpm, 120.0)

        self.assertAlmostEqual(stats.bpm_mean, 120.0)
        self.assertAlmostEqual(stats.bpm_std, 1.0)
        self.assertAlmostEqual(stats.nps_mean, 3.0, delta=0.5)
        self.assertAlmostEqual(stats.nps_std, 1.0)

    def test_bpm_nps_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]
        dataset = loading.ppc.PPC_AlignedOnsetDataset(stepfiles, features, 2, transform=
                                                      loading.ppc.AlignedNormalizedStatsTransform(
                                                          loading.ppc.AlignedOnsetDatasetStats.from_stepfiles(stepfiles)
                                                      ))

        x, nps, bpm, y = dataset[0]

        self.assertAlmostEqual(np.min(bpm), 0.0)
        self.assertAlmostEqual(np.max(bpm), 0.0)

        self.assertAlmostEqual(np.min(nps), 0.0)
        self.assertAlmostEqual(np.max(nps), 0.0)


    def test_is_compatible_with_model(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]
        dataset = loading.ppc.PPC_AlignedOnsetDataset(stepfiles, features, 2)

        x, nps, bpm, y = dataset[0]
        
        model = models.ppc.PumpPumpConvolutionAlignedOnset()

        with torch.no_grad():
            x = torch.tensor(x).float()[None, :]
            nps = torch.tensor([nps]).float()
            bpm = torch.tensor(bpm).float()[None, :]

        y = model(x, nps, bpm)

        self.assertEqual(y.shape, (x.shape[0], x.shape[1], 48))

    def test_x_context_works(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]
        dataset = loading.ppc.PPC_AlignedOnsetDataset(stepfiles, features, 1)

        for i in range(1, len(dataset)):
            x, _, _, _ = dataset[i]
            x_prev, _, _, _ = dataset[i-1]
            
            np.testing.assert_array_equal(x[0, 1], x_prev[0, 2])

        for i in range(len(dataset)-1):
            x, _, _, _ = dataset[i]
            x_next, _, _, _ = dataset[i+1]
           
            np.testing.assert_array_equal(x[0, 3], x_next[0, 2])

        for i in range(2, len(dataset)):
            x, _, _, _ = dataset[i]
            x_prev, _, _, _ = dataset[i-2]
            
            np.testing.assert_array_equal(x[0, 0], x_prev[0, 2])

        for i in range(len(dataset)-2):
            x, _, _, _ = dataset[i]
            x_next, _, _, _ = dataset[i+2]
            
            np.testing.assert_array_equal(x[0, 4], x_next[0, 2])
            

    def test_x_is_ok(self):
        stepfiles = [stepfile_of([self.chart])]
        features = [self.audio_view]
        dataset = loading.ppc.PPC_AlignedOnsetDataset(stepfiles, features, 1)

        for beat in range(len(dataset)):
            x, nps, bpm, y = dataset[beat]

            self.assertEqual(x.shape, (1, 5, 32, 80, 3))

            for i in range(32):
                self.assertEqual(x[0, 2, i, 0, 0], int(50*beat + 50 * i/32))


def irange(a, b):
    return range(a, b+1)

if __name__ == '__main__':
    unittest.main()
