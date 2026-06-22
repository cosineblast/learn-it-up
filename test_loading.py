
import unittest
import numpy as np
import loading
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
    return StepInfo(time_in_beats=0.0, time_in_seconds=time, stepcode='10000')

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
        self.dataset = loading.PumpItUpConvolutionCNNOnsetDataset(stepfiles, audios) 
        self.dataset1 = loading.PumpItUpConvolutionLSTMOnsetDataset(stepfiles, audios, 1)
        
        assert len(self.dataset) == len(self.dataset1)

        for i in range(len(self.dataset)):

            frames1, difficulty1, is_step1 = self.dataset1[i]
            frames, difficulty, is_step = self.dataset[i]

            np.testing.assert_array_equal(frames1[0], frames)
            np.testing.assert_array_equal(difficulty1, difficulty)
            np.testing.assert_array_equal(is_step1[0], is_step)
        


class TestLSTMOnsetDatasetUnrollOne(unittest.TestCase):

    def setUp(self):
        stepfiles, audios, audio = sample_input()

        self.dataset = loading.PumpItUpConvolutionLSTMOnsetDataset(stepfiles, audios, 1)
        self.audio = audio

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
        

    def test_stats_ok(self):
        self.assertEqual(self.dataset.chart_stats[0].len_blocks, 101)
        self.assertEqual(self.dataset.chart_stats[1].len_blocks, 201)
        self.assertEqual(self.dataset.chart_stats[2].len_blocks, 491)
        self.assertEqual(len(self.dataset), 793)

    def test_difficulty_first_chart_ok(self):
        for i in irange(0, 100):
            _, difficulty, _, = self.dataset[i]

            self.assertEqual(difficulty[1], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_first_chart_ok(self):
        step_indices = set()

        for i in irange(0, 100):
            _, _, y = self.dataset[i]

            self.assertEqual(y.shape, (1,))

            if y[0]:
                step_indices.add(i)

        self.assertEqual(step_indices, {0, 50, 75, 83, 94, 100})

    def test_frames_first_chart_ok(self):
        for i in irange(0, 100):
            frames, _, _ = self.dataset[i]
            f = self.context_around(i+100)
            np.testing.assert_array_equal(frames, f)

    ## second chart

    def test_difficulty_second_chart_ok(self):
        for i in irange(101,301):
            _, difficulty, _ = self.dataset[i]
            self.assertEqual(difficulty[7], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_frames_second_chart_ok(self):
        for i in irange(101,301):
            frames, _, _ = self.dataset[i]
            b = i-101
            f = self.context_around(b+150)
            np.testing.assert_array_equal(frames, f)

    def test_steps_second_chart_ok(self):
        step_indices = set()

        for i in irange(101,301):
            _, _, y = self.dataset[i]
            if y:
                step_indices.add(i)

        self.assertEqual(len(step_indices), 4)
        self.assertEqual(step_indices, {101, 201, 226, 301})

    ## third chart

    def test_difficulty_third_chart_ok(self):
        for i in irange(302, 792):
            _, difficulty, _ = self.dataset[i]
            self.assertEqual(difficulty[11], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_third_chart_ok(self):
        step_indices = set()

        for i in irange(302, 792):
            _, _, y = self.dataset[i]

            if y:
                step_indices.add(i)

        self.assertEqual(step_indices, {302, 351, 792})

    def test_frames_third_chart_ok(self):
        for i in irange(302,792):
            frames, _, _, = self.dataset[i]
            b = i-302
            f = self.context_around(b+1)
            np.testing.assert_array_equal(frames, f)


class TestLSTMOnsetDatasetUnrollN(unittest.TestCase):

    def test_all_ok(self):
        stepfiles, audios, _ = sample_input()
        
        dataset1 = loading.PumpItUpConvolutionLSTMOnsetDataset(stepfiles, audios, 1)
        dataset_block = loading.PumpItUpConvolutionLSTMOnsetDataset(stepfiles, audios, 20)

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

        dataset = loading.PumpItUpConvolutionLSTMOnsetDataset(stepfiles, audios, 20, loading.MaskAndPaddingTransform(20, skip=1))

        for i in range(len(dataset)):
            frames, diff, y, mask = dataset[i]

            self.assertEqual(frames.shape, (20, 15, 3))
            self.assertEqual(diff.shape, (25,))
            self.assertEqual(y.shape, (20,))
            self.assertEqual(mask.shape, (20,))



def irange(a, b):
    return range(a, b+1)

if __name__ == '__main__':
    unittest.main()
