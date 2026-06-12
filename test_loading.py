
import unittest
import numpy as np
import loading
from ssc_util import RefinedStepFile, RefinedChart, StepInfo

class TestCNNDataset(unittest.TestCase):

    def setUp(self):

        # This loader should not care about any beat-related info,
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

        def step_at(time):
            return StepInfo(time_in_beats=0.0, time_in_seconds=time, stepcode='10000')

        def stepfile_of(charts, title='some file'):
            return RefinedStepFile(
                info={},
                music='idk.mp3',
                title=title,
                charts=charts
            )

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

        paths = ['idk.ssc.bin', 'idk2.ssc.bin']

        # 5 seconds of audio features
        audio = np.zeros((3, 500)) + np.arange(500)
        audio = audio.transpose(1, 0)

        self.audio = audio
        audio_len = audio.shape[0]

        padding = np.ones((7, 3)) * loading.DEFAULT_VALUE
        padded_audio = np.concat([padding, audio, padding])

        def load_audio(path):
            return loading.FeatureView(padded_audio, 7, audio_len)

        dataset = loading.PumpItUpConvolutionCNNOnsetDataset(stepfiles, paths, load_audio)

        self.dataset = dataset

    def context_around(self, i):
        result = np.zeros((15, 3))

        k = i-7
        j = 0

        while k <= i+7:
            if k < 0 or k >= self.audio.shape[0]:
                result[j] = loading.DEFAULT_VALUE
            else:
                result[j] = self.audio[k]
            k += 1
            j += 1

        return result
        

    def test_stats_ok(self):
        self.assertEqual(self.dataset.stepfile_stats[0].chart_stats[0].len_frames, 101)
        self.assertEqual(self.dataset.stepfile_stats[0].chart_stats[1].len_frames, 201)
        self.assertEqual(self.dataset.stepfile_stats[0].len_frames, 302)
        self.assertEqual(len(self.dataset), 793)

    def test_difficulty_first_chart_ok(self):
        for i in irange(0, 100):
            (_, difficulty), _, = self.dataset[i]
            self.assertEqual(difficulty[1], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_first_chart_ok(self):
        step_indices = set()

        for i in irange(0, 100):
            (_, _), y, = self.dataset[i]
            if y:
                step_indices.add(i)

        self.assertEqual(step_indices, {0, 50, 75, 83, 94, 100})


    def test_frames_first_chart_ok(self):
        for i in irange(0, 100):
            (frames, _), _, = self.dataset[i]
            f = self.context_around(i+100)
            np.testing.assert_array_equal(frames, f)

    ## second chart

    def test_difficulty_second_chart_ok(self):
        for i in irange(101,301):
            (_, difficulty), _, = self.dataset[i]
            self.assertEqual(difficulty[7], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_frames_second_chart_ok(self):
        for i in irange(101,301):
            (frames, _), _, = self.dataset[i]
            b = i-101
            f = self.context_around(b+150)
            np.testing.assert_array_equal(frames, f)

    def test_steps_second_chart_ok(self):
        step_indices = set()

        for i in irange(101,301):
            (_, _), y, = self.dataset[i]
            if y:
                step_indices.add(i)

        self.assertEqual(len(step_indices), 4)
        self.assertEqual(step_indices, {101, 201, 226, 301})

    ## third chart

    def test_difficulty_third_chart_ok(self):
        for i in irange(302, 792):
            (_, difficulty), _, = self.dataset[i]
            self.assertEqual(difficulty[11], 1)
            self.assertEqual(np.sum(difficulty), 1)

    def test_steps_third_chart_ok(self):
        step_indices = set()

        for i in irange(302, 792):
            (_, _), y, = self.dataset[i]
            if y:
                step_indices.add(i)

        self.assertEqual(step_indices, {302, 351, 792})

    def test_frames_third_chart_ok(self):
        for i in irange(302,792):
            (frames, _), _, = self.dataset[i]
            b = i-302
            f = self.context_around(b+1)
            np.testing.assert_array_equal(frames, f)


def irange(a, b):
    return range(a, b+1)




        


if __name__ == '__main__':
    unittest.main()
