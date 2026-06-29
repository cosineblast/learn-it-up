import unittest
import ssc_util

import numpy as np

from hypothesis import given, strategies as st, assume, example

from operator import itemgetter

def st_beats():
    return st.floats(0.0, 100.0)
def st_bpms():
    return st.floats(1.0, 240.0)

@st.composite
def st_bpm_pairs(draw, *, start_at_zero=False):
    stuff = draw(st.lists(st.tuples(st_beats(), st_bpms()), min_size=1)) 

    if start_at_zero:
        min_entry = min(stuff, key=itemgetter(0))
        if min_entry[0] != 0.0:
            stuff.append((0.0, min_entry[1]))

    stuff = list({pair[0]: pair for pair in stuff}.values())

    stuff.sort(key=itemgetter(0))
    
    return stuff

    

EPS = 0.001

class TestSSCUtil(unittest.TestCase):

    @given(st.integers(1, 25))
    def test_get_difficulty_works_single_number(self, n):
        self.assertEqual(ssc_util._get_difficulty(f'S{n}'), n)

    @given(st.integers(1, 25))
    def test_get_difficulty_works_underscore_number(self, n):
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_V'), n)
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_H'), n)
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_VH'), n)

    @given(st.lists(st_beats(), min_size=1).map(sorted), st_bpms(), st.floats(1.0, 100.0))
    def test_average_bpm_constant_bpm_returns_constant(self, beats, bpm, end_offset):
        bpms = [(beat, bpm) for beat in beats]
        end = max(beats)+end_offset
        self.assertAlmostEqual(ssc_util.find_average_bpm(bpms, end), bpm, delta=EPS)

    @given(st_bpm_pairs(), st.floats(1.0, 100.0))
    def test_average_bpm_within_bpms(self, pairs, end_offset):
        beats, bpms = zip(*pairs)

        end = max(beats)+end_offset

        result = ssc_util.find_average_bpm(pairs, end)

        self.assertLessEqual(min(bpms)-EPS, result)
        self.assertLessEqual(result, max(bpms)+EPS)

    @given(st_bpm_pairs(), st.floats(1.0, 100.0))
    def test_average_bpm_immune_to_outliers(self, pairs, end_offset):
        beats, bpms = zip(*pairs)
        beats = list(beats)
        bpms = list(bpms)

        next_beat = max(beats)+end_offset

        old_avg = ssc_util.find_average_bpm(pairs, next_beat)

        beats.append(next_beat)
        bpms.append(1e8)

        # the new segment represents at most one percent of the full length
        end = next_beat + (next_beat-min(beats)) / 100

        new_avg = ssc_util.find_average_bpm(list(zip(beats, bpms)), end)

        self.assertLessEqual(old_avg, new_avg)
        self.assertLessEqual(new_avg, old_avg + old_avg/100 + EPS)

    @given(st_bpm_pairs(start_at_zero=True), st.floats(1.0, 100.0))
    def test_average_bpm_has_same_time_as_original(self, pairs, end_offset):
        beats, bpms = zip(*pairs)
        beats = list(beats)
        bpms = list(bpms)

        next_beat = max(beats)+end_offset

        result = ssc_util.find_average_bpm(pairs, next_beat)

        segments = ssc_util._compute_segment_durations(pairs)

        full_duration = ssc_util._compute_beat_absolute_time(0.0, pairs, segments, next_beat)

        self.assertAlmostEqual(full_duration, 60/result * next_beat, delta=EPS)


    @given(st_bpm_pairs(start_at_zero=True), st.lists(st.floats(0.0, 1.0)).map(sorted), st.floats(-2.0, 2.0))
    def test_compute_multiple_beats_works(self, pairs, target_ratios, offset):
        beats, bpms = zip(*pairs)
        beats = list(beats)
        bpms = list(bpms)

        target_beats = [max(beats) * ratio for ratio in target_ratios]

        segments = ssc_util._compute_segment_durations(pairs)

        expected_times = [ssc_util._compute_beat_absolute_time(offset, pairs, segments, beat) for beat in target_beats]

        computed_times = ssc_util.compute_multiple_beats_absolute_times(offset, pairs, target_beats)

        np.testing.assert_array_almost_equal(np.array(expected_times), np.array(computed_times))
        






if __name__ == '__main__':
    unittest.main()
