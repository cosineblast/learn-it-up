import unittest

import ssc_util

from hypothesis import given, strategies as st

class TestSSCUtil(unittest.TestCase):

    @given(st.integers(1, 25))
    def test_get_difficulty_works_single_number(self, n):
        self.assertEqual(ssc_util._get_difficulty(f'S{n}'), n)

    @given(st.integers(1, 25))
    def test_get_difficulty_works_underscore_number(self, n):
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_V'), n)
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_H'), n)
        self.assertEqual(ssc_util._get_difficulty(f'S{n}_VH'), n)

if __name__ == '__main__':
    unittest.main()
