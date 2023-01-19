
import unittest

import numpy
from scipy.stats import linregress
from regressor import linregress as linregress_fast

def make_correlated(n: int, r2: float):
    ''' generate numpy arrays, with varying correlation between them

    Args:
        n: length of arrays
        r2: approximate r-squared

    Returns:
        tuple of numpy arrays
    '''
    x = numpy.random.random(n)
    y = numpy.random.random(n)
    y -= r2 * (y - x)  # shrink values in 2nd array towards the first
    return x.astype(numpy.float32), y.astype(numpy.float32)

def get_delta(old, new):
    ''' get relative and absolute difference between two numeric values

    Ensures greater numeric similarity than what the standard checks permit.
    '''
    # check the values have the same sign
    assert ((old == new) & (old == 0)) | ((old >= 0) & (new >= 0)) | ((old * new) > 0)
    old = abs(old)
    new = abs(new)

    hi = max(old, new)
    lo = min(old, new)

    relative = 1.0 if lo == 0 else hi / lo
    absolute = hi - lo

    return relative, absolute

class TestRegressor(unittest.TestCase):
    ''' check that the results from this package match scipy.stats.linregress
    '''
    def test_linregress(self):
        ''' regressor.linregress matches scipy.stats.linregress
        '''
        # check across a wide range of array sizes and correlations
        for n in [3, 10, 100, 1000, 10000, 100000]:
            for r2 in numpy.linspace(0, 1, 1000):
                a, b = make_correlated(n, r2)
                b += 1

                old = linregress(a, b)
                new = linregress_fast(a, b)

                d_slope, a_slope = get_delta(old.slope, new.slope)
                d_intercept, a_intercept = get_delta(old.intercept, new.intercept)
                d_rvalue, a_rvalue = get_delta(old.rvalue, new.rvalue)
                d_pvalue, a_pvalue = get_delta(old.pvalue, new.pvalue)
                d_stderr, a_stderr = get_delta(old.stderr, new.stderr)

                self.assertTrue(d_slope < 1.00031 or a_slope < 1e-8)
                self.assertTrue(d_intercept < 1.002 or a_intercept < 1e-7)
                self.assertTrue(d_rvalue < 1.0004 or a_rvalue < 1e-8)
                self.assertTrue(d_pvalue < 1.003 or a_pvalue < 1e-8)
                self.assertTrue(a_stderr < 1e-5)

    def test_linregress_small(self):
        ''' check results from small arrays
        '''
        a = numpy.ones(0, dtype=numpy.float32)
        res = linregress_fast(a, a)
        self.assertTrue(numpy.isnan(res.slope))
        self.assertTrue(numpy.isnan(res.intercept))
        self.assertTrue(numpy.isnan(res.rvalue))
        self.assertTrue(numpy.isnan(res.pvalue))
        self.assertTrue(numpy.isnan(res.stderr))

        a = numpy.ones(1, dtype=numpy.float32)
        res = linregress_fast(a, a)
        self.assertTrue(numpy.isnan(res.slope))
        self.assertTrue(numpy.isnan(res.intercept))
        self.assertTrue(numpy.isnan(res.rvalue))
        self.assertTrue(numpy.isnan(res.pvalue))
        self.assertTrue(numpy.isnan(res.stderr))

        a = numpy.ones(2, dtype=numpy.float32)
        res = linregress_fast(a, a)
        self.assertTrue(numpy.isnan(res.slope))
        self.assertTrue(numpy.isnan(res.intercept))
        self.assertTrue(numpy.isnan(res.rvalue))
        self.assertEqual(res.pvalue, 1.0)
        self.assertEqual(res.stderr, 0.0)

    def test_linregress_no_variance(self):
        ''' results are nan when all array values are identical
        '''
        a = numpy.ones(20, dtype=numpy.float32)
        res = linregress_fast(a, a)
        self.assertTrue(numpy.isnan(res.slope))
        self.assertTrue(numpy.isnan(res.intercept))
        self.assertTrue(numpy.isnan(res.rvalue))
        self.assertTrue(numpy.isnan(res.pvalue))
        self.assertTrue(numpy.isnan(res.stderr))

    def test_linregress_small_deviation(self):
        ''' results are close when small arrays differ slightly at one value
        '''
        a = numpy.array([0.49789444, 0.12506859, 0.75386035, 0.025621228,
            0.00039564757, 0.71248668, 0.078348994, 0.62318009, 0.48770180],
            dtype=numpy.float32)

        eps = numpy.finfo(numpy.float32).eps

        for i in range(100):
            b = numpy.copy(a)
            b[0] += eps * i
            old = linregress(a, b)
            new = linregress_fast(a, b)

            d_slope, a_slope = get_delta(old.slope, new.slope)
            d_intercept, a_intercept = get_delta(old.intercept, new.intercept)
            d_rvalue, a_rvalue = get_delta(old.rvalue, new.rvalue)
            d_pvalue, a_pvalue = get_delta(old.pvalue, new.pvalue)
            d_stderr, a_stderr = get_delta(old.stderr, new.stderr)

            self.assertTrue(d_slope < 1.0002)
            self.assertTrue(a_intercept < 1e-7)
            self.assertTrue(d_rvalue < 1.0001)

            # if the pvalues are very low, then they can have a high relative
            # difference, up to 1.5X. but it's the difference between p=1e-45 vs
            # p=1.5e-45, so we allow a small absolute difference as well
            self.assertTrue(d_pvalue < 1.6 or a_pvalue < 1e-45)
            self.assertTrue(a_stderr < 1e-5)

    def test_linregress_large(self):
        ''' regressor.linregress matches scipy.stats.linregress
        '''
        # check across a wide range of correlation
        n = 1000000
        for x, r2 in enumerate(numpy.linspace(0, 1, 100)):
            a, b = make_correlated(n, r2)

            old = linregress(a, b)
            new = linregress_fast(a, b)

            self.assertAlmostEqual(old.slope, new.slope, places=4)
            self.assertAlmostEqual(old.intercept, new.intercept, places=4)
            self.assertAlmostEqual(old.rvalue, new.rvalue, places=4)
            self.assertAlmostEqual(old.pvalue, new.pvalue, places=4)
            self.assertAlmostEqual(old.stderr, new.stderr, delta=1e-6)
