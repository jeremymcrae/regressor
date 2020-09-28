
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

class TestRegressor(unittest.TestCase):
    ''' check that the results from this package match scipy.stats.linregress
    '''
    def test_linregress(self):
        ''' regressor.linregress matches scipy.stats.linregress
        '''
        # check across a wide range of correlation
        n = 500
        for x, r2 in enumerate(numpy.linspace(0, 1, 1000)):
            a, b = make_correlated(n, r2)

            old = linregress(a, b)
            new = linregress_fast(a, b)

            self.assertAlmostEqual(old.slope, new.slope, places=5)
            self.assertAlmostEqual(old.intercept, new.intercept, places=5)
            self.assertAlmostEqual(old.rvalue, new.rvalue, places=5)
            self.assertAlmostEqual(old.pvalue, new.pvalue, places=5)
            self.assertAlmostEqual(old.stderr, new.stderr, delta=1e-5)

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
    
    def test_linregress_no_variance(self)
        ''' results are nan when all array values are identical
        '''
        a = numpy.ones(20, dtype=numpy.float32)
        res = linregress_fast(a, a)
        self.assertTrue(numpy.isnan(res.slope))
        self.assertTrue(numpy.isnan(res.intercept))
        self.assertTrue(numpy.isnan(res.rvalue))
        self.assertTrue(numpy.isnan(res.pvalue))
        self.assertTrue(numpy.isnan(res.stderr))

    def test_linregress_large(self):
        ''' regressor.linregress matches scipy.stats.linregress
        '''
        # check across a wide range of correlation
        n = 500000
        for x, r2 in enumerate(numpy.linspace(0, 1, 100)):
            a, b = make_correlated(n, r2)

            old = linregress(a, b)
            new = linregress_fast(a, b)

            self.assertAlmostEqual(old.slope, new.slope, places=4)
            self.assertAlmostEqual(old.intercept, new.intercept, places=4)
            self.assertAlmostEqual(old.rvalue, new.rvalue, places=4)
            self.assertAlmostEqual(old.pvalue, new.pvalue, places=4)
            self.assertAlmostEqual(old.stderr, new.stderr, delta=1e-6)
