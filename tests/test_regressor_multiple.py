
import unittest
from time import time

import numpy
from scipy.linalg import lstsq
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression

from regressor import linregress

def difference(first, second):
    return abs(first - second)

def relative_diff(first, second):
    delta = first / second
    below = delta < 1
    delta[below] = 1 / delta[below]
    return delta

def make_arrays(n, k):
    ''' make two arrays or random numbers, one for x-values, one for y-values
    
    We make this easier for the fits by formatting as 32-bit floats in fortran 
    ordered arrays.
    
    Returns:
        tuple of (x_values, y_values, x_values with an intercept column)
    '''
    x = numpy.asfortranarray(numpy.random.random((n, k)).astype(numpy.float32))
    y = numpy.asfortranarray(numpy.random.random(n).astype(numpy.float32))
    x_with_intercept = numpy.concatenate([x, numpy.ones(len(x), dtype=numpy.float32)[:, None]], axis=1)
    return x, y, x_with_intercept

class TestRegressorMulitple(unittest.TestCase):
    ''' check that the results from this package match other regression systems
    '''
    def test_multiple_regression(self):
        ''' check a single large regression with many different linear models.
        
        Skip p-value checks for other tests, as statsmodels is too slow. Also,
        this uses arrasy of random numbers, i.e. random betas and p-values.
        '''
        x, y, x_with_intercept = make_arrays(n=5000, k=250)
        
        # mainly compare lstsq from scipy with regressor, but lstsq lacks p-values
        # so we need statsnmodels OLS for p-values, but that is very slow
        lstsq_fit = lstsq(x_with_intercept, y)
        regressor_fit = linregress(x, y)
        sm_fit = OLS(y, x_with_intercept).fit()
        sk_fit = LinearRegression().fit(x, y)
        
        # check that the betas are very tightly correlated
        corr = numpy.corrcoef(lstsq_fit[0][:-1], regressor_fit.coef_[:-1])[0, 1] ** 2
        self.assertTrue(corr > 0.99999)
        
        # check that lstsq betas correlate with sk_fit betas for extra sanity
        corr = numpy.corrcoef(lstsq_fit[0][:-1], sk_fit.coef_)[0, 1] ** 2
        self.assertTrue(corr > 0.99999)
        
        # check the beta values are very close. They aren't identical, as this
        # package uses 32-bit floats, but the others convert to 64-bit doubles.
        # Differences should be on the order of 1e-8, which is the usual delta 
        # between a 64-bit float and its 32-bit representation (for values 
        # around 0.5). Float differences accumulate to around 2e-6 at most, 
        # which makes a bigger relative difference for betas near zero.
        abs_diff = difference(lstsq_fit[0][:-1], regressor_fit.coef_[:-1])
        self.assertTrue(abs_diff.max() < 2e-5)
        
        # check the p-values are nearly identical in log10 space, and correlate
        p_delta = abs(numpy.log10(regressor_fit.pvalue) - numpy.log10(sm_fit.pvalues))
        self.assertTrue(p_delta.max() < 4e-3)
        corr = numpy.corrcoef(numpy.log10(regressor_fit.pvalue), numpy.log10(sm_fit.pvalues))[0, 1] ** 2
        self.assertTrue(corr > 0.99999)
    
    def test_multiple_regression_big_small(self):
        ''' test multiple regression with a small array, and a large array
        '''
        for n in [50, 500000]:
            x, y, x_with_intercept = make_arrays(n=n, k=10)
            
            lstsq_fit = lstsq(x_with_intercept, y)
            regressor_fit = linregress(x, y)
            
            # check the betas are tightly correlated
            corr = numpy.corrcoef(lstsq_fit[0][:-1], regressor_fit.coef_[:-1])[0, 1] ** 2
            self.assertTrue(corr > 0.997)
            
            abs_diff = difference(lstsq_fit[0][:-1], regressor_fit.coef_[:-1])
            self.assertTrue(abs_diff.max() < 1e-4, msg=f'max abs_diff={abs_diff.max()} for n={n}')
    
    def test_regression_correlated(self):
        ''' check multiple regresion, where y-values depend on the x columns
        '''
        x, y, x_with_intercept = make_arrays(n=50000, k=10)
        
        # define some effect sizes (which decline across the columns)
        betas = numpy.logspace(0, -10, num=x.shape[1])
        y = (x * betas).sum(axis=1)
        
        lstsq_fit = lstsq(x_with_intercept, y)
        regressor_fit = linregress(x, y)
        
        # check differences versus the predefined betas
        diff = difference(betas, regressor_fit.coef_[:-1])
        self.assertTrue(diff.max() < 5e-5)
        
        corr = numpy.corrcoef(betas, regressor_fit.coef_[:-1])[0, 1] ** 2
        self.assertTrue(corr > 0.9999999)
        
        # and check difference versus lstsq fit
        corr = numpy.corrcoef(lstsq_fit[0][:-1], regressor_fit.coef_[:-1])[0, 1] ** 2
        self.assertTrue(corr > 0.9999999)
