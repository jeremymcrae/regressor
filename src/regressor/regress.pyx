# cython: language_level=3, boundscheck=False, emit_linenums=True

from libcpp cimport bool
from libc.stdint cimport uint32_t

import numpy
from jax.lax_linalg import svd
from scipy.special import stdtr

cdef extern from "covariance.h" namespace "regressor":
    cdef struct covmeans:
        double x
        double y
    
    cdef struct covs:
        covmeans avg
        double s_xx
        double s_xy
        double s_yx
        double s_yy
        uint32_t size
    
    covs covariance(float * x, const uint32_t & size_x, float * y, const uint32_t & size_y, bool sampled) except +

class LinregressResult:
    def __init__(self, slope, intercept, rvalue, pvalue, stderr):
        self.slope = slope
        self.intercept = intercept
        self.rvalue = rvalue
        self.pvalue = pvalue
        self.stderr = stderr
    
    def __repr__(self):
        return f'LinregressResult(slope={self.beta}, intercept={self.intercept}, ' \
            f'rvalue={self.rvalue}, pvalue={self.pvalue}, stderr={self.stderr})'
    
    def __iter__(self):
        for x in [self.beta, self.intercept, self.rvalue, self.pvalue, self.stderr]:
            yield x
    
    @property
    def beta(self):
        ''' alternative name for the slope
        '''
        return self.slope
    @property
    def coef_(self):
        ''' alternative name for beta, used by scikit LinearRegression
        '''
        return self.beta
    @property
    def params(self):
        ''' alternative name for beta, used by statsmodels OLS
        '''
        return self.beta
    
    @property
    def bse(self):
        ''' alternative name for standard errors attribute, used by statsmodels
        '''
        return self.stderr

def linregress_simple(float[::1] x, float[::1] y, bool sampled_means=False):
    ''' perform simple linear regression on two float32 numpy arrays
    
    Args:
        x: numpy array of x-values (as numpy.float32 dtype)
        y: numpy array of y-values (as numpy.float32 dtype)
        sampled_means: estimate x and y means from samples. Use for faster speed,
            at the trade-off of slightly less precise beta and p-value.
    
    Returns:
        LinregressResult with slope, intercept, r-value, p-value and standard error
    '''
    vals = covariance(&x[0], len(x), &y[0], len(y), sampled_means)
    
    s_xx = vals.s_xx if vals.s_xx != 0 else float('nan')
    
    # the remainder is from the scipy.stats.linregress function
    r_num = vals.s_xy
    r_den = numpy.sqrt(s_xx * vals.s_yy)
    r = 0.0 if r_den == 0.0 else r_num / r_den
    
    # test for numerical error propagation
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0

    df = len(x) - 2
    slope = r_num / s_xx
    intercept = vals.avg.y - slope * vals.avg.x
    if len(x) == 2:
        # handle case when only two points are passed in
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        stderr = 0.0
    else:
        TINY = 1e-20
        t = r * numpy.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        prob = stdtr(df, -numpy.abs(t)) * 2
        stderr = numpy.sqrt((1 - r ** 2) * vals.s_yy / s_xx / df)
    
    return LinregressResult(slope, intercept, r, prob, stderr)

def linregress_full(float[:, ::1] endog, float[:] exog, bool has_intercept=False):
    ''' run a linear regression with covariates
    
    This is much faster than running an OLS regression with statsmodels, and
    on par with the scipy.linalg.lstsq(), except this also calculates standard
    errors of the betas and p-values. If stderrs are required, this function is 
    faster than running scipy.stats.lstsq before calculating stderrs, since those
    would require calculating the endog.T @ endog dot product again.
    '''
    if not has_intercept:
        endog = numpy.concatenate([endog, numpy.ones(len(endog), dtype=numpy.float32)[:, None]], axis=1)
    
    # compute betas. Store one dot product, to avoid later recomputation
    dotted = numpy.dot(endog.T, endog)
    betas = numpy.linalg.solve(dotted, numpy.dot(endog.T, exog))
    
    # use jax.lax_linalg.svd() for singular value decomposition, since this
    # in turn calls the BCDSVD function from Eigen, which is fast
    singular = svd(numpy.asarray(endog), full_matrices=False, compute_uv=False)
    
    # get the matrix rank from the SVD values. The shape is reversed compared to numpy code
    rank = endog.shape[0] if endog.shape[0] > endog.shape[1] else endog.shape[1]
    tol = singular.max(axis=-1, keepdims=True) * rank * numpy.finfo(singular.dtype).eps
    matrix_rank = (singular > tol).sum(axis=-1)
    
    # calculate the degrees of freedom of the residuals
    df_resid = endog.shape[0] - matrix_rank
    
    predicted = numpy.dot(endog, betas)
    residuals = exog - predicted
    scale = numpy.dot(residuals, residuals) / df_resid
    cov_params = numpy.linalg.inv(dotted)
    
    stderr = numpy.sqrt(numpy.diag(scale * cov_params))
    
    p_values = numpy.array([stdtr(df_resid, -numpy.abs(a / b)) for a, b in zip(betas, stderr)]) * 2
    
    # give nan values for intercepts and rvalues, which are just present to match
    # the simple linear regression attributes
    intercepts = numpy.array([float('nan') * len(stderr)])
    rvalues = numpy.array([float('nan') * len(stderr)])
    
    return LinregressResult(betas, intercepts, rvalues, p_values, stderr)

def linregress(endog, exog, bool has_intercept=False):
    ''' run linear regression on numpy arrays
    '''
    assert exog.ndim == 1
    if endog.ndim == 1:
        return linregress_simple(endog, exog)
    else:
        return linregress_full(endog, exog, has_intercept)
