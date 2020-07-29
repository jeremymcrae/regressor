# cython: language_level=3, boundscheck=False, emit_linenums=True

from collections import namedtuple

from libcpp cimport bool
from libc.stdint cimport uint32_t

cimport numpy
import numpy
from scipy.special import stdtr

LinResult = namedtuple('LinregressResult', ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr'])

cdef extern from "covariance.h" namespace "regresser":
    cdef struct covmeans:
        float x
        float y
    
    cdef struct covs:
        covmeans avg
        double s_xx
        double s_xy
        double s_yx
        double s_yy
        uint32_t size
    
    covs covariance(float * x, const uint32_t & size_x, float * y, const uint32_t & size_y, bool sampled)

def linregress(numpy.ndarray[numpy.float32_t, mode="c"] x, numpy.ndarray[numpy.float32_t, mode="c"] y, bool sampled_means=False):
    ''' perform simple linear regression on two float32 numpy arrays
    
    Args:
        x: numpy array of x-values (as numpy.float32 dtype)
        y: numpy array of y-values (as numpy.float32 dtype)
        sampled_means: estimate x and y means from samples. Use for faster speed,
            at the trade-off of slightly less precise beta and p-value.
    
    Returns:
        LinregressResult with slope, intercept, r-value, p-value and standard error
    '''
    x = numpy.ascontiguousarray(x)
    y = numpy.ascontiguousarray(y)
    
    vals = covariance(&x[0], len(x), &y[0], len(y), sampled_means)
    
    # the remainder is from the scipy.stats.linregress function
    r_num = vals.s_xy
    r_den = numpy.sqrt(vals.s_xx * vals.s_yy)
    r = 0.0 if r_den == 0.0 else r_num / r_den
    
    # test for numerical error propagation
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0

    df = len(x) - 2
    slope = r_num / vals.s_xx
    intercept = vals.avg.y - slope * vals.avg.x
    if len(x) == 2:
        # handle case when only two points are passed in
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        sterrest = 0.0
    else:
        TINY = 1e-20
        t = r * numpy.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        prob = stdtr(df, -numpy.abs(t)) * 2
        stderr = numpy.sqrt((1 - r ** 2) * vals.s_yy / vals.s_xx / df)
    
    return LinResult(slope, intercept, r, prob, stderr)
