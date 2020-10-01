### regressor

This is a small library that tries to run simple linear regression quickly on
modern x86 hardware. This uses vectorized operations to speed up calculating dot
products and means. The input numpy arrays need to be 1D with 32-bit floats.

As a result, this is ~20X faster than scipy.stats.linregress, but only runs
on x86-64 hardware with AVX extensions (most desktops and servers as of 2020).

### Install
```sh
pip install regressor
```

### Usage
```py
>>> import numpy as np
>>> from regressor import linregress
>>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
>>> y = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
>>> linregress(x, y)
LinregressResult(slope=-1.0, intercept=6.0, rvalue=-1.0, pvalue=1.2e-30, stderr=0.0)
```

### Performance
The graph below compares the times required for simple linear regressions between
this package and scipy.stats.linregress, starting from arrays with 10 elements
up to arrays with 100 million elements. This was run on a 2.6 GHz Skylake CPU.
In general, this package performs simple linear regression in about 1/20th of
the time required by scipy.stats.linregress.

![Performance](/docs/performance.svg)

#### Reliability
The regression results from this package match scipy.stats.linregress to within 
4 decimal places (for the slope, intercept, r-value, p-value and standard error).
The graphs below demonstrate this consistency by comparing betas, r-values and 
p-values from this package vs scipy.stats.linregress. These used randomly 
sampled values with varying correlation between the X and Y arrays to assess 
reliability across a wide range of P-values.

![Reliability](/docs/reliability.svg)

I could only find one scenario where the behavior of this package differs from
scipy.stats.linregress - when you regress a small array with itself, the p-value
is naturally very small. When regressing again with itself incremented slightly
(e.g. array + 1.2e-7), we expect the same slope and p-value, but the intercept
should be shifted up by the incremented value. However, the p-value can diverge
due to imprecision from float addition. scipy.stats.linregress is also affected 
by this, but to a lesser degree. The divergence only occurs with some input 
arrays of random numbers, about 55% of runs in my tests, depending on the
input array size. Here's some code to demonstrate the issue:

```py
>>> import numpy
>>> from scipy.stats import linregress
>>> from regressor import linregress as linreg2

>>> a = numpy.array([0.49789444, 0.12506859, 0.75386035, 0.025621228, 0.00039564757,
        0.71248668, 0.078348994, 0.62318009, 0.48770180], dtype=numpy.float32)
>>> b = numpy.copy(a)
>>> eps = numpy.finfo(numpy.float32).eps

>>> linreg2(a, b)
LinregressResult(slope=1.0, intercept=0.0, rvalue=1.0, pvalue=3.292585384803146e-70, stderr=0.0)
>>> linreg2(a, b + eps)
LinregressResult(slope=0.9999999999999959, intercept=9.271833784074701e-08, 
    rvalue=0.9999999999999959, pvalue=1.4627920285341798e-50, stderr=3.425878486341894e-08)

>>> linregress(a, b)
LinregressResult(slope=1.0, intercept=0.0, rvalue=1.0, pvalue=3.292585384803146e-70, stderr=0.0)
>>> linregress(a, b + eps)
LinregressResult(slope=1.0, intercept=1.1920928955078125e-07, rvalue=1.0, 
    pvalue=3.292585384803146e-70, stderr=0.0)
```

This behavior only occurs when the input arrays have at least 9 values (and
becomes irrelevant with arrays with more than 50 values, since those have p <
1e-323). It only matters if the input values are perfectly correlated, even if
one value differs slightly, then the results are very similar. Again, here's
some code to demonstrate:

```py
>>> b[0] += eps
>>> linreg2(a, b)
LinregressResult(slope=1.000000020527417, intercept=-7.537115487288304e-09, 
    rvalue=0.9999999999999909, pvalue=2.37040745003888e-49, stderr=5.100092205240057e-08)
>>> linreg2(a, b)
LinregressResult(slope=1.0000000205274189, intercept=2.2265206844895857e-08, 
    rvalue=0.9999999999999918, pvalue=1.6549532101768438e-49, stderr=4.844923917880737e-08)
```

Ignore the different intercepts, regressor or scipy.stats.linregress are both
wrong, using multiples of eps slowly converged, but with jumpy steps.

I won't work around the larger issue, since it only alters the result when the
regressed arrays are identical, but one has also been adjusted by adding a scalar
to all entries. I can't see any scenario where this would occur other than on
purpose.
