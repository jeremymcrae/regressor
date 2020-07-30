### regressor

This is a small library that tries to run simple linear regression quickly on
modern x86 hardware. This uses vectorized operations to speed up calculating dot
products and means. The input numpy arrays need to be 1D with 32-bit floats.

As a result, this is ~10X faster than scipy.stats.linregress, but only runs
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
this package and scipy.stats.linregress, starting from arrays with 100 elements
up to arrays with 100 million elements. In general, this package performs simple
linear regression in about 1/15th of the time required by scipy.stats.linregress.

![Performance](/docs/performance.svg)

#### Reliability
The regression results from this package match scipy.stats.linregress to within 
4 decimal places (for the slope, intercept, r-value, p-value and standard error).
The graphs below demonstrate this consistency by comparing betas, r-values and 
p-values from this package vs scipy.stats.linregress. These used randomly 
sampled values with varying correlation between the X and Y arrays to assess 
reliability across a wide range of P-values.

![Reliability](/docs/reliability.svg)
