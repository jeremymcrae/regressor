### regressor

This is a small library that tries to run simple linear regression quickly on
modern x86 hardware. This uses vectorized operations to speed up calculating dot products and means.

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
