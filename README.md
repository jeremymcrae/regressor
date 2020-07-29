### regresser

This is a small library that tries to run simple linear regression quickly on
64 bit x86 hardware. This uses AVX vectorization to speed up calculations.
This is ~10X faster than scipy.stats.linregress when using large arrays (>5000
elements per array).

### Install
```sh
pip install regresser
```

### Usage
```py
import numpy as np
from regresser import linregress
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
y = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
linregress(x, y)
```
