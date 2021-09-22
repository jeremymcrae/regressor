from pkg_resources import get_distribution

__version__ = get_distribution('regressor').version

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass

from regressor.regress import linregress
