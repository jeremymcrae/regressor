from importlib.metadata import version

__name__ == 'regressor'
__version__ = version(__name__)

try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass

from regressor.regress import linregress
