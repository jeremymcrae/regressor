from pkg_resources import get_distribution

__version__ = get_distribution('regresser').version

from regresser.cov import linregress
