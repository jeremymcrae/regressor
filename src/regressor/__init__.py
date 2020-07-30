from pkg_resources import get_distribution

__version__ = get_distribution('regressor').version

from regressor.regress import linregress
