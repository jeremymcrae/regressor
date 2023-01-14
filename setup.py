
import io
from setuptools import setup
import sys
import platform

from distutils.core import Extension
from Cython.Build import cythonize
import numpy

EXTRA_COMPILE_ARGS = ['-std=c++11', '-O2']
EXTRA_LINK_ARGS = []
if sys.platform == "darwin":
    EXTRA_COMPILE_ARGS += [
        "-stdlib=libc++",
        "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1",
        "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
        ]
    EXTRA_LINK_ARGS += [
        "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
        ]

if platform.machine() == 'x86_64':
    EXTRA_COMPILE_ARGS += ['-mavx', '-mavx2', '-mfma']

ext = cythonize([
    Extension('regressor.regress',
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        sources=['src/regressor/regress.pyx',
            'src/covariance.cpp',
            ],
        include_dirs=['src/', numpy.get_include()],
        language='c++'),
    ])

setup(name='regressor',
    description='Package for fast regression',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    version='1.0.7',
    author='Jeremy McRae',
    author_email='jmcrae@illumina.com',
    license="MIT",
    url='https://github.com/jeremymcrae/regressor',
    packages=['regressor'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
    ],
    tests_require=[
        'statsmodels',
        'scikit-learn',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    ext_modules=ext,
    test_loader='unittest:TestLoader',
    )
