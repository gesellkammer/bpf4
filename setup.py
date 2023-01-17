"""
BPF

A package implementing piece-wise interpolation functions in Cython
"""
from __future__ import print_function
from setuptools import setup, Extension
import os
import sys


def get_version():
    d = {}
    with open("bpf4/version.py") as f:    
        code = f.read()
    exec(code, d)        
    version = d.get('__version__', (0, 0, 0))
    return ("%d.%d.%d" % version).strip()

long_description = open('README.md').read()
    
versionstr = get_version()


class get_numpy_include(str):

    def __str__(self):
        import numpy
        return numpy.get_include()


def get_includes():
    return ['bpf4', get_numpy_include()]


setup(
    name = "bpf4",
    python_requires='>=3.9',
    ext_modules = [
        Extension(
            "bpf4.core",
            sources=["bpf4/core.pyx"],
            include_dirs=get_includes()
        )
    ],
    include_dirs = get_includes(),
    install_requires = ['numpy>=1.8', 'matplotlib', 'scipy'],
    packages = ['bpf4'],

    # metadata
    version          = versionstr,
    url              = 'https://github.com/gesellkammer/bpf4',
    download_url     = 'https://github.com/gesellkammer/bpf4', 
    author           = 'eduardo moguillansky',
    author_email     = 'eduardo.moguillansky@gmail.com',
    maintainer       = '',
    maintainer_email = '',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    description = "Piece-wise interpolation and lazy evaluation in cython"
)
