"""
BPF

A package implementing piece-wise interpolation functions in Cython
"""
from __future__ import print_function
from setuptools import setup
from setuptools import Extension
# from Cython.Distutils import build_ext
import os
import sys


def get_version():
    d = {}
    with open("bpf4/version.py") as f:    
        code = f.read()
    exec(code, d)        
    version = d.get('__version__', (0, 0, 0))
    return ("%d.%d.%d" % version).strip()

if not os.path.exists("README.md"):
    long_description = ""
else:
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except (IOError, ImportError):
        print("Could not convert README to RST")
        long_description = open('README.md').read()
    
compiler_args = [] 
versionstr = get_version()

if sys.platform == 'windows':
    compiler_args += ["-march=i686"]  # This is needed in windows to compile cleanly


class get_numpy_include(str):

    def __str__(self):
        import numpy
        return numpy.get_include()


def get_includes():
    return ['bpf4', get_numpy_include()]


setup(
    name = "bpf4",
    setup_requires = [
        'setuptools>=18.0', 
        'cython>=0.21', 
        'numpy>=1.8',
    ],
    ext_modules = [
        Extension("bpf4.core", 
        sources=["bpf4/core.pyx"], 
        include_dirs=get_includes(), 
        extra_link_args=compiler_args, 
        compiler_args=compiler_args),  
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
    description = "Peace-wise interpolation and lazy evaluation in cython"
)

print("Version: %s" % versionstr)
