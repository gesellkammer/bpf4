"""
BPF

A package implementing piece-wise interpolation functions in Cython
"""
from __future__ import print_function
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
import numpy as np
import os
import sys

USE_CYTHON = True  


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
    
cmdclass     = {}
ext_modules  = []
compile_args = [] 

if sys.platform == 'windows':
    compile_args += ["-march=i686"]  # This is needed in windows to compile cleanly

extra_link_args = compile_args 
versionstr = get_version()

include_dirs = [
    np.get_include(),
    'bpf4'
]

ext_common_args = {
    'extra_compile_args': compile_args,
    'extra_link_args': extra_link_args, 
    'include_dirs': include_dirs
}

ext_modules += [
    Extension(
        "bpf4.core", 
        [ "bpf4/core.pyx" ], 
        **ext_common_args
        ),
    ]

cmdclass.update({ 'build_ext': build_ext })

setup(
    name = "bpf4",
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    include_dirs = [np.get_include()],
    setup_requires = ['cython>=0.19', 'numpy>=1.7'],
    install_requires = ['numpy>=1.7', 'matplotlib'],
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
