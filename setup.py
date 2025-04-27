"""
BPF

A package implementing piece-wise interpolation functions in Cython
"""
from setuptools import setup, Extension
import numpy


setup(
    name = "bpf4",
    python_requires='>=3.10',
    ext_modules = [
        Extension(
            "bpf4.core",
            sources=["bpf4/core.pyx"],
            include_dirs = ["bpf4", numpy.get_include()]
        )
    ],
    install_requires = ['numpy>=1.8', 'matplotlib', 'scipy', 'visvalingamwyatt'],
    packages = ['bpf4'],
    # package_data={'bpf4': ['core.pyi', '__init__.pyi', 'py.typed']},
    package_data={'bpf4': ['__init__.pyi', 'core.pyo', 'py.typed']},
    include_package_data=True,
    license='GPLv3'
)
