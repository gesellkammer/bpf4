BPF4
====

About
-----

bpf4 is a library for curve fitting and break point functions in python.
It is mainly programmed in Cython for efficiency. It has been used in
itself for functional and numerical analysis.

Example
-------

Find the intersection between two curves

::

    >>> from bpf4 import bpf  # this imports the api
    >>> import pylab
    >>> a = bpf.spline((0, 0), (1, 5), (2, 3), (5, 10))  # each point (x, y)
    >>> b = bpf.expon((0, -10), (2,15), (5, 3), exp=3)
    >>> a.plot() # uses matplotlib
    >>> b.plot() 
    >>> zeros = (a - b).zeros()
    >>> pylab.plot(zeros, a.map(zeros), 'o')

.. figure:: https://github.com/gesellkammer/bpf4/raw/master/pics/zeros.png
   :alt: 1

   1

Features
--------

Many interpolation types besides linear:

-  spline
-  half-cosine
-  exponential
-  fibonacci
-  exponantial half-cosine
-  pchip
-  logarithmic
-  etc.

| Interpolation types can be mixed, so that each segment has a different
  interpolation (with the exception of spline interpolation)
| Curves can be combined non-destructively. Following from the example
  above.

::

    >>> c = (a + b).sin().abs()
    >>> c[1.5:4].plot()  # plot only the range (1.5, 4)

.. figure:: https://github.com/gesellkammer/bpf4/raw/master/pics/sinabs.png
   :alt: 2

   2

Syntax support for shifting, scaling and slicing a bpf

::

    >>> a >> 2        # a shifted to the right
    >>> (a * 5) ^ 2   # scale the x coord by 2, scale the y coord by 5
    >>> a[2:2.5]      # slice only a portion of the bpf
    >>> a[::0.01]     # sample the bpf with an interval of 0.01

| Derivation and Integration: ``c.derivative().plot()`` or
  ``c.integrated().integrated().plot()``
| Numerical integration: ``c.integrate_between(2, 4)``

Dependencies
------------

-  cython >= 0.19
-  numpy

Installation
------------

To install the latest version (recommended)

::

    $> git clone https://github.com/gesellkammer/bpf4.git
    $> cd bpf4
    $> python setup.py install

Otherwise, via pip or easy\_install

::

    $> pip install bpf4
