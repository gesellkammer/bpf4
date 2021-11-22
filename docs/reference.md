# Core


---------


## BlendShape

### BlendShape


```python

class BlendShape()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## BpfBase

### BpfBase


```python

class BpfBase()

```


---------


### Methods

#### \_\_init\_\_


xs and ys are arrays of points (x, y)


```python

def __init__() -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## BpfInterface


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## BpfInversionError


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### with\_traceback


set self.__traceback__ to tb and return self.


```python

def with_traceback() -> None

```


---------


### Attributes

**args**

## BpfPointsError


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### with\_traceback


set self.__traceback__ to tb and return self.


```python

def with_traceback() -> None

```


---------


### Attributes

**args**

## Const

### Const


```python

class Const()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Const.mapn_between(self, int n, double x0, double x1, ndarray out=None) -> ndarray


```python

def mapn_between() -> None

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Expon

### Expon


```python

class Expon()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Exponm

### Exponm


```python

class Exponm()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Fib

### Fib


```python

class Fib()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Halfcos

### Halfcos


```python

class Halfcos()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Halfcos

### Halfcos


```python

class Halfcos()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Halfcosm

### Halfcosm


```python

class Halfcosm()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Linear

### Linear


```python

class Linear()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### flat\_pairs


Returns a flat 1D array with x and y values interlaced


```python

def flat_pairs() -> None

```


a = linear(0, 0, 1, 10, 2, 20)
a.flat_pairs()
-> array([0, 0, 1, 10, 2, 20])

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


Linear.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new Linear bpf where x and y coordinates are inverted. This


```python

def inverted() -> None

```


is only possible if y never decreases in value

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sliced


cut this bpf at the given points, inserting points at


```python

def sliced() -> None

```


those coordinates, to limit this bpf to the range
x0:x1.

**NB**: this is different from crop, which is just a "view" into the underlying
bpf. In this case a real Linear bpf is returned. 

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Max

### Max


```python

class Max()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Min

### Min


```python

class Min()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Multi

### Multi


```python

class Multi()

```


---------


### Methods

#### \_\_init\_\_


xs: the sequence of x points


```python

def __init__() -> None

```


ys: the sequence of y points
interpolations: the interpolation used between these points

NB: len(interpolations) = len(xs) - 1

The interpelation is indicated via a string of the type:

'linear'      -> linear
'expon(2)'    -> exponential interpolation, exp=2
'halfcos'
'halfcos(0.5) -> half-cos exponential interpolation with exp=0.5
'nointerpol'  -> no interpolation (rect)

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


returns an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

exponent is only of value if the interpolation type makes use of it

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Nearest

### Nearest


```python

class Nearest()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## NoInterpol

### NoInterpol


```python

class NoInterpol()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Sampled

### Sampled


```python

class Sampled()

```


---------


### Methods

#### \_\_init\_\_


This class wraps a seq of values defined across a regular grid


```python

def __init__() -> None

```


(starting at x0, defined by x0 + i * dx)

When evaluated, values between the samples are interpolated with
the given interpolation

If 'samples' follow the ISampled interface, then it is not needed
to pass dx and

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### flat\_pairs


Returns a flat 1D array with x and y values interlaced


```python

def flat_pairs() -> None

```


a = linear(0, 0, 1, 10, 2, 20)
a.flat_pairs()
-> array([0, 0, 1, 10, 2, 20])

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


Sampled.fromseq(type cls, *args, **kws)


```python

def fromseq() -> None

```

----------

#### fromxy


Sampled.fromxy(type cls, *args, **kws)


```python

def fromxy() -> None

```

----------

#### integrate


return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


The same as integrate() but between the (included) bounds x0-x1


```python

def integrate_between() -> None

```


It is effectively the same as bpf[x0:x1].integrate(), but more efficient

NB : N has no effect. It is put here to comply with the signature of the function. 

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Sampled.inverted(self)


```python

def inverted() -> None

```

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Sampled.mapn_between(self, int n, double x0, double x1, ndarray out=None) -> ndarray


```python

def mapn_between() -> None

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


Sampled.points(self)


```python

def points() -> None

```

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


returns an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

exponent is only of value if the interpolation type makes use of it

----------

#### set\_interpolation


Sets the interpolation of this Sampled bpf, inplace


```python

def set_interpolation() -> None

```


NB: returns self, so you can do 
    sampled = bpf[x0:x1:dx].set_interpolation('expon(2)')

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**dx**

**interpolation**

**samplerate**

**x0**

**x1**

**xs**

**ys**

## Slope

### Slope


```python

class Slope()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**offset**: offset: 'double'

**slope**: slope: 'double'

**x0**

**x1**

## Smooth

### Smooth


```python

class Smooth()

```


---------


### Methods

#### \_\_init\_\_


A bpf with smoothstep interpolation.


```python

def __init__(numiter) -> None

```



**Args**

* **numiter**: the number of smoothstep steps

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Smoother

### Smoother


```python

class Smoother()

```


---------


### Methods

#### \_\_init\_\_


A bpf with smootherstep interpolation (perlin's variation of smoothstep)


```python

def __init__() -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### clone\_with\_new\_data


Create a new bpf with the same interpolation shape and any


```python

def clone_with_new_data() -> None

```


other attribute of this bpf but with new data

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### insertpoint


Return a copy of this bpf with the point (x, y) inserted


```python

def insertpoint() -> None

```

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


Return an array with the results of evaluating this bpf at a grid


```python

def mapn_between() -> None

```


of the form `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values (INPLACE) which are returned when this bpf is evaluated


```python

def outbound() -> None

```


outside its bounds.

The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```


##### Example

    >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
    >>> b.points()
    ([0, 1, 2], [0, 100, 50])

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### removepoint


Return a copy of this bpf with point at x removed


```python

def removepoint() -> None

```


Raises ValueError if x is not in this bpf

To remove elements by index, do::

    xs, ys = mybpf.points()
    xs = numpy.delete(xs, indices)
    ys = numpy.delete(ys, indices)
    mybpf = mybpf.clone_with_new_data(xs, ys)

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


Return an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

Exponent is only of value if the interpolation type makes use of it

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift() -> None

```


Use `shifted` to create a new bpf

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch() -> None

```


NB: Use `stretched` to create a new bpf

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## Spline

### Spline


```python

class Spline()

```


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


The same as map(self, xs) but faster


```python

def map() -> None

```


xs can also be a number, in which case it is interpreted as
the number of elements to calculate in an evenly spaced
grid between the bounds of this bpf.

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
>>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array

----------

#### mapn\_between


return a numpy array representing n values of this bpf between x0 and x1


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

>>> out = numpy.empty((100,), dtype=float)
>>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### points


returns (xs, ys)


```python

def points() -> None

```

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


returns an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

exponent is only of value if the interpolation type makes use of it

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## USpline

### USpline


BPF with univariate spline interpolation. This is implemented by


```python

class USpline()

```


wrapping a UnivariateSpline from scipy.


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


BpfInterface.abs(self) -> _BpfUnaryFunc


```python

def abs() -> None

```

----------

#### acos


BpfInterface.acos(self) -> _BpfUnaryFunc


```python

def acos() -> None

```

----------

#### amp2db


BpfInterface.amp2db(self) -> _Bpf_amp2db


```python

def amp2db() -> None

```

----------

#### apply


return a new bpf where func is applied to the result of it


```python

def apply() -> None

```


func(self(x))   -- see 'function composition'

##### Example

    >>> from math import sin
    >>> new_bpf = this_bpf.apply(sin)
    >>> assert new_bpf(x) == sin(this_bpf(x))

NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf

----------

#### asin


BpfInterface.asin(self) -> _BpfUnaryFunc


```python

def asin() -> None

```

----------

#### bounds


BpfInterface.bounds(self)


```python

def bounds() -> None

```

----------

#### ceil


BpfInterface.ceil(self) -> _BpfUnaryFunc


```python

def ceil() -> None

```

----------

#### clip


BpfInterface.clip(self, double y0=-inf, double y1=inf) -> _BpfLambdaClip


```python

def clip() -> None

```

----------

#### concat


Concatenate this bpf to other, so that the beginning of other is the end of this one


```python

def concat() -> None

```

----------

#### copy


Create a copy of this bpf


```python

def copy() -> None

```

----------

#### cos


BpfInterface.cos(self) -> _BpfUnaryFunc


```python

def cos() -> None

```

----------

#### db2amp


BpfInterface.db2amp(self) -> _Bpf_db2amp


```python

def db2amp() -> None

```

----------

#### debug


keys:


```python

def debug() -> None

```


* integrationmode

----------

#### derivative


Return a curve which represents the derivative of this curve


```python

def derivative() -> None

```


It implements Newtons difference quotiont, so that

derivative(x) = bpf(x + h) - bpf(x)
                -------------------
                          h

----------

#### dxton


Calculate the number of points as a result of dividing the


```python

def dxton() -> None

```


bounds of this bpf by the sampling period `dx`:

    n = (x1 + dx - x0) / dx

where x0 and x1 are the x coord start and end points and dx 
is the sampling period.

----------

#### exp


BpfInterface.exp(self) -> _BpfUnaryFunc


```python

def exp() -> None

```

----------

#### f2m


BpfInterface.f2m(self) -> _BpfF2M


```python

def f2m() -> None

```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


returns a new BPF which is the projection of this BPF


```python

def fit_between() -> None

```


to the interval x0:x1

This operation only makes sense if the current BPF is bounded
(none of its bounds is inf)

----------

#### floor


BpfInterface.floor(self) -> _BpfUnaryFunc


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor. In this variant points can be given as tuples
or as a flat sequence. For example, to create a Linear bpf, these
operations result in the same bpf:


    Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
    Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
    Linear((x0, x1, ...), (y0, y1, ...))



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf. If any of the bounds is inf,


```python

def integrate() -> None

```


the result is also inf.

Tip: to determine the limits of the integration, first crop the bpf via a slice
Example:

b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)

----------

#### integrate\_between


BpfInterface.integrate_between(self, double x0, double x1, size_t N=0) -> double


```python

def integrate_between() -> None

```

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a new BPF which is the inversion of this BPF, or None if the function is


```python

def inverted() -> None

```


not invertible.

In an inverted function the coordinates are swaped: the inverted version of a 
BPF indicates which x corresponds to a given y

`f.inverted()(f(x)) = x`

For a function to be invertible, it must be strictly increasing or decreasing,
with no local maxima or minima.

so if `y(1) = 2`, then `y.inverted()(2) = 1`

----------

#### keep\_slope


Return a new bpf which is a copy of this bpf when inside


```python

def keep_slope() -> None

```


bounds() but outside bounds() it behaves as a linear bpf
with a slope equal to the slope of this bpf at its extremes

----------

#### log


BpfInterface.log(self, double base=M_E) -> _BpfLambdaLog


```python

def log() -> None

```

----------

#### log10


BpfInterface.log10(self) -> _BpfUnaryFunc


```python

def log10() -> None

```

----------

#### m2f


BpfInterface.m2f(self) -> _BpfM2F


```python

def m2f() -> None

```

----------

#### map


USpline.map(self, xs, ndarray out=None) -> ndarray


```python

def map() -> None

```

----------

#### mapn\_between


USpline.mapn_between(self, int n, double x0, double x1, ndarray out=None) -> ndarray


```python

def mapn_between() -> None

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf. To constrain the calculation


```python

def mean() -> None

```


to a given portion, use::

    bpf.integrate_between(start, end) / (end-start)

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx` so that the bounds of this bpf


```python

def ntodx() -> None

```


are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

    >>> a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
    >>> a(-0.5)
    -1
    >>> a(1.1)
    0
    >>> a(0)
    1
    >>> a(1)
    10

    # fallback to another curve outside self
    >>> a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)

----------

#### periodic


return a new bpf which is is a copy of this bpf when inside


```python

def periodic() -> None

```


bounds() and outside it replicates it in a periodic way.
It has no bounds.

##### Example

    >>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
    >>> b(0.5)
    0
    >>> b(1.5)
    0
    >>> b(-10)
    -1

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> Any

```



**Args**

* **kind**: one of 'line', 'bar'
* **n**: the number of points to plot
* **show**: if the plot should be shown immediately after (default is True). If
    you         want to display multiple BPFs in one plot, for instance to
    compare them,         you can call plot on each of the bpfs with show=False,
    and then either         call the last one with plot=True or call
    bpf4.plot.show().
* **axes** (`matplotlib.pyplot.Axes`): if given, will be used to plot onto it,
    otherwise an ad-hoc axes is created
* **kws**: any keyword will be passed to plot.plot_coords

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;self

----------

#### preapply


return a new bpf where func is applied to the argument


```python

def preapply() -> None

```


before it is passed to the bpf: `bpf(func(x))`

##### Example

    >>> bpf = Linear((0, 1, 2), (0, 10, 20))
    >>> bpf(0.5)
    5

    >>> shifted_bpf = bpf.preapply(lambda x: x + 1)
    >>> shifted_bpf(0.5)
    15

**NB**: `A_bpf.preapply(B_bpf)` is the same as `B_bpf | A_bpf`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


return a NEW bpf representing this bpf


```python

def render(xs, interpolation) -> Any

```



**Args**

* **xs**: a seq of points at which this bpf is sampled or a number,         in
    which case an even grid is calculated with that number of points
* **interpolation**: the same interpolation types supported by `.sampled`

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a new bpf representing this bpf. Depending on the interpolation this new bpf will be a Linear or a NoInterpol bpf

----------

#### round


BpfInterface.round(self) -> _BpfLambdaRound


```python

def round() -> None

```

----------

#### sample\_between


Returns an array representing this bpf sampled at an interval of dx


```python

def sample_between(x0, x1, dx, out: ndarray) -> Any

```


between x0 and x1

**NB**: x0 and x1 are included

##### Example

>>> thisbpf = bpf.linear(0, 0, 10, 10)
>>> thisbpf.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]

This is the same as thisbpf.mapn_between(11, 0, 10)



**Args**

* **x0**: point to start sampling
* **x1**: point to stop sampling (included)
* **dx**: the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;an array with the values of this bpf at a regular grid of dx from x0 to x1 If out was given, the returned array is out

----------

#### sampled


Sample this bpf at an interval of dx (samplerate = 1 / dx)


```python

def sampled() -> None

```


returns a Sampled bpf with the given interpolation between the samples

interpolation can be any kind of interpolation, for example
'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

if you need to sample a portion of the bpf, use sampled_between

The same results can be achieved by the shorthand:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

See also: ntodx, dxton

----------

#### sampled\_between


sample a portion of this bpf at an interval of dx


```python

def sampled_between() -> None

```


returns a Sampled bpf with bounds=(x0, x1)

This is the same as thisbpf[x0:x1:dx]

----------

#### segments


returns an iterator where each item is


```python

def segments() -> None

```


(float x, float y, str interpolation_type, float exponent)

exponent is only of value if the interpolation type makes use of it

----------

#### shifted


the same as shift, but a NEW bpf is returned, which is a shifted


```python

def shifted() -> None

```


view on this bpf.

##### Example

    >>> a = bpf.linear(0, 1, 1, 5)
    >>> b = a.shifted(2)
    >>> b(3) == a(1)

----------

#### sin


BpfInterface.sin(self) -> _BpfUnaryFunc


```python

def sin() -> None

```

----------

#### sinh


BpfInterface.sinh(self) -> _BpfUnaryFunc


```python

def sinh() -> None

```

----------

#### sqrt


BpfInterface.sqrt(self) -> _BpfUnaryFunc


```python

def sqrt() -> None

```

----------

#### stretched


returns new bpf which is a projection of this bpf stretched


```python

def stretched(rx, fixpoint) -> None

```


over the x axis. 

NB: to stretch over the y-axis, just multiply this bpf
See also: fit_between

##### Example

    # stretch the shape of the bpf, but preserve the position
    >>> a = linear(1, 1, 2, 2)
    >>> a.stretched(4, fixpoint=a.x0).bounds()
    (1, 9)



**Args**

* **rx**: the stretch factor
* **fixpoint**: the point to use as reference

----------

#### tan


BpfInterface.tan(self) -> _BpfUnaryFunc


```python

def tan() -> None

```

----------

#### tanh


BpfInterface.tanh(self) -> _BpfUnaryFunc


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


the same as integrate() but within the bounds [x0, x1]


```python

def trapz_integrate_between(N) -> None

```



**Args**

* **N**: optional. a hint to the number of subdivisions used to calculate
    the integral. If not given, a default is used. This default is defined in
    CONFIG['integrate.trapz_intervals']

----------

#### zeros


Calculate the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]



**Args**

* **h**: the accuracy to scan for zeros-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if possitive, stop the search when this number of zeros have
    been found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## blend


blend(a, b, mix=0.5) -> BpfInterface


```python

def blend() -> None

```


blend these BPFs

if mix == 0: the result is *a*
if mix == 1: the result is *b*

mix can also be a bpf or any function

### Example

    # create a curve which is in between a halfcos and a linear interpolation
    a = bpf.halfcos(0, 0, 1, 1)
    b = bpf.linear(0, 0, 1, 1)
    a.blendwith(b, 0.5)

    # nearer to halfcos
    a.blendwith(b, 0.1)


---------


## bpf\_zero\_crossings


bpf_zero_crossings(BpfInterface b, double h=0.01, int N=0, double x0=NAN, double x1=NAN, int maxzeros=0) -> list


```python

def bpf_zero_crossings(b, h, N) -> Any

```


Return the zeros if b in the interval defined



**Args**

* **b**: a bpf
* **h**: the interval to scan for zeros. for each interval only one zero will be
    found
* **N**: alternatively you can give the number of intervals to scan. h will be
    calculated from that            N overrides h         x0, x1: the bounds to
    use. these, if given, override the bounds b

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list of zeros


---------


## brentq


brentq(bpf, x0, xa, xb, xtol=9.9999999999999998e-13, rtol=4.4408920985006262e-16, max_iter=100)


```python

def brentq(bpf, x0, xa, xb, xtol, rtol, max_iter) -> Any

```


calculate the zero of (bpf + x0) in the interval (xa, xb) using brentq algorithm

**NB**: to calculate all the zeros of a bpf, use the .zeros method


Example::

    # calculate the x where a == 0.5
    >>> a = bpf.linear(0, 0, 10, 1)
    >>> x_at_zero, numcalls = bpf_brentq(a, -0.5, 0, 1)
    >>> print x_at_zero
    5



**Args**

* **bpf**: the bpf to evaluate
* **x0**: an offset so that bpf(x) + x0 = 0
* **xa**: the starting point to look for a zero
* **xb**: the end point
* **xtol**: The computed root x0 will satisfy np.allclose(x, x0, atol=xtol,
    rtol=rtol)
* **rtol**: The computed root x0 will satisfy np.allclose(x, x0, atol=xtol,
    rtol=rtol)
* **max_iter**: the max. number of iterations

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a tuple (zero of the bpf, number of function calls)


---------


## setA4


setA4(double freq)


```python

def setA4() -> None

```