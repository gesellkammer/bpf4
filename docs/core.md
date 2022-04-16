# Core


---------


## BlendShape

### BlendShape


A bpf resulting of blending between two different bpfs


```python

class BlendShape()

```


---------


### Methods

#### \_\_init\_\_


```python

def __init__(xs: ndarray, ys: ndarray, shape0: str, shape1: str, mix: float
             ) -> None

```



**Args**

* **xs** (`ndarray`): x-coord data
* **ys** (`ndarray`): y-coord data
* **shape0** (`str`): first shape
* **shape1** (`str`): second shape
* **mix** (`float`): a float between 0 and 1 blending shape0 and shape1

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**descriptor**

**exp**

**x0**

**x1**

## BpfInterface

### BpfInterface


Base class for all BreakPointFunctions


```python

class BpfInterface()

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

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


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Expon

### Expon


A bpf with exponential interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with symmetrical exponential interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with fibonacci interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with half-cosine interpolation


```python

class Halfcos()

```


---------


### Methods

#### \_\_init\_\_


```python

def __init__(xs: array, ys: array, exp: float, numiter) -> None

```



**Args**

* **xs** (`array`): the x-coord data
* **ys** (`array`): the y-coord data
* **exp** (`float`): an exponent applied to the halfcosine interpolation
* **numiter**: how many times to apply the interpolation

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with half-cosine interpolation


```python

class Halfcos()

```


---------


### Methods

#### \_\_init\_\_


```python

def __init__(xs: array, ys: array, exp: float, numiter) -> None

```



**Args**

* **xs** (`array`): the x-coord data
* **ys** (`array`): the y-coord data
* **exp** (`float`): an exponent applied to the halfcosine interpolation
* **numiter**: how many times to apply the interpolation

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with half-cosine and exponent depending on the orientation of the interpolation


```python

class Halfcosm()

```


When interpolating between two y values, y0 and y1, if  y1 < y0 the exponent
is inverted, resulting in a symmetrical interpolation shape


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with linear interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### flat\_pairs


Returns a flat 1D array with x and y values interlaced


```python

def flat_pairs() -> None

```


```python

>>> a = linear(0, 0, 1, 10, 2, 20)
>>> a.flat_pairs()
array([0, 0, 1, 10, 2, 20])

```

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

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


Return a new Linear bpf where x and y coordinates are inverted.


```python

def inverted() -> None

```


This is only possible if y never decreases in value

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sliced


Cut this bpf at the given points


```python

def sliced() -> None

```


If needed it inserts points at the given coordinates to limit this bpf to 
the range `x0:x1`.

**NB**: this is different from crop, which is just a "view" into the underlying
bpf. In this case a real `Linear` bpf is returned. 

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Multi

### Multi


A bpf where each segment can have its own interpolation kind


```python

class Multi()

```


---------


### Methods

#### \_\_init\_\_


```python

def __init__(xs: ndarray, ys: ndarray, interpolations: list[str]) -> None

```


**NB**: `len(interpolations) == len(xs) - 1`

The interpelation is indicated via a string of the type:


* 'linear': linear
* 'expon(2)': exponential interpolation, exp=2
* 'halfcos'
* 'halfcos(0.5): half-cos exponential interpolation with exp=0.5
* 'nointerpol': no interpolation (rect)



**Args**

* **xs** (`ndarray`): the sequence of x points
* **ys** (`ndarray`): the sequence of y points
* **interpolations** (`list[str]`): the interpolation used between these points

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Multi.segments(self)


```python

def segments() -> None

```

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## Nearest

### Nearest


A bpf with nearest interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf without interpolation


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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with regularly sampled data


```python

class Sampled()

```


When evaluated, values between the samples are interpolated with
a given function: linear, expon(x), halfcos, halfcos(x), etc.


---------


### Methods

#### \_\_init\_\_


```python

def __init__(samples: ndarray, dx: float, x0: float, interpolation: str) -> None

```



**Args**

* **samples** (`ndarray`): the y-coord sampled data
* **dx** (`float`): the sampling **period**
* **x0** (`float`): the first x-value
* **interpolation** (`str`): the interpolation function used. One of 'linear',
    'nointerpol', 'expon(X)', 'halfcos', 'halfcos(X)', 'smooth',
    'halfcosm', etc.

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:


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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### flat\_pairs


Returns a flat 1D array with x and y values interlaced


```python

def flat_pairs() -> None

```


```python
>>> a = linear(0, 0, 1, 10, 2, 20)
>>> a.flat_pairs()
array([0, 0, 1, 10, 2, 20])
```

----------

#### floor


Returns a bpf representing the floor of this bpf


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


Return the result of the integration of this bpf.


```python

def integrate() -> None

```


If any of the bounds is inf, the result is also inf.

**NB**: to determine the limits of the integration, first crop the bpf via a slice

##### Example

Integrate this bpf from its lower bound to 10 (inclusive)

```python
b[:10].integrate()  
```

----------

#### integrate\_between


The same as integrate() but between the (included) bounds x0-x1


```python

def integrate_between() -> None

```


It is effectively the same as `bpf[x0:x1].integrate()`, but more efficient

**NB**: N has no effect. It is put here to comply with the signature of the function. 

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

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


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Sampled.points(self)


```python

def points() -> None

```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Returns an iterator over the segments of this bpf


```python

def segments() -> None

```


Each item is a tuple `(float x, float y, str interpolation_type, float exponent)`

**NB**: exponent is only relevant if the interpolation type makes use of it

----------

#### set\_interpolation


Sets the interpolation of this Sampled bpf, inplace


```python

def set_interpolation() -> None

```


Returns *self*, so you can do:

    sampled = bpf[x0:x1:dx].set_interpolation('expon(2)')

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with smoothstep interpolation.


```python

class Smooth()

```


---------


### Methods

#### \_\_init\_\_


```python

def __init__(xs: ndarray, ys: ndarray, numiter: int) -> None

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data
* **numiter** (`int`): the number of smoothstep steps

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


A bpf with smootherstep interpolation (perlin's variation of smoothstep)


```python

class Smoother()

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### clone\_with\_new\_data


Create a new bpf with the same attributes as self but with new data


```python

def clone_with_new_data(xs: ndarray, ys: ndarray) -> Any

```



**Args**

* **xs** (`ndarray`): the x-coord data
* **ys** (`ndarray`): the y-coord data

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The new bpf. It will be of the same class as self

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



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


**NB**: *self* is not modified

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Return an array of `n` elements resulting of evaluating this bpf regularly


```python

def mapn_between() -> None

```


The x coordinates at which this bpf is evaluated are equivalent to `linspace(xstart, xend, n)`

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Set the values **inplace** returned when this bpf is evaluated outside its bounds.


```python

def outbound() -> None

```


The default behaviour is to interpret the values at the bounds to extend to infinity.

In order to not change this bpf inplace, use `b.copy().outbound(y0, y1)`

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


Returns (xs, ys)


```python

def points() -> None

```


##### Example

```python

>>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
>>> b.points()
([0, 1, 2], [0, 100, 50])
```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

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


Raises `ValueError` if x is not in this bpf

To remove elements by index, do:

```python

xs, ys = mybpf.points()
xs = numpy.delete(xs, indices)
ys = numpy.delete(ys, indices)
mybpf = mybpf.clone_with_new_data(xs, ys)

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Return an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(x: float, y: float, interpoltype: str, exponent: float)`

Exponent is only of value if the interpolation type makes use of it.

----------

#### shift


Shift the bpf along the x-coords, **INPLACE**


```python

def shift(dx: float) -> None

```


Use `shifted` to create a new bpf



**Args**

* **dx** (`float`): the shift interval

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretch


Stretch or compress this bpf in the x-coordinate **INPLACE**


```python

def stretch(rx: float) -> None

```


**NB**: use `stretched` to create a new bpf



**Args**

* **rx** (`float`): the stretch/compression factor

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
```

----------

#### map


The same as map(self, xs) but faster


```python

def map(xs: ndarray | int, out: ndarray) -> None

```


```python

bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
```

##### Example

```python

>>> out = numpy.empty((100,), dtype=float)
>>> xs = numpy.linspace(0, 10, 100)
# This is the right way to pass an output array
>>> out = thisbpf.map(xs, out)   

```



**Args**

* **xs** (`ndarray | int`): the x coordinates at which to sample this bpf,
    or an integer representing the number of elements to calculate         in an
    evenly spaced grid between the bounds of this bpf
* **out** (`ndarray`): if given, an attempt will be done to use it as
    destination         for the result. The user should not trust that this
    actually happens         (see example)

----------

#### mapn\_between


Calculate an array of `n` values representing this bpf between `x0` and `x1`


```python

def mapn_between() -> None

```


x0 and x1 are included

If out is passed, an attempt will be done to use it as destination for the result
Nonetheless, you should NEVER trust that this actually happens. See example

##### Example

```python

out = numpy.empty((100,), dtype=float)
out = thisbpf.mapn_between(100, 0, 10, out)

```

----------

#### max


BpfInterface.max(self, b)


```python

def max() -> None

```

----------

#### mean


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### points


returns (xs, ys)


```python

def points() -> None

```

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


Returns an iterator over the segments of this bpf


```python

def segments() -> None

```


Each segment is a tuple `(float x, float y, str interpolation_type, float exponent)`

**NB**: exponent is only relevant if the interpolation type makes use of it

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a list with the zeros of this bpf


---------


### Attributes

**x0**

**x1**

## USpline

### USpline


bpf with univariate spline interpolation.


```python

class USpline()

```


This is implemented by wrapping a UnivariateSpline from scipy.


---------


### Methods

#### \_\_init\_\_


Initialize self.  See help(type(self)) for accurate signature.


```python

def __init__(self, args, kwargs) -> None

```

----------

#### abs


Returns a bpf representing the absolute value of this bpf


```python

def abs() -> None

```

----------

#### acos


Returns a bpf representing the arc cosine of this bpf


```python

def acos() -> None

```

----------

#### amp2db


Returns a bpf converting linear amplitudes to decibels


```python

def amp2db() -> None

```


##### Example

```python
>>> linear(0, 0, 1, 1).amp2db().map(10)
array([-280.        ,  -19.08485019,  -13.06425028,   -9.54242509,
       -7.04365036,   -5.1054501 ,   -3.52182518,   -2.18288939,
       -1.02305045,    0.        ])
```

----------

#### apply


Create a bpf where `func` is applied to the result of this pdf


```python

def apply() -> None

```


**NB**: `a.apply(b)` is the same as `a | b`

##### Example

```python

>>> from bpf4 import *
>>> from math import *
>>> a = linear(0, 0, 1, 10)
>>> def func(x):
...     return sin(x) + 1
>>> b = a.apply(func)
>>> b(1)
0.4559788891106302
>>> sin(a(1)) + 1
0.4559788891106302

```

----------

#### asin


Returns a bpf representing the arc sine of this bpf


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


Returns a bpf representing the ceil of this bpf


```python

def ceil() -> None

```

----------

#### clip


Return a bpf clipping the result between y0 and y1


```python

def clip() -> None

```


```python

>>> linear(0, -1, 1, 1).clip(0, 1).map(20)
array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.        ])
```

----------

#### concat


Concatenate this bpf to other


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


Returns a bpf representing the cosine of this bpf


```python

def cos() -> None

```

----------

#### db2amp


Returns a bpf converting decibels to linear amplitudes


```python

def db2amp() -> None

```


##### Example

```python
>>> linear(0, 0, 1, -60).db2amp().map(10)
array([1.        , 0.46415888, 0.21544347, 0.1       , 0.04641589,
       0.02154435, 0.01      , 0.00464159, 0.00215443, 0.001     ])
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


It implements Newtons difference quotiont, so that:

```

                bpf(x + h) - bpf(x)
derivative(x) = -------------------
                          h
```

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

#### expon


Returns a bpf representing the exp operation with this bpf


```python

def expon() -> None

```


##### Example

```python

>>> from bpf4 import *
>>> a = linear(0, 0, 1, 10)
>>> a(0.1)
1.0
>>> exp(1.0)
2.718281828459045
>>> a.expon()(0.1)
2.718281828459045

----------

#### f2m


Returns a bpf converting frequencies to midinotes


```python

def f2m() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> freqs = linear(0, 442, 1, 882)
>>> freqs.f2m().map(10)
array([69.        , 70.82403712, 72.47407941, 73.98044999, 75.3661766 ,
       76.64915905, 77.84358713, 78.96089998, 80.01045408, 81.        ])
```

----------

#### fib


BpfInterface.fib(self) -> _BpfLambdaFib


```python

def fib() -> None

```

----------

#### fit\_between


Returns a view of this bpf fitted within the interval x0:x1


```python

def fit_between(x0, x1) -> Any

```


This operation only makes sense if the bpf is bounded
(none of its bounds is inf)

##### Example

```python

>>> from bpf4 import *
>>> a = linear(1, 1, 2, 5)
>>> a.bounds()
(1, 5)
>>> b = a.fit_between(0, 10)
>>> b.bounds()
0, 10
>>> b(10)
5
```



**Args**

* **x0**: the lower bound to fit this bpf
* **x1**: the upper bound to fit this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;the projected bpf

----------

#### floor


Returns a bpf representing the floor of this bpf


```python

def floor() -> None

```

----------

#### fromseq


BpfInterface.fromseq(type cls, *points, **kws)


```python

def fromseq(points, kws) -> None

```


A helper constructor, in this variant points are given as tuples or as a flat sequence. 

For example, to create a Linear bpf, these operations result in the same bpf:


```python
Linear.fromseq(x0, y0, x1, y1, x2, y2, ...)
Linear.fromseq((x0, y0), (x1, y1), (x2, y2), ...)
Linear((x0, x1, ...), (y0, y1, ...))
```



**Args**

* **points**: either the interleaved x and y points, or each point as a
    2D tuple
* **kws**: any keyword will be passed to the default constructor (for
    example, exp in the case of a Expon bpf)

----------

#### integrate


Return the result of the integration of this bpf.


```python

def integrate() -> Any

```


If any of the bounds is inf, the result is also inf.

**NB**: to set the bounds of the integration, first crop the bpf via a slice

##### Example

```python

>>> linear(0, 0, 10, 10).sin()[0:2*pi].integrate()
-1.7099295055304798e-17

```



**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrate\_between


Integrate this bpf between x0 and x1


```python

def integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start x of the integration range
* **x1**: end x of the integration range
* **N**: number of intervals to use for integration

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### integrated


Return a bpf representing the integration of this bpf at a given point


```python

def integrated() -> None

```

----------

#### inverted


Return a view on this bpf with the coords inverted


```python

def inverted() -> None

```


In an inverted function the coordinates are swaped: the inverted version of a 
bpf indicates which *x* corresponds to a given *y*

Returns None if the function is not invertible. For a function to be invertible, 
it must be strictly increasing or decreasing, with no local maxima or minima.


    f.inverted()(f(x)) = x


So if `y(1) == 2`, then `y.inverted()(2) == 1`

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


Returns a bpf representing the log of this bpf


```python

def log() -> None

```

----------

#### log10


Returns a bpf representing the log10 of this bpf


```python

def log10() -> None

```

----------

#### m2f


Returns a bpf converting from midinotes to frequency


```python

def m2f() -> None

```


##### Example

```python
>>> from bpf4 import *
>>> midinotes = linear(0, 60, 1, 65)
>>> freqs = midinotes.m2f()
>>> freqs.map(10)
array([262.81477242, 271.38531671, 280.23535149, 289.37399111,
       298.81064715, 308.55503809, 318.61719934, 329.0074936 ,
       339.73662146, 350.81563248])
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


Calculate the mean value of this bpf.


```python

def mean() -> None

```


To constrain the calculation to a given portion, use:

```python

bpf.integrate_between(start, end) / (end-start)

```

----------

#### min


BpfInterface.min(self, b)


```python

def min() -> None

```

----------

#### ntodx


Calculate the sampling period `dx`


```python

def ntodx() -> None

```


Calculate `dx` so that the bounds of this bpf 
are divided into N parts: `dx = (x1-x0) / (N-1)`

----------

#### outbound


Return a new Bpf with the given values outside the bounds


```python

def outbound() -> None

```


##### Examples

```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 10).outbound(-1, 0)
>>> a(-0.5)
-1
>>> a(1.1)
0
>>> a(0)
1
>>> a(1)
10

# fallback to another curve outside self
>>> a = linear(0, 1, 1, 10).outbound(0, 0) + expon(-1, 2, 4, 10)
```

----------

#### periodic


Returns a new bpf which replicates this in a periodic way


```python

def periodic() -> None

```


The new bpf is a copy of this bpf when inside its bounds 
and outside it, it replicates it in a periodic way, with no bounds.

##### Example

```python

>>> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
>>> b(0.5)
0
>>> b(1.5)
0
>>> b(-10)
-1
```

----------

#### plot


Plot the bpf. Any key is passed to plot.plot_coords


```python

def plot(kind, n, show, axes: matplotlib.pyplot.Axes, kws) -> None

```


##### Example

```python

from bpf4 import *
a = linear(0, 0, 1, 10, 2, 0.5)
a.plot()

# Plot to a preexistent axes
ax = plt.subplot()
a.plot(axes=ax)
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

----------

#### preapply


Returns a bpf where `func` is applied to the argument before it is passed to this bpf


```python

def preapply(func: callable) -> Any

```


This is equivalent to `func(x) | self`

##### Example

```python

>>> bpf = Linear((0, 1, 2), (0, 10, 20))
>>> bpf(0.5)
5

>>> shifted_bpf = bpf.preapply(lambda x: x + 1)
>>> shifted_bpf(0.5)
15
```

**NB**: `bpf1.preapply(bpf2)` is the same as `bpf2 | bpf1`



**Args**

* **func** (`callable`): a function `func(x: float) -> float` which is applied
    to         the argument before passing it to this bpf

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;A bpf following the pattern `lambda x: bpf(func(x))`

----------

#### rand


BpfInterface.rand(self) -> _BpfRand


```python

def rand() -> None

```

----------

#### render


Create a new bpf representing this bpf rendered at the given points


```python

def render(xs: int | list | np.ndarray, interpolation: str) -> Any

```


The difference between `.render` and `.sampled` is that this method
creates a Linear/NoInterpol bpf whereas `.sampled` returns a 
`Sampled` bpf (a `Sampled` bpf works only for regularly sampled data,
a Linear or NoInterpol bpfs accept any data as its x coordinate)

##### See Also

* `BpfInterface.sampled`



**Args**

* **xs** (`int | list | np.ndarray`): a seq of points at which this bpf
    is sampled or a number, in which case an even grid is calculated
    with that number of points. In the first case a Linear or NoInterpol
    bpf is returned depending on the `interpolation` parameter (see below).
    In the second case a `Sampled` bpf is returned.
* **interpolation** (`str`): the interpoltation type of the returned bpf.
    One of 'linear', 'nointerpol'

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


Sample this bpf at an interval of dx between x0 and x1


```python

def sample_between(x0: float, x1: float, dx: float, out: ndarray) -> Any

```


**NB**: the interface is similar to numpy's `linspace`

##### Example

```python

>>> a = linear(0, 0, 10, 10)
>>> a.sample_between(0, 10, 1)
[0 1 2 3 4 5 6 7 8 9 10]
```

This is the same as `a.mapn_between(11, 0, 10)`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **out** (`ndarray`): if given, the result will be placed here and no new array
    will         be allocated

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;An array with the values of this bpf sampled at at a regular grid of period `dx` from `x0` to `x1`. If out is given the result is placed in it

----------

#### sampled


Sample this bpf at a regular interval, returns a `Sampled` bpf


```python

def sampled(dx: float, interpolation: str) -> None

```


Sample this bpf at an interval of dx (samplerate = 1 / dx)
returns a Sampled bpf with the given interpolation between the samples

**NB**: If you need to sample a portion of the bpf, use sampled_between

The same results can be achieved via indexing, in which case the resuling
bpf will be linearly interpolated:

    bpf[::0.1]    # returns a sampled version of this bpf with a dx of 0.1
    bpf[:10:0.1]  # samples this bpf between (x0, 10) at a dx of 0.1

##### See also

* `ntodx`
* `dxton`



**Args**

* **dx** (`float`): the sample interval
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function)

----------

#### sampled\_between


Sample a portion of this bpf, returns a `Sampled` bpf


```python

def sampled_between(x0: float, x1: float, dx: float, interpolation: str, 
                    example) -> Any

```


**NB**: This is the same as `thisbpf[x0:x1:dx]`



**Args**

* **x0** (`float`): point to start sampling (included)
* **x1** (`float`): point to stop sampling (included)
* **dx** (`float`): the sampling period
* **interpolation** (`str`): the interpolation kind. One of 'linear',
    'nointerpol', 'halfcos', 'expon(XX)', 'halfcos(XX)' (where         XX is an
    exponential passed to the interpolation function). For
* **example**: 'expon(2.0)' or 'halfcos(0.5)'

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The `Sampled` bpf, representing this bpf sampled at a grid of [x0:x1:dx] with the given interpolation

----------

#### segments


USpline.segments(self)


```python

def segments() -> None

```

----------

#### shifted


Returns a view of this bpf shifted by `dx` over the x-axes


```python

def shifted() -> None

```


This is the same as `shift`, but a new bpf is returned

##### Example


```python

>>> from bpf4 import *
>>> a = linear(0, 1, 1, 5)
>>> b = a.shifted(2)
>>> b(3) == a(1)
```

----------

#### sin


Returns a bpf representing the sine of this bpf


```python

def sin() -> None

```

----------

#### sinh


Returns a bpf representing the sinh of this bpf


```python

def sinh() -> None

```

----------

#### sqrt


Returns a bpf representing the sqrt of this bpf


```python

def sqrt() -> None

```

----------

#### stretched


Returns a view of this bpf stretched over the x axis.


```python

def stretched(rx: float, fixpoint: float) -> None

```


**NB**: to stretch over the y-axis, just multiply this bpf

**See also**: `fit_between`

##### Example

Stretch the shape of the bpf, but preserve the start position

```python

>>> a = linear(1, 1, 2, 2)
>>> a.stretched(4, fixpoint=a.x0).bounds()
(1, 9)
```



**Args**

* **rx** (`float`): the stretch factor
* **fixpoint** (`float`): the point to use as reference

----------

#### tan


Returns a bpf representing the tan of this bpf


```python

def tan() -> None

```

----------

#### tanh


Returns a bpf representing the tanh of this bpf


```python

def tanh() -> None

```

----------

#### trapz\_integrate\_between


Integrate this bpf between [x0, x1] using the trapt method


```python

def trapz_integrate_between(x0, x1, N) -> Any

```



**Args**

* **x0**: start of integration period
* **x1**: end of the integration period
* **N**: number of subdivisions used to calculate the integral. If not given,
    a default is used (default defined in `CONFIG['integrate.trapz_intervals']`)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;The result of the integration

----------

#### zeros


Find the zeros of this bpf


```python

def zeros(h, N, x0, x1, maxzeros) -> Any

```


##### Example

```python

>>> a = bpf.linear(0, -1, 1, 1)
>>> a.zeros()
[0.5]

```



**Args**

* **h**: the accuracy to scan for zero-crossings. If two zeros are within
    this distance, they will be resolved as one.
* **N**: alternatively, you can give the number of intervals to scan.         h
    will be derived from this
* **x0**: the point to start searching. If not given, the starting point of this
    bpf         will be used
* **x1**: the point to stop searching. If not given, the end point of this bpf
    is used
* **maxzeros**: if > 0, stop the search when this number of zeros have been
    found

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


Blend these BPFs

if mix == 0: the result is *a*
if mix == 1: the result is *b*

mix can also be a bpf or any function

### Example


```python
# create a curve which is in between a halfcos and a linear interpolation
>>> from bpf4 import *
a = halfcos(0, 0, 1, 1)
b = linear(0, 0, 1, 1)
a.blendwith(b, 0.5)

# nearer to halfcos
a.blendwith(b, 0.1)
```


---------


## bpf\_zero\_crossings


bpf_zero_crossings(BpfInterface b, double h=0.01, int N=0, double x0=NAN, double x1=NAN, int maxzeros=0) -> list


```python

def bpf_zero_crossings(b, h: float, N: int, maxzeros) -> Any

```


Return the zeros if b in the interval defined



**Args**

* **b**: a bpf
* **h** (`float`): the interval to scan for zeros. for each interval only one
    zero will be found
* **N** (`int`): alternatively you can give the number of intervals to scan. h
    will be calculated from that            N overrides h         x0, x1: the
    bounds to use. these, if given, override the bounds b
* **maxzeros**: if given, search will stop if this number of zeros is found

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


### Example

```python

# calculate the x where a == 0.5
>>> from bpf4 import *
>>> a = linear(0, 0, 10, 1)
>>> xzero, numcalls = brentq(a, -0.5, 0, 1)
>>> xzero
5
```



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


Set the reference freq used