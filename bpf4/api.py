"""
## High-level API for bpf4

The API allows a high-level and flexible interface to the core of bpf4,
which is implemented in cython for efficiency. 

### API vs core

These three curves, *a*, *b* and *c* define the same linear break-point-function
The first two definitions, *a* and *b*, use the high-level API, which allows for
points to be defined as a flat sequence, as tuples of (x, y). The *core* classes
need to be instantiated with two arrays of *x* and *y* values, as in *c*

```python

from bpf4 import *
a = linear(0, 0, 1, 2.5, 3, 10)
b = linear((0, 0), (1, 2.5), (3, 10))
c = core.Linear([0, 1, 3], [0, 2.5, 10])
```

"""
from __future__ import annotations

from . import core
from . import util



def linear(*args) -> core.Linear:
    """
    Construct a Linear bpf.

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    A bpf can be constructed in multiple ways, all of which result
    in the same Linear instance:

    ```python

    linear(x0, y0, x1, y1, ...)
    linear((x0, y0), (x1, y1), ...)
    linear({x0:y0, x1:y1, ...})
    ```

    Example
    -------

    ```python
    from bpf4 import *
    a = linear([0, 2, 3.5, 10], [0.1, 0.5, -3.5,  4])
    a.plot()
    ```
    ![](assets/Linear.png)
    """
    X, Y, kws = util.parseargs(*args)
    if kws:
        raise ValueError("linear does not take any keyword")
    return core.Linear(X, Y)
    

def expon(*args, **kws) -> core.Expon:
    """
    Construct an Expon bpf (a bpf with exponential interpolation)

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
        exp: the exponent to use
        numiter: Number of iterations. A higher number accentuates the effect
        
    A bpf can be constructed in multiple ways:

    ```python
    expon(x0, y0, x1, y1, ..., exp=exponent)
    expon(exponent, x0, y0, x1, y1, ...)
    expon((x0, y0), (x1, y1), ..., exp=exponent)
    expon({x0:y0, x1:y1, ...}, exp=exponent)
    ```
    
    Example
    -------

    ```python
    from bpf4 import *
    import matplotlib.pyplot as plt
    numplots = 5
    fig, axs = plt.subplots(2, numplots, tight_layout=True, figsize=(20, 8))
    for i in range(numplots):
        exp = i+1
        expon(0, 0, 1, 1, exp=exp).plot(show=False, axes=axs[0, i])
        expon(0, 0, 1, 1, exp=1/exp).plot(show=False, axes=axs[1, i])
        axs[0, i].set_title(f'{exp=}')
        axs[1, i].set_title(f'exp={1/exp:.2f}')
        
    plot.show()
    ```
    ![](assets/expon-grid.png)

    """
    X, Y, kws = util.parseargs(*args, **kws)
    assert "exp" in kws
    return core.Expon(X, Y, **kws)
    

def halfcos(*args, exp=1, numiter=1, **kws) -> core.Halfcos:
    """
    Construct a half-cosine bpf (a bpf with half-cosine interpolation)

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
        exp: the exponent to use
        numiter: Number of iterations. A higher number accentuates the effect
    
    A bpf can be constructed in multiple ways:

    ```python

    halfcos(x0, y0, x1, y1, ...)
    halfcos((x0, y0), (x1, y1), ...)
    halfcos({x0:y0, x1:y1, ...})
    ```

    ```python
    a = halfcos([0, 1, 3, 10], [0.1, 0.5, 3.5,  1])
    b = halfcos(*a.points(), exp=2)
    c = halfcos(*a.points(), exp=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1], show=False)
    c.plot(axes=axes[2])
    ```
    ![](assets/Halfcos.png)

    
    """
    X, Y, kws = util.parseargs(*args, **kws)
    return core.Halfcos(X, Y, exp=exp, numiter=numiter, **kws)  


halfcosexp = halfcos


def halfcosm(*args, **kws) -> core.Halfcosm:
    """
    Halfcos interpolation with symmetric exponent

    When used with an exponent, the exponent is inverted for downwards 
    segments `(y1 > y0)`

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
        exp: exponent to apply prior to cosine interpolation. The higher the exponent, the
            more skewed to the right the shape will be
        numiter: Number of iterations. A higher number accentuates the effect
    
    Returns:
        (core.Halfcosm) A bpf with symmetric cosine interpolation


    A bpf can be constructed in multiple ways:

    ```python

    halfcosm(x0, y0, x1, y1, ..., exp=2.0)
    halfcosm(2.0, x0, y0, x1, y1, ...)    # The exponent can be placed first
    halfcosm((x0, y0), (x1, y1), ...)
    halfcosm({x0:y0, x1:y1, ...})
    ```

    ```python
    from bpf4 import *
    a = halfcosm(0, 0.1,
                 1, 0.5,
                 3, 3.5,
                 10, 1, exp=2)
    b = halfcosm(*a.points(), exp=2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1])
    ```
    ![](assets/Halfcosm.png)


    """
    X, Y, kws = util.parseargs(*args, **kws)
    return core.Halfcosm(X, Y, **kws)


def spline(*args) -> core.Spline:
    """
    Construct a cubic-spline bpf 

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    Returns:
        (core.Spline) A Spline bpf
    

    A bpf can be constructed in multiple ways:

    ```python
    spline(x0, y0, x1, y1, ...)
    spline((x0, y0), (x1, y1), ...)
    spline({x0:y0, x1:y1, ...})
    ```

    ```python
    from bpf4 import *
    a = smooth(0, 0.1, 1, 0.5, 3, -3.5, 10, 1)
    b = spline(*a.points())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1])
    ```
    ![](assets/Spline.png)
    """
    X, Y, kws = util.parseargs(*args)
    return core.Spline(X, Y)


def uspline(*args) -> core.USpline: 
    """
    Construct a univariate cubic-spline bpf 

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    Returns:
        (core.USpline) A USpline bpf

    A bpf can be constructed in multiple ways:

    ```python
    uspline(x0, y0, x1, y1, ...)
    uspline((x0, y0), (x1, y1), ...)
    uspline({x0:y0, x1:y1, ...})
    ```

    ```python
    from bpf4 import *
    a = spline(0, 0.1, 1, 0.5, 3, -3.5, 10, 1)
    b = uspline(*a.points())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, tight_layout=True)
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1])
    ```
    ![](assets/Uspline.png)

    !!! info "See Also"

        * [spline](#spline)
        * [pchip](#pchip)

    """
    X, Y, kws = util.parseargs(*args)
    return core.USpline(X, Y)


def nointerpol(*args) -> core.NoInterpol:
    """
    A bpf with floor interpolation

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    A bpf can be constructed in multiple ways:

        nointerpol(x0, y0, x1, y1, ...)
        nointerpol((x0, y0), (x1, y1), ...)
        nointerpol({x0:y0, x1:y1, ...})

    ```python
    a = linear([0, 1, 3, 10], [0.1, 0.5, 3.5,  1])
    b = nointerpol(*a.points())
    c = nearest(*a.points())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1], show=False)
    c.plot(axes=axes[2])
    ```
    ![](assets/NoInterpol.png)
    """
    X, Y, kws = util.parseargs(*args)
    return core.NoInterpol(X, Y, **kws)
    

def nearest(*args) -> core.Nearest:
    """
    A bpf with floor interpolation

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
                a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
                or two arrays `xs` and `ys`

    A bpf can be constructed in multiple ways:

        nearest(x0, y0, x1, y1, ...)
        nearest((x0, y0), (x1, y1), ...)
        nearest({x0:y0, x1:y1, ...})

    ```python
    a = linear([0, 1, 3, 10], [0.1, 0.5, 3.5,  1])
    b = nointerpol(*a.points())
    c = nearest(*a.points())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1], show=False)
    c.plot(axes=axes[2])
    ```
    ![](assets/NoInterpol.png)
    """
    X, Y, kws = util.parseargs(*args)
    return core.Nearest(X, Y, **kws)


def smooth(*args, numiter=1) -> core.Smooth:
    """
    A bpf with smoothstep interpolation. 

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
        numiter: determines the number of smoothstep steps applied 
            (see https://en.wikipedia.org/wiki/Smoothstep)

    Returns:
        (core.Smooth) A bpf with smoothstep interpolation

    ```python

    from bpf4.api import *
    a = smooth((0, 0.1), (1, 0.5), (3, -3.5), (10, 1))
    a.plot()
    ```
    ![](assets/Smooth.png)

    !!! info "See Also"

        * [smoother](#smoother)
    """
    X, Y, kws = util.parseargs(*args)
    return core.Smooth(X, Y, numiter=numiter)


def smoother(*args) -> core.Smoother:
    """
    A bpf with smootherstep interpolation 

    This bpf uses Perlin's variation on smoothstep,
    see https://en.wikipedia.org/wiki/Smoothstep)

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    Returns:
        (core.Smoother) A bpf with smootherstep interpolation
    

    ```python
    from bpf4 import *
    a = smooth(0, 0.1, 
               1, 0.5, 
               3, -3.5, 
               10, 1)
    b = smoother(*a.points())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(axes=axes[0], show=False)
    b.plot(axes=axes[1])
    ```
    ![](assets/Smoother.png)
    """
    X, Y, kws = util.parseargs(*args)
    return core.Smoother(X, Y)


def multi(*args):
    """
    A bpf with a per-pair interpolation

    Example
    -------

    ```python
    
    # (0,0) --linear-- (1,10) --expon(3)-- (2,3) --expon(3)-- (10, -1) --halfcos-- (20,0)

    multi(0, 0,   'linear' 
          1, 10,  'expon(3)', 
          2, 3,   # assumes previous interpolation 
          10, -1, 'halfcos'      
          20, 0)
    
    # also the following syntax is possible
    multi((0, 0, 'linear')
          (1, 10, 'expon(3)'), 
          (2, 3), 
          (10, -1, 'halfcos'), 
          (20, 0))
    ```
    """
    xs, ys, interpolations = util.multi_parseargs(args)
    return core.Multi(xs, ys, interpolations)


def pchip(*args):
    """
    Monotonic Cubic Hermite Intepolation

    Args:
        args: either a flat list of coordinates in the form `x0, y0, x1, y1, ...`,
            a list of tuples `(x0, y0), (x1, y1), ...`, a dict `{x0:y0, x1:y1, ...}`
            or two arrays `xs` and `ys`
    
    A bpf can be constructed in multiple ways:

    ```python

    pchip(x0, y0, x1, y1, ...)
    pchip((x0, y0), (x1, y1), ...)
    pchip({x0:y0, x1:y1, ...})


    >>> a = core.Smoother([0, 1, 3, 10, 12, 12.5], [0.1, 0.5, -3.5,  1, 4.5, -1])
    >>> b = core.Spline(*a.points())
    >>> c = pchip(*a.points())

    >>> fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True, tight_layout=True)
    >>> a.plot(axes=axes[0], show=False)
    >>> b.plot(axes=axes[1], show=False)
    >>> c.plot()
    ``` 
    ![](assets/pchip.png)   
    """
    from . import pyinterp
    xs, ys, kws = util.parseargs(*args)
    return pyinterp.Pchip(xs, ys, **kws)


def const(value) -> core.Const:
    """
    A bpf which always returns a constant value

    Args:
        value: the constant value

    Example
    -------

    ```python
    
    >>> c5 = const(5)
    >>> c5(10) 
    5
    ```
    
    """
    return core.Const(value)


def slope(slope:float, offset=0., bounds: tuple[float, float] = None) -> core.Slope:
    """
    Generate a straight line with the given slope and offset 

    This is the same as linear(0, offset, 1, slope)

    Example
    -------

    ```python

    >>> a = slope(0.5, 1)
    >>> a
    Slope[-inf:inf]
    >>> a[0:10].plot()
    ```
    ![](assets/slope-plot.png)
    """
    return core.Slope(slope, offset, bounds=bounds)


def stack(*bpfs) -> core.Stack:
    """
    A bpf representing a stack of bpf

    Within a Stack, a bpf does not have outbound values. When evaluated
    outside its bounds the bpf below is used, iteratively until the
    lowest bpf is reached. Only the lowest bpf is evaluated outside its
    bounds

    Args:
        bpfs: a sequence of bpfs

    Returns:
        (core.Stack) A stacked bpf


    Example
    -------

    ```python
    # Interval    bpf
    # [0, 3]      a
    # (3, 4]      b
    # (4, 10]     c

    from bpf4 import *
    import matplotlib.pyplot as plt
    a = linear(0, 0, 3, 1)
    b = linear(2, 9, 4, 10)
    c = halfcos(0, 0, 10, 10)
    s = core.Stack((a, b, c))

    ax = plt.subplot(111)
    a.plot(color="#f00", alpha=0.4, axes=ax, linewidth=4, show=False)
    b.plot(color="#00f", alpha=0.4, axes=ax, linewidth=4, show=False)
    c.plot(color="#f0f", alpha=0.4, axes=ax, linewidth=4, show=False)
    s.plot(axes=ax, linewidth=2, color="#000", linestyle='dotted')
    ```
    ![](assets/stack2.png)

    """
    if len(bpfs) == 1 and isinstance(bpfs[0], (list, tuple)):
        bpfs = bpfs[0]
    return core.Stack(bpfs)


def blendshape(shape0:str, shape1:str, mix, points) -> core.BpfInterface:
    """
    Create a bpf blending two interpolation forms

    Args:
        shape0: a description of the first interpolation
        shape1: a description of the second interpolation
        mix (float | core.BpfInterface): blend factor. 
            A value between 0 (use only `shape0`)
            and 1 (use only `shape1`). A value of `0.5` will result in
            an average between the first and second interpolation kind.
            Can be a bpf itself, returning the mix value at any x value
        points: either a tuple `(x0, y0, x1, y1, ...)` or a tuple `(xs, ys)`
            where *xs* and *ys* are lists/arrays containing the *x* and *y*
            coordinates of the points

    Returns:
        (core.BpfInterface) A bpf blending two different interpolation kinds

    Example
    -------

    ```python

    from bpf4 import *
    a = blendshape('halfcos(2.0)', 'linear', mix=0.5, points=(0, 0, 1, 1))
    halfcos(0, 0, 1, 1, exp=2).plot(color='red')
    linear(0, 0, 1, 1).plot(color='blue')
    a.plot(color='green')
    ```
    ![](assets/blend1.png)
    """
    X, Y, kws = util.parseargs(*points)
    a = makebpf(shape0, X, Y)
    b = makebpf(shape1, X, Y)
    return core.blend(a, b, mix)


