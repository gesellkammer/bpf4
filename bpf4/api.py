"""
API for bpf4
"""
from __future__ import annotations

from . import core
from .util import (
    minimum,
    maximum, 
    max_,
    min_,
    sum_,
    select,
    asbpf,
    loadbpf,
    parseargs,
    multi_parseargs
)

from .core import (
    blend,
    brentq,
    BpfInterface,
    BpfBase
)


def linear(*args) -> core.Linear:
    """
    Construct a Linear bpf.

    A bpf can be constructed in multiple ways:

        linear(x0, y0, x1, y1, ...)
        linear((x0, y0), (x1, y1), ...)
        linear({x0:y0, x1:y1, ...})
    """
    X, Y, kws = parseargs(*args)
    if kws:
        raise ValueError("linear does not take any keyword")
    return core.Linear(X, Y)
    

def expon(*args, **kws) -> core.Expon:
    """
    Construct an Expon bpf (a bpf with exponential interpolation)

    A bpf can be constructed in multiple ways:

        expon(x0, y0, x1, y1, ..., exp=exponent)
        expon(exponent, x0, y0, x1, y1, ...)
        expon((x0, y0), (x1, y1), ..., exp=exponent)
        expon({x0:y0, x1:y1, ...}, exp=exponent)
    
    Keywords:
        numiter: Number of iterations. A higher number accentuates the effect
    """
    X, Y, kws = parseargs(*args, **kws)
    assert "exp" in kws
    return core.Expon(X, Y, **kws)
    

def halfcos(*args, exp=1, numiter=1, **kws) -> core.Halfcos:
    """
    Construct a half-cosine bpf (a bpf with half-cosine interpolation)

    A bpf can be constructed in multiple ways:

        halfcos(x0, y0, x1, y1, ...)
        halfcos((x0, y0), (x1, y1), ...)
        halfcos({x0:y0, x1:y1, ...})

    Keywords:
        exp: exponent to apply prior to cosine interpolation. The higher the exponent, the
            more skewed to the right the shape will be
        numiter: Number of iterations. A higher number accentuates the effect
    
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcos(X, Y, exp=exp, numiter=numiter, **kws)  


halfcosexp = halfcos


def halfcosm(*args, **kws) -> core.Halfcosm:
    """
    Similar to halfcos, but when used with an exponent, the exponent is inverted
    for downwards segments (y1 > y0)

    A bpf can be constructed in multiple ways:

        halfcosm(x0, y0, x1, y1, ...)
        halfcosm((x0, y0), (x1, y1), ...)
        halfcosm({x0:y0, x1:y1, ...})

    Keywords:
        exp: exponent to apply prior to cosine interpolation. The higher the exponent, the
            more skewed to the right the shape will be
        numiter: Number of iterations. A higher number accentuates the effect
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcosm(X, Y, **kws)


def spline(*args) -> core.Spline:
    """
    Construct a cubic-spline bpf 

    A bpf can be constructed in multiple ways:

        spline(x0, y0, x1, y1, ...)
        spline((x0, y0), (x1, y1), ...)
        spline({x0:y0, x1:y1, ...})
    
    """
    X, Y, kws = parseargs(*args)
    return core.Spline(X, Y)


def uspline(*args) -> core.USpline: 
    """
    Construct a univariate cubic-spline bpf 

    A bpf can be constructed in multiple ways:

        uspline(x0, y0, x1, y1, ...)
        uspline((x0, y0), (x1, y1), ...)
        uspline({x0:y0, x1:y1, ...})
    
    **NB**: This is implemented by wrapping a UnivariateSpline from scipy.
    """
    X, Y, kws = parseargs(*args)
    return core.USpline(X, Y)


def fib(*args):
    """
    A bpf with fibonacci interpolation

    A bpf can be constructed in multiple ways:

        fib(x0, y0, x1, y1, ...)
        fib((x0, y0), (x1, y1), ...)
        fib({x0:y0, x1:y1, ...})
    
    """
    X, Y, kws = parseargs(*args)
    return core.Fib(X, Y, **kws)    
    

def nointerpol(*args) -> core.NoInterpol:
    """
    A bpf with floor interpolation

    A bpf can be constructed in multiple ways:

        nointerpol(x0, y0, x1, y1, ...)
        nointerpol((x0, y0), (x1, y1), ...)
        nointerpol({x0:y0, x1:y1, ...})
    """
    X, Y, kws = parseargs(*args)
    return core.NoInterpol(X, Y, **kws)
    

def nearest(*args) -> core.Nearest:
    """
    A bpf with floor interpolation

    A bpf can be constructed in multiple ways:

        nearest(x0, y0, x1, y1, ...)
        nearest((x0, y0), (x1, y1), ...)
        nearest({x0:y0, x1:y1, ...})
    """
    X, Y, kws = parseargs(*args)
    return core.Nearest(X, Y, **kws)


def smooth(*args, numiter=1) -> core.Smooth:
    """
    A bpf with smoothstep interpolation. `numiter` determines the number
    of smoothstep steps applied (see https://en.wikipedia.org/wiki/Smoothstep)

    A bpf can be constructed in multiple ways:

        smooth(x0, y0, x1, y1, ...)
        smooth((x0, y0), (x1, y1), ...)
        smooth({x0:y0, x1:y1, ...})
    
    Keywords:
        numiter: number of smoothstep steps. 
    """
    X, Y, kws = parseargs(*args)
    return core.Smooth(X, Y, numiter=numiter)


def smoother(*args) -> core.Smoother:
    """
    A bpf with smootherstep interpolation (Perlin's variation on smoothstep,
    see https://en.wikipedia.org/wiki/Smoothstep)

    A bpf can be constructed in multiple ways:

        smoother(x0, y0, x1, y1, ...)
        smoother((x0, y0), (x1, y1), ...)
        smoother({x0:y0, x1:y1, ...})
    
    """
    X, Y, kws = parseargs(*args)
    return core.Smoother(X, Y)


def multi(*args):
    """
    A bpf with a per-pair interpolation

    Example
    ========

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
    """
    xs, ys, interpolations = multi_parseargs(args)
    return core.Multi(xs, ys, interpolations)


def pchip(*args):
    """
    Monotonic Cubic Hermite Intepolation

    A bpf can be constructed in multiple ways:

        pchip(x0, y0, x1, y1, ...)
        pchip((x0, y0), (x1, y1), ...)
        pchip({x0:y0, x1:y1, ...})
    
    """
    from . import pyinterp
    xs, ys, kws = parseargs(*args)
    return pyinterp.Pchip(xs, ys, **kws)


def const(value) -> core.Const:
    """
    A bpf which always returns a constant value

    Example
    =======
    
    >>> c5 = const(5)
    >>> c5(10) 
    5
    
    """
    return core.Const(value)


def slope(slope:float, offset=0., keepslope=True) -> core.Slope:
    """
    Generate a straight line with the given slope and offset (the 
    same as linear(0, offset, 1, slope)
    """
    return core.Slope(slope, offset)


def blendshape(shape0:str, shape1:str, mix:float, *args) -> core.BlendShape:
    X, Y, kws = parseargs(*args)
    return core.BlendShape(X, Y, shape0, shape1, mix)


