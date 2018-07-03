"""
API for bpf4
"""
from . import core
from .util import (
    _bpfconstr,
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
)


def linear(*args):
    # type: (...) -> core.Linear
    """
    Example: define an linear BPF with points = 0:0, 1:5, 2:20

    These all do the same thing:

    linear(0, 0, 1, 5, 2, 20)  # fast but rather unclear
    linear((0, 0), (1, 5), (2, 20))
    linear({0:0, 1:5, 2:20})
    """
    X, Y, kws = parseargs(*args)
    if kws:
        raise ValueError("linear does not take any keyword")
    return core.Linear(X, Y)
    # return _bpfconstr('linear', *args)


def expon(*args, **kws):
    # type: (...) -> core.Expon
    """
    Example: define an exponential BPF with exp=2 and points = 0:0, 1:5, 2:20

    These all do the same thing:

    expon(2, 0, 0, 1, 5, 2, 20)
    expon(2, (0, 0), (1, 5), (2, 20))
    expon(2, {0:0, 1:5, 2:20})
    expon(0, 0, 1, 5, 2, 20, exp=2)
    expon((0,0), (1, 5), (2, 20), exp=2)
    expon({0:0, 1:5, 2:20}, exp=2)

    other kws:
    * numiter=1  number of iterations. A higher number accentuates the effect
    """
    X, Y, kws = parseargs(*args, **kws)
    assert "exp" in kws
    return core.Expon(X, Y, **kws)
    # return _bpfconstr('expon', *args, **kws)


def halfcos(*args, **kws):
    # type: (...) -> core.Halfcos
    """
    halfcos(x0, y0, x1, y1, ..., xn, yn)
    halfcos((x0, y0), (x1, y1), ..., (xn, yn))
    halfcos({x0:y0, x1:y1, x2:y2})

    As a first parameter you can define an exp:

    halfcos(exp, x0, y0, x1, y1, ..., xn, yn)

    Or as a keyword argument at the end:
    halfcos(x0, y0, x1, y1, ..., xn, yn, exp=0.5)

    other kws:
    * numiter=1  number of iterations. A higher number accentuates the effect
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcos(X, Y, **kws)  


halfcosexp = halfcos


def halfcosm(*args, **kws):
    # type: (...) -> core.Halfcosm
    """
    Similar to halfcos, but when used with an exponent, the exponent is inverted
    for downwards segments (y1 > y0)

    See halfcos for doc
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcosm(X, Y, **kws)


def halfcos2(*args, **kws):
    # type: (...) -> core.Halfcos
    """
    A shortcut for halfcos(..., numiter=2)
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcos(X, Y, numiter=2, **kws)


def halfcos2m(*args, **kws):
    # type: (...) -> core.Halfcosm
    """
    A shortcut for halfcosm(..., numiter=2)
    """
    X, Y, kws = parseargs(*args, **kws)
    return core.Halfcosm(X, Y, numiter=2, **kws)


def spline(*args) -> core.Spline:
    """
    Example: define a spline BPF with points = 0:0, 1:5, 2:20

    These all do the same thing:

    spline(0, 0, 1, 5, 2, 20)  # fast but rather unclear
    spline((0, 0), (1, 5), (2, 20))
    spline({0:0, 1:5, 2:20})
    """
    X, Y, kws = parseargs(*args)
    return core.Spline(X, Y)


def uspline(*args):
    """
    BPF with univariate spline interpolation. This is implemented by
    wrapping a UnivariateSpline from scipy.
    """
    X, Y, kws = parseargs(*args)
    return core.USpline(X, Y)


def fib(*args):
    """
    Example: define a fib BPF with points = 0:0, 1:5, 2:20

    These all do the same thing:

    fib(0, 0, 1, 5, 2, 20)  # fast but rather unclear
    fib((0, 0), (1, 5), (2, 20))
    fib({0:0, 1:5, 2:20})
    """
    return _bpfconstr('fib', *args)


def nointerpol(*args):
    """
    Example: define an nointerpol BPF with points = 0:0, 1:5, 2:20

    These all do the same thing:

    nointerpol(0, 0, 1, 5, 2, 20)  # fast but rather unclear
    nointerpol((0, 0), (1, 5), (2, 20))
    nointerpol({0:0, 1:5, 2:20})

    nointerpol().fromseq(0, 0, 1, 5, 2, 20)
    nointerpol().fromxy([0, 1, 5], [0, 5, 20])
    """
    return _bpfconstr('nointerpol', *args)


def nearest(*args):
    """
    a BPF with nearest interpolation
    """
    return _bpfconstr('nearest', *args)


def smooth(*args, numiter=1):
    # type: (...) -> core.Smooth
    X, Y, kws = parseargs(*args)
    return core.Smooth(X, Y, numiter=numiter)


def recurse(N, bpf):
    if isinstance(N, int):
        out = bpf
        for _ in range(N):
            out = out | bpf
        return out
    else:
        raise TypeError("bpf: expected a BPF")


def multi(*args):
    """
    Example: define the following BPF  
    (0,0) --linear-- (1,10) --expon(3)-- (2,3) --expon(3)-- (10, -1) --halfcos-- (20,0)

    multi(
        0, 0, 
        1, 10, 'linear', 
        2, 3, 'expon(3)', 
        10, -1,       # assumes previus interpolation
        20, 0, 'halfcos')
    multi(
         0, 0, 
        (1, 10, 'linear'), 
        (2, 3, 'expon(3)), 
        (10, -1), 
        (20, 0, 'halfcos')
    )
    """
    xs, ys, interpolations = multi_parseargs(args)
    return core.Multi(xs, ys, interpolations)


def pchip(*args):
    """
    Monotonic Cubic Hermite Intepolation

    These all do the same thing:

    pchip(0, 0, 1, 5, 2, 20)  # fast but rather unclear
    pchip((0, 0), (1, 5), (2, 20))
    pchip({0:0, 1:5, 2:20})

    pchip().fromseq(0, 0, 1, 5, 2, 20)
    pchip().fromxy([0, 1, 5], [0, 5, 20])
    """
    from . import pyinterp
    args = parseargs(*args)
    return pyinterp.Pchip(args.xs, args.ys)


def const(value):
    """
    a constant bpf

    >>> c5 = const(5)
    >>> c5(10) 
    5
    """
    return core.Const(value)


def slope(slope, offset=0, keepslope=True):
    """
    generate a straight line with the given slope and offset (the 
    same as linear(0, offset, 1, slope)
    """
    return core.Slope(slope, offset)


def blendshape(shape0, shape1, mix, *args):
    X, Y, kws = parseargs(*args)
    return core.BlendShape(X, Y, shape0, shape1, mix)

BpfInterface = core._BpfInterface
