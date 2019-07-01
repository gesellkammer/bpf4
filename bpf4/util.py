"""
Utilities for bpf4
"""
import operator as _operator
import os as _os
import itertools as _itertools
from functools import reduce
from typing import Sequence as Seq

import numpy as np
from scipy.integrate import quad as _quad
from scipy.optimize import brentq as _brentq

from . import core

_CSV_COLUMN_NAMES = ('x', 'y', 'interpolation', 'exponent')

_CONSTRUCTORS = {
    'linear': core.Linear,
    'expon': core.Expon,
    'halfcos': core.Halfcos,
    'nointerpol': core.NoInterpol,
    # 'halfcosexp': core.HalfcosExp,
    'spline': core.Spline,
    'uspline': core.USpline,
    'fib': core.Fib,
    'slope': core.Slope,
    # 'halfcos2': core.Halfcos2,
    'nearest': core.Nearest,
    # 'halfcos2m': core.Halfcos2m,
    # 'halfcosexpm': core.HalfcosExpm,
    'halfcosm': core.Halfcosm,
    'smooth': core.Smooth
}


def _isiterable(obj) -> bool:
    try:
        iter(obj)
        return not isinstance(obj, str)
    except TypeError:
        return False


def _iflatten(seq):
    """
    return an iterator to the flattened items of sequence s
    strings are not flattened
    """
    try:
        iter(seq)
    except TypeError:
        yield seq
    else:
        for elem in seq:
            if isinstance(elem, str):
                yield elem
            else:
                for subelem in _iflatten(elem):
                    yield subelem

def csv_to_bpf(csvfile):
    """
    Read a bpf from a csv file
    """
    from emlib import csvtools
    rows = csvtools.readcsv(csvfile)
    interpolation = rows[0].interpolation
    if all(row.interpolation == interpolation for row in rows[:-1]):
        # all of the same type
        if interpolation in ('expon', 'halfcosexp', 'halfcos2'):
            exp = rows[0].exponent
            constructor = get_bpf_constructor("%s(%.3f)" % (interpolation, exp))
        else:
            constructor = get_bpf_constructor(interpolation)
        numrows = len(rows)
        xs = np.empty((numrows,), dtype=float)
        ys = np.empty((numrows,), dtype=float)
        for i in range(numrows):
            r = rows[i]
            xs[i] = r.x
            ys[i] = r.y
        return constructor(xs, ys)
    else:
        # multitype
        raise NotImplementedError("BPFs with multiple types not implemented YET")


def bpf_to_csv(bpf, csvfile):
    """
    Write this bpf as a csv representation
    """
    import csv
    csvfile = _os.path.splitext(csvfile)[0] + '.csv'
    try:
        # it follows the 'segments' protocol, returning a seq of (x, y, interpoltype, exp)
        segments = bpf.segments()
        with open(csvfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMN_NAMES)
            writer.writerows(segments)
    except AttributeError:
        raise TypeError("BPF must be rendered in order to be written as CSV")


def bpf_to_dict(bpf):
    """
    convert a bpf to a dict with the following format

    b = bpf.expon(3.0, 0, 0, 1, 10, 2, 20)
    bpf_to_dict(b)
    {
        'interpolation': 'expon(3.0)',
        'points': [0, 0, 1, 10, 20, 20]  # [x0, y0, x1, y1, ...]
    }

    b = bpf.multi(0, 0, 'linear',
                  1, 10, 'expon(2)',
                  3, 25)
    bpf_to_dict(b)
    {
        'interpolation': 'multi',
        'segments': [
            [0, 0, 'linear'],
            [1, 10, 'expon(2)',
            [3, 25, '']]
    }
    """
    try:
        segments = list(bpf.segments())
    except:
        raise TypeError("this kind of BPF cannot be translated. It must be rendered first.")
    d = {}
    interpolation = segments[0][2]
    # x y interpolation exp

    def normalize_segment(segment):
        """
        a segment as returned by .segments() is [x, y, interpl, exp]
        we want [x, y, interpol(exp)]
        """
        if len(segment) == 4:
            x, y, interpol, exp = segment
            if not interpol and exp == 0:
                # the last segment of a multi bpf
                segment = [x, y]
            else:
                interpol = "{interpol}({exp})".format(interpol=interpol, exp=exp)
                segment = [x, y, interpol]
        return segment

    if not all(segment[2] == interpolation for segment in segments[:-1]):
        # multi
        d['interpolation'] = 'multi'
        segments = [normalize_segment(seg) for seg in segments]
        d['segments'] = segments
    else:
        try:
            exp = bpf.exp
            if exp != 1:
                assert "(" not in interpolation
                interpolation = "{interp}({exp})".format(interp=interpolation, exp=exp)
        except AttributeError:
            pass
        d['interpolation'] = interpolation
        points = []
        for segment in segments:
            points.append(segment[0])
            points.append(segment[1])
        d['points'] = points
    return d


def bpf_to_json(bpf, outfile=None):
    """
    convert this bpf to json format.
    If outfile is not given, it returns a string, as in dumps

    kws are passed directly to json.dump
    """
    import json
    asdict = bpf_to_dict(bpf)
    if outfile is None:
        return json.dumps(asdict)
    else:
        with open(outfile, 'w') as f:
            json.dump(asdict, f)


def bpf_to_yaml(bpf, outfile=None):
    """
    convert this bpf to json format. if outfile is not given, it returns a string, as in dumps
    """
    from io import StringIO
    import yaml
    d = bpf_to_dict(bpf)
    if outfile is None:
        stream = StringIO()
    else:
        outfile = _os.path.splitext(outfile)[0] + '.yaml'
        stream = open(outfile, 'w')
    dumper = yaml.Dumper(stream)
    dumper.add_representer(tuple, lambda du, instance: du.represent_list(instance))
    dumper.add_representer(np.float64, lambda du, instance: du.represent_float(instance))
    dumper.open()
    dumper.represent(d)
    dumper.close()
    if outfile is None:
        return stream.getvalue()


def dict_to_bpf(d):
    """
    Format 1: 

    bpf = {
        'interpolation': 'expon(2)',
        10: 0.1,
        15: 1,
        25: -1
    }

    Format 2:

    bpf = {
        'interpolation': 'linear',
        'points': [x0, y0, x1, y1, ...]
    }

    Format 2b

    bpf = {
        'interpolation': 'linear',
        'points': [(x0, y0), (x1, y1), ...]
    }

    Format 3 (multi)

    bpf = {
        'interpolation': 'multi',
        'segments': [
            [x0, y0, 'descr0'],
            [x1, y1, 'descr1'],
            ...
            [xn, yn, '']
        ]
    }
    """

    # check format
    if 'points' in d:
        # format 2
        interpolation = d.get('interpolation', 'linear')
        constructor = get_bpf_constructor(interpolation)
        points = d['points']
        if isinstance(points[0], (int, float)):
            # format 2a
            X = points[::2]
            Y = points[1::2]
        elif isinstance(points[0], (list, tuple)):
            X = [point[0] for point in points]
            Y = [point[1] for point in points]
        return constructor.fromxy(X, Y)
    elif 'segments' in d and d['interpolation'] == 'multi':
        segments = d['segments']
        X = [s[0] for s in segments]
        Y = [s[1] for s in segments]
        interpolations = [s[2] for s in segments[:-1]]
        return core.Multi(X, Y, interpolations)
    else:
        # format 1
        interpolation = d.get('interpolation', 'linear')
        constructor = get_bpf_constructor(interpolation)
        points = [(k, v) for k, v in d.items() if isinstance(k, (int, float))]
        points.sort()
        X, Y = list(zip(*points))
        return constructor.fromxy(X, Y)


def loadbpf(path, fmt='auto'):
    """
    load a bpf saved with dumpbpf

    Possible formats: auto, csv, yaml, json
    """
    if fmt == 'auto':
        fmt = _os.path.splitext(path)[-1].lower()[1:]
    assert fmt in ('csv', 'yaml', 'json')
    if fmt == 'yaml':
        import yaml
        d = yaml.load(open(path))
    elif fmt == 'json':
        import json
        d = json.load(path)
    elif fmt == 'csv':
        return csv_to_bpf(path)
    return dict_to_bpf(d)  


def asbpf(obj, bounds=(-np.inf, np.inf)) -> core._BpfInterface:
    """
    Convert obj to a bpf

    obj can be a function, a dict, a constant, or a bpf (in which case it
    is returned as is)
    """
    if isinstance(obj, core._BpfInterface):
        return obj
    elif callable(obj):
        return core._FunctionWrap(obj, bounds)
    elif hasattr(obj, '__float__'):
        return core.Const(float(obj))
    else:
        raise TypeError("can't wrap %s" % str(obj))
 

def parseargs(*args, **kws):
    """
    Convert the args and kws to the canonical form:
    
    xs, ys, kws

    Raises ValueError if failed
    """
    L = len(args)
    if L == 0:
        raise ValueError("no arguments given")
    elif L == 1:
        d = args[0]
        assert isinstance(d, dict)
        items = list(d.items())
        items.sort()
        xs, ys = list(zip(*items))
    elif L == 2:
        if not all(map(_isiterable, args)):
            raise ValueError(f"parsing error: 2 args, expected a seq. [(x0, y1), (x1, y1)] or (xs, ys) but got {args}")
        if len(args[0]) > 2:   # <--  (xs, ys)
            xs, ys = args
        else:                  # <--  ((x0, y0), (x1, y1))
            xs, ys = list(zip(*args))    
    elif not any(map(_isiterable, args)):
        # (x0, y0, x1, y1, ...)
        if L % 2 == 0:  # even
            xs = args[::2]
            ys = args[1::2]
        else:
            if kws:
                raise ValueError(f"Uneven number of args. No keywords allowed in this case, but got {kws}")
            kws = {'exp': args[0]}
            xs = args[1::2]
            ys = args[2::2]
    elif all(map(_isiterable, args)):   # <-- ((x0, y0), (x1, y1), ...)
        xs, ys = list(zip(*args))
    else:
        raise ValueError("could not parse arguments")
    return xs, ys, kws
        

def makebpf(descr:str, X:Seq[float], Y:Seq[float]) -> core._BpfInterface:
    """
    - descr: a string descriptor of the interpolation
        * linear
        * expon(x)
        * etc.
    - X: the array of xs
    - Y: the array of ys
    """
    return _bpfconstr(descr).fromxy(X, Y)


def _bpfconstr(descr, *args, **kws):
    """
    given a descriptor of an interpolation function, return 
    a constructor for a BPF with the given interpolation

    Examples:

    different ways of defining a bpf with points (0, 0), (1, 100), (2, 20)
    
    bpf('linear', (x0, x1, x2, x3), (y0, y1, y2, y3))

    bpf('linear')
    bpf('linear', 0, 0, 1, 100, 2, 20)
    bpf('linear', {0:0, 1:100, 2:20})
    bpf('linear', (0, 0), (1, 100), (2, 20))
    bpf('linear')(0, 0, 1, 100, 2, 20)
    bpf('linear').fromxy([0, 1, 2], [0, 100, 20])
    bpf('expon', 2)(0, 0, 1, 100, 2, 20)
    bpf('expon', 2, {0:0, 1:100, 2:20})
    bpf('expon(2)', 0, 0, 1, 100, 2, 20)
    bpf('expon:2', 0, 0, 1, 100, 2, 20)
    bpf('expon', 2, 0, 0, 1, 100, 2, 20)
    bpf('expon', 0,0, 1,100, 2,20, exp=2)
    
    Or using the methods defined for each interpolation type:
    
    bpf.linear(0, 0, 1, 100, 2, 20)
    bpf.linear({0:0, 1:100, 2:20})
    bpf.expon(0, 0, 1, 100, 2, 20, exp=2)
    bpf.expon(2, 0, 0, 1, 100, 2, 20)
    bpf.expon(2, (0, 0), (1, 100), (2, 20))
    bpf.halfcos({0:0, 1:100, 2:20}, exp=2)  # if you want a halfcosexp with exp=2
    bpf.halfcos(2, {0:0, 1:100, 2:20})      # if you want a halfcosexp with exp=2
    bpf.halfcos(0, 0, 1, 100, 2, 20)        # a normal halfcos (ie. exp=1)
    
    etc.
    
    """
    param = None
    if '(' in descr:
        descr, param = descr.split('(')
        param = float(param[:-1])    
    elif ':' in descr:
        descr, param = descr.split(':')
    else:
        if kws:
            exp = kws.get('exp')
            if exp is not None:
                param = exp
            if isinstance(args[0], dict):
                args = list(args[0].items())
                args.sort()
        else:
            l = len(args)
            if l == 1:
                p = args[0]
                if isinstance(p, dict):
                    param = None
                    args = list(p.items())
                    args.sort()
                elif descr == 'linear':
                    return core.Slope(args[0])
                else:
                    param = p
                    args = ()
            elif l > 1:
                if l == 2 and all(_isiterable(arg) for arg in args) and all(len(arg) > 2 for arg in args):
                    # (x0, x1, x2, ..., xn), (y0, y1, y2, ..., yn)
                    constructor = _CONSTRUCTORS.get(descr)
                    return _BpfConstructor(constructor, param).fromxy(*args)
                elif any(_isiterable(arg) for arg in args):
                    if not _isiterable(args[0]):  # exp, (x0, y0), (x1, y1)
                        param = args[0]
                        args = args[1:]
                    elif not _isiterable(args[-1]):  # (x0, y0), (x1, y1), exp
                        param = args[-1]
                        args = args[:-1]
                    else:
                        param = None
                else:
                    if l % 2:  # exp, x0, y0, x1, y1
                        param = args[0]
                        args = args[1:]
                    else:
                        param = None
    constructor = _CONSTRUCTORS.get(descr)
    if not constructor:
        raise ValueError("no bpf of type %s" % descr)
    if not args:
        return _BpfConstructor(constructor, param)
    else:
        if isinstance(args[0], dict):
            return _BpfConstructor(constructor, param).fromdict(args[0])
        return _BpfConstructor(constructor, param).fromseq(*args)

    
get_bpf_constructor = _bpfconstr

    
class _BpfConstructor:
    def __init__(self, constructor, exp=None):
        # do minor error checking
        t = constructor.__name__.lower()
        if t in ('halfcos', 'halfcosm'):
            if exp is not None:
                constructor = core.HalfcosExp
            else:
                assert exp is None
        elif t in ('halfcos2', 'halfcos2m', 'halfcosexpm'):
            if exp is None:
                exp = 1
        elif t in ('linear', 'nointerpol', 'nearest', 'spline', 'fib', 'slope', 'uspline'):
            assert exp is None
        else:
            assert exp is not None
        self.constructor = constructor
        self.exp = exp

    def __call__(self, *args, **kws):
        X, Y, kws = parseargs(*args, **kws)
        return self.constructor(X, Y, **kws)

    def fromseq(self, *seq, **kws):
        k = {'exp':self.exp} if self.exp is not None else {}
        k.update(kws)
        return self.constructor.fromseq(*seq, **k)

    def fromxy(self, xs, ys, **kws):
        k = {'exp':self.exp} if self.exp is not None else {}
        k.update(kws)
        return self.constructor(xs, ys, **k)


def multi_parseargs(args):
    flattened_args = list(_iflatten(args))
    if len(flattened_args) == len(args):
        # it is already a flat list
        args = flattened_args
        points = []
        interpolations = []
        point_open = False
        last_interpolation = 'linear'
        interpol_mode = None
        xs = [args[0]]
        ys = [args[1]]
        lasty = ys[0]
        for i, arg in enumerate(args[2:]):
            if not point_open:
                point_open = True
                interpolation = None
                point = []
                if interpol_mode == 'first':
                    if isinstance(arg, str):
                        interpolation = arg
                    else:
                        point.append(float(arg))
                        interpolation = last_interpolation
                elif interpol_mode == 'last':
                    point.append(float(arg))
                else:
                    if isinstance(arg, str):
                        interpol_mode = 'first'
                        interpolation = arg
                    else:
                        interpol_mode = 'last'
                        point.append(float(arg))
            else:
                L = len(point)
                if interpol_mode == 'first':
                    assert interpolation is not None
                    if L == 0:
                        point.append(float(arg))
                    elif L == 1:
                        point.append(float(arg) if arg is not None else lasty)
                        points.append((point, interpolation))  # <--- 
                        point_open = False
                        lasty = point[1]
                        last_interpolation = interpolation
                        
                    else:
                        raise ValueError
                elif interpol_mode == 'last':
                    if L == 0:
                        raise ValueError
                    elif L == 1:
                        point.append(float(arg) if arg is not None else lasty)
                    elif L == 2:
                        if not isinstance(arg, str):
                            interpolation = last_interpolation
                            points.append((point, interpolation))  # <---
                            lasty = point[1]
                            last_interpolation = interpolation
                            point_open = False
                        else:
                            assert interpolation is None
                            interpolation = arg
                            points.append((point, interpolation))  # <---
                            lasty = point[1]
                            last_interpolation = interpolation
                            point_open = False
        if point_open:
            points.append((point, last_interpolation))
        for point, interpolation in points:
            x, y = point
            xs.append(x)
            ys.append(y)
            interpolations.append(interpolation)
        return xs, ys, interpolations
    else:
        # it is of the type (x0, y0), (x1, y1, interpolation), ...
        last_interpolation = 'linear'
        x0, y0 = args[0]
        xs = [x0]
        ys = [y0]
        lasty = y0
        interpolations = []
        for point in args[1:]:  # we skip the first point
            if len(point) == 2:
                interpolation = last_interpolation
                x, y = point
            else:
                x, y = (j for j in point if not isinstance(j, str))
                interpolation = next((j for j in point if isinstance(j, str)))
            xs.append(x)
            y = y if y is not None else lasty
            lasty = y
            ys.append(y)
            interpolations.append(interpolation)
        return xs, ys, interpolations    
        
# Bpf = _Bpf()

        
def _bpf_from_pairs(interpolation, pairs):
    xs, ys = list(zip(*pairs))
    return get_bpf_constructor(interpolation)(xs, ys)

    
def max_(*elements) -> core.Max:
    """
    Return a bpf representing the max over elements
    """
    bpfs = list(map(asbpf, elements))
    return core.Max(*bpfs)

    
def min_(*elements) -> core.Min:
    """
    Return a bpf representing the min over elements
    """
    bpfs = list(map(asbpf, elements))
    return core.Min(*bpfs)

    
def sum_(*elements):
    """
    Return a bpf representing the sum of elements
    """
    bpfs = list(map(asbpf, elements))
    return reduce(_operator.add, bpfs)


def select(which, bpfs) -> core._BpfSelect:
    """
    which returns at any x, which bpf from bpfs should return the result
    
    >>> which = nointerpol(0, 0, 5, 1)
    >>> bpfs = [linear(0, 0, 10, 10), linear(0, 10, 10, 0)]
    >>> s = select(which, bpfs)
    >>> s(1)     # at time=1, the first bpf will be selected
    0
    """
    return core._BpfSelect(asbpf(which), list(map(asbpf, bpfs)))

    
def dumpbpf(bpf, fmt='yaml', outfile=None):
    """
    Dump the data of this bpf as human readable text to a file 
    or to a string (if no outfile is given)
    
    If outfile is given, its extension will be used to determine
    the format.

    The bpf can then be reconstructed via `loadbpf`

    Formats supported: csv, yaml, json
    """
    if fmt == 'csv':
        if outfile is None:
            raise ValueError("need an outfile to dump to CSV")
        return bpf_to_csv(bpf, outfile)
    elif fmt == 'json':
        return bpf_to_json(bpf, outfile)
    elif fmt == 'yaml':
        return bpf_to_yaml(bpf, outfile)
    else:
        # we interpret it as a filename, the format should be the extention
        base, ext = _os.path.splitext(fmt)
        if ext in ('.csv', '.json', '.yaml'):
            outfile = fmt
            fmt = ext[1:]
            return dumpbpf(bpf, fmt, outfile)
        else:
            raise ValueError("format not understood or not supported.")

            
def concat_bpfs(bpfs, fadetime=0) -> core._BpfConcat:
    """
    glue these bpfs together, one after the other
    """
    if fadetime != 0:
        raise ValueError("fade not implemented")
    bpfs2 = [bpfs[0]]
    x0, x1 = bpfs[0].bounds()
    xs = [x0]
    for bpf in bpfs[1:]:
        bpf2 = bpf.fit_between(x1, x1 + (bpf._x1 - bpf._x0))
        bpfs2.append(bpf2)
        xs.append(bpf2._x0)
        x0, x1 = bpf2.bounds()
    return core._BpfConcat(xs, bpfs2)

    
def integrate(bpf, bounds=None):
    """
    integrate the given bpf between the given bounds (or use the bpf bounds if not given)
    
    It uses the quad algorithm in scipy.integrate
    """
    if bounds is None:
        x0, x1 = bpf.bounds()
    else:
        x0, x1 = bounds
    return _quad(bpf, x0, x1)[0]


def warped(bpf, dx=None, numpoints=1000):
    """
    bpf represents the curvature of a linear space. the result is a 
    warped bpf so that:
    
    position_bpf | warped_bpf = corresponding position after warping
    
    dx: the accuracy of the measurement
    numpoints: if dx is not given, the bpf is sampled `numpoints` times
               across its bounds
    
    Example:
    find the theoretical position of a given point according to a probability distribution
    
    distribution = bpf.halfcos(0,0, 0.5,1, 1, 0)
    w = warped(distribution)
    original_points = (0, 0.25, 0.33, 0.5)
    warped_points = w.map(original_points)
    """
    x0, x1 = bpf.bounds()
    if dx is None:
        dx = (x1 - x0) / numpoints
    integrated = bpf.integrated()[::dx]
    integrated_at_x1 = integrated(bpf.x1)
    # N = int((x1 + dx - x0) / dx + 0.5)
    xs = np.arange(x0, x1+dx, dx)    
    ys = np.ones_like(xs) * np.nan
    for i in range(len(xs)):
        try:
            ys[i] = _brentq(integrated - xs[i]*integrated_at_x1, x0, x1)
        except:
            pass
    return core.Sampled(ys, dx=dx, x0=x0)

        
def _minimize(bpf, N, func=min, debug=False):
    x0, x1 = bpf.bounds()
    xs = np.linspace(x0, x1, N)
    from scipy.optimize import brent
    mins = [brent(bpf, brack=(xs[i], xs[-i])) for i in range(int(len(xs) * 0.5 + 0.5))]
    mins2 = [(bpf(m), m) for m in mins]  # if x0 <= m <= x1]
    if debug:
        print(mins2)
    if mins2:
        return float(func(mins2)[1])
    return None

    
def minimum(bpf, N=10):
    """
    return the x where bpf(x) is the minimum of bpf
    
    N: number of estimates
    """
    return _minimize(bpf, N, min)


def maximum(bpf, N=10):
    """
    return the x where bpf(x) is the maximum of bpf
    """
    return _minimize(-bpf, N, min)
    

def rms(bpf, rmstime=0.1):
    bpf2 = bpf**2
    from math import sqrt

    def func(x):
        return sqrt(bpf2.integrate_between(x, x+rmstime) / rmstime)
    return asbpf(func).set_bounds(bpf.x0, bpf.x1)
 

def binarymask(mask, durs=None, offset=0, cycledurs=True):
    """
    Creates a binary mask

    mask: a sequence of states. A state is either 0 or 1
          and can be represented also by 'x'(1) or 'o', '-'(0)
    durs: a sequence of durations (default=[1])

    Example
    =======

    mask = create_mask("x--x-x---")

    """
    if durs is None:
        durs = [1]
    if cycledurs:
        durs = _itertools.cycle(durs)
    else:
        assert len(durs) == len(mask)

    def binvalue(x):
        d = {'x':1, 'o':0, '-':0, 1:1, 0:0, '1':1, '0':0}
        return d.get(x)

    mask = [binvalue(x) for x in mask]
    mask.append(mask[-1])
    assert all(x is not None for x in mask)
    t = offset
    times = []
    for i, dur in zip(range(len(mask)), durs):
        times.append(t)
        t += dur
    return core.NoInterpol(times, mask)


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = _itertools.tee(iterable)
    try:
        next(b)
    except StopIteration:
        pass
    return zip(a, b)

    
def jagged_band(xs, upperbpf, lowerbpf=0, curve='linear'):
    """
    create a jagged bpf between lowerbpf and upperbpf at the x
    values given by xs
    
    At each x in xs the, the value is equal to lowerbpf, sweeping
    with curvature 'curve' to upperbpf just before the next x
    """
    constructor = get_bpf_constructor(curve)
    upperbpf = asbpf(upperbpf)
    lowerbpf = asbpf(lowerbpf)
    EPSILON = 1e-12
    fragments = []
    if xs[0] > upperbpf.x0 > float('-inf'):
        xs = [upperbpf.x0] + xs
    if xs[-1] < upperbpf.x1 < float('inf'):
        xs.append(upperbpf.x1)
    for x0, x1 in _pairwise(xs[:-1]):
        x1 = x1 - EPSILON
        fragment = constructor(x0, lowerbpf(x0), x1, upperbpf(x1))[x0:x1].outbound(0, 0)
        fragments.append(fragment)
    x0 = xs[-2]
    x1 = xs[-1]
    fragments.append(constructor(x0, lowerbpf(x0), x1, upperbpf(x1))[x0:x1].outbound(0, 0))
    return sum_(*fragments)
    

def randombw(bw, center=1):
    """
    Create a random bpf
    
    bw: a (time-varying) bandwidth
    center  : the center of the random distribution
    
    if randombw is 0.1 and center is 1, the bpf will render values 
    between 0.95 and 1.05

    NB: this bpf will always be different, since the random numbers
    are calculated as needed. Sample it to freeze it to a known state.

    Example
    =======

    >>> l = bpf.linear(0, 0, 1, 1)
    >>> r = bpf.util.randombw(0.1)
    >>> l2 = (l*r)[::0.01]
    """
    bw = asbpf(bw)
    return (bw.rand() + (center - bw*0.5))[bw.x0:bw.x1]
    

def blendwithfloor(b, mix=0.5):
    return core.blend(b, asbpf(b(maximum(b))), mix)[b.x0:b.x1]
    # return b.blendwith(asbpf(b(minimum(b))), mix)[b.x0:b.x1]

    
def blendwithceil(b, mix=0.5):
    return core.blend(b, asbpf(b(maximum(b))), mix)[b.x0:b.x1]
    # return b.blendwith(asbpf(b(maximum(b))), mix)[b.x0:b.x1]


def smoothen2(bpf, window, numproj=7):
    """
    bpf: a bpf
    window: length (in the x coord) of the smoothing window
    numproj: (int) amount of projections to use. Should be an uneven value
    """
    n = numproj
    alpha = window / float(n)
    projs_left = [bpf << (alpha*i) for i in range(int(n / 2))]
    projs_right = [bpf >> (alpha*i) for i in range(int(n / 2))]
    bsmooth = (sum(projs_left) + sum(projs_right) + bpf) / (len(projs_left) + len(projs_right) + 1)
    return bsmooth[bpf.x0:bpf.x1]


def smoothen(b, window, N=1000):
    """
    b      : a bpf
    window : the width (in x coords) of the smoothing window
    N      : number of points to resample the bpf
    """
    dx = min((b.x1 - b.x0) / N, window/7)
    nwin = int(window / dx)
    box = np.ones(nwin)/nwin
    Y = b[::dx].ys
    Y2 = np.convolve(Y, box, mode="same")
    X = np.linspace(b.x0, b.x1, len(Y2))
    return core.Linear.fromxy(X, Y2)


def arrayslice(x0, x1, X, *Ys):
    """
    Slice a sorted array and linked arrays as if they where a bpf

    x0, x1: where to perform the slices
    x: the array used to perform the slice
    ys: one or more secondary arrays which represent a linear bpf,
        where y = f(x) 
    
    Example
    =======

    X = np.linspace(0, 10, 11, dtype=float)
    Y = X*2

    x, y = arrayslice(3.5, 7, X, Y)
   
    """
    if x0 >= x1:
        raise ValueError("x0 should be less than x1")

    if x0 > X[0]:
        i0 = np.searchsorted(X, x0) - 1    
    else:
        i0 = 0
        x0 = X[0]

    if x1 < X[-1]:
        i1 = np.searchsorted(X, x1) + 1
    else:
        i1 = len(X)
        x1 = X[-1]

    X2 = X[i0:i1].copy()
    X2[0] = x0
    X2[-1] = min(x1, X2[-1])
    out = [X2]  
    for Y in Ys:
        Y2 = Y[i0:i1].copy()
        y0, y1 = np.interp((x0, x1), X, Y)
        Y2[0] = y0
        Y2[-1] = y1
        out.append(Y2)
    return out 
    

def linearslice(linearbpf, x0, x1):
    X, Y = linearbpf.points()
    X, Y = arrayslice(x0, x1, X, Y)
    return core.Linear(X, Y)


def _linearslice(linearbpf, x0, x1):
    """
    Slice the given bpf, returning a new Linear bpf with endpoints
    x0 and x1.
    """
    assert isinstance(linearbpf, core.Linear)
    X, Y = linearbpf.points()
    insert_head = x0 > X[0]
    if insert_head:
        i = np.searchsorted(X, x0)
        X = X[i-1:]
        Y = Y[i-1:]
    insert_tail = x1 < X[-1]
    if insert_tail:
        i = np.searchsorted(X, x1)
        X = X[:i+1]
        Y = Y[:i+1]
    if insert_head or insert_tail:
        # we copy when we know exactly how much to copy
        X = X.copy()
        Y = Y.copy()
    if insert_head:
        X[0] = x0
        Y[0] = linearbpf(x0)
    if insert_tail:
        X[i] = x1
        Y[i] = linearbpf(x1)    
    return core.Linear(X, Y)


def linear_inverted(linearbpf):
    res = core._array_issorted(linearbpf.ys)
    if res == -1:  # not sorted
        raise ValueError(f"bpf can't be inverted, ys should be always increasing.\nys={linearbpf.ys}")
    elif res == 1: # sorted, no dups
        ys = linearbpf.ys 
        xs = linearbpf.xs
    elif res == 0: # sorted, has dups
        ys, unique_idx = np.unique(linearbpf.ys, return_index=True)
        xs = linearbpf.xs[unique_idx]
    return core.Linear(ys, xs)
