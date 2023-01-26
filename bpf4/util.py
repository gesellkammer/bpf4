"""
Utilities for bpf4
"""
from __future__ import annotations
import operator as _operator
import os as _os
import itertools as _itertools
from functools import reduce
from typing import Sequence

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
    'spline': core.Spline,
    'uspline': core.USpline,
    'slope': core.Slope,
    'nearest': core.Nearest,
    'halfcosm': core.Halfcosm,
    'smooth': core.Smooth,
    'smoother': core.Smoother
}


def _isiterable(obj) -> bool:
    try:
        iter(obj)
        return not isinstance(obj, str)
    except TypeError:
        return False


def _iflatten(seq):
    """
    Return an iterator to the flattened items of sequence s
    
    Strings are not flattened
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

def csv_to_bpf(csvfile: str) -> core.BpfInterface:
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


def bpf_to_csv(bpf: core.BpfInterface, csvfile: str) -> None:
    """
    Write this bpf as a csv representation

    Args:
        bpf: the bpf to write as csv
        csvfile: the output filename
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


def bpf_to_dict(bpf: core.BpfInterface) -> dict:
    """
    convert a bpf to a dict with the following format

    Args:
        bpf: the bpf to convert to a dict

    Returns:
        (dict) The bpf as a dictionary

    ```python

    >>> b = bpf.expon(3.0, 0, 0, 1, 10, 2, 20)
    >>> bpf_to_dict(b)
    {
        'interpolation': 'expon(3.0)',
        'points': [0, 0, 1, 10, 20, 20]  # [x0, y0, x1, y1, ...]
    }

    >>> b = bpf.multi(0, 0, 'linear',
                      1, 10, 'expon(2)',
                      3, 25)
    >>> bpf_to_dict(b)
    {
        'interpolation': 'multi',
        'segments': [
            [0, 0, 'linear'],
            [1, 10, 'expon(2)',
            [3, 25, '']]
    }
    ```
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


def bpf_to_json(bpf: core.BpfInterface, outfile: str = None) -> str:
    """
    convert this bpf to json format.

    If outfile is not given, it returns a string, as in dumps

    Args:
        bpf: the bpf to dump as json
        outfile: the output filename. If given, output is saved here

    Returns:
        (str) The json text
    """
    import json
    asdict = bpf_to_dict(bpf)
    jsontxt = json.dumps(asdict)
    if outfile:
        open(outfile, 'w').write(jsontxt)
    return jsontxt
        

def bpf_to_yaml(bpf: core.BpfInterface, outfile: str = None) -> str:
    """
    Convert this bpf to json format. 

    Args:
        bpf: the bpf to convert
        outfile: if given, the yaml text is saved to this file

    Returns:
        (str) The yaml text
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
    return stream.getvalue()


def dict_to_bpf(d: dict) -> bpf.BpfInterface:
    """
    Convert a dict to a bpf

    Args:
        d: the dictionary to convert to a bpf

    Returns:
        the converted bpf


    **Format 1** 

    ```python

    bpf = {
        'interpolation': 'expon(2)',
        10: 0.1,
        15: 1,
        25: -1
    }
    ```

    **Format 2**

    ```python
    bpf = {
        'interpolation': 'linear',
        'points': [x0, y0, x1, y1, ...]
    }
    ```

    **Format 2b**

    ```python
    bpf = {
        'interpolation': 'linear',
        'points': [(x0, y0), (x1, y1), ...]
    }
    ```

    **Format 3 (multi)**

    ```python

    bpf = {
        'interpolation': 'multi',
        'segments': [
            [x0, y0, 'descr0'],
            [x1, y1, 'descr1'],
            ...
            [xn, yn, '']
        ]
    }
    ```
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


def loadbpf(path: str, fmt='auto') -> core.BpfInterface:
    """
    Load a bpf saved with dumpbpf

    Possible formats: auto, csv, yaml, json

    Args:
        path: the path of the saved bpf
        fmt: the format used to save the bpf ('auto' to detect the format)

    Returns:
        a bpf
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


def asbpf(obj, bounds=(-np.inf, np.inf)) -> core.BpfInterface:
    """
    Convert obj to a bpf

    obj can be a function, a dict, a constant, or a bpf (in which case it
    is returned as is)
    """
    if isinstance(obj, core.BpfInterface):
        return obj
    elif callable(obj):
        return core._FunctionWrap(obj, bounds)
    elif hasattr(obj, '__float__'):
        return core.Const(float(obj))
    else:
        raise TypeError("can't wrap %s" % str(obj))
 

def parseargs(*args, **kws) -> tuple[list[float], list[float], dict]:
    """
    Convert the args and kws to the canonical form (xs, ys, kws)
    
    Returns:
        (tuple[list[float], list[float], dict]) A tuple `(xs, ys, kws)`
    
    Raises ValueError if failed

    All the following variants result in the same result:

    ```python

    x0, y0, x1, y1, …, exp=0.5
    (x0, y0), (x1, y1), …, exp=0.5           
    {x0:y0, x1:y1, …}, exp=0.5               
    [x0, x1, …], [y0, y1, …], exp=0.5        
    
    Result: [x0, x1, …], [y0, y1, …], {exp:0.5}
    ```
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
            raise ValueError(f"parsing error: 2 args, expected a seq. "
                             f"[(x0, y1), (x1, y1)] or (xs, ys) but got {args}")
        if len(args[0]) > 2:   # <--  (xs, ys)
            xs, ys = args
        else:                  # <--  ((x0, y0), (x1, y1), ...)
            xs, ys = list(zip(*args))    
    elif not any(map(_isiterable, args)):
        # (x0, y0, x1, y1, ...)
        if L % 2 == 0:  # even
            xs = args[::2]
            ys = args[1::2]
        else:
            if kws:
                raise ValueError(f"Uneven number of args. No keywords allowed in "
                                 f"this case, but got {kws}")
            kws = {'exp': args[0]}
            xs = args[1::2]
            ys = args[2::2]
    elif all(map(_isiterable, args)):   # <-- ((x0, y0), (x1, y1), ...)
        xs, ys = list(zip(*args))
    else:
        raise ValueError("could not parse arguments")
    return xs, ys, kws
        


def parsedescr(descr: str, validate=True) -> tuple[str, dict]:
    """
    Parse interpolation description


    | descr                   | output                             |
    |-------------------------|------------------------------------|
    | linear                  | linear, {}                         |
    | expon(0.4)              | expon, {'exp': 0.4}                |
    | halfcos(2.5, numiter=1) | halfcos, {'exp':2.5, 'numiter': 1} |

    """
    if "(" not in descr:
        classname = descr
        kws = {}
    else:
        classname, rest = descr.split("(")
        assert rest[-1] == ")"
        rest = rest[:-1]
        parts = rest.split(",")
        kws = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=")
                kws[key] = float(value)
            else:
                assert 'exp' not in kws
                kws['exp'] = float(part)
    if validate:
        assert classname in _CONSTRUCTORS
    return classname, kws


def makebpf(descr: str, X: Sequence[float], Y: Sequence[float]) -> core.BpfInterface:
    """
    Create a bpf from the given descriptor and points

    Args:
        descr: a string descriptor of the interpolation ("linear", "expon(xx)", ...)
        X: the array of xs
        Y: the array of ys

    Returns:
        the created bpf
    """
    interpolkind, kws = parsedescr(descr)
    bpfclass = _CONSTRUCTORS.get(interpolkind)
    if bpfclass is None:
        raise ValueError(f"descr {descr} is not valid")
    return bpfclass(X, Y, **kws)
    

def multi_parseargs(args) -> tuple[list[float], list[float], list[str]]:
    """
    Parse args of a multi bpf

    Given a list of args of the form (x0, y0, interpol) or (x0, y0) (or a flat
    version thereof), fills the possibly missing interpolation descriptions
    and returns a tuple `(xs, ys, interpolations)`

    Returns:
        a tuple (xs, ys, interpolations), where len(interpolations) == len(xs) - 1
    """
    xs = []
    ys = []
    interpolations = []
    last_interpolation = 'linear'
    if all(isinstance(arg, (int, float, str)) for arg in args):
        # a flat list
        accum = 0
        for arg in args:
            if accum == 0:
                xs.append(arg)
                accum += 1
            elif accum == 1:
                ys.append(arg)
                accum += 1
            else:
                if isinstance(arg, str):
                    interpolations.append(arg)
                    last_interpolation = arg
                    accum = 0
                else:
                    interpolations.append(last_interpolation)
                    xs.append(arg)
                    accum = 1
    else:
        # it is of the type (x0, y0, interpolation), (x1, y1), ...
        assert all(isinstance(arg, tuple) and 2 <= len(arg) <= 3 for arg in args)
        
        for arg in args[:-1]:
            if len(arg) == 2:
                x, y = arg
                interp = last_interpolation
            elif len(arg) == 3:
                x, y, interp = arg
            xs.append(x)
            ys.append(y)
            interpolations.append(interp)
        x, y = args[-1]
        xs.append(x)
        ys.append(y)
    assert all(isinstance(x, (int, float)) for x in xs)
    assert all(isinstance(y, (int, float)) for x in ys) 
    assert all(isinstance(i, str) for i in interpolations)
    assert len(xs) == len(ys) == len(interpolations)+1
    return xs, ys, interpolations    
    
    
def max_(*elements) -> core.Max:
    """
    Return a bpf representing the max over the given elements

    Args:
        elements: each element can be a bpf or a scalar

    Returns:
        a Max bpf
    """
    bpfs = list(map(asbpf, elements))
    return core.Max(*bpfs)

    
def min_(*elements) -> core.Min:
    """
    Return a bpf representing the min over elements

    Args:
        elements: each can be a bpf or a scalar

    Returns:
        a Min bpf
    """
    bpfs = list(map(asbpf, elements))
    return core.Min(*bpfs)

    
def sum_(*elements):
    """
    Return a bpf representing the sum of elements

    Args:
        elements: each can be a bpf or a scalar

    Returns:
        a bpf representing the sum of all elements
    
    """
    bpfs = list(map(asbpf, elements))
    return reduce(_operator.add, bpfs)


def select(which: core.BpfInterface, bpfs: Sequence[core.BpfInterface], shape='linear') -> core._BpfSelect:
    """
    Create a new bpf which interpolates between adjacent bpfs given
    
    Args:
        which: returns at any x, which bpf from bpfs should return the result
        bpfs: a list of bpfs
        shape: interpolation shape between consecutive bpfs

    Returns:
        a BpfSelect

    **Example**

    ```python   

    >>> which = nointerpol(0, 0, 5, 1)
    >>> bpfs = [linear(0, 0, 10, 10), linear(0, 10, 10, 0)]
    >>> s = select(which, bpfs)
    >>> s(1)     # at time=1, the first bpf will be selected
    0
    ```
    """
    return core._BpfSelect(asbpf(which), list(map(asbpf, bpfs)), shape)

    
def dumpbpf(bpf: core.BpfInterface, fmt='yaml') -> str:
    """
    Dump the data of this bpf as human readable text 

    
    Args:
        bpf: the bpf to dump
        fmt: the format, one of 'csv', 'yaml', 'json'
        
    Returns:
        the text representation according to the format 
    
    The bpf can then be reconstructed via `loadbpf`

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
        raise ValueError(f"Format {fmt} not supported")
        
            
def concat_bpfs(bpfs: list[core.BpfInterface]) -> core._BpfConcat:
    """
    Concatenate these bpfs together, one after the other
    """
    bpfs2 = [bpfs[0]]
    x0, x1 = bpfs[0].bounds()
    xs = [x0]
    for bpf in bpfs[1:]:
        bpf2 = bpf.fit_between(x1, x1 + (bpf._x1 - bpf._x0))
        bpfs2.append(bpf2)
        xs.append(bpf2._x0)
        x0, x1 = bpf2.bounds()
    return core._BpfConcat(xs, bpfs2)


def warped(bpf: core.BpfInterface, dx:float=None, numpoints=1000) -> core.Sampled:
    """
    Represents the curvature of a linear space. 

    The result is a warped bpf so that:
    
    ```
    position_bpf | warped_bpf = corresponding position after warping
    ```

    Args:
        dx: the accuracy of the measurement
        numpoints: if dx is not given, the bpf is sampled `numpoints` times
            across its bounds

    Returns:
        (core.Sampled) The warped bpf
    

    Example
    -------

    Find the theoretical position of a given point according to a 
    probability distribution
    
    ```python
    >>> from bpf4 import *
    >>> import matplotlib.pyplot as plt
    >>> distribution = halfcos(0,0, 0.5,1, 1, 0)
    >>> w = util.warped(distribution)
    >>> distribution.plot()
    >>> w.plot()

    ```
    ![](assets/warped.png)
    
    Now plot the histrogram of the warped bpf. It should resemble the
    original distribution
    ```python
    plt.hist(w.map(10000), bins=200, density=True)
    ```
    ![](assets/warped-hist.png)

    Using another distribution, notice that the histogram follows the
    distribution again:

    ```python
    distribution = halfcos(0,0, 0.8,1, 1, 0, exp=3.)
    w = util.warped(distribution)
    distribution.plot()
    w.plot()
    _ = plt.hist(w.map(10000), bins=200, density=True)[2]
    ```
    ![](assets/warped-hist2.png)
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

        
def _minimize(bpf, N: int, func=min, debug=False) -> float | None:
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

    
def minimum(bpf: core.BpfInterface, N=10) -> float | None:
    """
    Find the x where bpf(x) is minimized
    
    Args:
        bpf: the bpf to analyze
        N: the number of estimates

    Returns:
        the *x* value where bpf(x) is minimized. Returns `None` if
        no minimum found

    """
    return _minimize(bpf, N, min)


def maximum(bpf: core.BpfInterface, N=10) -> float | None:
    """
    return the x where bpf(x) is the maximum of bpf

    Args:
        bpf: the bpf to analyze
    
    Returns:
        the *x* value where bpf(x) is the maximum. Returns `None` if
        no maximum found

    """
    return _minimize(-bpf, N, min)
    

def rms(bpf: core.BpfInterface, rmstime=0.1) -> core.BpfInterface:
    """
    The rms of this bpf

    Args:
        bpf: the bpf 
        rmstime: the time to calculate the rms over

    Returns:
        a bpf representing the rms of this bpf at any x coord
    """
    bpf2 = bpf**2
    from math import sqrt

    def func(x):
        return sqrt(bpf2.integrate_between(x, x+rmstime) / rmstime)
    return asbpf(func, bounds=(bpf.x0, bpf.x1))


def rmsbpf(samples: np.ndarray, sr:int, dt=0.01, overlap=1) -> core.Sampled:
    """
    Return a bpf representing the rms of the given samples as a function of time

    Args:
        samples: the audio samples
        sr: the sample rate
        dt: analysis time period
        overlap: overlap of analysis frames

    Returns:
        a samples bpf
    """
    s = samples
    period = int(sr * dt + 0.5)
    hopsamps = period // overlap
    dt2 = hopsamps / sr
    numperiods = len(s) // hopsamps
    data = np.empty((numperiods,), dtype=float)
    for i in range(numperiods):
        idx0 = i * hopsamps
        chunk = s[idx0:idx0+period]
        data[i] = rms(chunk)
    return bpf4.core.Sampled(data, x0=0, dx=dt2)


def calculate_projection(x0, x1, p0, p1):
    """
    Calculate a projection needed to map the interval x0:x1 to p0:p1

    Returns:
        (tuple[float, float, float]) A tuple (rx, dx, offset)
    """
    rx = (x1-x0) / (p1-p0)
    dx = x0
    offset = p0
    return rx, dx, offset
    

def projection_fixedpoint(rx, dx, offset):
    """
    Returns the fixed point given the projection parameters

    x2 = (x-offset)*rx + dx

    For a fixed point, x2 == x
    """
    return (-offset*rx + dx)/(1-rx)
    
    

def binarymask(mask: str | list[int],
               durs: Sequence[float]=None,
               offset=0.,
               cycledurs=True
               ) -> core.NoInterpol:
    """
    Creates a binary mask

    Args:
        mask: a mask string ('x'=1, '-'=0) or a sequence of states (a state is either 0 or 1)
        durs: a sequence of durations (default=[1])

    Returns:
        (core.NoInterpol) A NoInterpol bpf representing the binary mask

    **Example**

    ```python
        
    >>> mask = binarymask("x--x-x---")
    ```

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

    
def jagged_band(xs: list[float], upperbpf: core.BpfInterface, lowerbpf=0, curve='linear'
                ) -> core.BpfInterface:
    """
    Create a jagged bpf between lowerbpf and upperbpf at the x values given
    
    At each x in xs the, the value is equal to lowerbpf, sweeping
    with curvature 'curve' to upperbpf just before the next x

    Example
    -------

    ```python
    
    from bpf4 import *
    import numpy as np
    a = expon(0, 0, 1, 10, exp=2)
    b = expon(0, 0, 1, 5, exp=4)
    j = util.jagged_band(list(np.arange(0, 1, 0.1)), a, b, curve='expon(1.5)')
    j.plot()
    ```
    ![](assets/jagged.png)
    """
    upperbpf = asbpf(upperbpf)
    lowerbpf = asbpf(lowerbpf)
    if not isinstance(xs, list): 
        xs = list(xs)
    EPSILON = 1e-12
    fragments = []
    if xs[0] > upperbpf.x0 > float('-inf'):
        xs = [upperbpf.x0] + xs
    if xs[-1] < upperbpf.x1 < float('inf'):
        xs.append(upperbpf.x1)
    for x0, x1 in _pairwise(xs[:-1]):
        x1 = x1 - EPSILON
        fragment = makebpf(curve, [x0, x1], [lowerbpf(x0), upperbpf(x1)])[x0:x1].outbound(0, 0)
        fragments.append(fragment)
    x0 = xs[-2]
    x1 = xs[-1]
    fragments.append(makebpf(curve, [x0, x1], [lowerbpf(x0), upperbpf(x1)])[x0:x1].outbound(0, 0))
    return max_(*fragments)
    

def randombw(bw: float | core.BpfInterface,
             center: float | core.BpfInterface
             ) -> core.BpfInterface:
    """
    Create a random bpf
    
    Args:
        bw: a (time-varying) bandwidth
        center: the center of the random distribution

    Returns:
        a bpf
        
    if randombw is 0.1 and center is 1, the bpf will render values 
    between 0.95 and 1.05

    **NB**: this bpf will always be different, since the random numbers
    are calculated as needed. Sample it to freeze it to a known state.

    **Example**
    
    ```python

    >>> l = bpf.linear(0, 0, 1, 1)
    >>> r = bpf.util.randombw(0.1)
    >>> l2 = (l*r)[::0.01]
    ```
    """
    bw = asbpf(bw)
    return (bw.rand() + (center - bw*0.5))[bw.x0:bw.x1]
    

def blendwithfloor(b: core.BpfInterface, mix=0.5) -> core._BpfBlend:
    return core.blend(b, asbpf(b(maximum(b))), mix)[b.x0:b.x1]
    
    
def blendwithceil(b, mix=0.5) -> core._BpfBlend:
    return core.blend(b, asbpf(b(maximum(b))), mix)[b.x0:b.x1]
    

def smoothen(b: core.BpfInterface, window:int, N=1000, interpol='linear') -> core.BpfInterface:
    """
    Return a linear bpf representing a smooth version of b

    Args:
        b: a bpf
        window: the width (in x coords) of the smoothing window
        N: number of points to resample the bpf
        interpol: the interpolation to use. One of 'linear', 'smooth', 'halfcos'

    Returns:
        a bpf representing a smoother version of b
    """
    dx = min((b.x1 - b.x0) / N, window/7)
    nwin = int(window / dx)
    box = np.ones(nwin)/nwin
    Y = b[::dx].ys
    Y2 = np.convolve(Y, box, mode="same")
    X = np.linspace(b.x0, b.x1, len(Y2))
    if interpol == 'linear':
        return core.Linear(X, Y2)
    elif interpol == 'smooth':
        return core.Smooth(X, Y2)
    elif interpol == 'halfcos':
        return core.Halfcos(X, Y2)
    else:
        raise ValueError(f"Interpolation {interpol} not supported here. "
                          "Possible values: linear, smooth, halfcos")


def zigzag(b0: core.BpfInterface,
           b1: core.BpfInterface,
           xs: Sequence[float],
           shape='linear'
           ) -> core.BpfInterface:
    """
    Creates a curve formed of lines from b0(x) to b1(x) for each x in xs

    Args:
        b0: a bpf
        b1: a bpf
        xs: a seq. of x values to evaluate b0 and b1
        shape: the shape of each segment

    Returns:
        The resulting bpf

    ::

       *.
        *...  b0
         *  ...
         *     ...
          *       ....
           *          ...
            *         :  ...
             *        :*    ...
             *        : *      ...
              *       :  **       ...
               *      :    *         :*.
                *     :     *        : **...
                 *    :      *       :   *  ...
                 *    :       *      :    *    ...
                  *   :        *     :     **     .:.
                   *  :         *    :       *     :**..
                    * :          **  :        **   :  ****.
                     *:            * :          *  :      ****
        -----------  *:             *:           * :          ****
          b1       ---*--------------*---         **:             ****
                                         -----------*----------      .**
                                                               -----------
        x0            x1              x2                       x3

    """
    curves = []
    for x0, x1 in pairwise(xs):
        X = [x0, x1]
        Y = [b0(x0), b1(x1)]
        curve = bpf.util.makebpf(shape, X, Y)
        curves.append(curve)
    jointcurve = bpf.max_(*[c.outbound(0, 0) for c in curves])
    return jointcurve


def bpfavg(b: core.BpfInterface,
           dx: float
           ) -> core.BpfInterface:
    """
    Return a Bpf which is the average of b over the range `dx`

    Args:
        b: the bpf
        dx: the period to average *b* over

    Returns:
        a bpf representing the average of *b* along the bounds of
        *b* over a sliding period of *dx*
    """
    dx2 = dx/2
    avg = ((b<<dx2)+b+(b>>dx2))/3.0
    return avg[b.x0:b.x1]


def histbpf(b: core.BpfInterface, numbins=10, numsamples=200, interpolation='linear'
            ) -> core.BpfInterface:
    """
    Create a historgram of *b*

    Args:
        b: the bpf
        numbins: the number of bins
        numsamples: how many samples to take to determine the histogram
        interpolation: the kind of interpolation of the returned bpf

    Returns:
        a bpf representing the percentile associated with a given value of *b*
        Percentiles are given between 0 and 1

    Example
    ~~~~~~~

    >>> from sndfileio import *
    >>> samples, sr = sndread("path/to/soundfile.wav")
    >>> dbcurve = rmsbpf(samples, sr=sr).amp2db()
    >>> dbhist = histbpf(dbcurve)
    # Find the db percentile at a given time, this gives a measurement of the
    # relative strength of the sound at a given moment
    >>> dur = len(samples)/sr
    >>> percentile = dbhist(dur*0.5)
    0.312

    This indicates that at the middle of the sound the amplitude is at percentile ~30

    """
    samples = b.map(numsamples)
    edges, hist = np.histogram(b, bins=numbins)
    percentile = np.linspace(0, 1, len(hist))
    if interpolation == 'linear':
        return core.Linear(hist, percentile)
    elif interpolation == 'nointerpol':
        return core.NoInterpol(hist, percentile)
    else:
        raise ValueError("Only 'linear' or 'nointerpol' are supported")
