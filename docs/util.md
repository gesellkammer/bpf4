# Util


Utilities for bpf4


---------


## asbpf


```python

def asbpf(obj, bounds: tuple = (-inf, inf)) -> core.BpfInterface

```


Convert obj to a bpf


obj can be a function, a dict, a constant, or a bpf (in which case it
is returned as is)



**Args**

* **obj**:
* **bounds** (`tuple`):  (default: (-inf, inf))


---------


## binarymask


```python

def binarymask(mask: Union[str, list[int]], durs: Sequence[float] = None, 
               offset: float = 0.0, cycledurs: bool = True) -> core.NoInterpol

```


Creates a binary mask


**Example**

```python

>>> mask = binarymask("x--x-x---")
```



**Args**

* **mask** (`Union[str, List[int]]`): a mask string ('x'=1, '-'=0) or a sequence
    of states (a state is either 0 or 1)
* **durs** (`Sequence[float]`): a sequence of durations (default=[1]) (default:
    None)
* **offset** (`float`):  (default: 0.0)
* **cycledurs** (`bool`):  (default: True)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.NoInterpol`) A NoInterpol bpf representing the binary mask


---------


## blendwithceil


```python

def blendwithceil(b, mix: float = 0.5) -> core._BpfBlend

```



**Args**

* **b**:
* **mix** (`float`):  (default: 0.5)


---------


## blendwithfloor


```python

def blendwithfloor(b: core.BpfInterface, mix: float = 0.5) -> core._BpfBlend

```



**Args**

* **b** (`core.BpfInterface`):
* **mix** (`float`):  (default: 0.5)


---------


## bpf\_to\_csv


```python

def bpf_to_csv(bpf: core.BpfInterface, csvfile: str) -> None

```


Write this bpf as a csv representation



**Args**

* **bpf** (`core.BpfInterface`):
* **csvfile** (`str`):


---------


## bpf\_to\_dict


```python

def bpf_to_dict(bpf: core.BpfInterface) -> dict

```


convert a bpf to a dict with the following format


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



**Args**

* **bpf** (`core.BpfInterface`):


---------


## bpf\_to\_json


```python

def bpf_to_json(bpf: core.BpfInterface, outfile: str = None) -> str

```


convert this bpf to json format.


If outfile is not given, it returns a string, as in dumps

kws are passed directly to json.dump



**Args**

* **bpf** (`core.BpfInterface`):
* **outfile** (`str`):  (default: None)


---------


## bpf\_to\_yaml


```python

def bpf_to_yaml(bpf, outfile: str = None) -> str

```


Convert this bpf to json format.



**Args**

* **bpf**: the bpf to convert
* **outfile** (`str`): if given, the yaml text is saved to this file (default:
    None)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`str`) the yaml text


---------


## concat\_bpfs


```python

def concat_bpfs(bpfs: list[core.BpfInterface]) -> core._BpfConcat

```


Concatenate these bpfs together, one after the other



**Args**

* **bpfs** (`List[core.BpfInterface]`):


---------


## csv\_to\_bpf


```python

def csv_to_bpf(csvfile: str) -> core.BpfInterface

```


Read a bpf from a csv file



**Args**

* **csvfile** (`str`):


---------


## dict\_to\_bpf


```python

def dict_to_bpf(d: dict) -> bpf.BpfInterface

```


Convert a dict to a bpf


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



**Args**

* **d** (`dict`):


---------


## dumpbpf


```python

def dumpbpf(bpf: core.BpfInterface, fmt: str = yaml) -> str

```


Dump the data of this bpf as human readable text


The bpf can then be reconstructed via `loadbpf`



**Args**

* **bpf** (`core.BpfInterface`): the bpf to dump
* **fmt** (`str`): the format, one of 'csv', 'yaml', 'json' (default: yaml)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`str`) the text representation according to the format


---------


## jagged\_band


```python

def jagged_band(xs, upperbpf: core.BpfInterface, lowerbpf: int = 0, 
                curve: str = linear) -> core.BpfInterface

```


Create a jagged bpf between lowerbpf and upperbpf at the x values given


At each x in xs the, the value is equal to lowerbpf, sweeping
with curvature 'curve' to upperbpf just before the next x



**Args**

* **xs**:
* **upperbpf** (`core.BpfInterface`):
* **lowerbpf** (`int`):  (default: 0)
* **curve** (`str`):  (default: linear)


---------


## loadbpf


```python

def loadbpf(path: str, fmt: str = auto) -> core.BpfInterface

```


Load a bpf saved with dumpbpf


Possible formats: auto, csv, yaml, json



**Args**

* **path** (`str`): the path of the saved bpf
* **fmt** (`str`): the format used to save the bpf ('auto' to detect the format)
    (default: auto)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.BpfInterface`) a bpf


---------


## makebpf


```python

def makebpf(descr: str, X: Sequence[float], Y: Sequence[float]
            ) -> core.BpfInterface

```


Create a bpf from the given descriptor and points



**Args**

* **descr** (`str`): a string descriptor of the interpolation ("linear",
    "expon(xx)", ...)
* **X** (`Sequence[float]`): the array of xs
* **Y** (`Sequence[float]`): the array of ys

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.BpfInterface`) the created bpf


---------


## max\_


```python

def max_(elements) -> core.Max

```


Return a bpf representing the max over the given elements



**Args**

* **elements**: each element can be a bpf or a scalar

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.Max`) a Max bpf


---------


## maximum


```python

def maximum(bpf: core.BpfInterface, N: int = 10) -> Optional[float]

```


return the x where bpf(x) is the maximum of bpf



**Args**

* **bpf** (`core.BpfInterface`): the bpf to analyze
* **N** (`int`):  (default: 10)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`Optional[float]`) is the maximum. Returns None if no maximum found


---------


## min\_


```python

def min_(elements) -> core.Min

```


Return a bpf representing the min over elements



**Args**

* **elements**: each can be a bpf or a scalar

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.Min`) a Min bpf


---------


## minimum


```python

def minimum(bpf: core.BpfInterface, N: int = 10) -> Optional[float]

```


return the x where bpf(x) is the minimum of bpf



**Args**

* **bpf** (`core.BpfInterface`): the bpf to analyze
* **N** (`int`): the number of estimates (default: 10)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`Optional[float]`) is the minimum. Returns None if no minimum found


---------


## multi\_parseargs


```python

def multi_parseargs(args) -> tuple[list[float], list[float], list[str]]

```


Parse args of a multi bpf


Given a list of args of the form (x0, y0, interpol) or (x0, y0) (or a flat
version thereof), fills the possibly missing interpolation descriptions
and returns a tuple (xs, ys, interpolations)



**Args**

* **args**:

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`Tuple[List[float], List[float], List[str]]`) a tuple (xs, ys, interpolations), where len(interpolations) == len(xs) - 1


---------


## parseargs


```python

def parseargs(args, kws) -> tuple[list[float], list[float], dict]

```


Convert the args and kws to the canonical form (xs, ys, kws)


Returns a tuple (xs:list[float], ys:list[float], kws:dict)

Raises ValueError if failed



**Args**

* **args**:
* **kws**:


---------


## parsedescr


```python

def parsedescr(descr: str, validate: bool = True) -> tuple[str, dict]

```


Parse interpolation description


| descr                   | output                             |
|-------------------------|------------------------------------|
| linear                  | linear, {}                         |
| expon(0.4)              | expon, {'exp': 0.4}                |
| halfcos(2.5, numiter=1) | halfcos, {'exp':2.5, 'numiter': 1} |



**Args**

* **descr** (`str`):
* **validate** (`bool`):  (default: True)


---------


## randombw


```python

def randombw(bw: Union[float, core.BpfInterface], center: Union[float, 
             core.BpfInterface]) -> core.BpfInterface

```


Create a random bpf


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



**Args**

* **bw** (`Union[float, core.BpfInterface]`): a (time-varying) bandwidth
* **center** (`Union[float, core.BpfInterface]`): the center of the random
    distribution

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.BpfInterface`) a bpf


---------


## rms


```python

def rms(bpf: core.BpfInterface, rmstime: float = 0.1) -> core.BpfInterface

```


The rms of this bpf



**Args**

* **bpf** (`core.BpfInterface`): the bpf
* **rmstime** (`float`): the time to calculate the rms over (default: 0.1)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.BpfInterface`) a bpf representing the rms of this bpf at any x coord


---------


## select


```python

def select(which: core.BpfInterface, bpfs: Sequence[core.BpfInterface], 
           shape: str = linear) -> core._BpfSelect

```


Create a new bpf which interpolates between adjacent bpfs given


**Example**

```python   

>>> which = nointerpol(0, 0, 5, 1)
>>> bpfs = [linear(0, 0, 10, 10), linear(0, 10, 10, 0)]
>>> s = select(which, bpfs)
>>> s(1)     # at time=1, the first bpf will be selected
0
```



**Args**

* **which** (`core.BpfInterface`): returns at any x, which bpf from bpfs should
    return the result
* **bpfs** (`Sequence[core.BpfInterface]`): a list of bpfs
* **shape** (`str`): interpolation shape between consecutive bpfs (default:
    linear)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core._BpfSelect`) a BpfSelect


---------


## smoothen


```python

def smoothen(b: core.BpfInterface, window: int, N: int = 1000, 
             interpol: str = linear) -> core.BpfInterface

```


Return a linear bpf representing a smooth version of b



**Args**

* **b** (`core.BpfInterface`): a bpf
* **window** (`int`): the width (in x coords) of the smoothing window
* **N** (`int`): number of points to resample the bpf (default: 1000)
* **interpol** (`str`): the interpolation to use. One of 'linear', 'smooth',
    'halfcos' (default: linear)

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;(`core.BpfInterface`) a bpf representing a smoother version of b


---------


## sum\_


```python

def sum_(elements) -> Any

```


Return a bpf representing the sum of elements



**Args**

* **elements**: each can be a bpf or a scalar

**Returns**

&nbsp;&nbsp;&nbsp;&nbsp;a bpf representing the sum of all elements


---------


## warped


```python

def warped(bpf: core.BpfInterface, dx: float = None, numpoints: int = 1000
           ) -> core.Sampled

```


bpf represents the curvature of a linear space. the result is a


warped bpf so that:

```
position_bpf | warped_bpf = corresponding position after warping
```


**Example**

Find the theoretical position of a given point according to a probability distribution

```python
>>> from bpf4 import *
>>> distribution = bpf.halfcos(0,0, 0.5,1, 1, 0)
>>> w = warped(distribution)
>>> original_points = (0, 0.25, 0.33, 0.5)
>>> warped_points = w.map(original_points)
```



**Args**

* **bpf** (`core.BpfInterface`):
* **dx** (`float`): the accuracy of the measurement (default: None)
* **numpoints** (`int`): if dx is not given, the bpf is sampled `numpoints`
    times         across its bounds (default: 1000)