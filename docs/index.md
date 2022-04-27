# bpf4

Welcome to the **bpf4** documentation!

**bpf4** is a library for curve fitting and break-point functions in python. It is mainly programmed
in cython for efficiency.

-----------------

## Installation

```bash
pip install bpf4
```

-----------------


## Quick Introduction


A BPF (Break-Point-Function) is defined by points in 2D space. Each different BPF defines
a specific interpolation type: linear, exponential, spline, etc. Operations on BPFs 
are lazy and result in new BPFs representing these operations. A BPF can be evaluated
at a specific x coord or an array of such coords or rendered with a given sampling 
period.

### Example 1

```python
>>> from bpf4 import *
>>> a = linear((0, 0), (1.5, 10), (3.4, 15))
>>> b = linear((1, 1), (4, 10))

# Construct a third BPF representing the average
>>> avgbpf = (a + b) / 2
>>> avgbpf
_BpfLambdaDivConst[0.0:4.0]

# Sample the bpf at a regular interval, generate 30 elements
>>> avgbpf.map(30)
array([ 0.5       ,  0.95977011,  2.67816092,  4.51724138,  6.35632184,
        7.29885057,  7.75862069,  8.2183908 ,  8.67816092,  9.13793103,
        9.59770115, 10.02268603, 10.20417423, 10.38566243, 10.56715064,
       10.74863884, 10.93012704, 11.11161525, 11.29310345, 11.47459165,
       11.65607985, 11.83756806, 12.01905626, 12.20054446, 12.38203267,
       12.5       , 12.5       , 12.5       , 12.5       , 12.5       ])

```

`avgbpf` is a BPF representing a set of mathematical operations. It is defined over a given range (the `bounds`
of the BPF). If not indicated otherwise outside of the bounds of the BPF its values remain constant:

```python
>>> a(3.4)
15
>>> a(10)
15
```

### Example 2: Intersection between two curves

```python

from bpf4 import *  
import matplotlib.pyplot as plt
a = spline((0, 0), (1, 5), (2, 3), (5, 10))  
b = expon((0, -10), (2,15), (5, 3), exp=3)
# plots are performed using matplotlib
a.plot() 
b.plot() 
zeros = (a - b).zeros()
plt.plot(zeros, a.map(zeros), 'o')
```

![1](https://github.com/gesellkammer/bpf4/raw/master/pics/zeros.png)

------------------


## Features


Many interpolation types besides linear:

* spline
* half-cosine
* exponential
* fibonacci
* exponantial half-cosine
* pchip
* logarithmic
* etc. 

Interpolation types can be mixed, so that each segment has a different interpolation (with the exception of spline interpolation)  
Curves can be combined non-destructively. Following from the example above.  


```pyton

c = (a + b).sin().abs()
# plot only the range (1.5, 4)
c[1.5:4].plot()  

```

![2](https://github.com/gesellkammer/bpf4/raw/master/pics/sinabs.png)

Syntax support for shifting, scaling and slicing a bpf

```python

a >> 2        # a shifted to the right
(a * 5) ^ 2   # scale the x coord by 2, scale the y coord by 5
a[2:2.5]      # slice only a portion of the bpf
a[::0.01]     # sample the bpf with an interval of 0.01

```

* Derivation and Integration: `c.derivative().plot()` or `c.integrated().integrated().plot()`  
* Numerical integration: `c.integrate_between(2, 4)`  


-------------------

## Mathematical operations

### Max / Min

```python
a = linear(0, 0, 1, 0.5, 2, 0)
b = expon(0, 0, 2, 1, exp=3)
a.plot(show=False, color="red", linewidth=4, alpha=0.3)
b.plot(show=False, color="blue", linewidth=4, alpha=0.3)
core.Max((a, b)).plot(color="black", linewidth=4, alpha=0.8, linestyle='dotted')
```
![](assets/Max.png)

```python
a = linear(0, 0, 1, 0.5, 2, 0)
b = expon(0, 0, 2, 1, exp=3)
a.plot(show=False, color="red", linewidth=4, alpha=0.3)
b.plot(show=False, color="blue", linewidth=4, alpha=0.3)
core.Min((a, b)).plot(color="black", linewidth=4, alpha=0.8, linestyle='dotted')
```
![](assets/Min.png)


### `+, -, *, /`

```
a = linear(0, 0, 1, 0.5, 2, 0)
b = expon(0, 0, 2, 1, exp=3)
a.plot(show=False, color="red", linewidth=4, alpha=0.3)
b.plot(show=False, color="blue", linewidth=4, alpha=0.3)
(a*b).plot(color="black", linewidth=4, alpha=0.8, linestyle='dotted')
```
![](assets/math-mul.png)

```python
a = linear(0, 0, 1, 0.5, 2, 0)
b = expon(0, 0, 2, 1, exp=3)
a.plot(show=False, color="red", linewidth=4, alpha=0.3)
b.plot(show=False, color="blue", linewidth=4, alpha=0.3)
(a**b).plot(color="black", linewidth=4, alpha=0.8, linestyle='dotted')
```
![](assets/math-pow.png)

```python
a = linear(0, 0, 1, 0.5, 2, 0)
b = expon(0, 0, 2, 1, exp=3)
a.plot(show=False, color="red", linewidth=4, alpha=0.3)
b.plot(show=False, color="blue", linewidth=4, alpha=0.3)
((a+b)/2).plot(color="black", linewidth=4, alpha=0.8, linestyle='dotted')
```
![](assets/math-avg.png)

### Building functions

A bpf can be used to build complex formulas

**Fresnel's Integral**: \( S(x) = \int_0^x {sin(t^2)} dt \)

```python
t = slope(1)
f = (t**2).sin()[0:10:0.001].integrated()
f.plot()
```

![](assets/fresnel.png)


#### Polar plots

Any kind of matplotlib plot can be used. For example, polar plots are possible
by creating an axes with *polar*=`True`

**Cardiod**: \(\rho = 1 + sin(-\theta) \)

```python

from math import *
theta = slope(1, bounds=(0, 2*pi))
r = 1 + (-theta).sin()

ax = plt.axes(polar=True)
ax.set_rticks([0.5, 1, 1.5, 2]); ax.set_rlabel_position(38)
r.plot(axes=ax)
```
![](assets/cardioid.png)


**Flower 5**: \(\rho = 3 + cos(5 * \theta) \)

```python
theta = core.Slope(1, bounds=(0, 2*pi))
r = 3 + (5*theta).cos()

ax = plt.axes(polar=True)
r.plot(axes=ax)

```
![](assets/polar1.png)