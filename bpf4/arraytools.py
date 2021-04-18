from __future__ import annotations
import numpy as np


def arrayslice(x0:float, x1:float, X:np.ndarray, *Ys:np.ndarray) -> np.ndarray:
    """
    Slice a sorted array and linked arrays as if they where a bpf

    Args:
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


def interlace_arrays(*arrays: np.ndarray) -> np.ndarray:
    """
    Interweave multiple arrays into a flat array in the form

    Example::

        A = [a0, a1, a2, ...]
        B = [b0, b1, b2, ...]
        C = [c0, c1, c2, ...]
        interlace(A, B, C)
        -> [a0, b0, c0, a1, b1, c1, ...]

    Args:
        *arrays (): the arrays to interleave. They should be 1D arrays of the
            same length

    Returns:
        a 1D array with the elements of the given arrays interleaved

    """
    assert all(a.size == arrays[0].size and a.dtype == arrays[0].dtype for a in arrays)
    size = arrays[0].size * len(arrays)
    out = np.empty((size,), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i::len(arrays)] = a
    return out

