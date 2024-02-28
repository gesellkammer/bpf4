from __future__ import annotations
from .config import CONFIG
import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Union
    from bpf4 import core


def plot_coords(xs: Union[list[float], np.ndarray],
                ys: Union[list[float], np.ndarray],
                show:bool=None, 
                kind='line', 
                axes: plt.Axes = None,
                figsize:tuple(float, float)=None,
                **keys
                ) -> plt.Axes:
    """
    Plot the points defined by xs and ys

    Args:
        xs: a seq. of x coords
        ys: a seq. of y coords
        kind: one of line or bar
        axes: if given, this axes is used
        figsize: as passed to pyplot.figure(...), a tuple (width, height). This setting
            only has effect if axes is not given
        keys: any keywords will be bassed to `axes.plot` (or `axes.bar` depending
            on `kind`)

    Returns:
        the matplotlib's Axes object used. If it was passed as the axes
        arg, then the returned axes will be this same Axes object

    """
    if not axes:
        if figsize:
            fig, axes = plt.subplots(figsize=figsize)
        else:
            fig, axes = plt.subplots()
    if kind == 'line':
        axes.plot(xs, ys, **keys)
    elif kind == 'bar':
        axes.bar(xs, ys, **keys)
    elif kind == 'stem':
        axes.stem(xs, ys, **keys) 
    plot_always_show = CONFIG['plot.always_show']
    if plot_always_show and (show is None or show is True):
        if axes:
            axes.show()
        else:
            plt.show()
    elif not plot_always_show and show is True:
        plt.show()
    return axes

        
def bpfplot(*bpfs: core.BpfInterface, npoints=400, show=True, **kws) -> None:
    """
    Plot one/multiple bpfs

    Args:
        bpfs: one or more bpfs to be plotted
        npoints: number of points to be used for the plot
        show: should the plot be plot immediately. else you can call show yourself
        kws: any keywords are passed to `plt.plot`
    """
    x0 = min(bpf.bounds()[0] for bpf in bpfs)
    x1 = max(bpf.bounds()[1] for bpf in bpfs)
    xs = np.linspace(x0, x1, npoints)
    yss = [bpf.map(xs) for bpf in bpfs]
    args = []
    for ys in yss:
        args.extend((xs, ys))
    plt.plot(*args, **kws)
    if show:
        plt.show()

        
def plot_stacked(*bpfs: core.BpfInterface, **kws) -> None:
    """
    Example
    -------

    ```python
    
    a = bpf.linear(0,0, 1,1)
    b = bpf.halfcos(0, 0, 0.5, 1, 1, 0)
    
    plot_stacked(a, (b, 'b'))
    ```

    """
    min_x = float('inf')
    max_x = -float('inf')
    bpfs2, labels = [], []
    for i, bpf in enumerate(bpfs):
        if isinstance(bpf, (tuple, list)):
            bpf, label = bpf
        else:
            label = str(i + 1)
        bpfs2.append(bpf)
        labels.append(label)
        x0, x1 = bpf.bounds()
        if x0 < min_x:
            min_x = x0
        if x1 > max_x:
            max_x = x1
    N = 2000
    xs = np.linspace(min_x, max_x, N)
    yss = [bpf[min_x:max_x].map(N) for bpf in bpfs2]
    y_data = np.row_stack(yss)
    y_data_stacked = np.cumsum(y_data, axis=0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    split_lines = [(0, y_data_stacked[0,:])]
    COLORTHEME = (
        '#CC6666',
        '#1DACD6',
        '#6E5160',
        '#FF22FF',
        '#66FF88',
        '#FF2266',
        '#444444'
    )
    for i in range(len(bpfs) - 1):
        split_lines.append((y_data_stacked[i,:], y_data_stacked[i+1,:]))
    facecolors = COLORTHEME[:len(bpfs)]
    for i in range(len(bpfs)):
        y0, y1 = split_lines[i]
        color = facecolors[i]
        ax1.fill_between(xs, y0, y1, facecolor=color, alpha=0.9, label=labels[i])
        ax1.plot(xs, y1, lw=3, label=labels[i],color=color)
    ax1.legend(loc='upper left')
    if kws.get('show'):
        plt.show()

        
def show() -> None:
    """Show the plotted bpfs"""
    plt.show()
