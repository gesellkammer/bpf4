# encoding: utf-8
from .config import CONFIG
from matplotlib import pyplot as plt
import numpy as np


def plot_coords(xs, ys, show=None, kind='line', **keys):
    if kind == 'line':
        plt.plot(xs, ys, **keys)
    elif kind == 'bar':
        plt.bar(xs, ys, **keys)
    plot_always_show = CONFIG['plot.always_show']
    if plot_always_show and (show is None or show is True):
        plt.show()
    elif not plot_always_show and show is True:
        plt.show()

        
def bpfplot(*bpfs, **keys):
    """
    bpfs: one or more bpfs to be plotted
    
    accepted keyword arguments:
        npoints:    number of points to be used for the plot
        show:       should the plot be plot immediately. else you can call show yourself
    """
    x0 = min(bpf.bounds()[0] for bpf in bpfs)
    x1 = max(bpf.bounds()[1] for bpf in bpfs)
    n = keys.pop('npoints', 400)
    xs = np.linspace(x0, x1, n)
    yss = [bpf.map(xs) for bpf in bpfs]
    args = []
    for ys in yss:
        args.extend((xs, ys))
    plt.plot(*args, **keys)
    if keys.pop('show'):
        plt.show()

        
def plot_stacked(*bpfs, **kws):
    """
    Example:
    
    a = bpf.linear(0,0, 1,1)
    b = bpf.halfcos(0, 0, 0.5, 1, 1, 0)
    
    plot_stacked(a, (b, 'b'))
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

        
def show():
    plt.show()