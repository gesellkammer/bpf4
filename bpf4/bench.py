import bpf3
import numpy
from bpf3 import bpf

def run1():
    xs = numpy.arange(0, 10, 0.1)
    ys = numpy.sin(xs)
    points = zip(xs, ys)

    a = bpf3.Linear(xs, ys)
    b = bpf.linear(*points)

    xs2 = numpy.arange(0, 10, 0.01)
    ys2 = a.map(xs2)
    ys3 = b.map(xs2)
    c = a * b
    ys4 = c.map(xs2)

if __name__ == '__main__':
    run1()

#a.plot(show=False)
#b.plot(show=False)
