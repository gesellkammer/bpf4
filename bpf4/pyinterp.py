from __future__ import absolute_import
from . import core
import numpy
OUTBOUND_CACHE = 1


class _NumpyInterp(core._FunctionWrap_Object):
    
    def __init__(self, interpolator, xs, ys):
        self._interpolator = interpolator
        super(_NumpyInterp, self).__init__(interpolator, (xs[0], xs[-1]))
        self.xs, self.ys = xs, ys

    def mapn_between(self, n, x0, x1, out=None):
        xs = numpy.linspace(x0, x1, n)
        out0 = self._interpolator(xs)
        if out is not None:
            out[...] = out0
            return out
        return out0
    
    def map(self, xs, out=None):
        if isinstance(xs, int):
            return self.mapn_between(xs, self.x0, self.x1)
        return self._interpolator(xs)
    
    def points(self):
        return [self.xs, self.ys]
    
    def __getstate__(self):
        return self.xs, self.ys
    
    def _slice(self, x0, x1):
        return core._BpfCrop_new(self, x0, x1, OUTBOUND_CACHE)


class Pchip(_NumpyInterp):
    """
    Monotonic Piecewise Cubit Hermite interpolation, similar to matlab's pchip
    """
    def __init__(self, xs, ys):
        from . import wafointerpol
        interpolator = wafointerpol.Pchip(xs, ys)
        super(Pchip, self).__init__(interpolator, xs, ys)

