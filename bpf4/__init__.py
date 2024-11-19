from . import core
from .core import (
    BpfInterface,
    BpfBase,
    Linear,
    Smooth,
    Smoother,
    Halfcos,
    Expon,
    Nearest,
    Sampled,
    Slope,
    Spline,
    Max,
    Min,
    blend,
)

from .api import (
    linear,
    expon,
    halfcos,
    spline,
    halfcosm,
    uspline,
    nointerpol,
    nearest,
    smooth,
    smoother,
    multi,
    pchip,
)


from .util import asbpf
