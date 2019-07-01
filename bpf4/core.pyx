#cython: boundscheck=False
#cython: embedsignature=True
#cython: wraparound=False
#cython: infer_types=True
#cython: profile=False
#cython: c_string_type=str, c_string_encoding=ascii

cdef extern from "math.h":
    double cos(double x) nogil
    double sin(double x) nogil
    double pow(double x, double y) nogil
    double ceil(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double floor(double x) nogil
    double fmod(double x, double y) nogil
    double hypot(double x, double y) nogil
    double atan2(double x, double y) nogil
    double tanh (double x ) nogil
    double fabs (double x) nogil
    double sqrt(double x) nogil
    double INFINITY, NAN, M_E, M_PI
    int isfinite(double x) nogil
    int isinf(double x) nogil
    int isnan(double x) nogil
    double acos(double x) nogil
    double asin(double x) nogil
    double cosh(double x) nogil
    double log10(double x) nogil
    double tan(double x) nogil
    double sinh(double x) nogil
    double log1p(double x) nogil
    
#cdef extern from "string.h":
#    ctypedef void* const_void_ptr "const void *"
#    void *memcpy(void *s1, const_void_ptr s2, size_t n) nogil

# ----------------------------------------------  cimports
from libc.stdlib cimport malloc, free # , realloc
from libc.stdlib cimport rand, srand, RAND_MAX
from cpython cimport PyList_GET_SIZE, PyList_GET_ITEM, PyTuple_New, PyTuple_SetItem
cimport cython
cimport numpy as c_numpy
from cython.view cimport array as cvarray
from numpy cimport (
    ndarray,
    npy_intp,
    PyArray_DIM,
    float_t,
    PyArray_ISCONTIGUOUS,
    import_array,
    PyArray_GETCONTIGUOUS,
    PyArray_ContiguousFromAny,  # object PyArray_ContiguousFromAny(op, int, int min_depth, int max_depth)
    NPY_DOUBLE,
    PyArray_SimpleNew,          # PyArray_SimpleNew(int nd, npy_intp *dims, int type_num)
    PyArray_SimpleNewFromData,  # PyArray_SimpleNewFromData(int nd, npy_intp *dims, int type_num, void *data)
    PyArray_EMPTY,              # PyArray_EMPTY(int nd, npy_intp* dims, int typenum, int fortran)
    PyArray_ISCARRAY,           # int PyArray_ISCARRAY( c_numpy.ndarray instance )
    PyArray_ZEROS,              # PyArray_ZEROS(int nd, npy_intp* dims, int type_num, int fortran)
    PyArray_FILLWBYTE           # PyArray_FILLWBYTE(obj, int val)
    )

# ---------------------------- import std-lib
# import math
import sys
import random

# ---------------------------- import others
import numpy
from numpy import array

# ---------------------------- own imports
from config import CONFIG
# from . import plot as _plot

# ---------------------------- init
import_array()
srand(random.randint(0, 99999))

# ---------------------------- platform specific constants

cdef double EPS = sys.float_info.epsilon
cdef double SQRT_EPS = sqrt(EPS)

# ---------------------------- DEFs
DEF PI = 3.141592653589793238462643383279502884197169399375105
DEF BRENTQ_ZERO = 2.221e-16
DEF BRENTQ_XTOL = 9.9999999999999998e-13
DEF BRENTQ_RTOL = 4.4408920985006262e-16
DEF BRENTQ_MAXITER = 100
DEF QUAD_LIMIT = 100
DEF SIMPSONS_ACCURACY = 1e-10
DEF SIMPSONS_MAXITER = 100


# -------------------------------------------------------------------
#       ERROR TYPES
# -------------------------------------------------------------------
class BpfPointsError(ValueError): pass
class BpfInversionError(ValueError): pass

# -------------------------------------------------------------------
#       INLINE FUNCS
# -------------------------------------------------------------------
ctypedef struct InterpolFunc

@cython.cdivision(True)
cdef inline double _interpol_linear(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

@cython.cdivision(True)
cdef inline double intrp_linear(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


cdef inline double _interpol_nointerpol(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    return y0 if x < x1 else y1

cdef inline double intrp_nointerpol(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    return y0 if x < x1 else y1

cdef inline double _interpol_nearest(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    if (x - x0) <= (x1 - x):
        return y0
    return y1

cdef inline double intrp_nearest(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    if (x - x0) <= (x1 - x):
        return y0
    return y1


@cython.cdivision(True)
cdef inline double _interpol_halfcos(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    cdef double dx
    dx = ((x - x0) / (x1 - x0)) * 3.14159265358979323846 + 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

@cython.cdivision(True)
cdef inline double intrp_halfcos(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef:
        double dx, y
        double x1x0 = x1 - x0
        int i
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            dx = (x - x0) / x1x0    
            dx = (dx + 1) * 3.14159265358979323846
            x = x0 + x1x0 * (1 + cos(dx)) / 2.0
    dx = (x - x0) / x1x0    
    dx = (dx + 1) * 3.14159265358979323846
    y = y0 + (y1 - y0) * (1 + cos(dx)) / 2.0
    return y


@cython.cdivision(True)
cdef inline double _interpol_halfcosexp(double x, double x0, double y0, double x1, double y1, double exp) nogil:
    cdef double dx
    dx = pow((x - x0) / (x1 - x0), exp)
    dx = (dx + 1.0) * 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

@cython.cdivision(True)
cdef inline double intrp_halfcosexp(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef:
        double dx
        double x1x0 = x1 - x0
        int i
        double exp = self.exp
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            dx = pow((x - x0) / x1x0, exp)    
            dx = (dx + 1) * 3.14159265358979323846
            x = x0 + x1x0 * (1 + cos(dx)) / 2.0
    dx = pow((x - x0) / x1x0, exp)    
    dx = (dx + 1) * 3.14159265358979323846        
    return y0 + (y1 - y0) * (1 + cos(dx)) / 2.0

    
@cython.cdivision(True)
cdef inline double _interpol_halfcosexpm(double x, double x0, double y0, double x1, double y1, double exp) nogil:
    cdef double dx
    if y1 < y0:
        exp = 1/exp
    dx = pow((x - x0) / (x1 - x0), exp)
    dx = (dx + 1.0) * 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)

@cython.cdivision(True)
cdef inline double intrp_halfcosexpm(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef:
        double dx
        double x1x0 = x1 - x0
        double exp = self.exp
        int i
    if y1 < y0:
        exp = 1/exp    
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            dx = pow((x - x0) / x1x0, exp)    
            dx = (dx + 1) * 3.14159265358979323846
            x = x0 + x1x0 * (1 + cos(dx)) / 2.0
    dx = pow((x - x0) / x1x0, exp)    
    dx = (dx + 1) * 3.14159265358979323846        
    return y0 + (y1 - y0) * (1 + cos(dx)) / 2.0

@cython.cdivision(True)
cdef inline double _interpol_halfcos2(double x, double x0, double y0, double x1, double y1, double exp) nogil:
    cdef double x2
    x2 = _interpol_halfcosexp(x, x0, x0, x1, x1, exp)
    return _interpol_halfcos(x2, x0, y0, x1, y1, 0)
    
    
@cython.cdivision(True)
cdef inline double _interpol_expon(double x, double x0, double y0, double x1, double y1, double exp) nogil:
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, exp) * (y1 - y0)

@cython.cdivision(True)
cdef inline double intrp_expon(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef: 
        double exp = self.exp
        double dx
        double x1x0 = x1 - x0
        int i
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            dx = (x - x0) / x1x0
            x = x0 + pow(dx, exp) * x1x0
    dx = (x - x0) / x1x0
    return y0 + pow(dx, exp) * (y1 - y0)


@cython.cdivision(True)
cdef inline double _interpol_exponm(double x, double x0, double y0, double x1, double y1, double exp) nogil:
    if y1 < y0:
        exp = 1/exp
    cdef double dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, exp) * (y1 - y0)

@cython.cdivision(True)
cdef inline double intrp_exponm(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef: 
        double exp = self.exp
        double dx
        double x1x0 = x1 - x0
        int i
    if y1 < y0:
        exp = 1/exp
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            dx = (x - x0) / x1x0
            x = x0 + pow(dx, exp) * x1x0
    dx = (x - x0) / x1x0
    return y0 + pow(dx, exp) * (y1 - y0)

cdef inline double _interpol_log(double x, double x0, double y0, double x1, double y1, double NOTUSED) nogil:
    pass

@cython.cdivision(True)
cdef inline double _fib(double x) nogil:
    """
    taken from home.comcast.net/~stuartmanderson/fibonacci.pdf
    fib at x = e^(x * ln(phi)) - cos(x * pi) * e^(x * ln(phi))
               -----------------------------------------------
                                     sqrt(5)
    """
    cdef double x_mul_log_phi = x * 0.48121182505960348 # 0.48121182505960348 = log(PHI)
    return (exp(x_mul_log_phi) - cos(x * PI) * exp(-x_mul_log_phi)) / 2.23606797749978969640917366873127623544
    
@cython.cdivision(True)
cdef inline double _interpol_fib(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    """
    fibonacci interpolation. it is assured that if x is equidistant to
    x0 and x1, then for the result y it should be true that

    y1 / y == y / y0 == ~0.618
    """
    cdef double dx = (x - x0) / (x1 - x0)
    cdef double dx2 = _fib(40 + dx * 2)
    cdef double dx3 = (dx2 - 102334155) / (165580141)
    return y0 + (y1 - y0) * dx3

@cython.cdivision(True)
cdef inline double intrp_fib(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    """
    fibonacci interpolation. it is assured that if x is equidistant to
    x0 and x1, then for the result y it should be true that

    y1 / y == y / y0 == ~0.618
    """
    cdef double dx = (x - x0) / (x1 - x0)
    cdef double dx2 = _fib(40 + dx * 2)
    cdef double dx3 = (dx2 - 102334155) / (165580141)
    return y0 + (y1 - y0) * dx3


@cython.cdivision(True)
cdef inline double _interpol_smooth(double x, double x0, double y0, double x1, double y1, double unused0) nogil:
    """
    #define SMOOTHSTEP(x) (x) * (x) * (3 - 2 * x)
    for (i = 0; i < N; i++) {
      v = i / N;
      v = SMOOTHSTEP(v);
      X = (A * v) + (B * (1 - v));
    }   --> http://sol.gfxile.net/interpolation/
    """
    cdef double v = (x - x0)/(x1 - x0)
    v = v*v*(3 - 2*v)
    return y0 + (y1 - y0) * v
    
@cython.cdivision(True)
cdef inline double intrp_smooth(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    """
    #define SMOOTHSTEP(x) (x) * (x) * (3 - 2 * x)
    for (i = 0; i < N; i++) {
      v = i / N;
      v = SMOOTHSTEP(v);
      X = (A * v) + (B * (1 - v));
    }   --> http://sol.gfxile.net/interpolation/
    """
    cdef:
        double x1x0 = x1 - x0
        double v
        int i
    if self.numiter > 1:
        for i in range(self.numiter - 1):
            v = (x - x0) / x1x0
            v = v*v*(3 - 2*v)
            x = x0 + x1x0 * v
    v = (x - x0) / x1x0
    v = v*v*(3 - 2*v)
    return y0 + (y1 - y0) * v


# ------------------------------------------------------------------------------------------------

@cython.cdivision(True)
cdef inline double _integr_trapz(double *xs, int xs_size, double dx) nogil:
    """
    area of a trapezium (trapezoid)

        a + b
    A = ----- h
          2 

    where
    A: area of the trapezium
    a: length of one of the sides
    b: length of the other side
    h: distance between a and b (height)

    In a function

      /|
     / |
    |  |
    |  |
    ----

    The curve is divided by a grid of dx. 
    a: f(x)
    b: f(x+dx)
    h: dx

    trapz = (f(x) + f(x+dx)) / 2 * dx
    """
    cdef int i
    cdef double x0, x1, accum
    cdef double r = 0.5 * dx
    x0 = xs[0]
    accum = 0
    for i in range(1, xs_size):
        x1 = xs[i]
        accum += (x0 + x1) * r
        x0 = x1
    return accum

@cython.cdivision(True)
cdef inline double _integr_trapz_between(double *xs, int xs_size, double dx, double offset, double a, double b) nogil:
    """
    integrate sampled values `xs`, spaced at `dx` with an offset of `offset` in the range
    [a, b] -- both a and b included

    NB: the definition of xs is used, no oversampling takes place for better accuracy
    in the cases where a and/or b do not fall exactly on the grid specified by offset+i*dx
    """
    cdef int i
    cdef double x0, x1, accum
    cdef double r = 0.5 * dx
    cdef int i0 = <int>((a - offset) / dx)
    cdef int i1 = <int>((b - offset) / dx) + 1
    accum = 0
    if i1 > xs_size:
        i1 = xs_size
    x0 = xs[i0]
    for i in range(i0+1, i1):
        x1 = xs[i]
        accum += (x0 + x1) * r
        x0 = x1
    return accum

@cython.cdivision(True)
cdef inline double _integr_trapz_between_exact(double *xs, int xs_size, double dx, double offset, double a, double b) nogil:
    """
    integrate sampled values `xs`, spaced at `dx` with an offset of `offset` in the range
    [a, b] -- both a and b included
    """
    cdef int i
    cdef double x0, x1, accum, y
    cdef double r = 0.5 * dx
    cdef int i0 = <int>((a - offset) / dx + 0.999999999)
    cdef int i1 = <int>((b - offset) / dx) + 1
    cdef double rest_a = ((a - offset) % dx) / dx
    cdef double rest_b = ((b - offset) % dx) / dx
    accum = 0
    if i1 > xs_size:
        i1 = xs_size
    x0 = xs[i0]
    for i in range(i0+1, i1):
        x1 = xs[i]
        accum += (x0 + x1) * r
        x0 = x1
    if rest_a > 0 and i0 > 0:
        y0 = xs[i0-1]
        y1 = xs[i0]
        y = y0 + (y1-y0) * rest_a
        accum += (y + y1) * 0.5 * dx * (1 - rest_a)
    if rest_b > 0 and (i1 + 1) < xs_size:
        y0 = xs[i1]
        y1 = xs[i1+1]
        y = y0 + (y1-y0) * rest_a
        accum += (y0 + y) * 0.5 * dx * rest_b
    return accum

cdef inline double _clip(double x, double x0, double x1) nogil:
    return x1 if x > x1 else x0 if x < x0 else x

cdef inline double _ntodx(size_t n, double x0, double x1):
    return (x1 - x0) / (n - 1)

cdef double _a4 = 442.0
DEF loge_2 = 0.6931471805599453094172321214581766


def setA4(double freq):
    global _a4
    _a4 = freq


@cython.cdivision(True)
cdef double m2f(double midinote) nogil:
    global _a4
    if 0.0 <= midinote:
        return _a4 * pow(2.0, (midinote - 69.0) / 12.0)
    return 0.

@cython.cdivision(True)
cdef double f2m(double freq) nogil:
    global _a4
    if 8.2129616379875419 < freq:    # this is the freq. of midi 0
        return 12 * (log(freq / _a4) / loge_2) + 69.0
    return 0

# ---------------------------- TYPES
ctypedef double(*t_unfunc)(double) nogil

cdef t_unfunc UNFUNCS[14]
UNFUNCS[:] = [
    cos,    # 0
    sin,    # 1
    ceil,   # 2
    log,    # 3
    exp,    # 4
    floor,  # 5
    tanh,   # 6
    fabs,   # 7
    sqrt,   # 8
    acos,   # 9
    asin,   # 10
    tan,    # 11
    sinh,   # 12
    log10,  # 13
    # m2f,    # 14
    # f2m     # 15
]

ctypedef double(*t_func0)(double, double, double, double, double, double) nogil
ctypedef double(*t_func)(InterpolFunc *, double, double, double, double, double) nogil

ctypedef struct InterpolFunc:
    t_func func
    double exp
    int numiter
    double mix
    InterpolFunc* blend_func
    char *name
    unsigned int needs_free

DTYPE = numpy.float #np.float64
ctypedef c_numpy.float_t DTYPE_t

# InterpolFunc
cdef inline void InterpolFunc_init(InterpolFunc *self, t_func func, double exp, char *name, unsigned int needs_free):
    self.func = func
    self.exp = exp
    self.numiter = 1
    self.mix = -1
    self.blend_func = NULL
    self.name = name
    self.needs_free = needs_free

cdef inline InterpolFunc* InterpolFunc_new(t_func func, double exp, char *name, unsigned int needs_free):
    cdef InterpolFunc* out
    out = <InterpolFunc *>malloc(sizeof(InterpolFunc))
    InterpolFunc_init(out, func, exp, name, needs_free)
    return out

cdef inline InterpolFunc* InterpolFunc_new_blend_from_descr(str descr0, str descr1, double mix):
    cdef InterpolFunc* out = InterpolFunc_new_from_descriptor(descr0, 1)  # force new 
    cdef InterpolFunc* blend = InterpolFunc_new_from_descriptor(descr1, 1)  # force new
    out.blend_func = blend
    out.mix = mix
    return out

cdef InterpolFunc* InterpolFunc_new_from_descriptor(str descr, int forcenew=0):
    cdef InterpolFunc* out = NULL
    cdef double exp = 1.0
    cdef str func_name
    if "(" in descr:
        func_name, param = descr.split("(")
        exp = float(param[:len(param)-1])
    else:
        func_name = descr
    if func_name == 'linear':
        if not forcenew:
            out = InterpolFunc_linear
        else:
            out = InterpolFunc_new(intrp_linear, 1, '', 1)
    elif func_name == 'expon':
        out = InterpolFunc_new(intrp_expon, exp, 'expon', 1)
    elif func_name == 'halfcos':
        if exp == 1.0 and not forcenew:
            out = InterpolFunc_halfcos
        else:
            out = InterpolFunc_new(intrp_halfcosexp, exp, 'halfcosexp', 1)
    elif func_name == 'halfcosexp':
        out = InterpolFunc_new(intrp_halfcosexp, exp, 'halfcosexp', 1)
    elif func_name == 'nointerpol':
        out = InterpolFunc_nointerpol
    elif func_name == 'fib':
        out = InterpolFunc_fib
    elif func_name == 'nearest':
        out = InterpolFunc_nearest
    elif func_name == 'smooth':
        out = InterpolFunc_smooth
    return out

cdef inline double InterpolFunc_call(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
    cdef double v0, v1
    if self.mix <= 0:
        return self.func(self, x, x0, y0, x1, y1)
    else:
        v0 = self.func(self, x, x0, y0, x1, y1)
        v1 = self.blend_func.func(self.blend_func, x, x0, y0, x1, y1)
        return v0 * (1 - self.mix) + v1 * self.mix

#cdef inline double InterpolFunc_call(InterpolFunc *self, double x, double x0, double y0, double x1, double y1) nogil:
#    cdef double y = self.func(self, x, x0, y0, x1, y1)
#    return y

cdef inline void InterpolFunc_free(InterpolFunc *self):
    if self is not NULL:
        if self.blend_func is not NULL:
            InterpolFunc_free(self.blend_func)
        if self.needs_free == 1:
            free(self)

cdef inline str InterpolFunc_get_descriptor(InterpolFunc *self):
    if self.exp != 1.0:
        return "%s(%s)" % (self.name, str(self.exp))
    else:
        return self.name   

# create the most used interpolation functions, which are shared across BPFs
# cdef InterpolFunc* InterpolFunc_linear  = InterpolFunc_new(_interpol_linear,  1.0, 'linear',  0)
cdef InterpolFunc* InterpolFunc_linear    = InterpolFunc_new(intrp_linear, 1.0, 'linear',  0)
cdef InterpolFunc* InterpolFunc_halfcos    = InterpolFunc_new(intrp_halfcos, 1.0, 'halfcos', 0)
cdef InterpolFunc* InterpolFunc_nointerpol = InterpolFunc_new(intrp_nointerpol,  1.0, 'nointerpol',  0)
cdef InterpolFunc* InterpolFunc_fib       = InterpolFunc_new(intrp_fib,  1.0, 'fib',  0)
cdef InterpolFunc* InterpolFunc_nearest   = InterpolFunc_new(intrp_nearest,  1.0, 'nearest',  0)
cdef InterpolFunc* InterpolFunc_smooth    = InterpolFunc_new(intrp_smooth, 1.0, 'smooth', 0)

cdef inline ndarray EMPTY1D(int size): #new_empty_doublearray_1D(int size):
    cdef npy_intp *dims = [size]
    return PyArray_EMPTY(1, dims, NPY_DOUBLE, 0)

DEF NUM_XS_FOR_RENDERING = 200
DEF DEFAULT_EPSILON = 1e-4
DEF INF = float('inf')
DEF INFNEG = float('-inf')
cdef double MAX_FLOAT = 3.40282346638528860e+38
CONST_XS_FOR_RENDERING = numpy.linspace(0., 1., NUM_XS_FOR_RENDERING)
DEF NOCOPY = False

# behaviour for cropping
DEF OUTBOUND_DEFAULT    = -1
DEF OUTBOUND_DONOTHING  = 0     # do nothing, just return the value of the original bpf. in this case, cropping only sets the bounds
DEF OUTBOUND_CACHE      = 1     # cache the value of the bpf at creation time and return this value outside the bounds
DEF OUTBOUND_SET        = 2     # set the value outside the bounds


cpdef int _array_issorted(double[:] xs):
    """
    Returns:
       -1 - not sorted
        0 - array is sorted, with dups
        1 - array is sorted, no dups

    """
    cdef int i
    cdef double x0, x1
    cdef int nodups = 1
    x1 = xs[0]
    with nogil:
        for i in range(1, xs.shape[0]):
            x0 = x1
            x1 = xs[i]
            if x1 < x0:
                return -1
            elif x1 == x0:
                nodups = 0
    return nodups


cdef inline int _searchsorted(double [:]xs, double x) nogil:
    cdef int imin = 0
    cdef int imax = xs.shape[0]
    cdef int imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if xs[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef inline int _csearchsorted(DTYPE_t *xs, int xs_length, DTYPE_t x) nogil:
    """
    equivalent to bisect_right. 
    xs[out] > x
    """
    cdef int imin = 0
    cdef int imax = xs_length
    cdef int imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if (<DTYPE_t *>(xs))[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef inline int _csearchsorted_left(DTYPE_t *xs, int xs_length, DTYPE_t x) nogil:
    cdef int imin = 0
    cdef int imax = xs_length
    cdef int imid
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if xs[imid] <= x:
            imin = imid + 1
        else:
            imax = imid
    return imin


cdef inline double* _seq_to_doubles(xs):
    cdef int size, i
    cdef double* out
    cdef double* data
    if isinstance(xs, ndarray):
        size = PyArray_DIM(<ndarray>xs, 0)
        out = <double *>malloc(sizeof(double) * size)
        if PyArray_ISCONTIGUOUS(<ndarray>xs):
            data = <DTYPE_t *>((<ndarray>xs).data)
            for i in range(size):
                out[i] = data[i]
        else:
            for i in range(size):
                out[i] = xs[i]
    else:
        if isinstance(xs, list):
            size = len(<list>xs)
            out = <double *>malloc(sizeof(double) * size)
            for i in range(size):
                out[i] = (<list>xs)[i]
        elif isinstance(xs, tuple):
            size = len(<tuple>xs)
            out = <double *>malloc(sizeof(double) * size)
            for i in range(size):
                out[i] = (<tuple>xs)[i]
        else:
            size = len(xs)
            out = <double *>malloc(sizeof(double) * size)
            for i in range(size):
                out[i] = xs[i]
    return out

def _get_bounds(a, b):
    cdef double start, end, b_start, b_end
    start, end = a.bounds()
    try:
        b_start, b_end = b.bounds()
        start = start if start < b_start else b_start
        end = end if end > b_end else b_end
    except:
        pass
    return (start, end)

cdef class _BpfInterface

cdef _BpfInterface _as_bpf(obj):
    try:
        return obj._asbpf()
    except:
        if hasattr(obj, '__call__'):
            return _FunctionWrap(obj, (INFNEG, INF))
        elif hasattr(obj, '__float__'):
            return Const(float(obj))
        else:
            return None

# ~~~~~~~~~~~~~~~~~~~ BpfInterface ~~~~~~~~~~~~~~~~~~~~~

cdef class _BpfInterface:
    cdef double _x0, _x1
    cdef int _integration_mode  # 0: dont use scipy, 1: use scipy, -1: calibrate
    cpdef _BpfInterface _asbpf(self): return self
    cdef void _bounds_changed(self): pass
    
    cdef inline void _set_bounds(self, double x0, double x1):
        self._x0 = x0
        self._x1 = x1
        self._integration_mode = CONFIG['integrate.default_mode']
    
    cdef inline void _set_bounds_like(self, _BpfInterface a):
        self._set_bounds(a._x0, a._x1)
    
    cpdef double ntodx(self, int N):
        """
        Calculate dx so that the bounds of this bpf are divided into N parts
        (x1-x0)/(N-1)
        """
        return (self._x1 - self._x0) / (N - 1)
    
    cpdef int dxton(self, double dx):
        """
        Calculate how many parts fit in this bpf with given dx
        (x1+dx - x0)/dx
        """
        return <int>(((self._x1 + dx) - self._x0) / dx)
    
    def bounds(self):
        return self._x0, self._x1
    
    property x0:
        def __get__(self): return self._x0
    
    property x1:
        def __get__(self): return self._x1
    
    def __add__(a, b):
        return _create_lambda_unordered(a, b, _BpfLambdaAdd, _BpfLambdaAddConst)
    
    def __sub__(a, b):
        return _create_rlambda(a, b, _BpfLambdaSub, _BpfLambdaSubConst, _BpfLambdaRSub, _BpfLambdaRSubConst)
    
    def __mul__(a, b):
        cdef float v
        try:
            v = float(b)  # are we a?
            if v == 0:
                return Const(0).set_bounds(a._x0, a._x1)
            elif v == 1:
                return a
            else:
                return _BpfLambdaMulConst(a, b, a.bounds())
        except (TypeError, ValueError):
            try:
                v = float(a) # are we b?
                if v == 0:
                    return Const(0).set_bounds(b._x0, b._x1)
                elif v == 1:
                    return b
                else:
                    return _BpfLambdaMulConst(b, a, b.bounds())
            except (TypeError, ValueError):
                return _create_lambda_unordered(a, b, _BpfLambdaMul, _BpfLambdaMulConst)  # two bpfs
    
    def __div__(a, b):
        cdef float v
        try:
            v = float(b)  # are we a?
            if v == 0:
                raise ZeroDivisionError("Can't divide by 0")
            elif v == 1:
                return a
            else:
                return _BpfLambdaDivConst(a, b, a.bounds())
        except (TypeError, ValueError):
            try:
                v = float(a) # are we b?
                if v == 0:
                    return 0
                else:
                    return _BpfLambdaRDivConst(b, a, b.bounds())
            except (TypeError, ValueError):
                return _create_rlambda(a, b, _BpfLambdaDiv, _BpfLambdaDivConst, _BpfLambdaRDiv, _BpfLambdaRDivConst)
    
    def __truediv__(a, b):
        return _create_rlambda(a, b, _BpfLambdaDiv, _BpfLambdaDivConst, _BpfLambdaRDiv, _BpfLambdaRDivConst)
    
    def __pow__(a, b, modulo):
        cdef double tmp
        if isinstance(a, _BpfInterface):
            if isinstance(b, _BpfInterface):
                return _BpfLambdaPow(a, b, _get_bounds(a, b))
            elif callable(b):
                return _BpfLambdaPow(a, _FunctionWrap(b), a.bounds())
            elif b == 2:
                return a*a
            elif b == 3:
                return a*a*a
            elif b == 4:
                tmp = a*a
                return tmp*tmp
            elif b == 0:
                return a
            elif b == -1:
                return 1/a
            elif b == -2:
                return 1/(a*a)
            else:
                return _BpfLambdaPowConst(a, b, a.bounds())
        elif isinstance(b, _BpfInterface):
            if callable(a):
                return _BpfLambdaPow(_FunctionWrap(a), b, b.bounds())
            return _BpfLambdaRPowConst(b, a, (INFNEG, INF))
        return NotImplemented

    def __neg__(self):
        return self * -1
    
    def __mod__(self, other):
        return _create_lambda(self, other, _BpfLambdaMod, _BpfLambdaModConst)
    
    def __abs__(self):
        return self.abs()
    
    def __or__(a, b):
        """
        a | b
        """
        if isinstance(a, _BpfInterface) and isinstance(b, _BpfInterface):
            out = _BpfCompose_new(a, b)
        elif isinstance(a, _BpfInterface) and callable(b):
            out = _BpfCompose_new(a, _FunctionWrap(b))
        elif callable(a) and isinstance(b, _BpfInterface):
            out = _BpfCompose_new(_FunctionWrap(b), a)
        else:
            return NotImplemented 
        return out
    
    def __rshift__(a, b):
        if isinstance(a, _BpfInterface):
            return a.shifted(b)
        return NotImplemented

    def __lshift__(a, b):
        if isinstance(a, _BpfInterface):
            return a.shifted(-b)
        return NotImplemented
        
    def __xor__(a, b): # ^
        if isinstance(a, _BpfInterface):
            return a.stretched(b)
        return NotImplemented

    def __richcmp__(_BpfInterface self, other, int t):
        if t == 0:      # <
            return _create_lambda(self, other, _BpfLambdaLowerThan, _BpfLambdaLowerThanConst)
        elif t == 2:    # ==
            return _create_lambda(self, other, _BpfLambdaEqual, _BpfLambdaEqualConst)
        elif t == 4:    # >
            return _create_lambda(self, other, _BpfLambdaGreaterThan, _BpfLambdaGreaterThanConst)
        elif t == 1:    # <=
            return _create_lambda(self, other, _BpfLambdaLowerOrEqualThan, _BpfLambdaLowerOrEqualThanConst)       # (self > other) == 0
        elif t == 3:    # !=
            return _create_lambda(self, other, _BpfLambdaUnequal, _BpfLambdaUnequalConst)
        elif t == 5:    # >=
            return _create_lambda(self, other, _BpfLambdaGreaterOrEqualThan, _BpfLambdaGreaterOrEqualThanConst) # (self < other) == 0
    
    def _get_xs_for_rendering(self, int n):
        cdef double x0, x1
        x0 = self.x0 if self.x0 != INFNEG else 0.0
        x1 = self.x1 if (self.x1 != INF and self.x1 > x0) else x0 + 1.0
        out = numpy.linspace(x0, x1, n)
        return out
    
    def _get_points_for_rendering(self, int n= -1):
        if n == -1:
            n = NUM_XS_FOR_RENDERING
        xs = self._get_xs_for_rendering(n)
        ys = self.map(xs)
        return xs, ys
    
    def render(self, xs, interpolation='linear'):
        """
        return a NEW bpf representing this bpf

        xs: a seq of points at which this bpf is sampled
            or a number, in which case an even grid is calculated
            with that number of points
        interpolation: the same interpolation types supported by
                       .sampled
        """
        if isinstance(xs, (int, long)):
            dx = (self._x1 - self._x0) / (xs - 1)
            return self.sampled(dx, interpolation=interpolation)
        else:
            ys = self.map(xs)
            if interpolation == 'linear':
                return Linear(xs, ys)
            elif interpolation == 'nointerpol':
                return NoInterpol(xs, ys)
            else:
                raise ValueError("interpolation %s not implemented" % interpolation)

    def plot(self, **keys):
        """
        plot the bpf. any key is passed to plot.plot_coords

        additional keywords:
            n       number of points to plot
            show    if the plot should be shown immediately after (default is True). If you
                    want to display multiple BPFs in one plot, for instance to compare them,
                    you can call plot on each of the BPFs with show=False, and then either
                    call the last one with plot=True or call bpf3.plot.show().
                    the module bpf3.plot is intended to be an abstraction over multiple plotting
                    backends, but at the moment only matplotlib is supported.
        """
        cdef int n = keys.pop('n', -1)
        xs, ys = self._get_points_for_rendering(n)
        from . import plot
        plot.plot_coords(xs, ys, **keys)
        return self
    
    cpdef _BpfInterface sampled(self, double dx, interpolation='linear'):
        """
        sample this bpf at an interval of dx (samplerate = 1 / dx)
        returns a Sampled bpf with the given interpolation between the samples

        interpolation can be any kind of interpolation, for example
        'linear', 'nointerpol', 'expon(2.4)', 'halfcos(0.5)', etc.

        if you need to sample a portion of the bpf, use sampled_between

        The same results can be achieved by the shorthand

        bpf[::0.1] will return a sampled version of this bpf with a dx of 0.1
        bpf[:10:0.1] will sample this bpf between (x0, 10) at a dx of 0.1
        """
        # we need to account for the edge (x1 IS INCLUDED)
        cdef int n = int((self._x1 - self._x0) / dx + 0.5) + 1
        ys = self.mapn_between(n, self._x0, self._x1) 
        return Sampled(ys, dx=dx, x0=self._x0, interpolation=interpolation)
    
    cpdef ndarray sample_between(self, double x0, double x1, double dx, ndarray out=None):
        """
        returns an array representing this bpf sampled at an interval of dx between x0 and x1

        x0 and x1 are included

        Example
        =======
        >>> thisbpf = bpf.linear(0, 0, 10, 10)
        >>> thisbpf.sample_between(0, 10, 1)
        [0 1 2 3 4 5 6 7 8 9 10]

        This is the same as thisbpf.mapn_between(11, 0, 10)
        """
        cdef int n
        n = int((x1 - x0) / dx + 0.5) + 1
        return self.mapn_between(n, x0, x1, out)
    
    cpdef _BpfInterface sampled_between(self, double x0, double x1, double dx, interpolation='linear'):
        """
        sample a portion of this bpf at an interval of dx
        returns a Sampled bpf with bounds=(x0, x1)

        This is the same as thisbpf[x0:x1:dx]
        """
        cdef int n = int((x1 - x0) / dx + 0.5) + 1
        ys = self.mapn_between(n, x0, x1)
        return Sampled(ys, dx=dx, x0=x0, interpolation=interpolation)

    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        """
        return a numpy array representing n values of this bpf between x0 and x1

        x0 and x1 are included

        If out is passed, an attempt will be done to use it as destination for the result
        Nonetheless, you should NEVER trust that this actually happens. See example

        Example
        =======

        >>> out = numpy.empty((100,), dtype=float)
        >>> out = thisbpf.mapn_between(100, 0, 10, out)   # <--- this is the right way to pass a result array
        """
        cdef ndarray[DTYPE_t, ndim=1] X = numpy.linspace(x0, x1, n)
        if out is None:
            out = X
        return self.map(X, out=out)

    cpdef ndarray map(self, xs, ndarray out=None):
        """
        The same as map(self, xs) but faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.

        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

        If out is passed, an attempt will be done to use it as destination for the result
        Nonetheless, you should NEVER trust that this actually happens. See example

        Example
        =======

        >>> out = numpy.empty((100,), dtype=float)
        >>> xs = numpy.linspace(0, 10, 100)
        >>> out = thisbpf.map(xs, out)   # <--- this is the right way to pass a result array
        """
        cdef ndarray [DTYPE_t, ndim=1] _xs
        cdef ndarray [DTYPE_t, ndim=1] result
        cdef DTYPE_t *data0
        cdef int i, nx
        cdef double x0,x1, dx
        if isinstance(xs, int):
            return self.mapn_between(xs, self._x0, self._x1, out)
        _xs = <ndarray>xs
        nx = PyArray_DIM(_xs, 0)
        if out is None:
            result = EMPTY1D(nx)
        else:
            result = <ndarray>out
        if PyArray_ISCONTIGUOUS(result):
            data0 = <DTYPE_t *>(result.data)
            with nogil:
                for i in range(nx):
                    data0[i] = self.__ccall__(_xs[i])
        else:
            for i in range(nx):
                result[i] = self.__ccall__(_xs[i])
        return result
    
    cpdef _BpfInterface concat(self, _BpfInterface other, double fadetime=0, fadeshape='expon(3)'):
        """
        glue this bpf to the other bpf, so that the beginning
        of the other is the end of this one
        """
        cdef _BpfInterface fade
        cdef _BpfInterface other2 = other.fit_between(self._x1, self._x1 + (other._x1 - other._x0))
        if fadetime == 0:
            return _BpfConcat2_new(self, other2, other2._x0)
        raise NotImplementedError("fade is not implemented")
    
    cpdef _BpfLambdaRound round(self):
        return _BpfLambdaRound(self)
    cpdef _BpfRand rand(self):
        return _BpfRand(self)
    
    cpdef _BpfUnaryFunc cos(self):  return _BpfUnaryFunc_new_from_index(self, 0)
    
    cpdef _BpfUnaryFunc sin(self):  return _BpfUnaryFunc_new_from_index(self, 1)
    
    cpdef _BpfUnaryFunc ceil(self): return _BpfUnaryFunc_new_from_index(self, 2)
    
    cpdef _BpfUnaryFunc exp(self):  return _BpfUnaryFunc_new_from_index(self, 4)
    
    cpdef _BpfUnaryFunc floor(self): return _BpfUnaryFunc_new_from_index(self, 5)
    
    cpdef _BpfUnaryFunc tanh(self): return _BpfUnaryFunc_new_from_index(self, 6)
    
    cpdef _BpfUnaryFunc abs(self):  return _BpfUnaryFunc_new_from_index(self, 7)
    
    cpdef _BpfUnaryFunc sqrt(self): return _BpfUnaryFunc_new_from_index(self, 8)
    
    cpdef _BpfUnaryFunc acos(self): return _BpfUnaryFunc_new_from_index(self, 9)
    
    cpdef _BpfUnaryFunc asin(self): return _BpfUnaryFunc_new_from_index(self, 10)
    
    cpdef _BpfUnaryFunc tan(self):  return _BpfUnaryFunc_new_from_index(self, 11)
    
    cpdef _BpfUnaryFunc sinh(self): return _BpfUnaryFunc_new_from_index(self, 12)
    
    cpdef _BpfUnaryFunc log10(self): return _BpfUnaryFunc_new_from_index(self, 13)
    
    cpdef _BpfLambdaLog log(self, double base=M_E): return _BpfLambdaLog(self, base, self.bounds())
    
    cpdef _BpfM2F m2f(self): return _BpfM2F(self)
    
    cpdef _BpfF2M f2m(self): return _BpfF2M(self)
    
    cpdef _Bpf_db2amp db2amp(self): return _Bpf_db2amp(self)
    
    cpdef _Bpf_amp2db amp2db(self): return _Bpf_amp2db(self)
    
    cpdef _BpfLambdaFib fib(self): return _BpfLambdaFib(self)
    
    cpdef _BpfLambdaClip clip(self, double y0=INFNEG, double y1=INF): return _BpfLambdaClip_new(self, y0, y1)

    cpdef _BpfInterface derivative(self):
        """
        Return a curve which represents the derivative of this curve

        It implements Newtons difference quotiont, so that

        derivative(x) = bpf(x + h) - bpf(x)
                        -------------------
                                  h
        """
        return _BpfDeriv(self)

    cpdef _BpfInterface integrated(self):
        """
        Return a bpf representing the integration of this bpf at a given point
        """
        if self._x0 == INFNEG:
            raise ValueError("Cannot integrate a function with an infinite negative bound")
        return _BpfIntegrate(self)
    
    cpdef double integrate(self):
        """
        Return the result of the integration of this bpf. If any of the bounds is inf,
        the result is also inf.

        Tip: to determine the limits of the integration, first crop the bpf via a slice
        Example:

        b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)
        
        """
        if isinf(self._x0) or isinf(self._x1):
            return INFINITY
        return self.integrate_between(self._x0, self._x1)
    
    cpdef double trapz_integrate_between(self, double x0, double x1, size_t N=0):
        """
        the same as integrate() but within the bounds [x0, x1]

        N: optional. a hint to the number of subdivisions used to calculate 
           the integral. If not given, a default is used. The default is defined in 
           CONFIG['integrate.trapz_intervals']
        """
        cdef:
            double dx
            double [::1] ys
        if N == 0:
            N = CONFIG['integrate.trapz_intervals']
        dx = (x1 - x0) / N
        if dx <= 0:
            return 0.0
        ys = self.sample_between(x0, x1, dx)
        return _integr_trapz(&ys[0], ys.shape[0], dx)
    
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        cdef double out
        cdef int get_mode
        cdef int mode = self._integration_mode
        cdef double outbound0, outbound1, inbound
        if x1 > self._x1:
            outbound1 = self.__ccall__(x1) * (x1 - self._x1)
            x1 = self._x1
        else:
            outbound1 = 0
        if x0 < self._x0:
            outbound0 = self.__ccall__(x0) * (self._x0 - x0)
            x0 = self._x0
        else:
            outbound0 = 0
        # override mode when N is given
        if N > 0:
            mode = 0
        if mode == 1 or mode == 2:
            inbound = integrate_simpsons(self, x0, x1, SIMPSONS_ACCURACY, SIMPSONS_MAXITER)
        else:
            inbound = self.trapz_integrate_between(x0, x1, N)
        return outbound0 + inbound + outbound1

    cpdef list zeros(self, double h=0.01, int N=0, double x0=NAN, double x1=NAN, int maxzeros=0):
        """
        return a list with the zeros of this bpf
        
        Parameters
        ----------
        
        h: the accuracy to scan for zeros-crossings. If two zeros are within this distance, they will be resolved as one.
        N: alternatively, you can give the number of intervals to scan. h will be derived from this
        x0, x1: limit the search to these boundaries, otherwise use the bounds of the bpf
        maxzeros: if possitive, stop the search when this number of zeros have been found
        
        Example
        -------
        
        >>> a = bpf.linear(0, -1, 1, 1)
        >>> a.zeros()
        [0.5]

        NB: to calculate one zero 
        """
        return bpf_zero_crossings(self, h=h, N=N, x0=x0, x1=x1, maxzeros=maxzeros)

    def max(self, b):
        if isinstance(b, _BpfInterface):
            return Max(self, b)
        return _BpfMaxConst(self, b)

    def min(self, b):
        if isinstance(b, _BpfInterface):
            return Min(self, b)
        return _BpfMinConst(self, b)

    def dump(self, str format='yaml', outfile=None):
        """
        dump the data of this bpf to a file (or as string if no path is given)
        if outfile is not given, a string with the output is returned (like dumps)

        formats supported: csv, yaml, json
        """
        import util
        return util.dumpbpf(self, format, outfile)

    def __reduce__(self):
        return type(self), self.__getstate__()

    cdef double __ccall__(self, double other) nogil:
        return 0.0

    def __call__(_BpfInterface self, double other):
        """
        _BpfInterface.__call__(self, double x)
        """
        return self.__ccall__(other)

    def keep_slope(self, double EPSILON=DEFAULT_EPSILON):
        """
        return a new bpf which is a copy of this bpf when inside
        bounds() but outside bounds() it behaves as a linear bpf
        with a slope equal to the slope of this bpf at its extremes
        """
        return _BpfKeepSlope(self, EPSILON)

    def outbound(self, double y0, double y1):
        """
        return a new Bpf with the given values outside the bounds

        Example 1:

        a = bpf.linear(0, 1, 1, 10).outbound(-1, 0)
        a(-0.5) --> -1
        a(1.1)  --> 0
        a(0)    --> 1
        a(1)    --> 10

        Example 2: fallback to another curve outside self

        a = bpf.linear(0, 1, 1, 10).outbound(0, 0) + bpf.expon(-1, 2, 4, 10)
        """
        return _BpfCrop_new(self, self._x0, self._x1, OUTBOUND_SET, y0, y1)

    def apply(self, func):
        """
        return a new bpf where func is applied to the result of it
        func(self(x))   -- see 'function composition'

        example:

        from math import sin
        new_bpf = this_bpf.apply(sin)
        assert new_bpf(x) == sin(this_bpf(x))

        NB: A_bpf.apply(B_bpf) is the same as A_bpf | B_bpf
        """
        return _BpfCompose_new(self, _FunctionWrap(func))

    def preapply(self, func):
        """
        return a new bpf where func is applied to the argument
        before it is passed to the bpf

        bpf(func(x))

        example:

        >> bpf = Linear((0, 1, 2), (0, 10, 20))
        >> bpf(0.5)
        5

        >> shifted_bpf = bpf.preapply(lambda x: x + 1)
        >> shifted_bpf(0.5)
        15

        NB: A_bpf.preapply(B_bpf) is the same as B_bpf | A_bpf
        """
        return _BpfCompose_new(_FunctionWrap(func), self)

    def periodic(self):
        """
        return a new bpf which is is a copy of this bpf when inside
        bounds() and outside it replicates it in a periodic way.
        It has no bounds.

        example:

        >> b = linear(xs=(0, 1), ys(-1, 1)Linear((0, 1), (-1, 1)).periodic()
        >> b(0.5)
        0
        >> b(1.5)
        0
        >> b(-10)
        -1
        """
        return _BpfPeriodic(self)

    def stretched(self, rx, fixpoint=0):
        """
        returns new bpf which is a projection of this bpf stretched
        over the x axis. 

        NB: to stretch over the y-axis, just multiply this bpf
        See also: fit_between

        Example: stretch the shape of the bpf, but preserve the position

        >>> mybpf = linear(1, 1, 2, 2)
        >>> mybpf.stretched(4, fixpoint=mybpf.x0).bounds()
        (1, 9)
        """
        if rx == 0:
            raise ValueError("the stretch factor cannot be 0")
        return _BpfProjection(self,rx=1.0/rx, dx=0, offset=-fixpoint)

    cpdef _BpfInterface fit_between(self, double x0, double x1):
        """
        returns a new BPF which is the projection of this BPF
        to the interval x0:x1

        This operation only makes sense if the current BPF is bounded
        (none of its bounds is inf)
        """
        cdef double rx
        if self._x0 == INF or self._x0 == INFNEG or self._x1 == INF or self._x1 == INFNEG:
            raise ValueError("This bpf is unbounded, cannot be fitted."
                             "Use thisbpf[x0:x1].fit_between(...)")
        rx = (self._x1 - self._x0) / (x1 - x0)
        dx = self._x0
        offset = x0
        return _BpfProjection(self, rx=rx, dx=dx, offset=offset)


    cpdef _BpfInterface shifted(self, dx):
        """
        the same as shift, but a NEW bpf is returned, which is a shifted
        view on this bpf.

        >>> a = bpf.linear(0, 1, 1, 5)
        >>> b = a.shifted(2)
        >>> b(3) == a(1)
        """
        return _BpfProjection(self, rx=1, dx=-dx)

    def inverted(self):
        """
        Return a new BPF which is the inversion of this BPF, or None if the function is 
        not invertible.

        In an inverted function the coordinates are swaped: the inverted version of a 
        BPF indicates which x corresponds to a given y
        
        so f.inverted()(f(x)) = x
        
        For a function to be invertible, it must be strictly increasing or decreasing,
        with no local maxima or minima.
        
        so if y(1) = 2, then y.inverted()(2) = 1
        """
        try:
            return _BpfInverted(self)
        except ValueError:
            return None

    cpdef _BpfInterface _slice(self, double x0, double x1):
        return _BpfCrop_new(self, x0, x1, OUTBOUND_DEFAULT, 0, 0)

    def __getitem__(self, slice):
        cdef double x0, x1
        cdef _BpfInterface out
        x0 = self._x0
        x1 = self._x1
        try:
            if slice.start is not None:
                x0 = slice.start
            if slice.stop is not None:
                x1 = slice.stop
            if slice.step is not None:
                return self.sampled_between(x0, x1, slice.step, 'linear')
            return self._slice(x0, x1)
        except AttributeError:
            raise ValueError("BPFs accept only slices, not single items.")
    
    @classmethod
    def fromseq(cls, *points, **kws):
        """
        Bpf.fromseq(x0, y0, x1, y1, x2, y2, ...) == Bpf((x0, x1, ...), (y0, y1, ...))
        Bpf.fromseq((x0, y0), (x1, y1), (x2, y2), ...) == Bpf((x0, x1, ...), (y0, y1, ...))
        """
        cdef ndarray data
        cdef int lenpoints
        if isinstance(points[0], (list, tuple)):
            # (x0, y0), (x1, y1), ...
            data = numpy.asarray(points, dtype=DTYPE)
            return cls(data[:,0], data[:,1], **kws)
        else:
            # x0, y0, x1, y1, ...
            lenpoints = len(points)
            if lenpoints % 2 != 0:
                raise ValueError(
                    "The instantiation form x0, y0, x1, y1 should have an even number of args"
                )
            if lenpoints == 2:
                return cls((points[0], points[1]), (points[0], points[1]), **kws)
            elif lenpoints == 1:
                return cls((0, 0), (0, points[0]), **kws)
            else:
                return cls(points[::2], points[1::2], **kws)

    def copy(self):
        state = self.__getstate__()
        obj = self.__class__(*state)
        try:
            obj.__setstate__(state)
        except:
            pass
        return obj
    
    def __repr__(self):
        x0, x1 = self.bounds()
        return "%s[%s:%s]" % (self.__class__.__name__, str(x0), str(x1))

    def debug(self, key, value=None):
        """
        keys:
            * integrationmode
        """
        if key == "integrationmode":
            if value is not None:
                self._integration_mode = value
            return self._integration_mode
        
cdef inline ndarray _asarray(obj): 
    return <ndarray>(PyArray_GETCONTIGUOUS(array(obj, DTYPE, False)))

cdef class _BpfBase(_BpfInterface):
    cdef ndarray xs, ys
    cdef DTYPE_t* xs_data
    cdef DTYPE_t* ys_data
    cdef int outbound_mode
    cdef double outbound0, outbound1
    cdef double cached_bounds_x0, cached_bounds_x1, cached_bounds_y0, cached_bounds_y1
    cdef InterpolFunc *interpol_func
    cdef Py_ssize_t xs_size
    cdef size_t cached_index1

    def __cinit__(self):
        self.interpol_func = NULL
        self.ys_data = NULL
        self.xs_data = NULL
        self.xs = None
        self.ys = None

    def __dealloc__(self):
        InterpolFunc_free(self.interpol_func)

    def __init__(_BpfBase self, xs, ys):
        """
        xs and ys are arrays of points (x, y)
        """
        cdef int len_xs, len_ys
        cdef ndarray [DTYPE_t, ndim=1] _xs = numpy.ascontiguousarray(xs, DTYPE)
        if _array_issorted(_xs) < 1:
            raise BpfPointsError(f"Points along the x coord should be sorted without duplicates.\n xs:\n{xs}")
        cdef ndarray [DTYPE_t, ndim=1] _ys = numpy.ascontiguousarray(ys, DTYPE)
        len_xs = PyArray_DIM(_xs, 0)
        len_ys = PyArray_DIM(_ys, 0)
        if len_xs != len_ys:
            raise ValueError("xs and ys must be of equal length")
        if len_xs < 1:
            raise ValueError("Can't creat a BPF of 0 points")
        self.xs = _xs
        self.ys = _ys
        self.xs_size = PyArray_DIM(_xs, 0)
        self._set_bounds(_xs[0], _xs[len_xs - 1])
        self.outbound_mode = OUTBOUND_CACHE
        self.outbound0 = _ys[0]
        self.outbound1 = _ys[len_ys - 1]
        self.xs_data = <DTYPE_t*>(self.xs.data)
        self.ys_data = <DTYPE_t*>(self.ys.data)
        self.cached_bounds_x0 = self.xs[0]
        self.cached_bounds_x1 = self.xs[1]
        self.cached_bounds_y0 = self.ys[0]
        self.cached_bounds_y1 = self.ys[1]
        self.cached_index1 = 0

    property descriptor:
        def __get__(self): return InterpolFunc_get_descriptor(self.interpol_func)

    cdef void _bounds_changed(self):
        self._invalidate_cache()

    cdef void _invalidate_cache(self):
        cdef int last_index = PyArray_DIM(self.xs, 0) - 1
        self.cached_bounds_x0 = 0
        self.cached_bounds_y0 = 0
        self.cached_bounds_x1 = 0
        self.cached_bounds_y1 = 0
        self.cached_index1 = 0
        if self.ys_data != NULL and self.outbound_mode == OUTBOUND_CACHE:
            self.outbound0 = (<DTYPE_t *>(self.ys_data))[0]
            self.outbound1 = (<DTYPE_t *>(self.ys_data))[last_index]
        
    def outbound(self, double y0, double y1):
        """
        Set the values (INPLACE) which are returned when this bpf is evaluated
        outside its bounds.

        The default behaviour is to interpret the values at the bounds to extend to infinity.

        In order to not change this bpf inplace, use

        b.copy().outbound(y0, y1)
        """
        self.outbound_mode = OUTBOUND_SET
        self.outbound0 = y0
        self.outbound1 = y1
        return self

    def __getstate__(self):
        return (self.xs, self.ys)

    def __setstate__(self, state):
        self.xs, self.ys = state

    cdef double __ccall__(_BpfBase self, double x) nogil:
        cdef double res, x0, y0, x1, y1
        cdef int index0, index1, nx
        cdef DTYPE_t *xs_data
        cdef DTYPE_t *ys_data
        if self.cached_bounds_x0 <= x < self.cached_bounds_x1:
            res = InterpolFunc_call(self.interpol_func, x, self.cached_bounds_x0, self.cached_bounds_y0, self.cached_bounds_x1, self.cached_bounds_y1)
        elif x < self._x0:
            res = self.outbound0
        elif x > self._x1:
            res = self.outbound1
        elif (self.cached_index1 < self.xs_size - 2) and (self.cached_bounds_x1 <= x < self.xs_data[self.cached_index1+1]):
            # usual situation: cross to next bin
            index1 = self.cached_index1 + 1
            x1 = self.xs_data[index1]
            y1 = self.ys_data[index1]
            res = InterpolFunc_call(self.interpol_func, x, self.cached_bounds_x1, self.cached_bounds_y1, x1, y1)
            self.cached_bounds_x0 = self.cached_bounds_x1
            self.cached_bounds_y0 = self.cached_bounds_y1
            self.cached_bounds_x1 = x1
            self.cached_bounds_y1 = y1
            self.cached_index1 = index1
        else:
            # out of cache call, find new boundaries and update cache
            index1 = _csearchsorted(self.xs_data, self.xs_size, x)
            index0 = index1 - 1
            self.cached_bounds_x0 = x0 = self.xs_data[index0]
            self.cached_bounds_y0 = y0 = self.ys_data[index0]
            self.cached_bounds_x1 = x1 = self.xs_data[index1]
            self.cached_bounds_y1 = y1 = self.ys_data[index1]
            self.cached_index1 = index1
            res = InterpolFunc_call(self.interpol_func, x, x0, y0, x1, y1)
        return res

    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef double[:] result = out if out is not None else EMPTY1D(n)
        # cdef ndarray [DTYPE_t, ndim=1] result
        cdef double dx = (x1 - x0) / (n - 1)   # we account for the edge (x1 IS INCLUDED)
        with nogil:        
            for i in range(n):
                result[i] = self.__ccall__(x0 + dx*i)
        return numpy.asarray(result)
        
    def __repr__(self):
        return "%s[%s:%s]" % (self.__class__.__name__, str(self._x0), str(self._x1))

    def _blendshape(self, double mix, descr):
        """
        Returns a BPF with the same points as this one, 
        but using a mixed shape as interpolation between the points

        mix: the mixing factor. 0=this shape, 1=shape given by `descr`
        descr: the descriptor of the blending shape
        
        Example:

        a = bpf.linear(0, 0, 1, 1)
        c = a.blendshape(0.5, 'expon(2)')

        This is the same as
        a = bpf.linear(0, 0, 1, 1)
        b = bpf.expon(0, 0, 1, 1, exp=2)
        c = blend

        """
        xs, ys = self.points()
        return BlendShape(xs, ys, self.descriptor, descr, mix)

    def stretch(self, double rx):
        """
        stretch or compress the bpf n the x-coords. it modifies the bpf itself
        use 'stretched' to create a new bpf

        bpf.xs *= rx
        """
        if rx == 0:
            raise ValueError("the stretch factor cannot be 0")
        self.xs *= rx
        self._recalculate_bounds()
        return self

    def shift(self, double dx):
        """
        shift the bpf along the x-coords. it modifies the bpf itself (INPLACE)
        use shifted to create a new bpf

        bpf.xs += dx
        """
        self.xs += dx
        self._recalculate_bounds()
        return self

    cdef void _recalculate_bounds(self):
        cdef DTYPE_t* data
        cdef int nx
        nx = PyArray_DIM(self.xs, 0)
        self.xs_data = <DTYPE_t*>self.xs.data
        self._x0 = self.xs_data[0]
        self._x1 = self.xs_data[nx - 1]
        self._invalidate_cache()

    def points(_BpfBase self):
        """
        returns (xs, ys)

        >>> b = Linear.fromseq(0, 0, 1, 100, 2, 50)
        >>> b.points()
        ([0, 1, 2], [0, 100, 50])

        if you want to iterate over each point (xn, yn),
        use iter(thisbpf), like

        >>> for x, y in b:
        >>>    print x, y

        (0, 0)
        (1, 100)
        (2, 50)

        """
        return self.xs, self.ys

    def insertpoint(self, double x, double y):
        """
        Return a copy of this bpf with the point(x, y) inserted
        """
        cdef int index = _searchsorted(self.xs, x)
        new_xs = numpy.insert(self.xs, index, x)
        new_ys = numpy.insert(self.ys, index, y)
        return self.__class__(new_xs, new_ys)

    def removepoint(self, double x):
        """
        Return a copy of this bpf with point at x removed

        * Raises ValueError if x is not in this bpf
        * To remove elements by index, do

        xs, ys = mybpf.points()
        xs = numpy.delete(xs, indices)
        ys = numpy.delete(ys, indices)
        mybpf = mybpf.__class__(xs, ys)
        """
        cdef int index = _csearchsorted_left(self.xs_data, self.xs.size, x)
        if self.xs_data[index] != x:
            raise ValueError("%f is not in points" % x)
        newxs = numpy.delete(self.xs, index)
        newys = numpy.delete(self.ys, index)
        return self.__class__(newxs, newys)

    def segments(self):
        """
        returns an iterator where each item is
        (float x, float y, str interpolation_type, float exponent)

        exponent is only of value if the interpolation type makes use of it
        """
        cdef size_t i
        cdef size_t num_segments
        num_segments = len(self.xs) - 1
        interpoltype = self.__class__.__name__.lower()
        for i in range(num_segments):
            yield (float(self.xs[i]), float(self.ys[i]), interpoltype, self.interpol_func.exp)
        yield (float(self.xs[num_segments]), float(self.ys[num_segments]), '', 0)

    property exp:
        def __get__(self):
            return self.interpol_func.exp

cdef class Smooth(_BpfBase):
    def __init__(self, xs, ys, int numiter=1):
        if numiter == 1:
            self.interpol_func = InterpolFunc_smooth
        else:
            self.interpol_func = InterpolFunc_new(intrp_smooth, 1, "smooth", 1)
            self.interpol_func.numiter = numiter
        _BpfBase.__init__(self, xs, ys)
    
cdef class Linear(_BpfBase):
    def __init__(self, xs, ys):
        self.interpol_func = InterpolFunc_linear
        _BpfBase.__init__(self, xs, ys)

    def _get_points_for_rendering(self, int n= -1):
        return self.xs, self.ys

    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        cdef double pre, mid, post
        cdef size_t index0, index1
        if x0 <= self._x0:
            pre = self.__ccall__(x0) * (self._x0 - x0)
            index0 = 0
        else:
            index0 = _csearchsorted(self.xs_data, self.xs_size, x0)
            # trapezium = (a+b)/2 * h (where h is the distance between a and b)
            pre = (self.ys_data[index0] + self.__ccall__(x0)) * 0.5 * (self.xs_data[index0] - x0)
        if x1 >= self._x1:
            post = self.__ccall__(x1) * (x1 - self._x1)
            index1 = self.xs_size - 1
        else:
            index1 = _csearchsorted_left(self.xs_data, self.xs_size, x1) - 1
            post = (self.ys_data[index1] + self.__ccall__(x1)) * 0.5 * (x1-self.xs_data[index1])
        mid = 0
        for i in range(index0, index1):
            #mid += (self.ys_data[index0] + self.ys_data[index1])*0.5*(self.xs_data[index1] - self.xs_data[index0])
            mid += (self.ys_data[i] + self.ys_data[i+1]) * 0.5 * (self.xs_data[i+1] - self.xs_data[i])
        return pre + mid + post

    cpdef Linear sliced(self, double x0, double x1):
        """
        cut this bpf at the given points, inserting points at 
        those coordinates, to limit this bpf to the range
        x0:x1.

        NB: this is different from crop. A real Linear bpf is returned.
        """
        import util
        return util.linearslice(self, x0, x1)

    def inverted(self):
        import util
        return util.linear_inverted(self)
        
        

cdef class Halfcos(_BpfBase):
    def __init__(self, xs, ys, double exp=1.0, int numiter=1):
        if exp == 1.0 and numiter == 1:
            self.interpol_func = InterpolFunc_halfcos
        elif exp == 1:
            self.interpol_func = InterpolFunc_new(intrp_halfcos, 1, 'halfcos', 1)
        else:
            self.interpol_func = InterpolFunc_new(intrp_halfcosexp, exp, 'halfcosexp', 1)
        self.interpol_func.numiter = numiter
        super().__init__(xs, ys)
    
    def __getstate__(self):
        return self.xs, self.ys, self.interpol_func.exp, self.interpol_func.numiter

    def __repr__(self):
        exp = self.interpol_func.exp
        return "%s[%s:%s] exp=%s" % (self.__class__.__name__, str(self._x0), str(self._x1), str(exp))
        
HalfcosExp = Halfcos
        
cdef class Halfcosm(Halfcos):
    def __init__(self, xs, ys, double exp=1.0, int numiter=1):
        self.interpol_func = InterpolFunc_new(intrp_halfcosexpm, exp, 'halfcosexpm', 1)
        self.interpol_func.numiter = numiter
        _BpfBase.__init__(self, xs, ys)

cdef class BlendShape(_BpfBase):
   def __init__(self, xs, ys, shape0, shape1, double mix):
       _BpfBase.__init__(self, xs, ys)
       self.interpol_func = InterpolFunc_new_blend_from_descr(shape0, shape1, mix)
       if self.interpol_func is NULL:
           raise ValueError("interpolation shape not understood")

cdef class Expon(_BpfBase):
    def __init__(self, xs, ys, double exp, int numiter=1):
        _BpfBase.__init__(self, xs, ys)
        self.interpol_func = InterpolFunc_new(intrp_expon, exp, 'expon', 1)  # must be freed
        self.interpol_func.numiter = numiter
    
    def __getstate__(self): return self.xs, self.ys, self.interpol_func.exp
    def __setstate__(self, state):
        self.xs, self.ys, exp = state
        self.interpol_func.exp = exp

    def __repr__(self):
        return "%s[%s:%s] exp=%s" % (self.__class__.__name__, str(self._x0), str(self._x1), str(self.interpol_func.exp))

cdef class Exponm(Expon):
    def __init__(self, xs, ys, double exp, int numiter=1):
        _BpfBase.__init__(self, xs, ys)
        self.interpol_func = InterpolFunc_new(intrp_exponm, exp, 'exponm', 1)  # must be freed
        self.interpol_func.numiter = numiter

cdef class Fib(_BpfBase):
    def __init__(self, xs, ys):
        _BpfBase.__init__(self, xs, ys)
        self.interpol_func = InterpolFunc_fib
    def __getstate__(self): return self.xs, self.ys
    def __repr__(self): return f"{self.__class__.__name__}[{self._x0}:{self._x1}]"

cdef class NoInterpol(_BpfBase):
    def __init__(self, xs, ys):
        _BpfBase.__init__(self, xs, ys)
        self.interpol_func = InterpolFunc_nointerpol
        
cdef class Nearest(_BpfBase):
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.interpol_func = InterpolFunc_nearest
    def __getstate__(self): return self.xs, self.ys
    
ctypedef struct SplineS:
    int use_low_slope
    int use_high_slope
    double low_slope
    double high_slope
    DTYPE_t *xs
    DTYPE_t *ys
    DTYPE_t *ys2
    int length
    npy_intp last_index

cdef SplineS* SplineS_new(xs, ys, Py_ssize_t xs_size, double low_slope=0, double high_slope=0, int use_low_slope=0, int use_high_slope=0):
    cdef SplineS *s = <SplineS *>malloc(sizeof(SplineS))
    cdef int length = xs_size
    cdef int i
    cdef DTYPE_t * _xs = <DTYPE_t *>malloc(sizeof(DTYPE_t) * length)
    cdef DTYPE_t * _ys = <DTYPE_t *>malloc(sizeof(DTYPE_t) * length)
    for i in range(length):
        _xs[i] = xs[i]
        _ys[i] = ys[i]
    s.xs = _xs
    s.ys = _ys
    s.low_slope = low_slope
    s.high_slope = high_slope
    s.use_low_slope = use_low_slope
    s.use_high_slope = use_high_slope
    s.last_index = length - 1
    s.length = length
    SplineS_calc_ypp(s)
    return s

cdef void SplineS_destroy(SplineS *self):
    free(self.xs)
    free(self.ys)
    free(self.ys2)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void SplineS_calc_ypp(SplineS *self) nogil:
    cdef DTYPE_t *x_vals = self.xs
    cdef DTYPE_t *y_vals = self.ys
    cdef int n = self.length
    cdef DTYPE_t *y2_vals = <DTYPE_t *>malloc(sizeof(DTYPE_t) * n)
    cdef DTYPE_t *u =       <DTYPE_t *>malloc(sizeof(DTYPE_t) * (n - 1))  # this is a temporary array
    cdef double x1_minus_x0, denom, sig, p, qn, un
    cdef int i, k
    if self.use_low_slope == 1:
        x1_minus_x0 = x_vals[1] - x_vals[0]
        # u[0] = (3.0/(x_vals[1]-x_vals[0])) * ((y_vals[1]-y_vals[0]) / (x_vals[1]-x_vals[0])-self.low_slope)
        x1_minus_x0 += 1e-12
        u[0] =(3.0 / x1_minus_x0) * ( (y_vals[1]-y_vals[0]) / x1_minus_x0 - self.low_slope)
        y2_vals[0] = -0.5
    else:
        u[0] = 0.0
        y2_vals[0] = 0.0   # natural spline
    for i in range(1, n-1):
        denom = x_vals[i+1] - x_vals[i-1] + 1e-12
        sig = (x_vals[i]-x_vals[i-1]) / denom
        p = sig*y2_vals[i-1]+2.0
        y2_vals[i] = (sig-1.0)/p
        if x_vals[i+1] == x_vals[i]:
            x_vals[i+1] += 1e-12
        u[i] = (y_vals[i+1]-y_vals[i]) / (x_vals[i+1]-x_vals[i]) - (y_vals[i]-y_vals[i-1]) / (x_vals[i]-x_vals[i-1])
        u[i] = (6.0 * u[i] / denom - sig * u[i-1]) / p
    if self.use_high_slope == 1:
        qn = 0.5
        un = (3.0/(x_vals[n-1]-x_vals[n-2])) * (self.high_slope - (y_vals[n-1]-y_vals[n-2]) / (x_vals[n-1]-x_vals[n-2]))
    else:
        qn = 0.0
        un = 0.0    # natural spline
    y2_vals[n-1] = (un-qn*u[n-2])/(qn*y2_vals[n-1]+1.0)
    for k in range(n-2, -1, -1):
        y2_vals[k] = y2_vals[k]*y2_vals[k+1]+u[k]
    free(u)
    self.ys2 = y2_vals

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double SplineS_at(SplineS *self, double x) nogil:
    # if out of range, return endpoint
    cdef double a, b, out
    if x <= self.xs[0]:
        return self.ys[0]
    if x >= self.xs[self.last_index]:
        return self.ys[self.last_index]
    cdef int pos = _csearchsorted(self.xs, self.length, x)
    cdef double h = self.xs[pos] - self.xs[pos-1]
    if h == 0.0:
        out = INFINITY
    else:
        a = (self.xs[pos] - x) / h
        b = (x - self.xs[pos-1]) / h
        out = (a* self.ys[pos-1] + b*self.ys[pos] + \
              ((a*a*a - a)*self.ys2[pos-1] + \
              (b*b*b - b)*self.ys2[pos]) * h*h/6.0)
    return out

cdef class Sampled(_BpfInterface):
    cdef readonly ndarray ys
    cdef double y0, y1
    cdef double grid_dx, grid_x0, grid_x1
    cdef int samples_size
    cdef int nointerpol
    cdef InterpolFunc* interpolfunc
    cdef DTYPE_t* data
    cdef ndarray _cached_xs

    def __init__(self, samples, double dx, double x0=0, interpolation='linear'):
        """
        This class wraps a seq of values defined across a regular grid
        (starting at x0, defined by x0 + i * dx)

        When evaluated, values between the samples are interpolated with
        the given interpolation

        If 'samples' follow the ISampled interface, then it is not needed
        to pass dx and
        """
        self.ys = numpy.ascontiguousarray(samples, dtype=DTYPE)
        self.data = <DTYPE_t *>self.ys.data
        l = PyArray_DIM(self.ys, 0)
        self.samples_size = l
        self.grid_x0 = x0
        self.grid_dx = dx
        self.grid_x1 = x0 + dx * (l - 1)
        self._set_bounds(x0, self.grid_x1)
        self._cached_xs = None
        if interpolation == 'nointerpol':
            self.nointerpol = 1
            self.interpolfunc = NULL
        elif interpolation == 'linear':
            self.nointerpol = 0
            self.interpolfunc = InterpolFunc_linear
        else:
            self.nointerpol = 0
            self.interpolfunc = InterpolFunc_new_from_descriptor(interpolation)
            if self.interpolfunc is NULL:
                raise ValueError("interpolation type not understood")
        self.y0 = self.data[0]
        self.y1 = self.data[l - 1]

    property samplerate:
        @cython.cdivision(True)
        def __get__(self): return 1.0 / self.grid_dx
    
    property xs:
        def __get__(self):
            if self._cached_xs is not None:
                return self._cached_xs
            self._cached_xs = numpy.linspace(self.grid_x0, self.grid_x1, self.samples_size, endpoint=False)
            return self._cached_xs
    
    property interpolation:
        def __get__(self):
            if self.interpolfunc is not NULL:
                return InterpolFunc_get_descriptor(self.interpolfunc)
            return 'nointerpol'
        def __set__(self, interpolation):
            self.set_interpolation(interpolation)

    property dx:
        def __get__(self): return self.grid_dx

    cpdef Sampled set_interpolation(self, interpolation):
        """
        Sets the interpolation of this Sampled bpf, inplace

        NB: returns self, so you can do 
            sampled = bpf[x0:x1:dx].set_interpolation('expon(2)')
        """
        InterpolFunc_free(self.interpolfunc)
        if interpolation == 'nointerpol':
            self.nointerpol = 1
            self.interpolfunc = NULL
        else:
            self.nointerpol = 0
            self.interpolfunc = InterpolFunc_new_from_descriptor(interpolation)
        return self

    def __dealloc__(self):
        InterpolFunc_free(self.interpolfunc)

    def __getstate__(self):
        return (self.ys, self.grid_dx, self.grid_x0, self.interpolation)

    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        cdef int index0
        cdef double y0, y1, x0
        cdef double out
        if x <= self.grid_x0:
            out = self.y0
        elif x >= self.grid_x1:
            out = self.y1
        else:
            index0 = int((x - self.grid_x0) / self.grid_dx)
            if self.nointerpol == 1:
                out = self.data[index0]
            else:
                y0 = self.data[index0]
                y1 = self.data[index0 + 1]
                x0 = self.grid_x0 + index0 * self.grid_dx
                out = InterpolFunc_call(self.interpolfunc, x, x0, y0, x0+self.grid_dx, y1)
        return out

    @cython.cdivision(True)
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef DTYPE_t *data
        cdef DTYPE_t *selfdata
        cdef int i, index0, nointerpol
        cdef double x, y
        cdef double grid_x0, grid_x1, self_y0, self_y1, grid_dx, interp_x0, interp_y0, interp_y1
        cdef double dx = (x1 - x0) / (n - 1) # we account for the edge (x1 IS INCLUDED)
        cdef double interpolfunc_exp
        cdef t_func interpolfunc
        if out is None:
            out = EMPTY1D(n)
            data = <DTYPE_t *>out.data
        elif PyArray_ISCONTIGUOUS(out):
            data = <DTYPE_t *>out.data
        else:
            for i in range(n):
                out[i] = self.__ccall__(x0 + dx*i)
            return out
        grid_x0 = self.grid_x0
        grid_x1 = self.grid_x1
        self_y0 = self.y0
        self_y1 = self.y1
        grid_dx = self.grid_dx
        selfdata = self.data
        nointerpol = self.nointerpol
        interpolfunc = self.interpolfunc.func
        interpolfunc_exp = self.interpolfunc.exp
        i = 0
        with nogil:
            while i < n:
                x = x0 + dx * i
                if x > grid_x0:
                    break
                data[i] = self_y0
                i += 1
            while i < n:
                x = x0 + dx * i
                if x >= grid_x1:
                    break
                index0 = int((x - grid_x0) / grid_dx)
                if nointerpol == 0:
                    y = selfdata[index0]
                else:
                    interp_x0 = grid_x0 + index0 * grid_dx
                    y = InterpolFunc_call(self.interpolfunc, x, interp_x0, selfdata[index0],
                                          interp_x0 + grid_dx, selfdata[index0+1])
                data[i] = y
                i += 1
            while i < n:
                data[i] = self_y1
                i += 1
        return out

    @classmethod
    def fromxy(cls, *args, **kws): raise NotImplementedError
    
    @classmethod
    def fromseq(cls, *args, **kws): raise NotImplementedError
    
    def _get_points_for_rendering(self, int n= -1): return self.xs, self.ys
    
    def segments(self):
        """
        returns an iterator where each item is
        (float x, float y, str interpolation_type, float exponent)

        exponent is only of value if the interpolation type makes use of it
        """
        cdef int i
        cdef double x0 = self.grid_x0
        cdef double dx = self.grid_dx
        descr = self.interpolation
        exp = self.interpolfunc.exp if self.interpolfunc is not NULL else 0.0
        for i in range(self.samples_size):
            yield (x0 + i*dx, self.data[i], descr, exp)

    cpdef double integrate(self):
        """
        return the result of the integration of this bpf. If any of the bounds is inf,
        the result is also inf.

        Tip: to determine the limits of the integration, first crop the bpf via a slice
        Example:

        b[:10].integrate()  -> integrate this bpf from its lower bound to 10 (inclusive)
        """
        return _integr_trapz(self.data, self.samples_size, self.grid_dx) 

    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        """
        The same as integrate() but between the (included) bounds x0-x1
        It is effectively the same as bpf[x0:x1].integrate(), but more efficient
        
        NB : N has no effect. It is put here to comply with the signature of the function. 
        """
        dx = self.grid_dx
        return _integr_trapz_between_exact(self.data, self.samples_size, dx, self.grid_x0, x0, x1)

    cpdef _BpfInterface derivative(self):
        """
        Return a curve which represents the derivative of this curve

        It implements Newtons difference quotiont, so that

        derivative(x) = bpf(x + h) - bpf(x)
                        -------------------
                                  h
        """
        return _BpfDeriv(self, self.grid_dx*0.99)

    def inverted(self):
        if self.interpolation == 'linear':
            import util
            return util.linear_inverted(self)
        return super().inverted()

    
cdef class Spline(_BpfInterface):
    cdef SplineS* _spline

    def __init__(self, xs, ys):
        cdef ndarray [DTYPE_t, ndim=1] _xs = numpy.asarray(xs, dtype=DTYPE)
        cdef ndarray [DTYPE_t, ndim=1] _ys = numpy.asarray(ys, dtype=DTYPE)
        cdef int N = len(xs)
        self._set_bounds(_xs[0], _xs[N - 1])
        self._integration_mode = 1  # use quad
        if _array_issorted(_xs) < 1:
            raise ValueError(f"Points along the x coord should be sorted without duplicates. \n{xs}")
        self._spline = SplineS_new(_xs, _ys, N)
    
    def __dealloc__(self):
        SplineS_destroy(self._spline)
    
    @cython.boundscheck(False)
    cdef double __ccall__(self, double x) nogil:
        return SplineS_at(self._spline, x)
    
    cdef inline tuple _points(self):
        cdef int i
        xs = [self._spline.xs[i] for i in range(self._spline.length)]
        ys = [self._spline.ys[i] for i in range(self._spline.length)]
        return (xs, ys)
    
    def __getstate__(self):
        return self._points()
    
    def points(self):
        """
        returns (xs, ys)
        """
        return self._points()

    def segments(self):
        """
        returns an iterator where each item is
        (float x, float y, str interpolation_type, float exponent)

        exponent is only of value if the interpolation type makes use of it
        """
        cdef size_t i
        cdef size_t num_segments
        exp = 0
        num_segments = self._spline.length - 1
        interpoltype = self.__class__.__name__.lower()
        for i in range(num_segments):
            yield (float(self._spline.xs[i]), float(self._spline.ys[i]), interpoltype, 0)
        yield (float(self._spline.xs[num_segments]), float(self._spline.ys[num_segments]), '', 0)

cdef class USpline(_BpfInterface):
    """
    BPF with univariate spline interpolation. This is implemented by
    wrapping a UnivariateSpline from scipy.
    """
    cdef object spline
    cdef object spline__call__
    cdef ndarray xs, ys
    
    def __init__(self, xs, ys):
        try:
            from scipy.interpolate import UnivariateSpline
        except ImportError:
            raise ImportError("could not import scipy. USpline is a wrapper of UnivariateSpline"
                              " and cant be used without scipy")
        self.spline = UnivariateSpline(xs, ys)
        self.spline__call__ = self.spline.__call__
        self.xs = numpy.asarray(xs)
        self.ys = numpy.asarray(ys)
        self._set_bounds(xs[0], xs[len(xs) - 1])
        self._integration_mode = 1  # use quad
    
    def __call__(self, x):
        return self.spline(x)
    
    cdef double __ccall__(self, double x) nogil:
        with gil:
            return self.spline__call__(x)
    
    cpdef ndarray map(self, xs, ndarray out=None):
        if out is not None:
            out[...] = self.spline__call__(xs)
            return out
        else:
            return self.spline__call__(xs)
    
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        xs = numpy.linspace(x0, x1, n)
        return self.map(xs, out)
    
    property _spline:
        def __get__(self):
            return self.spline
    
    def segments(self):
        """
        returns an iterator where each item is
        (float x, float y, str interpolation_type, float exponent)

        exponent is only of value if the interpolation type makes use of it
        """
        cdef size_t i
        cdef size_t num_segments
        exp = 0
        num_segments = len(self.xs) - 1
        interpoltype = self.__class__.__name__.lower()
        for i in range(num_segments):
            yield (float(self.xs[i]), float(self.ys[i]), interpoltype, 0)
        yield (float(self.xs[num_segments]), float(self.ys[num_segments]), '', 0)


cdef class _BpfConcat2(_BpfInterface):
    cdef _BpfInterface a
    cdef _BpfInterface b
    cdef double splitpoint
    cdef double __ccall__(self, double x) nogil:
        if x < self.splitpoint:
            return self.a.__ccall__(x)
        return self.b.__ccall__(x)
    
    def __getstate__(self):
        return self.a, self.b, self.splitpoint
    
    def __setstate__(self, state):
        self.a, self.b, self.splitpoint = state
        cdef double x0 = self.a._x0 if self.a._x0 < self.b._x0 else self.b._x0
        cdef double x1 = self.b._x1 if self.b._x1 > self.a._x1 else self.a._x1
        self._set_bounds(x0, x1)
    
cdef class Slope(_BpfInterface):
    cdef public double slope
    cdef public double offset
    
    def __init__(self, double slope, double offset=0, tuple bounds=None):
        self.slope = slope
        self.offset = offset
        if bounds is not None:
            self._set_bounds(bounds[0], bounds[1])
        else:
            self._set_bounds(INFNEG, INF)
        
    cdef double __ccall__(self, double x) nogil:
        return self.offset + x*self.slope

    def __add__(a, b):
        if isnumber(a):
            # we are b
            return Slope(b.slope, b.offset + a, b.bounds())
        elif isnumber(b):
            return Slope(a.slope, a.offset + b, a.bounds())
        else:
            return _create_lambda(a, b, _BpfLambdaAdd, _BpfLambdaAddConst)

    def __sub__(a, b):
        if isnumber(a):
            return b + (-a)
        elif isnumber(b):
            return a + (-b)
        else:
            return _create_rlambda(a, b, _BpfLambdaSub, _BpfLambdaSubConst, _BpfLambdaRSub, _BpfLambdaRSubConst)

    def __mul__(a, b):
        if isnumber(a):
            return Slope(b.slope*a, b.offset*a, b.bounds())
        elif isnumber(b):
            return Slope(a.slope*b, a.offset*b, a.bounds())
        return _BpfInterface.__mul__(a, b)

    def __getstate__(self):
        return self.slope, self.offset, self.bounds()

    
cdef class _BpfCompose(_BpfInterface):
    cdef _BpfInterface a
    cdef _BpfInterface b

    cdef double __ccall__(self, double x) nogil:
        x = self.a.__ccall__(x)
        return self.b.__ccall__(x)

    def __getstate__(self):
        return self.a, self.b

    def __setstate__(self, state):
        self.a, self.b = state
        self._set_bounds(self.a._x0, self.a._x1)

    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A
        if out is not None:
            self.a.mapn_between(n, x0, x1, out)
            self.b.map(out, out)
            return out
        else:
            A = self.a.mapn_between(n, x0, x1)
            self.b.map(A, A)
            return A
        
cdef _BpfCompose _BpfCompose_new(_BpfInterface a, _BpfInterface b):
    cdef _BpfCompose self = _BpfCompose()
    self.a = a
    self.b = b
    self._set_bounds(a._x0, a._x1)
    return self

cdef _BpfConcat2 _BpfConcat2_new(_BpfInterface bpf_a, _BpfInterface bpf_b, double splitpoint):
    cdef _BpfConcat2 self = _BpfConcat2()
    cdef double x0 = bpf_a._x0 if bpf_a._x0 < bpf_b._x0 else bpf_b._x0
    cdef double x1 = bpf_b._x1 if bpf_b._x1 > bpf_a._x1 else bpf_a._x1
    self._set_bounds(x0, x1)
    self.a = bpf_a
    self.b = bpf_b
    self.splitpoint = splitpoint
    return self

cdef class _BpfConcat(_BpfInterface):
    cdef list bpfs
    cdef double *xs
    cdef Py_ssize_t size
    cdef double last_x0, last_x1
    cdef _BpfInterface last_bpf, bpf0, bpf1

    def __cinit__(self, xs, bpfs):
        self.size = len(xs)
        self.xs = <double *>malloc(sizeof(double) * self.size)

    def __init__(self, xs, bpfs):
        cdef int i
        cdef _BpfInterface bpf
        self.bpfs = list(bpfs)
        i = 0
        for x in xs:
            self.xs[i] = x
            i += 1
        cdef double x0 = INF
        cdef double x1 = INFNEG
        for i in range(self.size):
            bpf = self.bpfs[i]
            if bpf._x0 < x0:
                x0 = bpf._x0
            if bpf._x1 > x1:
                x1 = bpf._x1
        self._set_bounds(x0, x1)
        self.bpf0 = self.bpfs[0]
        self.bpf1 = self.bpfs[self.size - 1]
        self.last_bpf = self.bpf0
        self.last_x0 = self.xs[0]
        self.last_x1 = self.xs[1]

    def __dealloc__(self):
        free(self.xs)

    cdef double __ccall__(self, double x) nogil:
        cdef int index
        if x <= self._x0:
            return self.bpf0.__ccall__(x)
        elif x >= self._x1:
            return self.bpf1.__ccall__(x)
        elif x >= self.last_x0 and x < self.last_x1:
            return self.last_bpf.__ccall__(x)
        else:
            index = _csearchsorted_left(self.xs, self.size, x) - 1
            with gil:
                self.last_bpf = self.bpfs[index]
            self.last_x0 = self.xs[index]
            self.last_x1 = self.xs[index+1]
            return self.last_bpf.__ccall__(x)

    cpdef _BpfInterface concat(self, _BpfInterface other, double fadetime=0, fadeshape='expon(3)'):
        cdef _BpfInterface other2 = other.fit_between(self._x1, self._x1 + (other._x1 - other._x0))
        cdef int i
        cdef list xs
        cdef list bpfs
        if fadetime == 0:
            xs = [self.xs[i] for i in range(self.size)]
            bpfs = self.bpfs[:]
            bpfs.append(other2)
            return _BpfConcat(xs, bpfs)
        raise NotImplementedError("fade is not implemented")

    def __getstate__(self):
        cdef list xs
        cdef int i
        xs = [self.xs[i] for i in range(self.size)]
        return xs, self.bpfs

cdef class _BpfBlend(_BpfInterface):
    cdef _BpfInterface a, b
    cdef _BpfInterface which

    def __init__(self, a, b, which):
        self.a = a
        self.b = b
        self.which = which
        cdef double x0 = min(a._x0, b._x0)
        cdef double x1 = max(a._x1, b._x1)
        self._set_bounds(x0, x1)

    cdef double __ccall__(self, double x) nogil:
        cdef double ya, yb, mix
        mix = self.which.__ccall__(x)
        ya = self.a.__ccall__(x) * (1 - mix)
        yb = self.b.__ccall__(x) * mix
        return ya + yb

    def __getstate__(self):
        return self.a, self.b, self.which
        
cdef class _BpfBlendConst(_BpfInterface):
    cdef _BpfInterface a, b
    cdef double which

    def __init__(_BpfBlendConst self, _BpfInterface a, _BpfInterface b, double which):
        self.a = a
        self.b = b
        self.which = which
        cdef double x0 = min(a._x0, b._x0)
        cdef double x1 = max(a._x1, b._x1)
        self._set_bounds(x0, x1)

    cdef double __ccall__(self, double x) nogil:
        return self.a.__ccall__(x) * (1 - self.which) + self.b.__ccall__(x) * self.which

    def __getstate__(self): return self.a, self.b, self.which

        
cdef class Multi(_BpfInterface):
    cdef DTYPE_t* xs
    cdef DTYPE_t* ys
    cdef InterpolFunc** interpolations
    cdef int size
    cdef DTYPE_t y0, y1, last_x0, last_x1, last_y0, last_y1
    cdef InterpolFunc* last_interpol

    def __cinit__(self, xs, ys, interpolations):
        self.size = len(xs)
        self.interpolations = <InterpolFunc **>malloc(sizeof(InterpolFunc*) * (self.size - 1))

    def __init__(self, xs, ys, interpolations):
        """
        xs: the sequence of x points
        ys: the sequence of y points
        interpolations: the interpolation used between these points

        NB: len(interpolations) = len(xs) - 1

        The interpelation is indicated via a string of the type:

        'linear'      -> linear
        'expon(2)'    -> exponential interpolation, exp=2
        'halfcos'
        'halfcos(0.5) -> half-cos exponential interpolation with exp=0.5
        'nointerpol'  -> no interpolation (rect)
        """
        cdef int i
        cdef list interpolations_list
        self.xs = _seq_to_doubles(xs)
        self.ys = _seq_to_doubles(ys)
        assert len(interpolations) == self.size - 1
        i = 0
        for interpolation in interpolations:
            self.interpolations[i] = InterpolFunc_new_from_descriptor(interpolation)
            i += 1
        self._set_bounds(self.xs[0], self.xs[self.size - 1])
        self.y0 = self.ys[0]
        self.y1 = self.ys[self.size - 1]
        self.last_x0 = self.xs[0]
        self.last_x1 = self.xs[1]
        self.last_y0 = self.ys[0]
        self.last_y1 = self.ys[1]
        self.last_interpol = self.interpolations[0]

    def __dealloc__(self):
        free(self.xs)
        free(self.ys)
        cdef int i
        for i in range(self.size - 1):
            InterpolFunc_free(self.interpolations[i])
        free(self.interpolations)

    def __getstate__(self):
        cdef int i
        xs = [self.xs[i] for i in range(self.size)]
        ys = [self.ys[i] for i in range(self.size)]
        interpolations = [InterpolFunc_get_descriptor(self.interpolations[i]) for i in range(self.size - 1)]
        return xs, ys, interpolations

    cdef double __ccall__(self, double x) nogil:
        cdef double res, x0, x1, y0, y1
        cdef int index1, index0
        if x <= self._x0:
            res = self.y0
        elif x >= self._x1:
            res = self.y1
        else:
            if self.last_x0 <= x < self.last_x1:
                # res = self.last_interpol.func(x, self.last_x0, self.last_y0, self.last_x1, self.last_y1, self.last_interpol.exp)
                res = InterpolFunc_call(self.last_interpol, x, self.last_x0, self.last_y0, self.last_x1, self.last_y1)
            else:
                index1 = _csearchsorted(self.xs, self.size, x)
                index0 = index1 - 1
                self.last_x0 = x0 = self.xs[index0]
                self.last_x1 = x1 = self.xs[index1]
                self.last_y0 = y0 = self.ys[index0]
                self.last_y1 = y1 = self.ys[index1]
                self.last_interpol = self.interpolations[index0]
                # res = self.last_interpol.func(x, x0, y0, x1, y1, self.last_interpol.exp)
                res = InterpolFunc_call(self.last_interpol, x, x0, y0, x1, y1)
        return res

    def segments(self):
        """
        returns an iterator where each item is
        (float x, float y, str interpolation_type, float exponent)

        exponent is only of value if the interpolation type makes use of it
        """
        cdef int i
        cdef InterpolFunc* func
        for i in range(self.size - 1):
            func = self.interpolations[i]
            yield self.xs[i], self.ys[i], func.name, func.exp
        yield (self.xs[self.size-1], self.ys[self.size-1], '', 0)

ctypedef double(*dfunc)(double) nogil

def _FunctionWrap(f, bounds=(INFNEG, INF)):
    return _FunctionWrap_Object(f, bounds)


cdef class _FunctionWrap_Object(_BpfInterface):
    cdef object f

    def __init__(self, f, bounds=(INFNEG, INF)):
        self._set_bounds(bounds[0], bounds[1])
        self.f = f.__call__

    cdef double __ccall__(self, double x) nogil:
        with gil:
            return self.f(x)

    def __getstate__(self):
        return (self.f, (self._x0, self._x1))

    cpdef _BpfInterface _slice(self, double x0, double x1):
        return _FunctionWrap_Object_OutboundConst_new(self, x0, x1)

    cpdef ndarray map(self, xs, ndarray out=None):
        """
        the same as map(self, xs) but somewhat faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.
        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))

        if out is given, the result if put into it. it must have the same shape as xs
        """
        cdef ndarray [DTYPE_t, ndim=1] _xs
        cdef ndarray [DTYPE_t, ndim=1] result
        cdef DTYPE_t *data1
        cdef DTYPE_t *data0
        cdef list xs_as_list
        cdef tuple xs_as_tuple
        cdef int i, nx
        cdef double x0,x1, dx
        cdef object f = self.f
        if isinstance(xs, (int, long)):
            return self.mapn_between(xs, self._x0, self._x1, out)
        else:
            if out is None:
                nx = len(xs)
                result = EMPTY1D(nx)
            else:
                result = <ndarray>out
                nx = PyArray_DIM(<ndarray>result, 0)
            if isinstance(xs, ndarray):
                if PyArray_ISCONTIGUOUS(<ndarray>xs):
                    data1 = <DTYPE_t *>((<ndarray>xs).data)
                    if PyArray_ISCONTIGUOUS(result):
                        data0 = <DTYPE_t *>result.data
                        for i in range(nx):
                            data0[i] = f(data1[i])
                    else:
                        nx = PyArray_DIM(<ndarray>xs, 0)
                        for i in range(nx):
                            result[i] = f(data1[i])
                else:
                    _xs = <ndarray>xs
                    for i in range(PyArray_DIM(xs, 0)):
                        result[i] = f(_xs[i])
            else:
                if isinstance(xs, list):
                    for i in range(nx):
                        result[i] = f((<list>xs)[i])
                elif isinstance(xs, tuple):
                    xs_as_tuple = xs
                    for i in range(len(xs_as_tuple)):
                        result[i] = f(xs_as_tuple[i])
                else:
                    for i in range(nx):
                        result[i] = f(xs[i])
        return result
        
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        xs = numpy.linspace(x0, x1, n)
        return self.map(xs, out=out)

cdef class _FunctionWrap_Object_OutboundConst(_FunctionWrap_Object):
    cdef double y0, y1
    cdef double __ccall__(self, double x) nogil:
        if x < self._x0:
            return self.y0
        elif x > self._x1:
            return self.y1
        else:
            with gil:
                return self.f(x)

cdef _FunctionWrap_Object_OutboundConst _FunctionWrap_Object_OutboundConst_new(_FunctionWrap_Object bpf, double x0, double x1):
    cdef _FunctionWrap_Object_OutboundConst out = _FunctionWrap_Object_OutboundConst(bpf.f, (x0, x1))
    out.y0 = out.f(x0)
    out.y1 = out.f(x1)
    return out

cdef class Const(_BpfInterface):
    cdef double value
    def __init__(self, double value):
        self._set_bounds(INFNEG, INF)
        self.value = value
    def __call__(self, x): return self.value
    cdef double __ccall__(self, double x) nogil:
        return self.value
    def __getstate__(self):
        return (self.value,)
    def _get_xs_for_rendering(self, int n):
        return CONST_XS_FOR_RENDERING
    def __getitem__(self, slice):
        cdef double x0, x1
        cdef _BpfInterface out
        x0 = self._x0
        x1 = self._x1
        if hasattr(slice, 'start'):
            if slice.start is not None: 
                x0 = slice.start
            if slice.stop is not None:  
                x1 = slice.stop
            out = Const(self.value)
            out._set_bounds(x0, x1)
            return out
        else:
            raise ValueError("BPFs accept only slices, not single items.")
    
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        if out is not None:
            out[...] = self.value
            return out
        return numpy.ones([n], dtype=float) * self.value
        
cdef _create_lambda_unordered(a, b, class_bin, class_const):
    if isinstance(a, _BpfInterface):        
        if isinstance(b, _BpfInterface):
            out = class_bin(a, b, _get_bounds(a, b))
        elif callable(b):
            out = class_bin(a, _FunctionWrap(b), a.bounds())
        else:
            out = class_const(a, b, a.bounds())
        return out
    elif isinstance(b, _BpfInterface):
        if callable(a):
            out = class_bin(b, _FunctionWrap(a), a.bounds())
        else:
            out = class_const(b, a, b.bounds())
        return out

cdef _create_lambda(_BpfInterface a, object b, class_bin, class_const):
    if isinstance(b, _BpfInterface):
        out = class_bin(a, b, _get_bounds(a, b))
    elif callable(b):
        out = class_bin(a, _FunctionWrap(b), a.bounds())
    else:
        out = class_const(a, b, a.bounds())
    return out

cdef _create_rlambda(object a, object b, class_bin, class_const, class_rbin=None, class_rconst=None):
    if isinstance(a, _BpfInterface):
        if isinstance(b, _BpfInterface):
            out = class_bin(a, b, _get_bounds(a, b))
        elif callable(b):
            out = class_bin(a, _FunctionWrap(b), a.bounds())
        else:
            out = class_const(a, b, a.bounds())
        return out
    else:
        if callable(a):
            out = class_rbin(_FunctionWrap(a), b, b.bounds())
        else:
            out = class_rconst(b, a, b.bounds())
    return out

cdef class _BpfBinOp(_BpfInterface):
    cdef _BpfInterface a, b

    def __init__(self, _BpfInterface a, _BpfInterface b, tuple bounds):
        self.a = a
        self.b = b
        self._x0, self._x1 = bounds
    
    def __getstate__(self):
        return self.a, self.b, (self._x0, self._x1)
    
    def _get_xs_for_rendering(self, int n):
        return numpy.unique(numpy.append(self.a._get_xs_for_rendering(n), self.b._get_xs_for_rendering(n)))

    cdef double __ccall__(self, double x) nogil:
        cdef double A = self.a.__ccall__(x)
        cdef double B = self.b.__ccall__(x)
        self._apply(&A, &B, 1)
        return A

    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        # the result should be put in A
        pass
    
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] ys_a = self.a.mapn_between(n, x0, x1, out)
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] ys_b = self.b.mapn_between(n, x0, x1)
        with nogil:
            self._apply(<DTYPE_t *>(ys_a.data), <DTYPE_t *>(ys_b.data), n)
        return ys_a
        
    cpdef ndarray map(self, xs, ndarray out=None):
        """
        the same as map(self, xs) but somewhat faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.
        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
        ( this is the same as bpf.mapn_between(10, bpf.x0, bpf.x1) )
        """
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] B
        cdef int len_xs
        if isinstance(xs, (int, long)):
            return self.mapn_between(xs, self._x0, self._x1, out)
        A = self.a.map(xs, out)
        B = self.b.map(xs)
        len_xs = len(xs)
        with nogil:
            self._apply(<DTYPE_t *>(A.data), <DTYPE_t *>(B.data), len_xs)
        return A

cdef class _BpfUnaryFunc(_BpfInterface):
    cdef _BpfInterface a
    cdef t_unfunc func
    cdef int funcindex
    
    def __reduce__(self):
        return type(self), (), self.a, self.funcindex 
    
    def __setstate__(self, state):
        bpf, funcindex = state
        return _BpfUnaryFunc_new_from_index(bpf, funcindex)
    
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        cdef t_unfunc func = self.func
        for i in range(n):
            A[i] = func(A[i])
    
    cdef double __ccall__(self, double x) nogil:
        return self.func(self.a.__ccall__(x))
    
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A = self.a.mapn_between(n, x0, x1, out)
        cdef t_unfunc func = self.func
        with nogil:
            self._apply(<DTYPE_t *>(A.data), n)
        return A
    
    cpdef ndarray map(self, xs, ndarray out=None):
        """
        the same as map(self, xs) but somewhat faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.
        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
        ( this is the same as bpf.mapn_between(10, bpf.x0, bpf.x1) )
        """
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A
        if isinstance(xs, (int, long)):
            return self.mapn_between(xs, self._x0, self._x1, out)
        A = self.a.map(xs, out)
        self._apply(<DTYPE_t *>(A.data), len(xs))
        return A
     
cdef _BpfUnaryFunc _BpfUnaryFunc_new(_BpfInterface a, t_unfunc func, int funcindex):
    cdef _BpfUnaryFunc self = _BpfUnaryFunc()
    self.a = a
    self.func = func
    self.funcindex = funcindex
    cdef double x0, x1
    x0, x1 = a.bounds()
    self._set_bounds(x0, x1)
    return self
    
cdef _BpfUnaryFunc _BpfUnaryFunc_new_from_index (_BpfInterface a, int funcindex):
    cdef t_unfunc func = _unfunc_from_index(funcindex)
    return _BpfUnaryFunc_new(a, func, funcindex)
    
cdef t_unfunc _unfunc_from_index(int funcindex):
    return UNFUNCS [funcindex]

cdef class _BpfUnaryOp(_BpfInterface):
    cdef _BpfInterface a

    def __init__(self, _BpfInterface a):
        self.a = a
        cdef double x0, x1
        x0, x1 = a.bounds()
        self._set_bounds(x0, x1)
    
    cdef double __ccall__(self, double x) nogil:
        cdef double X = self.a.__ccall__(x)
        self._apply(&X, 1)
        return X
    
    def __getstate__(self):
        return (self.a,)
    
    def _get_xs_for_rendering(self, int n):
        return self.a._get_xs_for_rendering(n)
    
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        pass
    
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A = self.a.mapn_between(n, x0, x1, out)
        with nogil:
            self._apply(<DTYPE_t *>(A.data), n)
        return A
    
    cpdef ndarray map(self, xs, ndarray out=None):
        """
        the same as map(self, xs) but somewhat faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.
        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
        ( this is the same as bpf.mapn_between(10, bpf.x0, bpf.x1) )
        """
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A
        if isinstance(xs, (int, long)):
            return self.mapn_between(xs, self._x0, self._x1, out)
        A = self.a.map(xs, out)
        self._apply(<DTYPE_t *>(A.data), len(xs))
        return A

cdef class _BpfBinOpConst(_BpfInterface):
    cdef double b_const
    cdef _BpfInterface a
    
    def __init__(_BpfBinOpConst self, _BpfInterface a, double b, tuple bounds, str op=''):
        self.a = a
        self.b_const = b
        self._x0, self._x1 = bounds
    
    def __getstate__(self):
        return self.a, self.b_const, (self._x0, self._x1)
    
    def _get_xs_for_rendering(self, int n):
        return self.a._get_xs_for_rendering(n)
    
    cdef void _apply(self, DTYPE_t *A, int n, double x) nogil:
        pass
    
    cpdef ndarray map(self, xs, ndarray out=None):
        """
        the same as map(self, xs) but somewhat faster

        xs can also be a number, in which case it is interpreted as
        the number of elements to calculate in an evenly spaced
        grid between the bounds of this bpf.
        bpf.map(10) == bpf.map(numpy.linspace(x0, x1, 10))
        ( this is the same as bpf.mapn_between(10, bpf.x0, bpf.x1) )
        """
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A
        cdef int len_xs
        if isinstance(xs, (int, long)):
            return self.mapn_between(xs, self._x0, self._x1, out)
        A = self.a.map(xs, out)
        len_xs = len(xs)
        with nogil:
            self._apply(<DTYPE_t *>(A.data), len_xs, self.b_const)
        return A
    
    cdef double __ccall__(self, double x) nogil:
        cdef double A = self.a.__ccall__(x)
        self._apply(&A, 1, self.b_const)
        return A

cdef class _BpfLambdaAdd(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] += B[i]
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.a.integrate_between(x0, x1, N) + self.b.integrate_between(x0, x1, N)
    
cdef class _BpfLambdaAddConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] += b
    def __add__(a, b):
        cdef double c
        if isinstance(a, _BpfInterface):
            try:
                c = float(b)
                return _BpfLambdaAddConst(a.a, a.b + c)
            except:
                return _BpfBinOpConst.__add__(a, b)
        else:
            try:
                c = float(a)
                return _BpfLambdaAddConst(b.a, b.b + a)
            except:
                return _BpfBinOpConst.__add__(b, a)
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.a.integrate_between(x0, x1, N) + (x1 - x0) * self.b_const

cdef class _BpfLambdaSub(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] -= B[i]
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.a.integrate_between(x0, x1, N) - self.b.integrate_between(x0, x1, N) 

cdef class _BpfLambdaRSub(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            B[i] -= A[i]
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.b.integrate_between(x0, x1, N) - self.a.integrate_between(x0, x1, N)

cdef class _BpfLambdaSubConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] -= b
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.a.integrate_between(x0, x1, N) - (x1-x0)*self.b_const

cdef class _BpfLambdaRSubConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = b - A[i]
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.b_const*(x1-x0) - self.a.integrate_between(x0, x1, N)

cdef class _BpfLambdaMul(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] *= B[i]
            
cdef class _BpfLambdaMulConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] *= b
    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        return self.a.integrate_between(x0, x1, N) * self.b_const

cdef class _BpfLambdaPow(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] ** B[i]

cdef class _BpfLambdaPowConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] ** b

cdef class _BpfLambdaRPowConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = b ** A[i]

cdef class _BpfLambdaDiv(_BpfBinOp):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] /= B[i]

cdef class _BpfLambdaDivConst(_BpfBinOpConst):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] /= b

cdef class _BpfLambdaMod(_BpfBinOp):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = fmod(A[i], B[i])

cdef class _BpfLambdaModConst(_BpfBinOpConst):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = fmod(A[i], b)

cdef class _BpfLambdaRDiv(_BpfBinOp):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = B[i] / A[i]

cdef class _BpfLambdaRDivConst(_BpfBinOpConst):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = b / A[i]

cdef class _BpfLambdaGreaterThan(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] > B[i]
    
cdef class _BpfLambdaGreaterOrEqualThan(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] >= B[i]

cdef class _BpfLambdaGreaterThanConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] > b

cdef class _BpfLambdaGreaterOrEqualThanConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] >= b

cdef class _BpfLambdaLowerThan(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] < B[i]

cdef class _BpfLambdaLowerThanConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] < b

cdef class _BpfLambdaLowerOrEqualThan(_BpfBinOp):
    #cdef double __ccall__(self, double x) nogil:
    #    return <double>(self.a.__ccall__(x)) <= <double>(self.b.__ccall__(x))
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] <= B[i]

cdef class _BpfLambdaLowerOrEqualThanConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] <= b

cdef class _BpfLambdaEqual(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] == B[i]

cdef class _BpfLambdaEqualConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] == b

cdef class _BpfLambdaUnequal(_BpfBinOp):
    cdef void _apply(self, DTYPE_t *A, DTYPE_t *B, int n) nogil:
        for i in range(n):
            A[i] = A[i] != B[i]

cdef class _BpfLambdaUnequalConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = A[i] != b
            
cdef class _BpfMaxConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        cdef double y
        for i in range(n):
            y = A[i]
            A[i] = y if y > b else b
        
cdef class _BpfMinConst(_BpfBinOpConst):
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        cdef double y
        for i in range(n):
            y = A[i]
            A[i] = y if y < b else b

cdef class _BpfLambdaLog(_BpfBinOpConst):
    def __init__(_BpfBinOpConst self, _BpfInterface a, double b, tuple bounds, str op=''):
        _BpfBinOpConst.__init__(self, a, b, bounds, op)
        self.b_const = m_log(b)
    
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, int n, double b) nogil:
        for i in range(n):
            A[i] = m_log(A[i]) / b


cdef class _BpfLambdaRound(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = floor(A[i] + 0.5)

cdef class _BpfRand(_BpfUnaryOp):
    @cython.cdivision(True)
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = (rand()/<double>RAND_MAX) * A[i]

cdef class _BpfLambdaFib(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = _fib(A[i])

cdef class _BpfM2F(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = m2f(A[i])

cdef class _BpfF2M(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = f2m(A[i])

cdef class _Bpf_db2amp(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        for i in range(n):
            A[i] = 10.0 ** (0.05*A[i])

cdef class _Bpf_amp2db(_BpfUnaryOp):
    cdef void _apply(self, DTYPE_t *A, int n) nogil:
        cdef double x
        for i in range(n):
            x = max(A[i], 1e-14)
            A[i] = log10(x) * 20.0
            

cdef class _BpfLambdaClip(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double y0, y1
    cdef double __ccall__(self, double x) nogil:
        cdef double y = self.bpf.__ccall__(x)
        if y > self.y1:
            return self.y1
        elif y < self.y0:
            return self.y0
        return y
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef:
            ndarray [DTYPE_t, ndim=1] result
            DTYPE_t *data
            int i
            double dx = (x1 - x0) / (n - 1) # we account for the edge (x1 IS INCLUDED)
            double y1 = self.y1
            double y0 = self.y0
            double y
            _BpfInterface bpf = self.bpf
        if out is None:
            result = EMPTY1D(n)
            data = <DTYPE_t *>result.data
            with nogil:
                for i in range(n):
                    y = bpf.__ccall__(x0 + dx * i)
                    y = _clip(y, y0, y1)
                    data[i] = y
            return result
        else:
            if PyArray_ISCONTIGUOUS(out):
                data = <DTYPE_t *>out.data
                with nogil:
                    for i in range(n):
                        y = bpf.__ccall__(x0 + dx * i)
                        y = _clip(y, y0, y1)
                        data[i] = y
            else:
                for i in range(n):
                    for i in range(n):
                        y = bpf.__ccall__(x0 + dx * i)
                        y = _clip(y, y0, y1)
                        out[i] = y
            return out

    def __reduce__(self):
        return type(self), (), (self.bpf, self.y0, self.y1)
    
    def __setstate__(self, state):
        self.bpf, self.y0, self.y1 = state
        self._set_bounds_like(self.bpf)


cdef _BpfLambdaClip _BpfLambdaClip_new(_BpfInterface bpf, double y0, double y1):
    cdef _BpfLambdaClip self = _BpfLambdaClip()
    self._set_bounds_like(bpf)
    self.bpf = bpf
    self.y0 = y0
    self.y1 = y1
    return self


cdef class _BpfDeriv(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double h
    def __init__(self, _BpfInterface bpf, double h=0):
        self.h = h
        self.bpf = bpf
        self._x0, self._x1 = bpf.bounds()

    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        cdef double h = self.h if self.h > 0 else (SQRT_EPS if x == 0 else SQRT_EPS*x)
        cdef double f1 = self.bpf.__ccall__(x+h)
        cdef double f0 = self.bpf.__ccall__(x)
        return (f1 - f0) / h   
       
    def __getstate__(self):
        return (self.bpf,)


cdef class _BpfInverted(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double bpf_x0, bpf_x1
    def __init__(self, _BpfInterface bpf):
        cdef double x0, x1
        self.bpf = bpf
        self.bpf_x0, self.bpf_x1 = bpf.bounds()
        x0 = bpf(self.bpf_x0)
        x1 = bpf(self.bpf_x1)
        if x0 >= x1:
            raise BpfInversionError("could not invert bpf")
        self._set_bounds(x0, x1)

    cdef double __ccall__(self, double x) nogil:
        cdef double out
        cdef int outerror, funcalls
        if x < self._x0:
            return self.bpf_x0
        elif x > self._x1:
            return self.bpf_x1
        out = _bpf_brentq(self.bpf, -x, self.bpf_x0, self.bpf_x1, &outerror, BRENTQ_XTOL, BRENTQ_RTOL, BRENTQ_MAXITER, &funcalls)
        if outerror == 1:   # error
            return NAN
        return out
    def __getstate__(self): return (self.bpf,)


cdef class _BpfIntegrate(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double bpf_at_x0, width, min_N_ratio, Nexp
    cdef int N, N0, Nwidth
    cdef public int debug
    cdef public size_t oversample
    def __init__(self, _BpfInterface bpf, N=None, bounds=None, double min_N_ratio=0.3, double Nexp=0.8, int oversample=0):
        cdef double x0, x1, dx
        cdef int i
        cdef _BpfInterface tmpbpf
        cdef int _N
        if bounds is None:
            bounds = bpf.bounds()
        _N = N if N is not None else CONFIG['integrate.trapz_intervals']
        x0, x1 = bounds
        self._set_bounds(x0, x1)
        self.bpf = bpf
        self.N = _N
        self.bpf_at_x0 = bpf.__ccall__(x0)
        self.min_N_ratio = min_N_ratio
        self.N0 = <int>(_N * min_N_ratio)
        self.Nexp = Nexp
        self.width = x1 - x0
        self.Nwidth = _N - self.N0
        self.oversample = oversample if oversample > 0 else CONFIG['integrate.oversample']
        if x0 == INFNEG or x0 == INF:
            raise ValueError("cannot integrate a function with an infinite lower bound")
        self.debug = 0
    
    @cython.cdivision(True)
    cdef int get_integration_steps(self, double x) nogil:
        return <int>(self.N0 + pow((x - self._x0) / self.width, self.Nexp) * self.Nwidth)
    
    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        with gil:
            return self.bpf.integrate_between(self._x0, x)
        
    @cython.cdivision(True)
    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef double [::1] accums
        cdef double dx0, dx1, dy
        cdef double dx = (x1 - x0) / (n-1)
        cdef int i, return_out
        cdef double accum = self.bpf.integrate_between(self._x0, x0) if x0 > self._x0 else self.__ccall__(self._x0)
        cdef size_t subn = self.oversample
        if out is not None and out.shape[0] == n and PyArray_ISCONTIGUOUS(out):
            accums = out
            return_out = 1
        else:
            accums = EMPTY1D(n)
            return_out = 0
        accums[0] = accum
        for i in range(1, n):
            dx1 = x0 + i*dx
            dx0 = dx1 - dx            
            dy = self.bpf.integrate_between(dx0, dx1, subn)
            accum += dy
            accums[i] = accum
        if return_out:
            return out
        else:
            return numpy.asarray(accums)

    cpdef _BpfInterface derivative(self):
        return self.bpf

    def __getstate__(self):
        return (self.bpf, self.N, self.bounds(), self.min_N_ratio, self.Nexp)
        
cdef class _BpfPeriodic(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double x0
    cdef double period
    def __init__(self, _BpfInterface bpf):
        self._set_bounds(INFNEG, INF)
        self.bpf = bpf
        cdef double x0, x1
        x0, x1 = bpf.bounds()
        self.x0 = x0
        self.period = x1 - x0
    
    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        # the C equivalent of self.x0 + ((x - self.x0) % self.period)
        # python n % M | c ((n % M) + M) % M
        cdef double n = x - self.x0
        cdef double p = self.x0 + ( (n % self.period + self.period) % self.period )
        return self.bpf.__ccall__( p )

    def __getstate__(self): return (self.bpf,)

cdef class _BpfProjection(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double bpf_x0
    cdef readonly double dx, rx, offset
    def __init__(self, _BpfInterface bpf, double rx, double dx=0, double offset=0):
        """
        equation:

        x2 = (x - offset) * rx + dx
        self(x) = bpf(x2)

        stretched 3: rx=3, offset=0, dx=0
        shifted 2: rx=1, offset=0, dx=1
        """
        self.bpf = bpf
        self.rx = rx
        self.dx = dx
        self.offset = offset
        cdef double x0 = (bpf._x0 - dx)/rx + offset
        cdef double x1 = (bpf._x1 - dx)/rx + offset
        if x0 < x1:
            self._set_bounds(x0, x1)
        else:
            self._set_bounds(x1, x0)
    
    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        x = (x - self.offset) * self.rx + self.dx
        return self.bpf.__ccall__(x)

    def __getstate__(self): return (self.bpf, self.rx, self.dx)


cdef double m_log(double x) nogil:
    if isfinite(x): # INFNEG > x < INF:
        if x > 0:
            return log(x)
        if x == 0:
            return INFNEG
        else:
            return NAN
    elif isnan(x):
        return x
    elif x > 0:
        return x
    else:
        return NAN
"""
    Taken from the python implementation (in C)

    static double
    m_log(double x)
    {
        if (Py_IS_FINITE(x)) {
            if (x > 0.0)
                return log(x);
            errno = EDOM;
            if (x == 0.0)
                return -Py_HUGE_VAL; /* log(0) = -inf */
            else
                return Py_NAN; /* log(-ve) = nan */
        }
        else if (Py_IS_NAN(x))
            return x; /* log(nan) = nan */
        else if (x > 0.0)
            return x; /* log(inf) = inf */
        else {
            errno = EDOM;
            return Py_NAN; /* log(-inf) = nan */
        }
    }
"""

cdef class _BpfKeepSlope(_BpfInterface):
    cdef _BpfInterface bpf
    cdef double EPSILON
    def __init__(self, _BpfInterface bpf, double EPSILON=DEFAULT_EPSILON):
        #_BpfInterface.__init__(self, INFNEG, INF)
        self._set_bounds(INFNEG, INF)
        self.bpf = bpf
        self.EPSILON = EPSILON

    @cython.cdivision(True)
    cdef double __ccall__(self, double x) nogil:
        cdef double slope
        cdef double x0 = self.bpf._x0
        cdef double x1 = self.bpf._x1
        if x0 <= x <= x1:
            return self.bpf.__ccall__(x)
        elif x > x1:
            slope = (self.bpf.__ccall__(x1) - self.bpf.__ccall__(x1 - self.EPSILON)) / self.EPSILON
            return self.bpf.__ccall__(x1) + slope * (x - x1)
        else:
            slope = (self.bpf.__ccall__(x0 + self.EPSILON) - self.bpf.__ccall__(x0)) / self.EPSILON
            return self.bpf.__ccall__(x0) + slope * (x - x0)

    def __getstate__(self): return self.bpf, self.EPSILON

cdef class _BpfCrop(_BpfInterface):
    cdef _BpfInterface bpf
    cdef readonly double _y0, _y1
    cdef readonly int outbound_mode

    cdef double __ccall__(self, double x) nogil:
        if self.outbound_mode == 0:
            return self.bpf.__ccall__(x)
        else:
            if x < self._x0:
                return self._y0
            elif x > self._x1:
                return self._y1
            return self.bpf.__ccall__(x)

    def __reduce__(self):
        return type(self), (), (self.bpf, self._x0, self._x1, self.outbound_mode, self._y0, self._y1)

    def __setstate__(self, state):
        self.bpf, x0, x1, self.outbound_mode, self._y0, self._y1 = state
        self._set_bounds(x0, x1)

    cpdef _BpfCrop outbound_set(self, double y0, double y1):
        """
        set the value returned by this BPF outside its defined bounds (inplace)
        """
        self.outbound_mode = OUTBOUND_SET
        self._y0 = y0
        self._y1 = y1
        return self

    def outbound(self, y0, y1=None):
        """
        return a new Bpf with the given values outside the bounds

        NB: you can specify one value for lower and one for upper bounds, or just one value for both
        """
        if y1 is None:
            y1 = y0
        return _BpfCrop_new(self.bpf, self._x0, self._x1, OUTBOUND_SET, y0, y1)

    cpdef double integrate_between(self, double x0, double x1, size_t N=0):
        if x0 >= self.bpf._x0 and x1 <= self.bpf._x1:
            return self.bpf.integrate_between(x0, x1)
        cdef double integr0, integr1, integr2, _x0, _x1
        if x0 < self.bpf._x0:
            integr0 = self.__ccall__(x0) * (self.bpf._x0 - x0)
            _x0 = self.bpf._x0
        else:
            integr0 = 0
            _x0 = x0
        if self._x1 > self.bpf._x1:
            integr2 = self.__ccall__(x1) * (x1 - self.bpf._x1)
            _x1 = self.bpf._x1
        else:
            integr2 = 0
            _x1 = x1
        integr1 = self.bpf.integrate_between(_x0, _x1)
        return integr0 + integr1 + integr2

    cpdef ndarray mapn_between(self, int n, double x0, double x1, ndarray out=None):
        cdef double x, y0, y1, intersect_x0, intersect_x1, dx
        cdef int i, i0, i1, intersect_n, intersect_i0, intersect_i
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] A = out if out is not None else EMPTY1D(n)
        cdef c_numpy.ndarray[DTYPE_t, ndim=1] intersection
        cdef DTYPE_t *data = <DTYPE_t*>(A.data)
        cdef DTYPE_t *data_intersection
        if self.outbound_mode == OUTBOUND_DONOTHING:
            return self.bpf.mapn_between(n, x0, x1, out)
        elif x0 >= self._x1:
            return numpy.ones((n,), dtype=float) * self.__ccall__(x0)
        elif x1 <= self._x0:
            return numpy.ones((n,), dtype=float) * self.__ccall__(x1)
        dx = (x1 - x0) / (n - 1)
        intersect_x0 = x0 if x0 > self.bpf.x0 else self.bpf.x0
        intersect_x0_quant = intersect_x0 - ((intersect_x0 - x0) % dx)
        intersect_x1 = x1 if x1 < self.bpf.x1 else self.bpf.x1
        intersect_x1_quant = intersect_x1 - ((intersect_x1 - x0) % dx)
        intersect_n = <int>((intersect_x1_quant - intersect_x0_quant) / dx) + 1
        intersect_i0 = <int>((intersect_x0_quant - x0) / dx)
        intersect_i1 = <int>((intersect_x1_quant - x0) / dx)
        intersection = self.bpf.mapn_between(intersect_n, intersect_x0_quant, intersect_x1_quant)
        data_intersection = <DTYPE_t*>(intersection.data)
        y0 = self._y0
        y1 = self._y1
        x = x0
        i = 0
        while x < self._x0:
            data[i] = y0
            i += 1
            x = x0 + dx*i
        for intersect_i in range(intersect_n):
            A[i] = data_intersection[intersect_i]
            i += 1
        x = x0 + dx*i
        while x <= x1:
            data[i] = y1
            i += 1
            x = x0 + dx*i
        return A    

cpdef _BpfCrop _BpfCrop_new(_BpfInterface bpf, double x0, double x1, int outbound_mode, double outbound0=0, double outbound1=0):
    """
    create a cropped bpf. 
    
    outbound_mode:
        -1: use the default
         0: do nothing. in this case, the bpf is evaluated at the cropping point each time it is called outside the bounds
         1: cache the values. the values at the bounds are cached and returned when an outbound call takes place
         2: set. in this case the last parameters outbount0 and outbound1 are used when called outside the bounds
    """
    self = _BpfCrop()
    self._set_bounds(x0, x1)
    self.bpf = bpf
    # -1: use the default, 0: do nothing, call __ccall__ each time, 1: cache y0 and y1 for values outside the bounds, 2: set y0 and y1 for values outside the bounds
    if outbound_mode == OUTBOUND_DEFAULT:
        outbound_mode = CONFIG['crop.outbound_mode']
    self.outbound_mode = outbound_mode
    if outbound_mode == OUTBOUND_CACHE:
        self._y0 = self.bpf.__ccall__(x0)
        self._y1 = self.bpf.__ccall__(x1)
    elif outbound_mode == OUTBOUND_SET:
        self._y0 = outbound0
        self._y1 = outbound1
    return self
    
cdef class _MultipleBpfs(_BpfInterface):
    cdef tuple _bpfs
    cdef void** bpfpointers
    cdef _BpfInterface tmp
    cdef int _numbpfs
    def __init__(self, bpfs):
        self._numbpfs = len(bpfs)
        self._bpfs = tuple(bpfs)
        self._calculate_bounds()
        self.bpfpointers = <void**>malloc(sizeof(void*) * self._numbpfs)
        cdef int i
        for i in range(self._numbpfs):
            self.bpfpointers[i] = <void*>bpfs[i]
    def __dealloc__(self):
        free(self.bpfpointers)
    def _calculate_bounds(self):
        cdef double bounds0, bounds1
        cdef double x0=INF, x1=INFNEG
        for bpf in self._bpfs:
            bound0, bound1 = bpf.bounds()
            if bound0 < x0:
                x0 = bound0
            if bound1 > x1:
                x1 = bound1
        self._set_bounds(x0, x1)
    def __getstate__(self): return (self._bpfs,)
    cdef double __ccall__(self, double x) nogil:
        with gil:
            raise NotImplementedError

cdef class Max(_MultipleBpfs):
    def __init__(self, *bpfs):
        if len(bpfs) == 1 and isinstance(bpfs[0], (list, tuple)):
            bpfs = bpfs[0]
        _MultipleBpfs.__init__(self, bpfs)

    cdef double __ccall__(self, double x) nogil:
        cdef double y = INFNEG
        cdef double res
        cdef int i
        for i in range(self._numbpfs):
            with gil:
                self.tmp = <_BpfInterface>(self.bpfpointers[i])
            res = self.tmp.__ccall__(x)
            if res > y:
                y = res
        return y

cdef class Min(_MultipleBpfs):
    def __init__(self, *bpfs):
        if len(bpfs) == 1 and isinstance(bpfs[0], (list, tuple)):
            bpfs = bpfs[0]
        _MultipleBpfs.__init__(self, bpfs)
    
    cdef double __ccall__(self, double x) nogil:
        cdef double y = INF
        cdef double res
        cdef int i
        for i in range(self._numbpfs):
            with gil:
                tmp = <_BpfInterface>(self.bpfpointers[i])
            res = tmp.__ccall__(x)
            if res < y:
                y = res
        return y


cdef class _BpfSelect(_MultipleBpfs):
    """
    No interpolation between the bpfs
    """
    cdef _BpfInterface which
    def __init__(self, which, bpfs):
        self.which = which
        _MultipleBpfs.__init__(self, bpfs)
    
    cdef double __ccall__(self, double x) nogil:
        cdef int index = <int>(self.which.__ccall__(x))
        with gil:
            tmp = <_BpfInterface>(self.bpfpointers[index])
        return tmp.__ccall__(x)

cdef class _BpfSelectX(_MultipleBpfs):
    cdef _BpfInterface which
    cdef InterpolFunc* func
    cdef int numbpfs
    def __init__(self, which, bpfs, shape='linear'):
        self.which = which
        self.numbpfs = len(bpfs)
        self.func = InterpolFunc_new_from_descriptor(shape)
        _MultipleBpfs.__init__(self, bpfs)
    
    cdef double __ccall__(self, double x) nogil:
        cdef double index = self.which.__ccall__(x)
        cdef double y0, y1, x0
        if index <= 0:
            with gil:
                b0 = <_BpfInterface>(self.bpfpointers[0])
            return b0.__ccall__(x)
        elif index >= self.numbpfs - 1:
            with gil:
                b0 = <_BpfInterface>(self.bpfpointers[self.numbpfs-1])
            return b0.__ccall__(x)
        else:
            x0 = floor(index)
            with gil:
                b0 = <_BpfInterface>(self.bpfpointers[<int>x0])
                b1 = <_BpfInterface>(self.bpfpointers[<int>x0 + 1])
            y0 = b0.__ccall__(x)
            y1 = b1.__ccall__(x)

            return InterpolFunc_call(self.func, index, x0, y0, x0+1, y1)

cdef inline aslist(obj): return obj if isinstance(obj, list) else list(obj)
    
def brentq(bpf, x0, xa, xb, xtol=9.9999999999999998e-13, rtol=4.4408920985006262e-16, max_iter=100):
    """
    calculate the zero of (bpf + x0) in the interval (xa, xb) using brentq algorithm

    NB: to calculate all the zeros of a bpf, use the .zeros method

    Returns
    =======

    (zero of the bpf, number of function calls)

    Example
    =======

    # calculate the x where a == 0.5
    >>> a = bpf.linear(0, 0, 10, 1)
    >>> x_at_zero, numcalls = bpf_brentq(a, -0.5, 0, 1)
    >>> print x_at_zero
    5
    """
    cdef int outerror, funcalls
    cdef double result
    result = _bpf_brentq(bpf, x0, xa, xb, &outerror, xtol, rtol, max_iter, &funcalls)
    if outerror:
        raise ValueError("zero of function cannot be found within the interval given")
    return result, funcalls

cpdef _BpfInterface blend(a, b, mix=0.5):
    """
    blend these BPFs
    
    if mix == 0: the result is *a*
    if mix == 1: the result is *b*
    
    mix can also be a bpf or any function
    
    Example
    -------
    
    # create a curve which is in between a halfcos and a linear interpolation
    a = bpf.halfcos(0, 0, 1, 1)
    b = bpf.linear(0, 0, 1, 1)
    a.blendwith(b, 0.5)
    
    # nearer to halfcos
    a.blendwith(b, 0.1)
    """
    if isinstance(mix, (int, float)):
        return _BpfBlendConst(_as_bpf(a), _as_bpf(b), mix)
    return _BpfBlend(_as_bpf(a), _as_bpf(b), _as_bpf(mix))
        


cdef inline int isnumber(obj):
    return isinstance(obj, (int, float))


@cython.cdivision(True)
cdef inline double _bpf_brentq(_BpfInterface bpf, double x0, double xa, double xb, int* outerror, double xtol, double rtol, int max_iter, int *outfuncalls) nogil:
    # original values: xtol=9.9999999999999998e-13, rtol=4.4408920985006262e-16, max_iter=100
    # calculate the 0 of the function bpf + x0 in the interval (xa, xb)
    # if it is not possible, outerror is set to 1, otherwise it is 0
    cdef:
        double xpre = xa
        double xcur = xb
        double xblk = 0
        double fpre, fcur, stry, dpre, dblk, sbis, tol
        double fblk = 0
        double spre = 0
        double scur = 0
        double a, b
        int funcalls = 2
        int i
    outerror[0] = 0
    fpre = bpf.__ccall__(xpre) + x0
    fcur = bpf.__ccall__(xcur) + x0
    outfuncalls[0] = 2
    if fpre * fcur > 0:
        outerror[0] = 1
        return 0
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    for i in range(max_iter):
        if fpre * fcur < 0:
            xblk = xpre; fblk = fpre
            spre = scur = xcur - xpre
        if fabs(fblk) < fabs(fcur):
            xpre = xcur; xcur = xblk; xblk = xpre; fpre = fcur
            fcur = fblk; fblk = fpre
        tol = xtol + rtol * fabs(xcur)
        sbis = (xblk - xcur) / 2
        if fabs(fcur) == 0 or fabs(sbis) < tol:
            return xcur
        if fabs(spre) > tol and fabs(fcur) < fabs(fpre):
            if xpre == xblk:    # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:               # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre) / (dblk*dpre*(fblk - fpre))     
            # if (2*fabs(stry) < DMIN(fabs(spre), 3*fabs(sbis) - tol)):
            a = fabs(spre)
            b = 3*fabs(sbis) - tol
            if (2*fabs(stry) < (a if a < b else b)):
            #if (2*fabs(stry) < DMIN(fabs(spre), 3*fabs(sbis) - tol)):   
                spre = scur; scur = stry  # good short step
            else:
                spre = sbis; scur = sbis  # bisect 
        else:
            spre = sbis; scur = sbis
        xpre = xcur
        fpre = fcur;
        if fabs(scur) > tol:
            xcur += scur
        else:
            xcur += tol if sbis > 0 else -tol
        fcur = bpf.__ccall__(xcur) + x0
        funcalls += 1
    outfuncalls[0] = funcalls
    return xcur

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef list bpf_zero_crossings(_BpfInterface b, double h=0.01, int N=0, double x0=NAN, double x1=NAN, int maxzeros=0):
    """
    return the zeros if b in the interval defined

    b: a bpf
    h: the interval to scan for zeros. for each interval only one zero will be found
    N: alternatively you can give the number of intervals to scan. h will be calculated from that
       N overrides h
    x0, x1: the bounds to use. these, if given, override the bounds b
    """
    if isnan(x0):
        x0 = b._x0
    if isnan(x1):
        x1 = b._x1
    if h > (x1 - x0) * 0.5:
        h = (x1 - x0) * 0.25
    if N == 0:
        N = <int>((x1 - x0) / h) + 1
    else:
        h = (x1 - x0) / (N - 1)
    cdef list out = []
    cdef double y0, y1, xa, xb, zero
    cdef int outerror, funcalls
    y1 = b.__ccall__(x0)
    cdef double last_zero = 0
    cdef int numzeros = 0
    cdef int add_it
    for i in range(N - 1):
        xa = x0 + i * h
        xb = xa + h - EPS
        y0 = b.__ccall__(xa)
        y1 = b.__ccall__(xb)
        add_it = 0
        # print(xa, xb, y0, y1)
        if y0 * y1 < 0:
            # print("y0*y1")
            outerror = -1
            zero = _bpf_brentq(b, 0, xa, xb, &outerror, 9.9999999999999998e-13, 4.4408920985006262e-16, 100, &funcalls)
            if outerror == 0:
                add_it = 1
        elif y1 == 0 and y0 != 0:
            zero = xb
            add_it = 1
        elif y0 == 0 and y1 != 0 and xa > last_zero:
            zero = xa
            add_it = 1
        if add_it:
            last_zero = zero
            out.append(zero)
            numzeros += 1
            if maxzeros > 0 and numzeros >= maxzeros:
                break
    return out

cdef inline double _integrate_adaptive_simpsons_inner(_BpfInterface f, double a, double b, double epsilon, 
                                                      double S, double fa, double fb, double fc, int bottom):
    cdef: 
        double c = (a + b) / 2
        double h = b - a
        double d = (a + c) / 2
        double g = (c + b) / 2
        double fd = f.__ccall__(d)
        double fe = f.__ccall__(g)
        double Sleft = (h / 12) * (fa + 4 * fd + fc)
        double Sright = (h / 12) * (fc + 4 * fe + fb)
    S2 = Sleft + Sright
    if bottom <= 0 or abs(S2 - S) <= 15 * epsilon:
        return S2 + (S2 - S) / 15.
    return _integrate_adaptive_simpsons_inner(f, a, c, epsilon / 2, Sleft, fa, fc, fd, bottom - 1) + \
           _integrate_adaptive_simpsons_inner(f, c, b, epsilon / 2, Sright, fc, fb, fe, bottom - 1)

cdef double integrate_simpsons(_BpfInterface f, double a, double b, double accuracy=10e-10, int max_iterations=50):
    cdef:
        double c = (a + b) / 2
        double h = b - a
        double fa = f.__ccall__(a)
        double fb = f.__ccall__(b)
        double fc = f.__ccall__(c)
        double S = (h / 6) * (fa + 4 * fc + fb)
    return _integrate_adaptive_simpsons_inner(f, a, b, accuracy, S, fa, fb, fc, max_iterations)
