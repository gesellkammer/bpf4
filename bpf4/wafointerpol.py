#!/usr/bin/env python
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import warnings
import numpy as np
import scipy.signal
import scipy.sparse as sp
from numpy.ma.core import prod
from numpy.lib.shape_base import vstack
from scipy.interpolate import PiecewisePolynomial
import pylab as plb
from numpy import pi, any, all, array, asarray, r_, dot, sign, conj, inf
from scipy.misc.common import pade
from six.moves import range
from six.moves import zip

__all__ = ['PPform', 'savitzky_golay', 'savitzky_golay_piecewise', 'sgolay2d','SmoothSpline', 
           'slopes','pchip_slopes','slopes2','stineman_interp', 'Pchip','StinemanInterp', 'CubicHermiteSpline']

def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = savitzky_golay(y, window_size=31, order=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> plt.legend()
    >>> plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

def savitzky_golay_piecewise(xvals, data, kernel=11, order =4):
    '''
    One of the most popular applications of S-G filter, apart from smoothing UV-VIS 
    and IR spectra, is smoothing of curves obtained in electroanalytical experiments. 
    In cyclic voltammetry, voltage (being the abcissa) changes like a triangle wave. 
    And in the signal there are cusps at the turning points (at switching potentials) 
    which should never be smoothed. In this case, Savitzky-Golay smoothing should be 
    done piecewise, ie. separately on pieces monotonic in x
    
    Example
    -------
    >>> n = 1e3
    >>> x = np.linspace(0, 25, n)
    >>> y = np.round(sin(x))
    >>> sig2 = np.linspace(0,0.5,50)
    
    # As an example, this figure shows the effect of an additive noise with a variance 
    # of 0.2 (original signal (black), noisy signal (red) and filtered signal (blue dots)).

    >>> yn = y + sqrt(0.2)*np.random.randn(*x.shape)
    >>> yr = savitzky_golay_piecewise(x, yn, kernel=11, order=4)
    >>> plt.plot(x, yn, 'r', x, y, 'k', x, yr, 'b.')
    '''
    turnpoint=0
    last=len(xvals)
    if xvals[1]>xvals[0] : #x is increasing?
        for i in range(1,last) : #yes
            if xvals[i]<xvals[i-1] : #search where x starts to fall
                turnpoint=i
                break
    else: #no, x is decreasing
        for i in range(1,last) : #search where it starts to rise
            if xvals[i]>xvals[i-1] :
                turnpoint=i
                break
    if turnpoint==0 : #no change in direction of x
        return savitzky_golay(data, kernel, order)
    else:
        #smooth the first piece
        firstpart=savitzky_golay(data[0:turnpoint],kernel,order)
        #recursively smooth the rest
        rest=savitzky_golay_piecewise(xvals[turnpoint:], data[turnpoint:], kernel, order)
        return np.concatenate((firstpart,rest))

def sgolay2d ( z, window_size, order, derivative=None):
    """
    Savitsky - Golay filters can also be used to smooth two dimensional data affected 
    by noise. The algorithm is exactly the same as for the one dimensional case, only 
    the math is a bit more tricky. The basic algorithm is as follow:
    for each point of the two dimensional matrix extract a sub - matrix, centered at 
    that point and with a size equal to an odd number "window_size".
    for this sub - matrix compute a least - square fit of a polynomial surface, defined as
    p(x, y) = a0 + a1 * x + a2 * y + a3 * x2 + a4 * y2 + a5 * x * y + ... . 
    Note that x and y are equal to zero at the central point.
    replace the initial central point with the value computed with the fit.
    Note that because the fit coefficients are linear with respect to the data spacing, they can pre - computed for efficiency. Moreover, it is important to appropriately pad the borders of the data, with a mirror image of the data itself, so that the evaluation of the fit at the borders of the data can happen smoothly.
    Here is the code for two dimensional filtering.

    Example
    -------
    # create some sample twoD data
    >>> x = np.linspace(-3,3,100)
    >>> y = np.linspace(-3,3,100)
    >>> X, Y = np.meshgrid(x,y)
    >>> Z = np.exp( -(X**2+Y**2))
    
    # add noise
    >>> Zn = Z + np.random.normal( 0, 0.2, Z.shape )
    
    # filter it
    >>> Zf = sgolay2d( Zn, window_size=29, order=4)
    
    # do some plotting
    >>> import matplotlib.pyplot as plt
    >>> plt.matshow(Z)
    >>> plt.matshow(Zn)
    >>> plt.matshow(Zf)
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size ** 2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k - n, n) for k in range(order + 1) for n in range(k + 1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2,)

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

class PPform(object):
    """The ppform of the piecewise polynomials is given in terms of coefficients
    and breaks.  The polynomial in the ith interval is
    x_{i} <= x < x_{i+1}

    S_i = sum(coefs[m,i]*(x-breaks[i])^(k-m), m=0..k)
    where k is the degree of the polynomial.

    Example
    -------
    >>> coef = np.array([[1,1]]) # unit step function
    >>> coef = np.array([[1,1],[0,1]]) # linear from 0 to 2
    >>> coef = np.array([[1,1],[1,1],[0,2]]) # linear from 0 to 2
    >>> breaks = [0,1,2]
    >>> self = PPform(coef, breaks)
    >>> x = np.linspace(-1,3)
    >>> plot(x,self(x))
    """
    def __init__(self, coeffs, breaks, fill=0.0, sort=False, a=None, b=None):
        if sort:
            self.breaks = np.sort(breaks)
        else:
            self.breaks = np.asarray(breaks)
        if a is None:
            a = self.breaks[0]
        if b is None:
            b = self.breaks[-1]
        self.coeffs = np.asarray(coeffs)
        self.order = self.coeffs.shape[0]
        self.fill = fill
        self.a = a
        self.b = b

    def __call__(self, xnew):
        saveshape = np.shape(xnew)
        xnew = np.ravel(xnew)
        res = np.empty_like(xnew)
        mask = (self.a <= xnew) & (xnew <= self.b)
        res[~mask] = self.fill
        xx = xnew.compress(mask)
        indxs = np.searchsorted(self.breaks[:-1], xx) - 1
        indxs = indxs.clip(0, len(self.breaks))
        pp = self.coeffs
        dx = xx - self.breaks.take(indxs)
        if True:
            v = pp[0, indxs]
            for i in range(1, self.order):
                v = dx * v + pp[i, indxs]
            values = v
        else:
            V = np.vander(dx, N=self.order)
            # values = np.diag(dot(V,pp[:,indxs]))
            dot = np.dot
            values = np.array([dot(V[k, :], pp[:, indxs[k]]) for k in range(len(xx))])
        
        res[mask] = values
        res.shape = saveshape
        return res
    
    def linear_extrapolate(self, output=True):
        '''
        Return a 1D PPform which extrapolate linearly outside its basic interval
        '''
    
        max_order = 2
    
        if self.order <= max_order:
            if output:
                return self
            else: 
                return
        breaks = self.breaks.copy()
        coefs = self.coeffs.copy()
        
        # Add new breaks beyond each end
        breaks2add = breaks[[0, -1]] + np.array([-1, 1])
        newbreaks = np.hstack([breaks2add[0], breaks, breaks2add[1]])
    
        dx = newbreaks[[0, -2]] - breaks[[0, -2]]
    
        dx = dx.ravel()
       
        # Get coefficients for the new last polynomial piece (a_n)
        # by just relocate the previous last polynomial and
        # then set all terms of order > maxOrder to zero
        
        a_nn = coefs[:, -1]
        dxN = dx[-1]
         
        a_n = polyreloc(a_nn, -dxN) # Relocate last polynomial
        #set to zero all terms of order > maxOrder 
        a_n[0:self.order - max_order] = 0
    
        #Get the coefficients for the new first piece (a_1)
        # by first setting all terms of order > maxOrder to zero and then
        # relocate the polynomial.

    
        #Set to zero all terms of order > maxOrder, i.e., not using them
        a_11 = coefs[self.order - max_order::, 0]
        dx1 = dx[0]
    
        a_1 = polyreloc(a_11, -dx1) # Relocate first polynomial 
        a_1 = np.hstack([zeros(self.order - max_order), a_1])
      
        newcoefs = np.hstack([ a_1.reshape(-1, 1), coefs, a_n.reshape(-1, 1)])
        if output:
            return PPform(newcoefs, newbreaks, a= -inf, b=inf)
        else:
            self.coeffs = newcoefs
            self.breaks = newbreaks
            self.a = -inf
            self.b = inf
    
    def derivative(self):
        """
        Return first derivative of the piecewise polynomial
        """
        
        cof = polyder(self.coeffs)
        brks = self.breaks.copy()
        return PPform(cof, brks, fill=self.fill)


    def integrate(self):
        """
        Return the indefinite integral of the piecewise polynomial
        """
        cof = polyint(self.coeffs)        

        pieces = len(self.breaks) - 1
        if 1 < pieces :
            # evaluate each integrated polynomial at the right endpoint of its interval
            xs = np.diff(self.breaks[:-1, ...], axis=0)
            index = np.arange(pieces - 1)
            
            vv = xs * cof[0, index]
            k = self.order
            for i in range(1, k):
                vv = xs * (vv + cof[i, index])
          
            cof[-1] = np.hstack((0, vv)).cumsum()

        return PPform(cof, self.breaks, fill=self.fill)

class SmoothSpline(PPform):
    """
    Cubic Smoothing Spline.

    Parameters
    ----------
    x : array-like
        x-coordinates of data. (vector)
    y : array-like
        y-coordinates of data. (vector or matrix)
    p : real scalar
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    lin_extrap : bool
        if False regular smoothing spline 
        if True a smoothing spline with a constraint on the ends to
        ensure linear extrapolation outside the range of the data (default)
    var : array-like
        variance of each y(i) (default  1)

    Returns
    -------
    pp : ppform
        If xx is not given, return self-form of the spline.

    Given the approximate values

        y(i) = g(x(i))+e(i)

    of some smooth function, g, where e(i) is the error. SMOOTH tries to
    recover g from y by constructing a function, f, which  minimizes

      p * sum (Y(i) - f(X(i)))^2/d2(i)  +  (1-p) * int (f'')^2


    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(0,1)
    >>> y = np.exp(x)+1e-1*np.random.randn(x.shape)
    >>> pp9 = SmoothSpline(x, y, p=.9)
    >>> pp99 = SmoothSpline(x, y, p=.99, var=0.01)
    >>> plot(x,y, x,pp99(x),'g', x,pp9(x),'k', x,exp(x),'r')

    See also
    --------
    lc2tr, dat2tr


    References
    ----------
    Carl de Boor (1978)
    'Practical Guide to Splines'
    Springer Verlag
    Uses EqXIV.6--9, self 239
    """
    def __init__(self, xx, yy, p=None, lin_extrap=True, var=1):
        coefs, brks = self._compute_coefs(xx, yy, p, var)
        super(SmoothSpline, self).__init__(coefs, brks)
        if lin_extrap:
            self.linear_extrapolate(output=False)
        
    def _compute_coefs(self, xx, yy, p=None, var=1):
        x, y = np.atleast_1d(xx, yy)
        x = x.ravel()
        dx = np.diff(x)
        must_sort = (dx < 0).any()
        if must_sort:
            ind = x.argsort()
            x = x[ind]
            y = y[..., ind]
            dx = np.diff(x)
    
        n = len(x)
    
        #ndy = y.ndim
        szy = y.shape
    
        nd = prod(szy[:-1])
        ny = szy[-1]
       
        if n < 2:
            raise ValueError('There must be >=2 data points.')
        elif (dx <= 0).any():
            raise ValueError('Two consecutive values in x can not be equal.')
        elif n != ny:
            raise ValueError('x and y must have the same length.')
    
        dydx = np.diff(y) / dx
    
        if (n == 2) : #% straight line
            coefs = np.vstack([dydx.ravel(), y[0, :]])
        else:
           
            dx1 = 1. / dx
            D = sp.spdiags(var * np.ones(n), 0, n, n)  # The variance
    
            u, p = self._compute_u(p, D, dydx, dx, dx1, n)
            dx1.shape = (n - 1, -1)
            dx.shape = (n - 1, -1)
            zrs = np.zeros(nd)
            if p < 1:
                ai = (y - (6 * (1 - p) * D * np.diff(vstack([zrs,
                                               np.diff(vstack([zrs, u, zrs]), axis=0) * dx1,
                                               zrs]), axis=0)).T).T #faster than yi-6*(1-p)*Q*u
            else:
                ai = y.reshape(n, -1)
    
            # The piecewise polynominals are written as
            # fi=ai+bi*(x-xi)+ci*(x-xi)^2+di*(x-xi)^3
            # where the derivatives in the knots according to Carl de Boor are:
            #    ddfi  = 6*p*[0;u] = 2*ci;
            #    dddfi = 2*diff([ci;0])./dx = 6*di;
            #    dfi   = np.diff(ai)./dx-(ci+di.*dx).*dx = bi;
    
            ci = np.vstack([zrs, 3 * p * u])  
            di = (np.diff(np.vstack([ci, zrs]), axis=0) * dx1 / 3);
            bi = (np.diff(ai, axis=0) * dx1 - (ci + di * dx) * dx)
            ai = ai[:n - 1, ...] 
            if nd > 1:
                di = di.T
                ci = ci.T
                ai = ai.T
            if not any(di):
                if not any(ci):
                    coefs = np.vstack([bi.ravel(), ai.ravel()])
                else:
                    coefs = np.vstack([ci.ravel(), bi.ravel(), ai.ravel()]) 
            else:
                coefs = np.vstack([di.ravel(), ci.ravel(), bi.ravel(), ai.ravel()]) 
                
        return coefs, x
       
    def _compute_u(self, p, D, dydx, dx, dx1, n):
        if p is None or p != 0:
            data = [dx[1:n - 1], 2 * (dx[:n - 2] + dx[1:n - 1]), dx[:n - 2]]
            R = sp.spdiags(data, [-1, 0, 1], n - 2, n - 2)
        
        if p is None or p < 1:
            Q = sp.spdiags([dx1[:n - 2], -(dx1[:n - 2] + dx1[1:n - 1]), dx1[1:n - 1]], [0, -1, -2], n, n - 2)
            QDQ = (Q.T * D * Q) 
            if p is None or p < 0:
                # Estimate p
                p = 1. / (1. + QDQ.diagonal().sum() / (100. * R.diagonal().sum()** 2));
            
            if p == 0:
                QQ = 6 * QDQ
            else:
                QQ = (6 * (1 - p)) * (QDQ) + p * R
        else:
            QQ = R 
            
        # Make sure it uses symmetric matrix solver
        ddydx = np.diff(dydx, axis=0)
        sp.linalg.use_solver(useUmfpack=True)
        u = 2 * sp.linalg.spsolve((QQ + QQ.T), ddydx)
        return u.reshape(n - 2, -1), p
 
def _edge_case(m0, d1):
    return np.where((d1==0) | (m0==0), 0.0, 1.0/(1.0/m0+1.0/d1))

def pchip_slopes(x, y):
    # Determine the derivatives at the points y_k, d_k, by using
    #  PCHIP algorithm is:
    # We choose the derivatives at the point x_k by
    # Let m_k be the slope of the kth segment (between k and k+1)
    # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
    # else use weighted harmonic mean:
    #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
    #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
    #   where h_k is the spacing between x_k and x_{k+1}

    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    smk = np.sign(mk)
    condition = ((smk[1:] != smk[:-1]) | (mk[1:]==0) | (mk[:-1]==0))

    w1 = 2*hk[1:] + hk[:-1]
    w2 = hk[1:] + 2*hk[:-1]
    whmean = 1.0/(w1+w2)*(w1/mk[1:] + w2/mk[:-1])

    dk = np.zeros_like(y)
    dk[1:-1][condition] = 0.0
    dk[1:-1][~condition] = 1.0/whmean[~condition]

    # For end-points choose d_0 so that 1/d_0 = 1/m_0 + 1/d_1 unless
    #  one of d_1 or m_0 is 0, then choose d_0 = 0

    dk[0] = _edge_case(mk[0],dk[1])
    dk[-1] = _edge_case(mk[-1],dk[-2])
    return dk

def slopes2(x,y, method='parabola', tension=0, monotone=True):
    '''
    Return estimated slopes y'(x) 
    
    Parameters
    ----------
    x, y : array-like
        array containing the x- and y-data, respectively.
        x must be sorted low to high... (no repeats) while
        y can have repeated values.
    method : string
        defining method of estimation for yp. Valid options are:
        'Catmull-Rom'  yp = (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        'Cardinal'     yp = (1-tension) * (y[k+1]-y[k-1])/(x[k+1]-x[k-1])
        'parabola'
        'secant' average secants 
            yp = 0.5*((y[k+1]-y[k])/(x[k+1]-x[k]) + (y[k]-y[k-1])/(x[k]-x[k-1]))
    tension : real scalar between 0 and 1.
        tension parameter used in Cardinal method
    monotone : bool
        If True modifies yp to preserve monoticity
    
    Returns
    -------
    yp : ndarray
        estimated slope
        
    References:
    -----------
    Wikipedia:  Monotone cubic interpolation
                Cubic Hermite spline

    '''
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)
    yp = np.zeros(y.shape, np.float_)

    
    dx = x[1:] - x[:-1]
    # Compute the slopes of the secant lines between successive points
    dydx = (y[1:] - y[:-1]) / dx
    
    method = method.lower()
    if method.startswith('parabola'):
        yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / (dx[1:] + dx[:-1])
        yp[0] = 2.0 * dydx[0] - yp[1]
        yp[-1] = 2.0 * dydx[-1] - yp[-2]
    else:
        # At the endpoints - use one-sided differences
        yp[0] = dydx[0]
        yp[-1] = dydx[-1]
        if method.startswith('secant'):
            # In the middle - use the average of the secants
            yp[1:-1] = (dydx[:-1] + dydx[1:]) / 2.0
        else: # Cardinal or Catmull-Rom method
            yp[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            if method.startswith('cardinal'):
                yp = (1-tension) * yp
    
    if monotone:
        # Special case: intervals where y[k] == y[k+1]    
        # Setting these slopes to zero guarantees the spline connecting
        # these points will be flat which preserves monotonicity
        ii, = (dydx == 0.0).nonzero()
        yp[ii] = 0.0
        yp[ii+1] = 0.0
        
        alpha = yp[:-1]/dydx
        beta  = yp[1:]/dydx
        dist  = alpha**2 + beta**2
        tau   = 3.0 / np.sqrt(dist)
        
        # To prevent overshoot or undershoot, restrict the position vector
        # (alpha, beta) to a circle of radius 3.  If (alpha**2 +  beta**2)>9,
        # then set m[k] = tau[k]alpha[k]delta[k] and m[k+1] =  tau[k]beta[b]delta[k]
        # where tau = 3/sqrt(alpha**2 + beta**2).
        
        # Find the indices that need adjustment
        indices_to_fix, = (dist > 9.0).nonzero() 
        for ii in indices_to_fix:
            yp[ii]   = tau[ii] * alpha[ii] * dydx[ii]
            yp[ii+1] = tau[ii] * beta[ii]  * dydx[ii]
    
    return yp

def slopes(x, y):
    """
    :func:`slopes` calculates the slope *y*'(*x*)

    The slope is estimated using the slope obtained from that of a
    parabola through any three consecutive points.

    This method should be superior to that described in the appendix
    of A CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russel
    W. Stineman (Creative Computing July 1980) in at least one aspect:

      Circles for interpolation demand a known aspect ratio between
      *x*- and *y*-values.  For many functions, however, the abscissa
      are given in different dimensions, so an aspect ratio is
      completely arbitrary.

    The parabola method gives very similar results to the circle
    method for most regular cases but behaves much better in special
    cases.

    Norbert Nemec, Institute of Theoretical Physics, University or
    Regensburg, April 2006 Norbert.Nemec at physik.uni-regensburg.de

    (inspired by a original implementation by Halldor Bjornsson,
    Icelandic Meteorological Office, March 2006 halldor at vedur.is)
    """
    # Cast key variables as float.
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)

    yp = np.zeros(y.shape, np.float_)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dydx = dy / dx
    yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / (dx[1:] + dx[:-1])
    yp[0] = 2.0 * dy[0] / dx[0] - yp[1]
    yp[-1] = 2.0 * dy[-1] / dx[-1] - yp[-2]
    return yp

class StinemanInterp(object):
    '''
    Returns the values of an interpolating function that runs through a set of points according to the algorithm of Stineman (1980).

    Parameters
    ---------    
    x,y : array-like
        coordinates of points defining the interpolating function.     
    yp : array-like
        slopes of the interpolating function at x. Optional: only given if they are known, else the argument is not used.
    method : string    
        method for computing the slope at the given points if the slope is not known. With method=
            "parabola" calculates the slopes from a parabola through every three points. 
    Notes
    -----
    The interpolation method is described in an article by Russell W. Stineman (1980)  
    
    According to Stineman, the interpolation procedure has "the following properties:
    
    If values of the ordinates of the specified points change monotonically, and the slopes of the line segments joining
    the points change monotonically, then the interpolating curve and its slope will change monotonically.
    If the slopes of the line segments joining the specified points change monotonically, then the slopes of the interpolating 
    curve will change monotonically. Suppose that the conditions in (1) or (2) are satisfied by a set of points, but a small 
    change in the ordinate or slope at one of the points will result conditions (1) or (2) being not longer satisfied. Then 
    making this small change in the ordinate or slope at a point will cause no more than a small change in the interpolating 
    curve." The method is based on rational interpolation with specially chosen rational functions to satisfy the above three 
    conditions.
    
    Slopes computed at the given points with the methods provided by the `StinemanInterp' function satisfy Stineman's requirements. 
    The original method suggested by Stineman (method="scaledstineman", the default, and "stineman") result in lower slopes near 
    abrupt steps or spikes in the point sequence, and therefore a smaller tendency for overshooting. The method based on a second
    degree polynomial (method="parabola") provides better approximation to smooth functions, but it results in in higher slopes 
    near abrupt steps or spikes and can lead to some overshooting where Stineman's method does not. Both methods lead to much 
    less tendency for `spurious' oscillations than traditional interplation methods based on polynomials, such as splines 
    (see the examples section).
    
    Stineman states that "The complete assurance that the procedure will never generate `wild' points makes it attractive as a 
    general purpose procedure".
    
    This interpolation method has been implemented in Matlab and R in addition to Python.
    
    Examples
    --------
    >>> import wafo.interpolate as wi
    >>> x = np.linspace(0,2*pi,20)
    >>> y = np.sin(x); yp = np.cos(x)
    >>> xi = np.linspace(0,2*pi,40);
    >>> yi = wi.StinemanInterp(x,y)(xi)
    >>> yi1 = wi.CubicHermiteSpline(x,y, yp)(xi)
    >>> yi2 = wi.Pchip(x,y, method='parabola')(xi)
    >>> plt.subplot(211)
    >>> plt.plot(x,y,'o',xi,yi,'r', xi,yi1, 'g', xi,yi1, 'b')
    >>> plt.subplot(212)
    >>> plt.plot(xi,np.abs(sin(xi)-yi), 'r', xi,  np.abs(sin(xi)-yi1), 'g', xi, np.abs(sin(xi)-yi2), 'b')
    
    References
    ----------
    Stineman, R. W. A Consistently Well Behaved Method of Interpolation. Creative Computing (1980), volume 6, number 7, p. 54-57.
    
    See Also
    --------
    slopes, Pchip
    '''
    def __init__(self, x,y,yp=None,method='parabola'):
        if yp is None:
            yp = slopes2(x, y, method)
        self.x = np.asarray(x, np.float_)
        self.y = np.asarray(y, np.float_)
        self.yp = np.asarray(yp, np.float_)
        
    def __call__(self, xi):
        xi = np.asarray(xi, np.float_)
        x = self.x
        y = self.y
        yp = self.yp
        # calculate linear slopes
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        s = dy / dx  #note length of s is N-1 so last element is #N-2
    
        # find the segment each xi is in
        # this line actually is the key to the efficiency of this implementation
        idx = np.searchsorted(x[1:-1], xi)
    
        # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
        # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] > x[-1]
    
        # the y-values that would come out from a linear interpolation:
        sidx = s.take(idx)
        xidx = x.take(idx)
        yidx = y.take(idx)
        xidxp1 = x.take(idx + 1)
        yo = yidx + sidx * (xi - xidx)
    
        # the difference that comes when using the slopes given in yp
        dy1 = (yp.take(idx) - sidx) * (xi - xidx)       # using the yp slope of the left point
        dy2 = (yp.take(idx + 1) - sidx) * (xi - xidxp1) # using the yp slope of the right point
    
        dy1dy2 = dy1 * dy2
        # The following is optimized for Python. The solution actually
        # does more calculations than necessary but exploiting the power
        # of numpy, this is far more efficient than coding a loop by hand
        # in Python
        dy1mdy2 = np.where(dy1dy2,dy1-dy2,np.inf)
        dy1pdy2 = np.where(dy1dy2,dy1+dy2,np.inf)
        yi = yo + dy1dy2 * np.choose(np.array(np.sign(dy1dy2), np.int32) + 1,
                                     ((2 * xi - xidx - xidxp1) / ((dy1mdy2) * (xidxp1 - xidx)),
                                      0.0,
                                      1 / (dy1pdy2)))
        return yi

        
def stineman_interp(xi, x, y, yp=None):
    """
    Given data vectors *x* and *y*, the slope vector *yp* and a new
    abscissa vector *xi*, the function :func:`stineman_interp` uses
    Stineman interpolation to calculate a vector *yi* corresponding to
    *xi*.

    Here's an example that generates a coarse sine curve, then
    interpolates over a finer abscissa::

      x = np.linspace(0,2*pi,20);  y = np.sin(x); yp = np.cos(x)
      xi = np.linspace(0,2*pi,40);
      yi = stineman_interp(xi,x,y,yp);
      plot(x,y,'o',xi,yi)

    The interpolation method is described in the article A
    CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russell
    W. Stineman. The article appeared in the July 1980 issue of
    Creative Computing with a note from the editor stating that while
    they were:

      not an academic journal but once in a while something serious
      and original comes in adding that this was
      "apparently a real solution" to a well known problem.

    For *yp* = *None*, the routine automatically determines the slopes
    using the :func:`slopes` routine.

    *x* is assumed to be sorted in increasing order.

    For values ``xi[j] < x[0]`` or ``xi[j] > x[-1]``, the routine
    tries an extrapolation.  The relevance of the data obtained from
    this, of course, is questionable...

    Original implementation by Halldor Bjornsson, Icelandic
    Meteorolocial Office, March 2006 halldor at vedur.is

    Completely reworked and optimized for Python by Norbert Nemec,
    Institute of Theoretical Physics, University or Regensburg, April
    2006 Norbert.Nemec at physik.uni-regensburg.de
    """

    # Cast key variables as float.
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)
    assert x.shape == y.shape
    #N = len(y)

    if yp is None:
        yp = slopes(x, y)
    else:
        yp = np.asarray(yp, np.float_)

    xi = np.asarray(xi, np.float_)
    #yi = np.zeros(xi.shape, np.float_)

    # calculate linear slopes
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = dy / dx  #note length of s is N-1 so last element is #N-2

    # find the segment each xi is in
    # this line actually is the key to the efficiency of this implementation
    idx = np.searchsorted(x[1:-1], xi)

    # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
    # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] > x[-1]

    # the y-values that would come out from a linear interpolation:
    sidx = s.take(idx)
    xidx = x.take(idx)
    yidx = y.take(idx)
    xidxp1 = x.take(idx + 1)
    yo = yidx + sidx * (xi - xidx)

    # the difference that comes when using the slopes given in yp
    dy1 = (yp.take(idx) - sidx) * (xi - xidx)       # using the yp slope of the left point
    dy2 = (yp.take(idx + 1) - sidx) * (xi - xidxp1) # using the yp slope of the right point

    dy1dy2 = dy1 * dy2
    # The following is optimized for Python. The solution actually
    # does more calculations than necessary but exploiting the power
    # of numpy, this is far more efficient than coding a loop by hand
    # in Python
    dy1mdy2 = np.where(dy1dy2,dy1-dy2,np.inf)
    dy1pdy2 = np.where(dy1dy2,dy1+dy2,np.inf)
    yi = yo + dy1dy2 * np.choose(np.array(np.sign(dy1dy2), np.int32) + 1,
                                 ((2 * xi - xidx - xidxp1) / ((dy1mdy2) * (xidxp1 - xidx)),
                                  0.0,
                                  1 / (dy1pdy2)))
    return yi

class CubicHermiteSpline(PiecewisePolynomial):
    '''
    Piecewise Cubic Hermite Interpolation using Catmull-Rom
    method for computing the slopes.
    '''
    def __init__(self, x, y, yp=None, method='Catmull-Rom'):
        if yp is None:
            yp = slopes2(x, y, method, monotone=False)
        super(CubicHermiteSpline, self).__init__(x, list(zip(y,yp)), orders=3)

class Pchip(PiecewisePolynomial):
    """PCHIP 1-d monotonic cubic interpolation

    Description
    -----------
    x and y are arrays of values used to approximate some function f:
       y = f(x)
    This class factory function returns a callable class whose __call__ method
    uses monotonic cubic, interpolation to find the value of new points.

    Parameters
    ----------
    x : array
        A 1D array of monotonically increasing real values.  x cannot
        include duplicate values (otherwise f is overspecified)
    y : array
        A 1-D array of real values.  y's length along the interpolation
        axis must be equal to the length of x.
    yp : array
        slopes of the interpolating function at x. Optional: only given if they are known, else the argument is not used.
    method : string    
        method for computing the slope at the given points if the slope is not known. With method=
            "parabola" calculates the slopes from a parabola through every three points. 

    Assumes x is sorted in monotonic order (e.g. x[1] > x[0])
    
    Example
    -------
    >>> import wafo.interpolate as wi
    # Create a step function (will demonstrate monotonicity)
    >>> x = np.arange(7.0) - 3.0
    >>> y = np.array([-1.0, -1,-1,0,1,1,1])
    
    # Interpolate using monotonic piecewise Hermite cubic spline
    >>> xvec = np.arange(599.)/100. - 3.0
    >>> yvec = wi.Pchip(x, y)(xvec)
    
    # Call the Scipy cubic spline interpolator
    >>> from scipy.interpolate import interpolate
    >>> function = interpolate.interp1d(x, y, kind='cubic')
    >>> yvec1 = function(xvec)
    
    # Non-montonic cubic Hermite spline interpolator using
    # Catmul-Rom method for computing slopes...
    >>> yvec2 = wi.CubicHermiteSpline(x,y)(xvec)
    
    >>> yvec3 = wi.StinemanInterp(x, y)(xvec)
    
    # Plot the results
    >>> plt.plot(x,    y,     'ro')
    >>> plt.plot(xvec, yvec,  'b')
    >>> plt.plot(xvec, yvec1, 'k')
    >>> plt.plot(xvec, yvec2, 'g')
    >>> plt.plot(xvec, yvec3, 'm')
    >>> plt.title("pchip() step function test")

    >>> plt.xlabel("X")
    >>> plt.ylabel("Y")
    >>> plt.title("Comparing pypchip() vs. Scipy interp1d() vs. non-monotonic CHS")
    >>> legends = ["Data", "pypchip()", "interp1d","CHS", 'SI']
    >>> plt.legend(legends, loc="upper left")
    >>> plt.show()

    """
    def __init__(self, x, y, yp=None, method='secant'):
        if yp is None:
            yp = slopes2(x, y, method=method, monotone=True)
        super(Pchip, self).__init__(x, list(zip(y,yp)), orders=3)
                
def test_smoothing_spline():
    x = np.linspace(0, 2 * pi + pi / 4, 20) 
    y = np.sin(x) #+ np.random.randn(x.size)
    pp = SmoothSpline(x, y, p=1)
    x1 = np.linspace(-1, 2 * pi + pi / 4 + 1, 20) 
    y1 = pp(x1)
    pp1 = pp.derivative()
    pp0 = pp1.integrate()
    dy1 = pp1(x1)
    y01 = pp0(x1)
    import pylab as plb
    plb.plot(x, y, x1, y1, '.', x1, dy1, 'ro', x1, y01, 'r-')
    plb.show()
    

__all__ = __all__ + np.lib.polynomial.__all__
__all__ = __all__ + ['pade', 'padefit', 'polyreloc', 'polyrescl', 'polytrim', 'poly2hstr', 'poly2str',
    'polyshift', 'polyishift', 'map_from_intervall', 'map_to_intervall',
    'cheb2poly', 'chebextr', 'chebroot', 'chebpoly', 'chebfit', 'chebval',
    'chebder', 'chebint', 'Cheb1d', 'dct', 'idct']

def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : {array_like, poly1d}
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : {None, list of `m` scalars, scalar}, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method

    Examples
    --------
    The defining property of the antiderivative:

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
    poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ])
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3.        ,  5.        ,  3.        ])

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = np.zeros(m, float)
    k = np.atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * np.ones(m, float)
    if len(k) < m:
        raise ValueError("k must be a scalar or a rank-1 array of length 1 or >m.")
    truepoly = isinstance(p, poly1d)
    p = asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:  
        ix = np.arange(len(p), 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
            pieces = p.shape[-1]
            k0 = k[0] * np.ones((1, pieces), dtype=int)
        else:
            k0 = [k[0]]
        y = np.concatenate((p.__truediv__(ix), k0), axis=0)

        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val

def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([ 0.])

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    truepoly = isinstance(p, poly1d)
    p = asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        n = len(p) - 1
        ix = np.arange(n, 0, -1)
        if p.ndim > 1:
            ix = ix[..., newaxis]
        y = ix * p[:-1]
        val = polyder(y, m - 1)
        if truepoly:
            return poly1d(val)
        return val

def unfinished_polydeg(x,y):
    '''
    Return optimal degree for polynomial fitting.
    N = POLYDEG(X,Y) finds the optimal degree for polynomial fitting
    according to the Akaike's information criterion.

    Assuming that you want to find the degree N of a polynomial that fits
    the data Y(X) best in a least-squares sense, the Akaike's information
    criterion is defined by:
        2*(N+1) + n*(log(2*pi*RSS/n)+1)
    where n is the number of points and RSS is the residual sum of squares.
    The optimal degree N is defined here as that which minimizes <a
    href="matlab:web('http://en.wikipedia.org/wiki/Akaike_Information_Criterion')">AIC</a>.

    Notes:
    -----
    If the number of data is small, POLYDEG may tend to return:
    N = (number of points)-1.

    ORTHOFIT is more appropriate than POLYFIT for polynomial fitting with
    relatively high degrees.

    Examples:
    --------
    load census
    n = polydeg(cdate,pop)

    x = np.linspace(0,10,300);
    y = np.sin(x.^3/100).^2 + 0.05*randn(size(x));
    n = polydeg(x,y)
    ys = orthofit(x,y,n);
    plot(x,y,'.',x,ys,'k')

    Damien Garcia, 02/2008, revised 01/2010

    See also POLYFIT, ORTHOFIT.
    '''
    x, y = np.atleast_1d(x, y)

    N = len(x)


    ## Search the optimal degree minimizing the Akaike's information criterion
    # ---
    #  y(x) are fitted in a least-squares sense using a polynomial of degree n
    #  developed in a series of orthogonal polynomials.


    p = y.mean()
    ys = np.ones((N,))*p
    AIC = 2+N*(np.log(2*pi*((ys-y)**2).sum()/N)+1)+ 4/(N-2)  #correction for small sample sizes

    p = np.zeros((2,2))
    p[1,0] = x.mean()
    PL = np.ones((2,N))
    PL[1] = x-p[1,0]

    n = 1
    nit = 0

    # While-loop is stopped when a minimum is detected. 3 more steps are
    # required to take AIC noise into account and to ensure that this minimum
    # is a (likely) global minimum.

    while nit<3:
        if n>0:
            p[0,n] = sum(x*PL[:,n]**2)/sum(PL[:,n]**2)
            p[1,n] = sum(x*PL[:,n-1]*PL[:,n])/sum(PL[:,n-1]**2)
            PL[:,n] = (x-p[0,n+1])*PL[:,n]-p[1,n+1]*PL[:,n-1]
        #end

        tmp = sum(y*PL)/sum(PL**2)
        ys = sum(PL*tmp,axis=-1)

        # -- Akaike's Information Criterion
        aic = 2*(n+1)+N*(np.log(2*pi*sum((ys-y.ravel()**2)/N)+1)) + 2*(n+1)*(n+2)/(N-n-2)


        if aic>=AIC:
            nit += 1
        else:
            nit = 0
            AIC = aic

        n = n+1

        if n>=N:
            break
    n = n-nit-1

    return n

def unfinished_orthofit(x,y,n):
    '''
    ORTHOFIT Fit polynomial to data.
    YS = ORTHOFIT(X,Y,N) smooths/fits data Y(X) in a least-squares sense
    using a polynomial of degree N and returns the smoothed data YS.

    [YS,YI] = ORTHOFIT(X,Y,N,XI) also returns the values YI of the fitting
    polynomial at the points of a different XI array.

    YI = ORTHOFIT(X,Y,N,XI) returns only the values YI of the fitting
    polynomial at the points of the XI array.

    [YS,P] = ORTHOFIT(X,Y,N) returns the polynomial coefficients P for use
    with POLYVAL.

    Notes:
    -----
    ORTHOFIT smooths/fits data using a polynomial of degree N developed in
    a sequence of orthogonal polynomials. ORTHOFIT is more appropriate than
    POLYFIT for polynomial fitting and smoothing since this method does not
    involve any matrix linear system but a simple recursive procedure.
    Degrees much higher than 30 could be used with orthogonal polynomials,
    whereas badly conditioned matrices may appear with a classical
    polynomial fitting of degree typically higher than 10.

    To avoid using unnecessarily high degrees, you may let the function
    POLYDEG choose it for you. POLYDEG finds an optimal polynomial degree
    according to the Akaike's information criterion (available <a
    href="matlab:web('http://www.biomecardio.com/matlab/polydeg.html')">here</a>).

    Example:
    -------
    x = np.linspace(0,10,300);
    y = np.sin(x.^3/100).^2 + 0.05*randn(size(x));
    ys = orthofit(x,y,25);
    plot(x,y,'.',x,ys,'k')
     try POLYFIT for comparison...

     Automatic degree determination with <a
    href="matlab:web('http://www.biomecardio.com/matlab/polydeg.html')">POLYDEG</a>
    n = polydeg(x,y);
    ys = orthofit(x,y,n);
    plot(x,y,'.',x,ys,'k')

    Reference: Methodes de calcul numerique 2. JP Nougier. Hermes Science
    Publications, 2001. Section 4.7 pp 116-121

    Damien Garcia, 09/2007, revised 01/2010

    See also POLYDEG, POLYFIT, POLYVAL.
    '''
    x, y = np.atleast_1d(x,y)
    # Particular case: n=0
    if n==0:
        p = y.mean()
        ys = np.ones(y.shape)*p   
        return p, ys

    # Reshape
    x = x.ravel()
    siz0 = y.shape
    y = y.ravel()

    # Coefficients of the orthogonal polynomials
    p = np.zeros((3,n+1))
    p[1,1] = x.mean()

    N = len(x)
    PL = np.ones((N,n+1))
    PL[:,1] = x-p[1,1]

    for i in range(2,n+1):
        p[1,i] = sum(x*PL[:,i-1]**2)/sum(PL[:,i-1]**2)
        p[2,i] = sum(x*PL[:,i-2]*PL[:,i-1])/sum(PL[:,i-2]**2)
        PL[:,i] = (x-p[1,i])*PL[:,i-1]-p[2,i]*PL[:,i-2]
    #end
    p[0,:] = sum(PL*y)/sum(PL**2);

    # ys = smoothed y
    #ys = sum(PL*p(0,:) axis=1)
    #ys.shape = siz0


    # Coefficients of the polynomial in its final form

    yi = np.zeros((n+1,n+1))
    yi[0,n] = 1
    yi[1,n-1:n+1] = 1 -p[1,1]
    for i in range(2, n+1):
        yi[i,:] = np.hstack((yi[i-1,1:], 0))-p[1,i]*yi[i-1,:]-p[2,i]*yi[i-2,:];

    p = sum(p[0,:]*yi, axis=0)
    return p

def polyreloc(p, x, y=0.0):
    """
    Relocate polynomial

    The polynomial `p` is relocated by "moving" it `x`
    units along the x-axis and `y` units along the y-axis.
    So the polynomial `r` is relative to the point (x,y) as
    the polynomial `p` is relative to the point (0,0).

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients to relocate.
        (Polynomial coefficients are in decreasing order.)
    x : scalar
        distance to relocate P along x-axis
    y : scalar
        distance to relocate P along y-axis (default 0)

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of relocated polynomial coefficients.

    See also
    --------
    polyrescl

    Example
    -------
    >>> import numpy as np
    >>> p = np.arange(6); p.shape = (2,-1)
    >>> np.polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(p,1)
    array([3, 5, 7])
    >>> r = polyreloc(p,-1) # move to the left along x-axis
    >>> np.polyval(r,-1)    # = polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(r,0)     # = polyval(p,1)
    array([3, 5, 7])
    """

    truepoly = isinstance(p, poly1d)
    r = np.atleast_1d(p).copy()
    n = r.shape[0]

    # Relocate polynomial using Horner's algorithm
    for ii in range(n, 1, -1):
        for i in range(1, ii):
            r[i] = r[i] - x * r[i - 1]
    r[-1] = r[-1] + y
    if r.ndim > 1 and r.shape[-1] == 1:
        r.shape = (r.size,)
    if truepoly:
        r = poly1d(r)
    return r

def polyrescl(p, x, y=1.0):
    """
    Rescale polynomial.

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients to rescale.
        (Polynomial coefficients are in decreasing order.)
    x,y : scalars
        defining the factors to rescale the polynomial `p`  in
        x-direction and y-direction, respectively.

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of rescaled polynomial coefficients.

    See also
    --------
    polyreloc

    Example
    -------
    >>> import numpy as np
    >>> p = np.arange(6); p.shape = (2,-1)
    >>> np.polyval(p,0)
    array([3, 4, 5])
    >>> np.polyval(p,1)
    array([3, 5, 7])
    >>> r = polyrescl(p,2)  # scale by 2 along x-axis
    >>> np.polyval(r,0)     # = polyval(p,0)
    array([ 3.,  4.,  5.])
    >>> np.polyval(r,2)     # = polyval(p,1)
    array([ 3.,  5.,  7.])
    """

    truepoly = isinstance(p, poly1d)
    r = np.atleast_1d(p)
    n = r.shape[0]

    xscale = (float(x) ** np.arange(1 - n , 1))
    if r.ndim == 1:
        q = y * r * xscale
    else:
        q = y * r * xscale[:, newaxis]
    if truepoly:
        q = poly1d(q)
    return q

def polytrim(p):
    """
    Trim polynomial by stripping off leading zeros.

    Parameters
    ----------
    p : array-like, poly1d
        vector or matrix of column vectors of polynomial coefficients in
        decreasing order.

    Returns
    -------
    r : ndarray, poly1d
        vector/matrix/poly1d of trimmed polynomial coefficients.

    Example
    -------
    >>> p = [0,1,2]
    >>> polytrim(p)
    array([1, 2])
    >>> p1 = [[0,0],[1,2],[3,4]]
    >>> polytrim(p1)
    array([[1, 2],
           [3, 4]])
    """

    truepoly = isinstance(p, poly1d)
    if truepoly:
        return p
    else:
        r = np.atleast_1d(p).copy()
        # Remove leading zeros
        is_not_lead_zeros = np.logical_or.accumulate(r != 0, axis=0)
        if r.ndim == 1:
            r = r[is_not_lead_zeros]
        else:
            is_not_lead_zeros = any(is_not_lead_zeros, axis=1)
            r = r[is_not_lead_zeros, :]
        return r

def poly2hstr(p, variable='x'):
    """
    Return polynomial as a Horner represented string.

    Parameters
    ----------
    p : array-like poly1d
        vector of polynomial coefficients in decreasing order.
    variable : string
        display character for variable

    Returns
    -------
    p_str : string
        consisting of the polynomial coefficients in the vector P multiplied
        by powers of the given `variable`.

    Examples
    --------
    >>> poly2hstr([1, 1, 2], 's' )
    '(s + 1)*s + 2'

    See also
    --------
    poly2str
    """
    var = variable

    coefs = polytrim(atleast_1d(p))
    order = len(coefs) - 1 # Order of polynomial.
    s = ''    # Initialize output string.
    ix = 1;
    for expon in range(order, -1, -1):
        coef = coefs[order - expon]
        #% There is no point in adding a zero term (except if it's the only
        #% term, but we'll take care of that later).
        if coef == 0:
            ix += 1
        else:
        #% Append exponent if necessary.
            if ix > 1:
                exponstr = '%.0f' % ix
                s = '%s**%s' % (s, exponstr);
                ix = 1
            #% Is it the first term?
            isfirst = s == ''

            # We need the coefficient only if it is different from 1 or -1 or
            # when it is the constant term.
            needcoef = ((abs(coef) != 1) | (expon == 0) & isfirst) | 1 - isfirst

            # We need the variable except in the constant term.
            needvar = (expon != 0)

            #% Add sign, but we don't need a leading plus-sign.
            if isfirst:
                if coef < 0:
                    s = '-'  #        % Unary minus.
            else:
                if coef < 0:
                    s = '%s - ' % s  #    % Binary minus (subtraction).
                else:
                    s = '%s + ' % s  #  % Binary plus (addition).


            #% Append the coefficient if it is different from one or when it is
            #% the constant term.
            if needcoef:
                coefstr = '%.20g' % np.abs(coef)
                s = '%s%s' % (s, coefstr)

            #% Append variable if necessary.
            if needvar:
                #% Append a multiplication sign if necessary.
                if needcoef:
                    if 1 - isfirst:
                        s = '(%s)' % s
                    s = '%s*' % s
                s = '%s%s' % (s, var)

    #% Now treat the special case where the polynomial is zero.
    if s == '':
        s = '0'
    return s

def poly2str(p, variable='x'):
    """
    Return polynomial as a string.

    Parameters
    ----------
    p : array-like poly1d
        vector of polynomial coefficients in decreasing order.
    variable : string
        display character for variable

    Returns
    -------
    p_str : string
        consisting of the polynomial coefficients in the vector P multiplied
        by powers of the given `variable`.

    See also
    --------
    poly2hstr

    Examples
    --------
    >>> poly2str([1, 1, 2], 's' )
    's**2 + s + 2'
    """
    thestr = "0"
    var = variable

    # Remove leading zeros
    coeffs = polytrim(atleast_1d(p))

    N = len(coeffs) - 1

    for k in range(len(coeffs)):
        coefstr = '%.4g' % np.abs(coeffs[k])
        if coefstr[-4:] == '0000':
            coefstr = coefstr[:-5]
        power = (N - k)
        if power == 0:
            if coefstr != '0':
                newstr = '%s' % (coefstr,)
            else:
                if k == 0:
                    newstr = '0'
                else:
                    newstr = ''
        elif power == 1:
            if coefstr == '0':
                newstr = ''
            elif coefstr == 'b' or coefstr == '1':
                newstr = var
            else:
                newstr = '%s*%s' % (coefstr, var)
        else:
            if coefstr == '0':
                newstr = ''
            elif coefstr == 'b' or coefstr == '1':
                newstr = '%s**%d' % (var, power,)
            else:
                newstr = '%s*%s**%d' % (coefstr, var, power)

        if k > 0:
            if newstr != '':
                if coeffs[k] < 0:
                    thestr = "%s - %s" % (thestr, newstr)
                else:
                    thestr = "%s + %s" % (thestr, newstr)
        elif (k == 0) and (newstr != '') and (coeffs[k] < 0):
            thestr = "-%s" % (newstr,)
        else:
            thestr = newstr
    return thestr

def polyshift(py, a= -1, b=1):
    """
    Polynomial coefficient shift

    Polyshift shift the polynomial coefficients by a variable shift:

    Y = 2*(X-.5*(b+a))/(b-a)

    i.e., the interval -1 <= Y <= 1 is mapped to the interval a <= X <= b

    Parameters
    ----------
    py : array-like
        polynomial coefficients for the variable y.
    a,b : scalars
        lower and upper limit.

    Returns
    -------
    px : ndarray
        polynomial coefficients for the variable x.

    See also
    --------
    polyishift

    Example
    -------
    >>> py = [1, 0]
    >>> px = polyshift(py,0,5)
    >>> polyval(px,[0, 2.5, 5])  #% This is the same as the line below
    array([-1.,  0.,  1.])
    >>> polyval(py,[-1, 0, 1 ])
    array([-1,  0,  1])
    """

    if (a == -1) & (b == 1):
        return py
    L = b - a
    return polyishift(py, -(2. + b + a) / L, (2. - b - a) / L)

def polyishift(px, a= -1, b=1):
    """
    Inverse polynomial coefficient shift

    Polyishift does the inverse of Polyshift,
    shift the polynomial coefficients by a variable shift:

    Y = 2*(X-.5*(b+a)/(b-a)

    i.e., the interval a <= X <= b is mapped to the interval -1 <= Y <= 1

    Parameters
    ----------
    px : array-like
        polynomial coefficients for the variable x.
    a,b : scalars
        lower and upper limit.

    Returns
    -------
    py : ndarray
        polynomial coefficients for the variable y.

    See also
    --------
    polyishift

    Example
    -------
    >>> px = [1, 0]
    >>> py = polyishift(px,0,5);
    >>> polyval(px,[0, 2.5, 5])  #% This is the same as the line below
    array([ 0. ,  2.5,  5. ])
    >>> polyval(py,[-1, 0, 1])
    array([ 0. ,  2.5,  5. ])
    """
    if (a == -1) & (b == 1):
        return px
    L = b - a
    xscale = 2. / L
    xloc = -float(a + b) / L
    return polyreloc(polyrescl(px, xscale), xloc)

def map_from_interval(x, a, b) :
    """F(x), where F: [a,b] -> [-1,1]."""
    return (x - (b + a) / 2.0) * (2.0 / (b - a))

def map_to_interval(x, a, b) :
    """F(x), where F: [-1,1] -> [a,b]."""
    return (x * (b - a) + (b + a)) / 2.0

def poly2cheb(p, a= -1, b=1):
    """
    Convert polynomial coefficients into Chebyshev coefficients

    Parameters
    ----------
    p : array-like
        polynomial coefficients
    a,b : real scalars
        lower and upper limits (Default -1,1)

    Returns
    -------
    ck : ndarray
        Chebychef coefficients

    POLY2CHEB do the inverse of CHEB2POLY: given a vector of polynomial
    coefficients AK, returns an equivalent vector of Chebyshev
    coefficients CK.

    This is useful for economization of power series.
    The steps for doing so:
    1. Convert polynomial coefficients to Chebychev coefficients, CK.
    2. Truncate the CK series to a smaller number of terms, using the
    coefficient of the first neglected Chebychev polynomial as an error
    estimate.
    3 Convert back to a polynomial by CHEB2POLY

    See also
    --------
    cheb2poly
    chebval
    chebfit

    Examples
    --------
    >>> import numpy as np
    >>> p = np.arange(5)
    >>> ck = poly2cheb(p)
    >>> cheb2poly(ck)
    array([ 1.,  2.,  3.,  4.])

    Reference
    ---------
    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 184-194
    """
    f = poly1d(p)
    n = len(f.coeffs)
    return chebfit(f, n, a, b)

def cheb2poly(ck, a= -1, b=1):
    """
    Converts Chebyshev coefficients to polynomial coefficients

    Parameters
    ----------
    ck : array-like
        Chebychef coefficients
    a,b : real, scalars
        lower and upper limits (Default -1,1)

    Returns
    -------
    p : ndarray
        polynomial coefficients

    It is not advised to do this for len(ck)>10 due to numerical cancellations.

    See also
    --------
    chebval
    chebfit

    Examples
    --------
    >>> import numpy as np
    >>> p = np.arange(5)
    >>> ck = poly2cheb(p)
    >>> cheb2poly(ck)
    array([ 1.,  2.,  3.,  4.])


    References
    ----------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    http://en.wikipedia.org/wiki/Chebyshev_form
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    """

    n = len(ck)

    b_Nmi = np.zeros(1)
    b_Nmip1 = np.zeros(1)
    y = r_[2 / (b - a), -(a + b) / (b - a)]
    y2 = 2. * y

    # Clenshaw recurence
    for ix in range(n - 1):
        tmp = b_Nmi
        b_Nmi = polymul(y2, b_Nmi) # polynomial multiplication
        nb = len(b_Nmip1)
        b_Nmip1[-1] = b_Nmip1[-1] - ck[ix]
        b_Nmi[-nb::] = b_Nmi[-nb::] - b_Nmip1
        b_Nmip1 = tmp

    p = polymul(y, b_Nmi) # polynomial multiplication
    nb = len(b_Nmip1)
    b_Nmip1[-1] = b_Nmip1[-1] - ck[n - 1]
    p[-nb::] = p[-nb::] - b_Nmip1
    return polytrim(p)

def chebextr(n):
    """
    Return roots of derivative of Chebychev polynomial of the first kind.

    All local extreme values of the polynomial are either -1 or 1. So,
    CHEBPOLY( N, CHEBEXTR(N) ) ) return the same as (-1).^(N:-1:0)
    except for the numerical noise in the former.

    Because the extreme values of Chebychev polynomials of the first
    kind are either -1 or 1, their roots are often used as starting
    values for the nodes in minimax approximations.


    Parameters
    ----------
    n : scalar, integer
        degree of Chebychev polynomial.

    Examples
    --------
    >>> x = chebextr(4)
    >>> chebpoly(4,x)
    array([ 1., -1.,  1., -1.,  1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    return - np.cos((pi * np.arange(n + 1)) / n);

def chebroot(n, kind=1):
    """
    Return roots of Chebychev polynomial of the first or second kind.

    The roots of the Chebychev polynomial of the first kind form a particularly
    good set of nodes for polynomial interpolation because the resulting
    interpolation polynomial minimizes the problem of Runge's phenomenon.

    Parameters
    ----------
    n : scalar, integer
        degree of Chebychev polynomial.
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)

    Examples
    --------
    >>> import numpy as np
    >>> x = chebroot(3)
    >>> np.abs(chebpoly(3,x))<1e-15
    array([ True,  True,  True], dtype=bool)
    >>> chebpoly(3)
    array([ 4.,  0., -3.,  0.])
    >>> x2 = chebroot(4,kind=2)
    >>> np.abs(chebpoly(4,x2,kind=2))<1e-15
    array([ True,  True,  True,  True], dtype=bool)
    >>> chebpoly(4,kind=2)
    array([ 16.,   0., -12.,   0.,   1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if kind not in (1, 2):
        raise ValueError('kind must be 1 or 2')
    return - np.cos(pi * (arange(n) + 0.5 * kind) / (n + kind - 1));


def chebpoly(n, x=None, kind=1):
    """
    Return Chebyshev polynomial of the first or second kind.

    These polynomials are orthogonal on the interval [-1,1], with
    respect to the weight function w(x) = (1-x^2)^(-1/2+kind-1).

    chebpoly(n) returns the coefficients of the Chebychev polynomial of degree N.
    chebpoly(n,x) returns the Chebychev polynomial of degree N evaluated in X.

    Parameters
    ----------
    n : integer, scalar
        degree of Chebychev polynomial.
    x : array-like, optional
        evaluation points
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)

    Returns
    -------
    p : ndarray
        polynomial coefficients if x is None.
        Chebyshev polynomial evaluated at x otherwise

    Examples
    --------
    >>> import numpy as np
    >>> x = chebroot(3)
    >>> np.abs(chebpoly(3,x))<1e-15
    array([ True,  True,  True], dtype=bool)
    >>> chebpoly(3)
    array([ 4.,  0., -3.,  0.])
    >>> x2 = chebroot(4,kind=2)
    >>> np.abs(chebpoly(4,x2,kind=2))<1e-15
    array([ True,  True,  True,  True], dtype=bool)
    >>> chebpoly(4,kind=2)
    array([ 16.,   0., -12.,   0.,   1.])


    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if x is None:  # Calculate coefficients.
        if n == 0:
            p = np.ones(1)
        else:
            p = np.round(pow(2, n - 2 + kind) * poly(chebroot(n, kind=kind)))
            p[1::2] = 0;
        return p
    else: #   Evaluate polynomial in chebychev form
        ck = np.zeros(n + 1)
        ck[0] = 1.
        return _chebval(atleast_1d(x), ck, kind=kind)

def chebfit(fun, n=10, a= -1, b=1, trace=False):
    """
    Computes the Chebyshevs coefficients

    so that f(x) can be approximated by:

                  n-1
           f(x) = sum ck*Tk(x)
                  k=0

    where Tk is the k'th Chebyshev polynomial of the first kind.

    Parameters
    ----------
    fun : callable
        function to approximate
    n : integer, scalar, optional
        number of base points (abscissas). Default n=10 (maximum 50)
    a,b : real, scalars, optional
        integration limits

    Returns
    -------
    ck : ndarray
        polynomial coefficients in Chebychev form.

    Examples
    --------
    Fit np.exp(x)

    >>> import pylab as pb
    >>> a = 0; b = 2
    >>> ck = chebfit(pb.exp,7,a,b);
    >>> x = pb.linspace(0,4);
    >>> h=pb.plot(x,pb.exp(x),'r',x,chebval(x,ck,a,b),'g.')
    >>> x1 = chebroot(9)*(b-a)/2+(b+a)/2
    >>> ck1 = chebfit(pb.exp(x1))
    >>> h=pb.plot(x,pb.exp(x),'r',x,chebval(x,ck1,a,b),'g.')

    >>> pb.close()

    See also
    --------
    chebval

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://mathworld.wolfram.com/ChebyshevApproximationFormula.html

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

    if (n > 50):
        warnings.warn('CHEBFIT should only be used for n<50')

    if hasattr(fun, '__call__'):
        x = map_to_interval(chebroot(n), a, b)
        f = fun(x);
        if trace:
            plb.plot(x, f, '+')
    else:
        f = fun
        n = len(f)
        #raise ValueError('Function must be callable!')
    #                     N-1
    #       c(k) = (2/N) sum w(n) f(n)*cos(pi*k*(2n+1)/(2N)), 0 <= k < N.
    #                    n=0
    #
    # w(0) = 0.5, w(n)=1 for n>0
    ck = dct(f[::-1]) / n
    ck[0] = ck[0] / 2.
    return ck[::-1]

def dct(x, n=None):
    """
    Discrete Cosine Transform

                      N-1
           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                      n=0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """

    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = np.hstack((x, np.zeros(n_shape)))
    else:
        xx = x[..., :n]

    real_x = all(isreal(xx))
    if (real_x and (remainder(n, 2) == 0)):
        xp = 2 * fft(hstack((xx[..., ::2], xx[..., ::-2])))
    else:
        xp = fft(hstack((xx, xx[..., ::-1])))
        xp = xp[..., :n]

    w = np.exp(-1j * np.arange(n) * pi / (2 * n))

    y = xp * w

    if real_x:
        return y.real
    else:
        return y

def idct(x, n=None):
    """
    Inverse Discrete Cosine Transform

                       N-1
           x[k] = 1/N sum w[n]*y[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                       n=0

           w(0) = 1/2
           w(n) = 1 for n>0

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.abs(x-idct(dct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> np.abs(x-dct(idct(x)))<1e-14
    array([ True,  True,  True,  True,  True], dtype=bool)

    Reference
    ---------
    http://en.wikipedia.org/wiki/Discrete_cosine_transform
    http://users.ece.utexas.edu/~bevans/courses/ee381k/lectures/
    """


    x = np.atleast_1d(x)

    if n is None:
        n = x.shape[-1]

    w = np.exp(1j * np.arange(n) * pi / (2 * n))

    if x.shape[-1] < n:
        n_shape = x.shape[:-1] + (n - x.shape[-1],)
        xx = np.hstack((x, np.zeros(n_shape))) * w
    else:
        xx = x[..., :n] * w

    real_x = all(isreal(x))
    if (real_x and (remainder(n, 2) == 0)):
        xx[..., 0] = xx[..., 0] * 0.5
        yp = ifft(xx)
        y = np.zeros(xx.shape, dtype=complex)
        y[..., ::2] = yp[..., :n / 2]
        y[..., ::-2] = yp[..., n / 2::]
    else:
        yp = ifft(hstack((xx, np.zeros_like(xx[..., 0]), conj(xx[..., :0:-1]))))
        y = yp[..., :n]

    if real_x:
        return y.real
    else:
        return y

def _chebval(x, ck, kind=1):
    """
    Evaluate polynomial in Chebyshev form.

    A polynomial of degree N in Chebyshev form is a polynomial p(x) of the form:

                 N
        p(x) =  sum ck*Tk(x)
                k=0
    or
                 N
        p(x) =  sum ck*Uk(x)
                k=0

    where Tk and Uk are the k'th Chebyshev polynomial of the first and second
    kind, respectively.

    References
    ----------
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
    """
    n = len(ck)
    b_Nmi = np.zeros(x.shape) # b_(N-i)
    b_Nmip1 = b_Nmi.copy()    # b_(N-i+1)
    x2 = 2 * x
    # Clenshaw reccurence
    for ix in range(n - 1):
        tmp = b_Nmi
        b_Nmi = x2 * b_Nmi - b_Nmip1 + ck[ix]
        b_Nmip1 = tmp
    return kind * x * b_Nmi - b_Nmip1 + ck[n - 1]


def chebval(x, ck, a= -1, b=1, kind=1, fill=None):
    """
    Evaluate polynomial in Chebyshev form at X

    A polynomial of degree N in Chebyshev form is a polynomial p(x) of the form:

             N
    p(x) =  sum ck*Tk(x)
            k=0

    where Tk is the k'th Chebyshev polynomial of the first or second kind.

    Paramaters
    ----------
    x : array-like
        points to evaluate
    ck : array-like
        polynomial coefficients in Chebyshev form ordered from highest degree to zero
    a,b : real, scalars, optional
        limits for polynomial (Default -1,1)
    kind: 1 or 2, optional
        kind of Chebychev polynomial (default 1)
    fill : scalar, optional
        If provided, define value to return for `x < a` or `b < x`.

    Examples
    --------
    Plot Chebychev polynomial of the first kind and order 4:
    >>> import pylab as pb
    >>> x = pb.linspace(-1,1)
    >>> ck = pb.zeros(5); ck[-1]=1
    >>> h = pb.plot(x,chebval(x,ck),x,chebpoly(4,x),'.')
    >>> pb.close()

    Fit exponential function:
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4);
    >>> h=pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x))
    >>> pb.close()

    See also
    --------
    chebfit

    References
    ----------
    http://en.wikipedia.org/wiki/Clenshaw_algorithm
    http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
    """

    y = map_from_interval(atleast_1d(x), a, b)
    if fill is None:
        f = _chebval(y, ck, kind=kind)
    else:
        cond = (abs(y) <= 1)
        f = np.where(cond, 0, fill)
        if any(cond):
            yk = np.extract(cond, y)
            f[cond] = _chebval(yk, ck, kind=kind)
    return f


def chebder(ck, a= -1, b=1):
    """
    Differentiate Chebyshev polynomial

    Parameters
    ----------
    ck : array-like
        polynomial coefficients in Chebyshev form of function to differentiate
    a,b : real, scalars
        limits for polynomial(Default -1,1)

    Return
    ------
    cder : ndarray
        polynomial coefficients in Chebyshev form of the derivative

    Examples
    --------

    Fit exponential function:
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4)
    >>> ck2 = chebder(ck,0,2);
    >>> h = pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x),'r')
    >>> pb.close()

    See also
    --------
    chebint
    chebfit

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

    n = len(ck) - 1
    cder = np.zeros(n, dtype=asarray(ck).dtype)
    cder[0] = 2 * n * ck[0]
    cder[1] = 2 * (n - 1) * ck[1]
    for j in range(2, n):
        cder[j] = cder[j - 2] + 2 * (n - j) * ck[j]

    return cder * 2. / (b - a) # Normalize to the interval b-a.

def chebint(ck, a= -1, b=1):
    """
    Integrate Chebyshev polynomial

    Parameters
    ----------
    ck : array-like
        polynomial coefficients in Chebyshev form of function to integrate.
    a,b : real, scalars
        limits for polynomial(Default -1,1)

    Return
    ------
    cint : ndarray
        polynomial coefficients in Chebyshev form of the integrated function

    Examples
    --------
    Fit exponential function:
    >>> import pylab as pb
    >>> ck = chebfit(pb.exp,7,0,2)
    >>> x = pb.linspace(0,4)
    >>> ck2 = chebint(ck,0,2);
    >>> h=pb.plot(x,chebval(x,ck,0,2),'g',x,pb.exp(x),'r.')
    >>> pb.close()

    See also
    --------
    chebder
    chebfit

    Reference
    ---------
    http://en.wikipedia.org/wiki/Chebyshev_polynomials

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """

# int T0(x) = T1(x)+1
# int T1(x) = 0.5*(T2(x)/2-T0/2)
# int Tn(x) dx = 0.5*{Tn+1(x)/(n+1) - Tn-1(x)/(n-1)}
#             N
#    p(x) =  sum cn*Tn(x)
#            n=0

# int p(x) dx = sum cn * int(Tn(x)dx) = 0.5*sum cn *{Tn+1(x)/(n+1) - Tn-1(x)/(n-1)}
# = 0.5 sum (cn-1-cn+1)*Tn/n n>0

    n = len(ck)

    cint = np.zeros(n)
    con = 0.25 * (b - a)

    dif1 = np.diff(ck[-1::-2])
    ix1 = r_[1:n - 1:2]
    cint[ix1] = -(con * dif1) / ix1
    if n > 3:
        dif2 = np.diff(ck[-2::-2])
        ix2 = r_[2:n - 1:2]
        cint[ix2] = -(con * dif2) / ix2
    cint = cint[::-1]
    #% cint(n) is a special case
    cint[-1] = (con * ck[n - 2]) / (n - 1)
    cint[0] = 2 * np.sum((-1) ** r_[0:n - 1] * cint[-2::-1]) # Set integration constant    
    return cint

class Cheb1d(object):
    coeffs = None
    order = None
    a = None
    b = None
    kind = None
    def __init__(self, ck, a= -1, b=1, kind=1):
        if isinstance(ck, Cheb1d):
            for key in ck.__dict__.keys():
                self.__dict__[key] = ck.__dict__[key]
            return
        cki = np.trim_zeros(atleast_1d(ck), 'b')
        if len(cki.shape) > 1:
            raise ValueError("Polynomial must be 1d only.")
        self.__dict__['coeffs'] = cki
        self.__dict__['order'] = len(cki) - 1
        self.__dict__['a'] = a
        self.__dict__['b'] = b
        self.__dict__['kind'] = kind


    def __call__(self, x):
        return chebval(x, self.coeffs, self.a, self.b, self.kind)

    def __array__(self, t=None):
        if t:
            return asarray(self.coeffs, t)
        else:
            return asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "Cheb1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        pass
    def __neg__(self):
        new = Cheb1d(self)
        new.coeffs = -self.coeffs
        return new

    def __pos__(self):
        return self


    def __add__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polyadd(self.coeffs, other.coeffs)
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polysub(self.coeffs, other.coeffs)
        return new

    def __rsub__(self, other):
        other = Cheb1d(other)
        new = Cheb1d(self)
        new.coeffs = polysub(other.coeffs, new.coeffs)
        return new

    def __eq__(self, other):
        other = Cheb1d(other)
        return (all(self.coeffs == other.coeffs) and (self.a == other.a)
        and (self.b == other.b) and (self.kind == other.kind))

    def __ne__(self, other):
        return any(self.coeffs != other.coeffs) or (self.a != other.a) or (self.b != other.b) or (self.kind != other.kind)

    def __setattr__(self, key, val):
        raise ValueError("Attributes cannot be changed this way.")

    def __getattr__(self, key):
        if key in ['c', 'coef', 'coefficients']:
            return self.coeffs
        elif key in ['o']:
            return self.order
        elif key in ['a']:
            return self.a
        elif key in ['b']:
            return self.b
        elif key in ['k']:
            return self.kind
        else:
            try:
                return self.__dict__[key]
            except KeyError:
                raise AttributeError("'%s' has no attribute '%s'" % (self.__class__, key))
    def __getitem__(self, val):
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[val]

    def __setitem__(self, key, val):
        #ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = np.zeros(key - self.order, self.coeffs.dtype)
            self.__dict__['coeffs'] = np.concatenate((self.coeffs, zr))
            self.__dict__['order'] = key
        self.__dict__['coeffs'][key] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `chebint` for full documentation.

        See Also
        --------
        chebint : equivalent function

        """
        integ = Cheb1d(self)
        integ.coeffs = chebint(self.coeffs, self.a, self.b)
        return integ

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `chebder` for full documentation.

        See Also
        --------
        chebder : equivalent function

        """
        der = Cheb1d(self)
        der.coeffs = chebder(self.coeffs, self.a, self.b)
        return der

def padefit(c, m=None):
    """
    Rational polynomial fitting from polynomial coefficients

    Parameters
    ----------
    c : array-like
        coefficients of power series expansion from highest degree to zero.
    m : scalar integer
        order of denominator polynomial. (Default np.floor((len(c)-1)/2))

    Returns
    -------
    num, den : poly1d
        numerator and denominator polynomials for the pade approximation

    If the function is well approximated by
              M+N+1
       f(x) = sum c(2*n+2-k)*x^k
              k=0

    then the pade approximation is given by
               M
              sum c1(n-k+1)*x^k
              k=0
    f(x) = ------------------------
              N
              sum c2(n-k+1)*x^k
              k=0

    Note: c must be ordered for direct use with polyval

    Example
    -------
    Pade approximation to np.exp(x)
    >>> import scipy.special as sp
    >>> import pylab as plb
    >>> c = poly1d(1./sp.gamma(plb.r_[6+1:0:-1]))  #polynomial coeff exponential function
    >>> [p, q] = padefit(c)
    >>> p; q
    poly1d([ 0.00277778,  0.03333333,  0.2       ,  0.66666667,  1.        ])
    poly1d([ 0.03333333, -0.33333333,  1.        ])

    >>> x = plb.linspace(0,4);
    >>> h = plb.plot(x,c(x),x,p(x)/q(x),'g-', x,plb.exp(x),'r.')
    >>> plb.close()

    See also
    --------
    scipy.misc.pade

    """
    if not m:
        m = int(floor((len(c) - 1) * 0.5))
    c = asarray(c)
    return pade(c[::-1], m)

def test_pade():
    cof = array(([1.0, 1.0, 1.0 / 2, 1. / 6, 1. / 24]))
    p, q = pade(cof, 2)
    t = np.arange(0, 2, 0.1)
    assert(all(abs(p(t) / q(t) - np.exp(t)) < 0.3))

def padefitlsq(fun, m, k, a= -1, b=1, trace=False, x=None, end_points=True):
    """
    Rational polynomial fitting. A minimax solution by least squares.

    Parameters
    ----------
    fun : callable or or a two column matrix
           f=[x,f(x)]  where length(x)>(m+k+1)*8.
    m, k : integer
        number of coefficients of the numerator and denominater, respectively.
    a, b : real scalars
        evaluation limits, (default a=-1,b=1)

    Returns
    -------
    num, den : poly1d
        numerator and denominator polynomials for the pade approximation
    dev : ndarray
        maximum absolute deviation of the approximation

    The pade approximation is given by
               m
              sum c1[m-i]*x**i
              i=0
    f(x) = ------------------------
               k
              sum c2[k-i]*x**i
              i=0

    If F is a two column matrix, [x f(x)], a good choice for x is:

    x = np.cos(pi/(N-1)*(N-1:-1:0))*(b-a)/2+ (a+b)/2, where N = (m+k+1)*8;

    Note: c1 and c2 are ordered for direct use with polyval

    Example
    -------

    Pade approximation to np.exp(x) between 0 and 2
    >>> import pylab as plb
    >>> [c1, c2] = padefitlsq(plb.exp,3,3,0,2)
    >>> c1; c2
    poly1d([ 0.01443847,  0.128842  ,  0.55284547,  0.99999962])
    poly1d([-0.0049658 ,  0.07610473, -0.44716929,  1.        ])

    >>> x = plb.linspace(0,4)
    >>> h = plb.plot(x, polyval(c1,x)/polyval(c2,x),'g')
    >>> h = plb.plot(x, plb.exp(x), 'r')

    See also
    --------
    padefit

    Reference
    ---------
    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 197-20
    """

    NFAC = 8
    BIG = 1e30
    MAXIT = 5

    smallest_devmax = BIG
    ncof = m + k + 1
    npt = NFAC * ncof # % Number of points where function is evaluated, i.e. fineness of mesh

    if x is None:
        if end_points:
            # Use the location of the local extreme values of
            # the Chebychev polynomial of the first kind of degree NPT-1.
            x = map_to_interval(chebextr(npt - 1), a, b)
        else:
            # Use the roots of the Chebychev polynomial of the first kind of degree NPT.
            # Note this is useful if there are singularities close to the endpoints.
            x = map_to_interval(chebroot(npt, kind=1), a, b)


    if hasattr(fun, '__call__'):
        fs = fun(x)
    else:
        fs = fun
        n = len(fs)
        if n < npt:
            warnings.warn('Check the result! Number of function values should be at least: %d' % npt)

    if trace:
        import pylab as plb
        plb.plot(x, fs, '+')

    wt = np.ones((npt))
    ee = np.ones((npt))
    mad = 0

    u = np.zeros((npt, ncof))
    for ix in range(MAXIT):
        #% Set up design matrix for least squares fit.
        pow = wt
        bb = pow * (fs + np.abs(mad) * sign(ee))

        for jx in range(m + 1):
            u[:, jx] = pow
            pow = pow * x

        pow = -bb
        for jx in range(m + 1, ncof):
            pow = pow * x
            u[:, jx] = pow


        [u1, w, v] = np.linalg.svd(u, full_matrices=False)
        cof = np.where(w == 0, 0.0, dot(bb, u1) / w)
        cof = dot(cof, v)

        #% Tabulate the deviations and revise the weights
        ee = polyval(cof[m::-1], x) / polyval(cof[ncof:m:-1].tolist() + [1, ], x) - fs

        wt = np.abs(ee)
        devmax = max(wt)
        mad = wt.mean() #% mean absolute deviation

        if (devmax <= smallest_devmax): #% Save only the best coefficients found
            smallest_devmax = devmax
            c1 = cof[m::-1]
            c2 = cof[ncof:m:-1].tolist() + [1, ]

        if trace:
            print(('Iteration=%d,  max error=%g' % (ix, devmax)))
            plb.plot(x, fs, x, ee + fs)
    return poly1d(c1), poly1d(c2)
