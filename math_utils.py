
import numpy as np
import numba


@numba.njit
def nth_polynomial_fit(x, y, n):
    """
    Fits an nth order polynomial to x and y
    """
    xmat = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            xmat[i,j] = x[i]**((n-j))
    return np.linalg.solve(xmat, y)


@numba.njit
def quad(x, p):
    """
    Evaluates the quadratic p at x
    """
    return p[0]*x**2 + p[1]*x + p[2]


@numba.njit
def bisection(p, start, stop):
    """
    Bisects start -> stop looking for roots of qudratic p
    """
    fstart = quad(start, p)
    for __ in range(1000):
        half = start + (stop - start)/2
        fhalf = quad(half, p)
        if abs(fhalf) < 1e-12:
            return half
        if fhalf * fstart > 0:
            start = half
            fstart = quad(start, p)
        else:
            stop = half
    return half


@numba.njit
def get_root(x, y):
    """
    Finds the root of y over the range of x
    """
    poly = nth_polynomial_fit(x, y, 2)
    return bisection(poly, x[0], x[2])
