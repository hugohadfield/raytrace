
from clifford.g3c import *
from clifford.tools.g3c import *
import clifford as cf
import numpy as np
import functools
import numpy.polynomial.polynomial as npp
import numba


class MultiVectorPolynomial(npp.Polynomial):
    """
    A class to represent a polynomial with multivector coefficients.

    Note we can no longer use __call__ for grade selection syntax as
    this is used by numpy.polynomial.polynomial.Polynomial to evaluate
    the polynomial, instead we provide a .grade(g: int) method.
    """
    def grade(self, other):
        """ Return a new polynomial with mv coefficients of the desired grade only """
        return MultiVectorPolynomial(cf.MVArray(self.coef)(other))

    def __hash__(self):
        """ A hacky hash function to allow the polynomial to be used in functools.lru_cache """
        return hash(str(self.coef.data))

    @property
    def scalar_poly(self):
        """ Return a polynomial in the scalar part of the multivectors only """
        return npp.Polynomial(cf.MVArray(self.coef).value[:, 0])

    def scalar_roots(self) -> np.ndarray:
        """ Evaluate the roots of a polynomial in the scalar part of the multivector """
        return self.scalar_poly.roots()

    def __invert__(self):
        """ Reversion overload """
        return MultiVectorPolynomial(~cf.MVArray(self.coef))

    def conv(self, other):
        """
        Discrete 1d convolution of the coefficient arrays of two polynomials,
        this is equivalent to __mul__ for polynomials except it jumps past
        all the numpy sequence checking overhead etc etc.
        """
        return MultiVectorPolynomial(np.convolve(self.coef, other.coef))



@functools.lru_cache(maxsize=128)
def poly_sigma(polyXdash):
    return -polyXdash*~polyXdash


@functools.lru_cache(maxsize=128)
def poly_norm_sigma_sqrd(polySigma):
    poly_zero_sigma = polySigma.grade(0)
    poly_four_sigma = polySigma.grade(4)
    return (poly_zero_sigma*poly_zero_sigma - poly_four_sigma*poly_four_sigma).grade(0)


@functools.lru_cache(maxsize=128)
def poly_S_xdash(sigma, xdash):
    return (sigma.grade(0) - sigma.grade(4))*xdash


def poly_pp_meet_L(p, L):
    return MultiVectorPolynomial(I5*(L^cf.MVArray(p.coef)))


def poly_meet_L(p, L):
    return MultiVectorPolynomial(I5*((I5*L)^(I5*cf.MVArray(p.coef))))


def gen_full_poly_point_pairs(polyXdash: MultiVectorPolynomial, L: cf.MultiVector):
    """
    Generate the scalar polynomial associated with the intersection of
    the line L with the point-pair surface formed by reprojection of the
    polynomial polyXdash onto the blade manifold
    """
    sigma = poly_sigma(polyXdash)
    S_xdash = poly_S_xdash(sigma, polyXdash)

    lmeetsigmaxdash = poly_pp_meet_L(S_xdash, L)
    leftside_sqrd = lmeetsigmaxdash.conv(lmeetsigmaxdash)

    lmeetxdash = poly_pp_meet_L(polyXdash, L)
    right_side_sqrd = poly_norm_sigma_sqrd(sigma).conv(lmeetxdash.conv(lmeetxdash))

    final_poly = MultiVectorPolynomial(leftside_sqrd.coef - right_side_sqrd.coef).scalar_poly
    return final_poly


@numba.njit
def _val_gen_full_poly_point_pairs(val_polyXdash, val_L):
    """
    Generate the scalar polynomial associated with the intersection of
    the line L with the point-pair surface formed by reprojection of the
    polynomial polyXdash onto the blade manifold.

    This is a jitted version of gen_full_poly_point_pairs operating
    directly on value arrays for speed.
    """
    val_sigma = _val_mvconv(val_polyXdash, val_polyXdash)
    sig4 = val_sigma @ mask4
    if np.sum(np.abs(sig4)) < 1E-8: # Degenerate case, the pps are on a circle
        final_poly = np.zeros(val_polyXdash.shape[0])
        for i in range(val_polyXdash.shape[0]):
            final_poly[i] = omt_func(val_L, val_polyXdash[i, :])[31]
        return final_poly
    sig0 = val_sigma @ mask0
    val_S_xdash = _val_mvconv(sig0 - sig4, val_polyXdash)
    for i in range(val_S_xdash.shape[0]):
        val_S_xdash[i, :] = dual_func(omt_func(val_L, val_S_xdash[i, :]))
    leftside_sqrd = _val_mvconv(val_S_xdash, val_S_xdash)[:, 0]

    val_lmeetxdash = np.zeros(val_polyXdash.shape)
    for i in range(val_polyXdash.shape[0]):
        val_lmeetxdash[i, :] = dual_func(omt_func(val_L, val_polyXdash[i, :]))

    sig02 = _val_mvconv(sig0, sig0)
    sig42 = _val_mvconv(sig4, sig4)
    right_side_sqrd = _val_mvconv((sig02 - sig42), _val_mvconv(val_lmeetxdash, val_lmeetxdash))[:, 0]

    final_poly = leftside_sqrd - right_side_sqrd
    return final_poly


def jitted_gen_full_poly_point_pairs(polyXdash, L):
    """
    Generate the scalar polynomial associated with the intersection of
    the line L with the point-pair surface formed by reprojection of the
    polynomial polyXdash onto the blade manifold.

    This is a jitted version of gen_full_poly_point_pairs operating
    directly on value arrays for speed.
    """
    return npp.Polynomial(_val_gen_full_poly_point_pairs(polyXdash.value, L.value))


@numba.njit
def _val_mvconv(a, b):
    """
    Performs the 1D discrete convolution of two 1D sequences of
    multivectors a and b
    returns y of shape (a.shape[0] + b.shape[0] - 1, 32)
    """
    lena = a.shape[0]
    lenb = b.shape[0]
    nconv = lena + lenb - 1
    y = np.zeros((nconv, 32))
    for i in range(nconv):
        b_start = max(0,i-lena+1)
        b_end   = min(i+1,lenb)
        a_start = min(i,lena-1)
        for j in range(b_start, b_end):
            y[i, :] += gmt_func(a[a_start, :], b[j, :])
            a_start -= 1
    return y


def mvconv(a: cf.MVArray, b: cf.MVArray):
    """
    Performs the 1D discrete convolution of two 1D sequences of
    multivectors a and b
    returns y of shape (a.shape[0] + b.shape[0] - 1, 32)
    """
    valconv = _val_mvconv(a.value, b.value)
    return cf.MVArray([layout.MultiVector(valconv[i, :], copy=False) for i in range(valconv.shape[0])])


def potential_roots_point_pairs(X0, X1, L):
    """
    Point pairs have a maximum of 6 roots
    """

    # Set up the polynomial
    coef_array = cf.MVArray([X0, X1-X0])
    polyXdash = MultiVectorPolynomial(coef_array)
    final_poly = gen_full_poly_point_pairs(polyXdash, L)

    # Solve the equation
    root_list = final_poly.roots()

    # Take only the real roots between 0 and 1
    real_valued = root_list.real[abs(root_list.imag)<1e-5]
    potential_roots = [r for r in real_valued if r >= 0 and r <= 1]

    return potential_roots


@numba.njit
def jitted_comp(c):
    """
    Generate the companion matrix for polynomial root finding.
    From the numpy source code:
    https://github.com/numpy/numpy/blob/d9b1e32cb8ef90d6b4a47853241db2a28146a57d/numpy/polynomial/polynomial.py#L1339
    """
    if len(c) == 2:
        return np.array([[-c[0] / c[1]]])
    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    bot = mat.reshape(-1)[n::n + 1]
    bot[...] = 1
    mat[:, -1] -= c[:-1] / c[-1]
    return mat


@numba.njit
def val_filter_roots_point_pair(potential_roots, X0_val, X1_val, L_val):
    """ Filters the potential roots to ensure that the meet really is zero there... """
    filtered_roots = -np.ones_like(potential_roots)
    i = 0
    mv_array = np.zeros((32, 2))
    mv_array[:, 0] = X0_val
    mv_array[:, 1] = X1_val
    alpha_array = np.zeros(2)
    for alpha in potential_roots:
        alpha_array[0] = 1.0-alpha
        alpha_array[1] = alpha
        avobj = neg_twiddle_root_val((mv_array @ alpha_array))[0, :]
        if np.abs(omt_func(avobj, L_val)[31]) < 1E-6:
            filtered_roots[i] = alpha
            i = i + 1
    return filtered_roots


def val_jitted_potential_roots_point_pairs(X0_val, X1_val, L_val):
    """
    Generates the potential roots of the intersection polynomial in the range 0 to 1
    for the linear inetrpolation of point pairs X0 and X1 intersected with line L

    This version operates on value arrays for speed and calls jitted functions
    internally.

    NOTE: DESPITE THE NAME THIS FUNCTION IS PURPOSEFULLY NOT JITTED IN ITS ENTIRITY!
    Jitting the entire function did not appear to produce any speed up and simply
    caused problems with the eigvals function in the polynomial solving.
    """
    coef_array = np.zeros((2, 32))
    coef_array[0, :] = X0_val
    coef_array[1, :] = (X1_val - X0_val)
    final_poly = _val_gen_full_poly_point_pairs(coef_array, L_val)
    comp = jitted_comp(final_poly)
    root_list = np.linalg.eigvals(comp)

    # Take only the real roots between 0 and 1
    real_valued = np.real(root_list)[np.abs(np.imag(root_list)) < 1e-5]
    potential_roots = real_valued[(real_valued >= 0)&(real_valued <= 1)]
    filtered_roots = val_filter_roots_point_pair(potential_roots, X0_val, X1_val, L_val)
    return filtered_roots


def jitted_potential_roots_point_pairs(X0: cf.MultiVector, X1: cf.MultiVector, L: cf.MultiVector) -> np.ndarray:
    """ Calls val_jitted_potential_roots_point_pairs """
    return val_jitted_potential_roots_point_pairs(X0.value, X1.value, L.value)


def gen_full_poly_circles(Xdash: MultiVectorPolynomial, L: cf.MultiVector) -> npp.Polynomial:
    """
    Generates the full polynomial for root finding evolved circle-ray
    intersection
    """
    sigma = poly_sigma(Xdash)

    sig0 = sigma.grade(0)
    sig4 = sigma.grade(4)

    Lvsig0Xdash = poly_meet_L(sig0*(Xdash), L)
    lvsig4Xdash = poly_meet_L(sig4*(Xdash), L)
    lvXdash = poly_meet_L(Xdash, L)

    left = ((Lvsig0Xdash - lvsig4Xdash) ** 2 + poly_norm_sigma_sqrd(sigma) * lvXdash ** 2) ** 2
    right = poly_norm_sigma_sqrd(sigma) * (
                (Lvsig0Xdash - lvsig4Xdash) * lvXdash + lvXdash * (Lvsig0Xdash - lvsig4Xdash)) ** 2
    return (left - right).scalar_poly


def potential_roots_circles(X0: cf.MultiVector, X1: cf.MultiVector, L: cf.MultiVector) -> list:
    """
    Interpolated circles form of polynomial of order 12
    """
    Xdash = MultiVectorPolynomial(cf.MVArray([X0, X1 - X0]))
    final_poly = gen_full_poly_circles(Xdash, L)
    # Solve the equation
    root_list = final_poly.roots()
    # Take only the real roots between 0 and 1
    real_valued = root_list.real[abs(root_list.imag)<1e-5]
    potential_roots = [r for r in real_valued if r >= 0 and r <= 1]
    return potential_roots

