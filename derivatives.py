
from clifford.g3c import *
from clifford.tools.g3c import *
import numba


@numba.njit
def val_d_sigma_d_alpha(Xdash, dXdashdalpha):
    return -gmt_func(dXdashdalpha, adjoint_func(Xdash)) - gmt_func(Xdash, adjoint_func(dXdashdalpha))


def d_sigma_d_alpha(Xdash, dXdashdalpha):
    """
    Derivative of sigma wrt alpha

    return -dXdashdalpha*~Xdash - Xdash*~dXdashdalpha
    """
    return layout.MultiVector(value=val_d_sigma_d_alpha(Xdash.value, dXdashdalpha.value))


@numba.njit
def val_sqsq(sigma):
    s4 = project_val(sigma, 4)
    return np.sqrt(sigma[0]**2 - imt_func(s4, s4)[0])


def sqsq(sigma):
    """
    The function [[Sigma]]

    return np.sqrt(sigma[0]**2 - (sigma(4)**2)[0])
    """
    return val_sqsq(sigma.value)


@numba.njit
def val_differential_sqsq(sigma, dsda):
    left = dsda[0] * sigma[0] + sigma[0] * dsda[0]
    s4 = project_val(sigma, 4)
    ds4 = project_val(dsda, 4)
    right = (gmt_func(ds4, s4) + gmt_func(s4, ds4))[0]
    return (left - right) / (2 * val_sqsq(sigma))


def differential_sqsq(sigma, dsda):
    """
    The differential of [[Sigma]]

    left = dsda[0]*sigma[0] + sigma[0]*dsda[0]
    right = (dsda(4)*sigma(4) + sigma(4)*dsda(4))[0]
    return (left - right)/(2*sqsq(sigma))
    """
    return val_differential_sqsq(sigma.value, dsda.value)


@numba.njit
def val_differential_root_sigma(sigma, dsda):
    dsqsq = val_differential_sqsq(sigma, dsda)
    sqsqsig = val_sqsq(sigma)
    left = +dsda
    left[0] += dsqsq
    left = left * ( 1.0 / ( np.sqrt(2)*np.sqrt( sigma[0] + sqsqsig ) ) )
    rightdiff = ((-1.0/(2*np.sqrt(2)))*( sigma[0] + sqsqsig )**(-3.0/2))*( dsda[0] + dsqsq )
    right = +sigma
    right[0] += sqsqsig
    right = right*rightdiff
    total = left + right
    return total


def differential_root_sigma(sigma, dsda):
    """
    The differential of square root of sigma

    dsqsq = differential_sqsq(sigma, dsda)
    sqsqsig = sqsq(sigma)
    left = ( dsda + dsqsq ) * ( 1.0 / ( np.sqrt(2)*np.sqrt( sigma[0] + sqsqsig ) ) )
    rightdiff = ((-1.0/(2*np.sqrt(2)))*( sigma[0] + sqsqsig )**(-3.0/2))*( dsda[0] + dsqsq )
    right = ( sigma + sqsqsig ) * rightdiff
    total = left + right
    return total
    """
    return layout.MultiVector(value=val_differential_root_sigma(sigma.value, dsda.value))


@numba.njit
def val_calculate_projector(root_sigma):
    num = -project_val(root_sigma, 4)
    num[0] += root_sigma[0]
    return num/(gmt_func(num, root_sigma))[0]


def calculate_projector(root_sigma):
    """
    Calculates the projector

    num = root_sigma(0) - root_sigma(4)
    return (num)/(num*root_sigma)
    """
    return layout.MultiVector(value=val_calculate_projector(root_sigma.value))


@numba.njit
def val_differential_projector(root_sigma, droot_sigma):
    # Calculate the right hand differential bit
    rs0 = project_val(root_sigma, 0)
    rs4 = project_val(root_sigma, 4)
    r04 = rs0 - rs4
    drs0 = project_val(droot_sigma, 0)
    drs4 = project_val(droot_sigma, 4)
    dr04 = drs0 - drs4
    left = gmt_func(dr04, root_sigma) + gmt_func(r04, droot_sigma)
    basert = gmt_func(r04, root_sigma)
    right = -1.0/(gmt_func(basert, basert)[0])
    rdiff = left*right
    # Calculate the rest
    total = (dr04/gmt_func(r04, root_sigma)[0]) + gmt_func(r04, rdiff)
    return total


def differential_projector(root_sigma, droot_sigma):
    """
    The differential of the projector

    # Calculate the right hand differential bit
    r04 = root_sigma(0) - root_sigma(4)
    dr04 = droot_sigma(0) - droot_sigma(4)
    left = dr04*root_sigma + r04*droot_sigma
    right = -1.0/((r04*root_sigma)**2)[0]
    rdiff = left*right
    # Calculate the rest
    total = (dr04/(r04*root_sigma)[0]) + r04*rdiff
    return total
    """
    return layout.MultiVector(value=val_differential_projector(root_sigma.value, droot_sigma.value))


@numba.njit
def val_differential_manifold_projection(Xdash, dXdashdalpha):
    """
    The differential of the projected object on the manifold
    """
    Sigma = gmt_func(-Xdash, adjoint_func(Xdash))
    root_sigma = positive_root_val(Sigma)
    dsda = val_d_sigma_d_alpha(Xdash, dXdashdalpha)
    droot_sigma = val_differential_root_sigma(Sigma, dsda)
    dS = val_differential_projector(root_sigma, droot_sigma)
    S = val_calculate_projector(root_sigma)
    return gmt_func(dS, Xdash) + gmt_func(S, dXdashdalpha)


def differential_manifold_projection(Xdash, dXdashdalpha):
    """
    The differential of the projected object on the manifold

    Sigma = -Xdash*~Xdash
    root_sigma = positive_root(Sigma)
    dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
    droot_sigma = differential_root_sigma(Sigma, dsda)
    dS = differential_projector(root_sigma, droot_sigma)
    S = calculate_projector(root_sigma)
    return dS*Xdash + S*dXdashdalpha
    """
    return layout.MultiVector(value=val_differential_manifold_projection(Xdash.value, dXdashdalpha.value))


@numba.njit
def val_differentiateLinearCircle(alpha, C1_val, C2_val):
    Xdash = (1-alpha)*C2_val + alpha*C1_val
    dXdashdalpha = C1_val - C2_val
    dXda = val_normalised(project_val(val_differential_manifold_projection(Xdash, dXdashdalpha), 3))
    return dXda


@numba.njit
def val_differentiateLinearPointPair(alpha, C1_val, C2_val):
    Xdash = (1-alpha)*C2_val + alpha*C1_val
    dXdashdalpha = C1_val - C2_val
    dXda = val_normalised(project_val(val_differential_manifold_projection(Xdash, dXdashdalpha), 2))
    return dXda

