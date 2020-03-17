
from clifford.g3c import *
from clifford.tools.g3c import *


def d_sigma_d_alpha(Xdash, dXdashdalpha):
    """
    Derivative of sigma wrt alpha
    """
    return -dXdashdalpha*~Xdash - Xdash*~dXdashdalpha


def test_d_sigma_d_alpha():
    for i in range(10000):
        C1 = random_circle()
        C2 = random_circle()
        alpha = 0.2
        delta_alpha = 0.00001
        Xdash = (1-alpha)*C1 + alpha*C2
        dXdashdalpha = -C1 + C2

        # Analytic solution
        analytic = d_sigma_d_alpha(Xdash, dXdashdalpha)

        # Central differences
        Xdashplus = (1-alpha-delta_alpha)*C1 + (alpha+delta_alpha)*C2
        Xdashminus = (1-alpha+delta_alpha)*C1 + (alpha-delta_alpha)*C2
        Sigmaplus = -Xdashplus*~Xdashplus
        Sigmaminus = -Xdashminus*~Xdashminus
        delta_Sigma = Sigmaplus - Sigmaminus

        res0 = (delta_Sigma/(2*delta_alpha)).value
        res1 = analytic.value
        np.testing.assert_almost_equal(res0, res1)
        print(i)


def sqsq(sigma):
    """
    The function [[Sigma]]
    """
    return np.sqrt(sigma[0]**2 - (sigma(4)**2)[0])


def differential_sqsq(sigma, dsda):
    """
    The differential of [[Sigma]]
    """
    left = dsda[0]*sigma[0] + sigma[0]*dsda[0]
    right = (dsda(4)*sigma(4) + sigma(4)*dsda(4))[0]
    return (left - right)/(2*sqsq(sigma))


def test_differential_sqsq():
    for i in range(10000):
        C1 = random_circle()
        C2 = random_circle()
        alpha = 0.2
        delta_alpha = 0.00001
        Xdash = (1-alpha)*C1 + alpha*C2
        dXdashdalpha = -C1 + C2

        Sigma = -Xdash*~Xdash
        
        dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
        res0 = differential_sqsq(Sigma, dsda)

        # Central differences
        Xdashplus = (1-alpha-delta_alpha)*C1 + (alpha+delta_alpha)*C2
        Xdashminus = (1-alpha+delta_alpha)*C1 + (alpha-delta_alpha)*C2
        Sigmaplus = -Xdashplus*~Xdashplus
        Sigmaminus = -Xdashminus*~Xdashminus
        sqsqplus = sqsq(Sigmaplus)
        sqsqminus = sqsq(Sigmaminus)
        res1 = (sqsqplus - sqsqminus)/(2*delta_alpha)

        np.testing.assert_almost_equal(res0, res1, 5)
        print(i)


def differential_root_sigma(sigma, dsda):
    """
    The differential of square root of sigma
    """
    dsqsq = differential_sqsq(sigma, dsda)
    sqsqsig = sqsq(sigma)
    left = ( dsda + dsqsq ) * ( 1.0 / ( np.sqrt(2)*np.sqrt( sigma[0] + sqsqsig ) ) )
    rightdiff = ((-1.0/(2*np.sqrt(2)))*( sigma[0] + sqsqsig )**(-3.0/2))*( dsda[0] + dsqsq )
    right = ( sigma + sqsqsig ) * rightdiff
    total = left + right
    return total


def test_differential_root_sigma():
    for i in range(10000):
        C1 = random_circle()
        C2 = random_circle()
        alpha = 0.2
        delta_alpha = 0.00001
        Xdash = (1-alpha)*C1 + alpha*C2
        dXdashdalpha = -C1 + C2

        Sigma = -Xdash*~Xdash

        dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
        diff_root = differential_root_sigma(Sigma, dsda)

        Xdashplus = (1-alpha-delta_alpha)*C1 + (alpha+delta_alpha)*C2
        Xdashminus = (1-alpha+delta_alpha)*C1 + (alpha-delta_alpha)*C2
        Sigmaplus = -Xdashplus*~Xdashplus
        Sigmaminus = -Xdashminus*~Xdashminus

        rp = positive_root(Sigmaplus)
        rm = positive_root(Sigmaminus)
        res0 = (rp - rm).value/(2*delta_alpha)
        res1 = (diff_root).value

        np.testing.assert_almost_equal(res0, res1, 5)
        print(i)


def calculate_projector(root_sigma):
    """
    Calculates the projector
    """
    num = root_sigma(0) - root_sigma(4)
    return (num)/(num*root_sigma)


def test_calculate_projector():
    for i in range(10000):
        C1 = random_circle()
        C2 = random_circle()
        alpha = 0.2
        Xdash = (1-alpha)*C1 + alpha*C2
        Sigma = -Xdash*~Xdash
        root_sigma = positive_root(Sigma)
        S = calculate_projector(root_sigma)
        Xest = S*Xdash
        Xcalc = average_objects([C1,C2], [0.8, 0.2])
        np.testing.assert_almost_equal(Xest.value, Xcalc.value, 5)
        print(i)


def differential_projector(root_sigma, droot_sigma):
    """
    The differential of the projector 
    """
    # Calculate the right hand differential bit
    r04 = root_sigma(0) - root_sigma(4)
    dr04 = droot_sigma(0) - droot_sigma(4)
    left = dr04*root_sigma + r04*droot_sigma
    right = -1.0/((r04*root_sigma)**2)[0]
    rdiff = left*right
    # Calculate the rest
    total = (dr04/(r04*root_sigma)[0]) + r04*rdiff
    return total


def test_differential_projector():
    for i in range(10000):
        C1 = random_circle()
        C2 = random_circle()
        alpha = 0.2
        delta_alpha = 0.00001
        Xdash = (1-alpha)*C1 + alpha*C2
        dXdashdalpha = -C1 + C2

        Sigma = -Xdash*~Xdash

        root_sigma = positive_root(Sigma)
        dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
        diff_root = differential_root_sigma(Sigma, dsda)
        dproj = differential_projector(root_sigma, diff_root)

        Xdashplus = (1-alpha-delta_alpha)*C1 + (alpha+delta_alpha)*C2
        Xdashminus = (1-alpha+delta_alpha)*C1 + (alpha-delta_alpha)*C2
        Sigmaplus = -Xdashplus*~Xdashplus
        Sigmaminus = -Xdashminus*~Xdashminus
        Splus = calculate_projector(positive_root(Sigmaplus))
        Sminus = calculate_projector(positive_root(Sigmaminus))
        res0 = (Splus - Sminus)/(2*delta_alpha)

        np.testing.assert_almost_equal(res0.value, dproj.value, 4)
        print(i)


def differential_manifold_projection(Xdash, dXdashdalpha):
    """
    The differential of the projected object on the manifold
    """
    Sigma = -Xdash*~Xdash
    root_sigma = positive_root(Sigma)
    dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
    droot_sigma = differential_root_sigma(Sigma, dsda)
    dS = differential_projector(root_sigma, droot_sigma)
    S = calculate_projector(root_sigma)
    return dS*Xdash + S*dXdashdalpha


def test_differential_manifold_projection():
    for i in range(100):
        C1 = random_circle()
        C2 = random_circle()
        for j, alpha in enumerate(np.linspace(0,1,100)):
            delta_alpha = 0.00001
            Xdash = (1-alpha)*C1 + alpha*C2
            dXdashdalpha = -C1 + C2

            dXda = differential_manifold_projection(Xdash, dXdashdalpha)

            Xtrue =average_objects([C1, C2], [1 - alpha, alpha])
            Xplus = average_objects([C1, C2], [1 - alpha - delta_alpha, alpha + delta_alpha])
            Xminus = average_objects([C1, C2], [1 - alpha + delta_alpha, alpha - delta_alpha])

            res0 = (Xplus - Xminus)/(2*delta_alpha)

            # Assert that the results are basically the same
            np.testing.assert_allclose(res0.value, dXda.value, 1E-3, 1E-6)
            # Assert it dots with the point to zero
            np.testing.assert_allclose((dXda|Xtrue).value, 0, 1E-3, 1E-6)
            # Assert it anticommutes
            np.testing.assert_allclose((dXda*Xtrue + Xtrue*dXda).value, 0, 1E-3, 1E-6)  
            # Assert it is a bivector
            np.testing.assert_allclose((dXda*Xtrue)(2).value, (dXda*Xtrue).value, 1E-3, 1E-6)
            
            print(i*100 + j)


def val_differentiateLinearCircle(alpha, C1_val, C2_val):
    C1 = layout.MultiVector(value=C1_val)
    C2 = layout.MultiVector(value=C2_val)
    Xdash = (1-alpha)*C2 + alpha*C1
    dXdashdalpha = C1 - C2
    dXda = differential_manifold_projection(Xdash, dXdashdalpha)(3).normal()
    return dXda.value


def val_differentiateLinearPointPair(alpha, C1_val, C2_val):
    C1 = layout.MultiVector(value=C1_val)
    C2 = layout.MultiVector(value=C2_val)
    Xdash = (1-alpha)*C2 + alpha*C1
    dXdashdalpha = C1 - C2
    dXda = differential_manifold_projection(Xdash, dXdashdalpha)(2).normal()
    return dXda.value


if __name__ == '__main__':
    test_d_sigma_d_alpha()
    test_differential_sqsq()
    test_differential_root_sigma()
    test_calculate_projector()
    test_differential_projector()
    test_differential_manifold_projection()
    pass
