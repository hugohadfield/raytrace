import unittest
from derivatives import *


class TestSquareRootDerivatives(unittest.TestCase):

    def test_differential_sqsq(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            delta_alpha = 0.00001
            Xdash = (1 - alpha) * C1 + alpha * C2
            dXdashdalpha = -C1 + C2

            Sigma = -Xdash * ~Xdash

            np.testing.assert_almost_equal(np.sqrt(Sigma[0] ** 2 - (Sigma(4) ** 2)[0]), sqsq(Sigma))

            dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
            res0 = differential_sqsq(Sigma, dsda)

            # Central differences
            Xdashplus = (1 - alpha - delta_alpha) * C1 + (alpha + delta_alpha) * C2
            Xdashminus = (1 - alpha + delta_alpha) * C1 + (alpha - delta_alpha) * C2
            Sigmaplus = -Xdashplus * ~Xdashplus
            Sigmaminus = -Xdashminus * ~Xdashminus
            sqsqplus = sqsq(Sigmaplus)
            sqsqminus = sqsq(Sigmaminus)
            res1 = (sqsqplus - sqsqminus) / (2 * delta_alpha)

            np.testing.assert_almost_equal(res0, res1, 5)
            print(i)

    def test_differential_root_sigma(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            delta_alpha = 0.00001
            Xdash = (1 - alpha) * C1 + alpha * C2
            dXdashdalpha = -C1 + C2

            Sigma = -Xdash * ~Xdash

            dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
            diff_root = differential_root_sigma(Sigma, dsda)

            Xdashplus = (1 - alpha - delta_alpha) * C1 + (alpha + delta_alpha) * C2
            Xdashminus = (1 - alpha + delta_alpha) * C1 + (alpha - delta_alpha) * C2
            Sigmaplus = -Xdashplus * ~Xdashplus
            Sigmaminus = -Xdashminus * ~Xdashminus

            rp = positive_root(Sigmaplus)
            rm = positive_root(Sigmaminus)
            res0 = (rp - rm).value / (2 * delta_alpha)
            res1 = (diff_root).value

            np.testing.assert_almost_equal(res0, res1, 5)
            print(i)


class TestManifoldProjectionDerivatives(unittest.TestCase):

    def test_d_sigma_d_alpha(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            delta_alpha = 0.00001
            Xdash = (1 - alpha) * C1 + alpha * C2
            dXdashdalpha = -C1 + C2

            # Analytic solution
            analytic = d_sigma_d_alpha(Xdash, dXdashdalpha)

            # Central differences
            Xdashplus = (1 - alpha - delta_alpha) * C1 + (alpha + delta_alpha) * C2
            Xdashminus = (1 - alpha + delta_alpha) * C1 + (alpha - delta_alpha) * C2
            Sigmaplus = -Xdashplus * ~Xdashplus
            Sigmaminus = -Xdashminus * ~Xdashminus
            delta_Sigma = Sigmaplus - Sigmaminus

            res0 = (delta_Sigma / (2 * delta_alpha)).value
            res1 = analytic.value
            np.testing.assert_almost_equal(res0, res1)
            print(i)

    def test_calculate_projector(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            Xdash = (1 - alpha) * C1 + alpha * C2
            Sigma = -Xdash * ~Xdash
            root_sigma = positive_root(Sigma)
            S = calculate_projector(root_sigma)
            Xest = (S * Xdash).normal()
            Xcalc = average_objects([C1, C2], [0.8, 0.2])
            print(Xest * Xest)
            print(Xest)
            print(Xcalc * Xcalc)
            print(Xcalc)
            np.testing.assert_almost_equal(Xest.value, Xcalc.value, 5)
            print(i)

    def test_differential_projector(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            delta_alpha = 1E-6
            Xdash = (1 - alpha) * C1 + alpha * C2
            dXdashdalpha = -C1 + C2

            Sigma = -Xdash * ~Xdash

            root_sigma = positive_root(Sigma)
            dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
            diff_root = differential_root_sigma(Sigma, dsda)
            dproj = differential_projector(root_sigma, diff_root)

            Xdashplus = (1 - alpha - delta_alpha) * C1 + (alpha + delta_alpha) * C2
            Xdashminus = (1 - alpha + delta_alpha) * C1 + (alpha - delta_alpha) * C2
            Sigmaplus = -Xdashplus * ~Xdashplus
            Sigmaminus = -Xdashminus * ~Xdashminus
            Splus = calculate_projector(positive_root(Sigmaplus))
            Sminus = calculate_projector(positive_root(Sigmaminus))
            res0 = (Splus - Sminus) / (2 * delta_alpha)

            # print(res0)
            # print(dproj, flush=True)
            np.testing.assert_allclose(res0.value, dproj.value, 1E-3, 1E-4)
            print(i)

    def test_jit_differential_projector(self):
        for i in range(10000):
            C1 = random_circle()
            C2 = random_circle()
            alpha = 0.2
            delta_alpha = 1E-8
            Xdash = (1 - alpha) * C1 + alpha * C2
            dXdashdalpha = -C1 + C2

            Sigma = -Xdash * ~Xdash

            root_sigma = positive_root(Sigma)
            dsda = d_sigma_d_alpha(Xdash, dXdashdalpha)
            droot_sigma = differential_root_sigma(Sigma, dsda)
            dproj = differential_projector(root_sigma, droot_sigma)

            # Calculate the right hand differential bit
            r04 = root_sigma(0) - root_sigma(4)
            dr04 = droot_sigma(0) - droot_sigma(4)
            left = dr04 * root_sigma + r04 * droot_sigma
            right = -1.0 / ((r04 * root_sigma) ** 2)[0]
            rdiff = left * right
            # Calculate the rest
            total = (dr04 / (r04 * root_sigma)[0]) + r04 * rdiff

            np.testing.assert_allclose(dproj.value, total.value, 1E-4, 1E-6)

    def test_differential_manifold_projection(self):
        for i in range(100):
            C1 = random_circle()
            C2 = random_circle()
            for j, alpha in enumerate(np.linspace(0, 1, 100)):
                delta_alpha = 0.00001
                Xdash = (1 - alpha) * C1 + alpha * C2
                dXdashdalpha = -C1 + C2

                dXda = differential_manifold_projection(Xdash, dXdashdalpha)

                Xtrue = average_objects([C1, C2], [1 - alpha, alpha])
                Xplus = average_objects([C1, C2], [1 - alpha - delta_alpha, alpha + delta_alpha])
                Xminus = average_objects([C1, C2], [1 - alpha + delta_alpha, alpha - delta_alpha])

                res0 = (Xplus - Xminus) / (2 * delta_alpha)

                # Assert that the results are basically the same
                np.testing.assert_allclose(res0.value, dXda.value, 1E-3, 1E-6)
                # Assert it dots with the point to zero
                np.testing.assert_allclose((dXda | Xtrue).value, 0, 1E-3, 1E-6)
                # Assert it anticommutes
                np.testing.assert_allclose((dXda * Xtrue + Xtrue * dXda).value, 0, 1E-3, 1E-6)
                # Assert it is a bivector
                np.testing.assert_allclose((dXda * Xtrue)(2).value, (dXda * Xtrue).value, 1E-3, 1E-6)

                print(i * 100 + j)


if __name__ == '__main__':
    unittest.main()
