import unittest
from clifford.tools.g3c import *
import clifford as cf
import time

from polynomial_root import *


class TestMultiVectorPolynomial(unittest.TestCase):

    def test_MultiVectorPolynomial_grade(self):
        pol = MultiVectorPolynomial(cf.MVArray([e1, e12, 1.0*e123]))
        print(pol)
        print(pol.grade(0))
        print(pol.grade(1))
        print(pol.grade(2))
        print(pol.grade(3))


    def test_MultiVectorPolynomial_scalar_poly(self):
        pol = MultiVectorPolynomial(cf.MVArray([-1.0 + 0*e1, 0*e1, 2.0 + 0*e1 + e13]))
        print(pol)
        print(pol.grade(0))
        print(pol.scalar_poly)
        print(pol.scalar_roots())


    def test_MultiVectorPolynomial_eval(self):
        X0 = random_point_pair()
        X1 = random_point_pair()
        coef_array = cf.MVArray([X1, X0-X1])
        poly0 = MultiVectorPolynomial(coef_array)
        print(X1)
        print(poly0(0))
        print('\n')
        print(X0)
        print(poly0(1.0))


    def test_MultiVectorPolynomial_mult(self):
        X0 = random_point_pair()
        X1 = random_point_pair()

        coef_array = cf.MVArray([X1, X0-X1])

        poly0 = MultiVectorPolynomial(coef_array)

        print((X0-X1)*(X0-X1), (X0-X1)*X1 + X1*(X0-X1), X1*X1)
        poly2 = poly0*poly0
        print(poly2.coef[2], poly2.coef[1], poly2.coef[0])


    def test_MultiVectorPolynomial_reversion(self):
        X0 = random_point_pair()
        X1 = random_point_pair()
        coef_array = cf.MVArray([X1, X0-X1])
        poly0 = MultiVectorPolynomial(coef_array)
        print((poly0))
        print((~poly0))


class TestPointPairs(unittest.TestCase):

    def test_jitted_gen_full_poly_point_pairs(self):

        for i in range(50):
            X0 = random_point_pair()
            X1 = random_point_pair()
            S = average_objects([X1, X0], [0.2, (1 - 0.2)])
            L = ((S * einf * S) ^ up(0.0) ^ einf).normal()

            coef_array = cf.MVArray([X1, X0 - X1])
            polyXdash = MultiVectorPolynomial(coef_array)
            final_poly = gen_full_poly_point_pairs(polyXdash, L)

            final_poly_jit = jitted_gen_full_poly_point_pairs(coef_array, L)

#             print('\n\n')
#             print(final_poly.coef)
#             print(final_poly_jit.coef)
#             print('\n\n', flush=True)
            np.testing.assert_allclose(final_poly.coef, final_poly_jit.coef)

        nrepeats = 100

        start_time = time.time()
        for i in range(nrepeats):
            gen_full_poly_point_pairs(polyXdash, L)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)

        start_time = time.time()
        for i in range(nrepeats):
            jitted_gen_full_poly_point_pairs(coef_array, L)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)



    def test_eval_final_poly_point_pairs(self):
        from scene_objects import PointPairSurface
        for i in range(10):
            X0 = random_point_pair()
            X1 = random_point_pair()
            coef_array = cf.MVArray([X0, X1-X0])
            polyXdash = MultiVectorPolynomial(coef_array)

            S = average_objects([X1,X0],[0.2,(1-0.2)])
            L = ((S*einf*S)^up(0.0)^einf).normal()
            final_poly = gen_full_poly_point_pairs(polyXdash, L)

            # Pick an alpha
            alpha = 0.4
            polyoutput = final_poly(alpha)

            Xdash = alpha*X1 + (1-alpha)*X0
            sigma = -Xdash*~Xdash
            left_res =  (I5*(L^((sigma(0) - sigma(4))*Xdash)))**2
            right_res = (I5*(L^(np.sqrt((sigma[0]**2 - (sigma(4)**2))[0])*Xdash)))**2
            manual_output = left_res - right_res

            np.testing.assert_allclose(polyoutput, manual_output.value[0], 1E-3, 1E-6)

#         from pyganja import GanjaScene, Color
#         gs = GanjaScene()
#         gs.add_objects([X0, X1], color=Color.BLACK)
#         gs.add_objects([S], color=Color.RED)
#         gs.add_objects([L])
#         draw(gs, scale=0.1, browser_window=True)

        print(final_poly.degree())
        r = potential_roots_point_pairs(X0, X1, L)

        surf = PointPairSurface(X0, X1)
        print( surf.intersection_func(L.value) )
        print(r)

        # test the value at each
        import matplotlib.pyplot as plt
        plt.plot(surf.probe_alphas, surf.probe_func(L.value))
        plt.plot(surf.probe_alphas, [final_poly(alpha) for alpha in surf.probe_alphas],'r')
        plt.plot(surf.probe_alphas, surf.probe_alphas*0,'k')
        plt.legend(["true meet", "polynomial function", "zero"])
        plt.savefig('point_pair_root_func.png')


    def test_time_root_finding_point_pairs(self):
        import time
        from scene_objects import PointPairSurface

        X0 = random_point_pair()
        X1 = random_point_pair()
        S = average_objects([X1,X0],[0.2,(1-0.2)])
        L = ((S*einf*S)^up(0.0)^einf).normal()

        nrepeats = 200

        print(potential_roots_point_pairs(X0,X1,L))
        print(jitted_potential_roots_point_pairs(X0,X1,L))
        surf = PointPairSurface(X0, X1)
        print(surf.intersection_func(L.value))

        # Polynomial method
        start_time = time.time()
        for i in range(nrepeats):
            potential_roots_point_pairs(X0, X1, L)
        end_time = time.time()
        print('Poly ms per eval: ', 1000*(end_time - start_time)/nrepeats)

        # Polynomial method jitted version
        start_time = time.time()
        for i in range(nrepeats):
            jitted_potential_roots_point_pairs(X0,X1,L)
        end_time = time.time()
        print('Jitted poly ms per eval: ', 1000*(end_time - start_time)/nrepeats)

        # The actual form in use
        start_time = time.time()
        for i in range(nrepeats):
            surf.intersection_func(L.value)
        end_time = time.time()
        print('Full production ms per eval: ', 1000*(end_time - start_time)/nrepeats)


class TestUtility(unittest.TestCase):

    def test_mvconv(self):
        X0 = random_point_pair()
        X1 = random_point_pair()

        coef_array = cf.MVArray([X1, X0 - X1])

        poly0 = MultiVectorPolynomial(coef_array)

        print((X0 - X1) * (X0 - X1), (X0 - X1) * X1 + X1 * (X0 - X1), X1 * X1)
        poly2 = poly0 * poly0
        print(poly2.coef[2], poly2.coef[1], poly2.coef[0])

        poly3 = mvconv(coef_array, coef_array)
        print(poly3[2], poly3[1], poly3[0])

        import time

        nrepeats = 10000

        start_time = time.time()
        for i in range(nrepeats):
            poly0 * poly0
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)

        start_time = time.time()
        for i in range(nrepeats):
            poly0.conv(poly0)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)

        start_time = time.time()
        for i in range(nrepeats):
            mvconv(coef_array, coef_array)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)

    def test_poly_norm_sigma_sqrd(self):
        X0 = random_point_pair()
        X1 = random_point_pair()
        coef_array = cf.MVArray([X1, X0 - X1])
        polyXdash = MultiVectorPolynomial(coef_array)

        alpha = 0.2
        Xdash = alpha * X0 + (1 - alpha) * X1
        sigma = -Xdash * ~Xdash

        print(poly_norm_sigma_sqrd(poly_sigma(polyXdash))(alpha))
        print((sigma[0] ** 2 - (sigma(4) ** 2))[0])


class TestCircle(unittest.TestCase):

    def test_eval_final_poly_circles(self):
        from scene_objects import CircleSurface
        X0 = random_circle()
        X1 = random_circle()
        S = average_objects([X1, X0], [0.2, (1 - 0.2)])
        L = ((S * einf * S) ^ up(0.0) ^ einf).normal()

        Xdash = MultiVectorPolynomial(cf.MVArray([X0, X1 - X0]))
        final_poly = gen_full_poly_circles(Xdash, L)

        # from pyganja import GanjaScene, Color
        # gs = GanjaScene()
        # gs.add_objects([X0, X1], color=Color.BLACK)
        # gs.add_objects([S], color=Color.RED)
        # gs.add_objects([L])
        # draw(gs, scale=0.1)

        r = potential_roots_circles(X0, X1, L)

        surf = CircleSurface(X0, X1)
        print(surf.intersection_func(L.value))
        print(r)

        # test the value at each
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(surf.probe_alphas, surf.probe_func(L.value))
        plt.plot(surf.probe_alphas, [final_poly(alpha) for alpha in surf.probe_alphas],'r')
        plt.plot(surf.probe_alphas, surf.probe_alphas*0, 'k')
        plt.legend(["true meet", "polynomial function", "zero"])
        plt.savefig('circle_root_func.png')
        plt.show()

    def test_time_root_finding_circle(self):
        import time
        from scene_objects import CircleSurface

        X0 = random_circle()
        X1 = random_circle()
        S = average_objects([X1,X0],[0.2,(1-0.2)])
        L = ((S*einf*S)^up(0.0)^einf).normal()

        nrepeats = 200

        print(potential_roots_circles(X0,X1,L))
        print(jitted_potential_roots_circles(X0,X1,L))
        surf = CircleSurface(X0, X1)
        print(surf.intersection_func(L.value))

        # Polynomial method
        start_time = time.time()
        for i in range(nrepeats):
            potential_roots_circles(X0, X1, L)
        end_time = time.time()
        print('Poly ms per eval: ', 1000*(end_time - start_time)/nrepeats)

        # Polynomial method jitted version
        start_time = time.time()
        for i in range(nrepeats):
            jitted_potential_roots_circles(X0,X1,L)
        end_time = time.time()
        print('Jitted poly ms per eval: ', 1000*(end_time - start_time)/nrepeats)

        # The actual form in use
        start_time = time.time()
        for i in range(nrepeats):
            surf.intersection_func(L.value)
        end_time = time.time()
        print('Full production ms per eval: ', 1000*(end_time - start_time)/nrepeats)


    def test_jitted_gen_full_poly_circles(self):

        for i in range(1000):
            X0 = random_point_pair()
            X1 = random_point_pair()
            S = average_objects([X1, X0], [0.2, (1 - 0.2)])
            L = ((S * einf * S) ^ up(0.0) ^ einf).normal()

            coef_array = cf.MVArray([X1, X0 - X1])
            polyXdash = MultiVectorPolynomial(coef_array)
            final_poly = gen_full_poly_circles(polyXdash, L)

            final_poly_jit = jitted_gen_full_poly_circles(coef_array, L)
            np.testing.assert_allclose(final_poly.coef, final_poly_jit.coef)

        nrepeats = 100

        start_time = time.time()
        for i in range(nrepeats):
            gen_full_poly_circles(polyXdash, L)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)

        start_time = time.time()
        for i in range(nrepeats):
            jitted_gen_full_poly_circles(coef_array, L)
        end_time = time.time()
        print('ms per eval: ', 1000 * (end_time - start_time) / nrepeats)



if __name__ == '__main__':
    unittest.main()
