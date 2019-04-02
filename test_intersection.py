
from clifftrace import *
from clifford.g3c import *
from clifford.tools.g3c import *
import matplotlib.pyplot as plt
import scipy.interpolate

import unittest


def get_intersect_surf(surf, L,origin, diff=False):
    P_val,alpha = pointofXsurface(L, surf, origin)
    if alpha is None:
        return None, None, None
    P = normalise_n_minus_1(layout.MultiVector(value=P_val))
    # Calpha = interp_objects_root(C1,C2,alpha)
    # P = project_points_to_circle([P], Calpha)[0]
    # P = 0.5*(P + P*L*P)
    # P = P*einf*P
    # P = project_points_to_circle([P], Calpha)[0]
    # P = 0.5 * (P + P * L * P)
    # P = normalise_n_minus_1(P * einf * P)
    if diff:
        N = -get_analytic_normal(surf.first,surf.second,alpha,P)
    else:
        N = -get_numerical_normal(surf.first,surf.second,alpha,P)
    return P, N, alpha


class TestCombo(unittest.TestCase):

    def test_surface_point_hit(self):
        n_rays = 1500

        origin = 3*e3+1*e2+1*e1
        C1 = (up(e1 + e2)^up(e1 -e2)^up(e1 + e3)).normal()
        C2 = (up(-e1 + e2)^up(-e1 -e2)^up(-e1 + e3)).normal()
        S = unsign_sphere(C1.join(C2)(4)).normal()

        surf = Interp_Surface(C1, C2, np.array([0., 0., 1.]), 1., 100., .5, 1., 0.)

        hit_error = 0
        normal_error = 0
        total_hits = 0
        double_error = 0
        for i in range(n_rays):
            L = (up(origin)^up(0.1*random_euc_mv())^einf).normal()
            if (meet(S,L)**2)[0] > 0:
                #point_pair_to_end_points(meet(S,L))
                P,N,alpha = get_intersect_surf(surf,L,origin)
                total_hits += 1
                if N is not None:
                    P_tru = point_pair_to_end_points(meet(L,S))[0]
                    P_tru = P_tru*einf*P_tru
                    truN = ((S*I5)^P_tru^einf).normal()
                    if (not np.max(np.abs((truN-N).value)) < 10**-6) and (not np.max(np.abs((P_tru|P).value)) < 10**-6):
                        double_error += 1
                        # print(P)
                        # print(P_tru)
                        # draw([L,P,C1,C2,P_tru, interp_objects_root(C1,C2,alpha), N, truN])
                        # exit()
                    elif not np.max(np.abs((P_tru|P).value)) < 10**-6:
                        hit_error += 1
                    elif not np.max(np.abs((truN-N).value)) < 10**-6:
                        normal_error += 1
                else:
                    pps = point_pair_to_end_points(meet(S, L))
                    if not (point_beyond_plane(pps[0], C1 ^ einf) or point_beyond_plane(pps[1], C1 ^ einf)):
                        if not (point_beyond_plane(pps[0], C2 ^ einf) or point_beyond_plane(pps[1], C2 ^ einf)):
                            hit_error += 1

        print('\n\n')
        print('Hit errors: ', 100 * hit_error / total_hits)
        print('Normal errors: ', 100 * normal_error / total_hits)
        print('Double errors: ', 100 * double_error / total_hits)

    def test_surface_point_hit_diff(self):
        n_rays = 1500

        origin = 3*e3+1*e2+1*e1
        C1 = (up(e1 + e2)^up(e1 -e2)^up(e1 + e3)).normal()
        C2 = (up(-e1 + e2)^up(-e1 -e2)^up(-e1 + e3)).normal()
        S = unsign_sphere(C1.join(C2)(4)).normal()

        surf = Interp_Surface(C1, C2, np.array([0., 0., 1.]), 1., 100., .5, 1., 0.)

        hit_error = 0
        normal_error = 0
        total_hits = 0
        double_error = 0
        for i in range(n_rays):
            L = (up(origin)^up(0.1*random_euc_mv())^einf).normal()
            if (meet(S,L)**2)[0] > 0:
                P,N,alpha = get_intersect_surf(surf,L,origin,diff=True)
                if N is not None:
                    total_hits += 1
                    P_tru = point_pair_to_end_points(meet(L,S))[0]
                    P_tru = P_tru*einf*P_tru
                    truN = ((S*I5)^P_tru^einf).normal()
                    if (not np.max(np.abs((truN-N).value)) < 10**-6) and (not np.max(np.abs((P_tru|P).value)) < 10**-6):
                        double_error += 1
                        # draw([L, P, C1, C2, P_tru, interp_objects_root(C1, C2, alpha), N, truN])
                        # exit()
                    elif not np.max(np.abs((P_tru|P).value)) < 10**-6:
                        hit_error += 1
                    elif not np.max(np.abs((truN-N).value)) < 10**-6:
                        normal_error += 1
                else:
                    pps = point_pair_to_end_points(meet(S,L))
                    if not(point_beyond_plane(pps[0],C1^einf) or point_beyond_plane(pps[1],C1^einf)):
                        if not (point_beyond_plane(pps[0], C2 ^ einf) or point_beyond_plane(pps[1], C2 ^ einf)):
                            hit_error += 1

        print('\n\n')
        print('Hit errors: ', 100 * hit_error / total_hits)
        print('Normal errors: ', 100 * normal_error / total_hits)
        print('Double errors: ', 100 * double_error / total_hits)


    def test_spline_hit_diff(self):

        origin = (10*e3+5*e2-e1)
        C1 = (up(e1 + e2)^up(e1 -e2)^up(e1 + e3)).normal()
        C2 = (up(-e1 + e2 + 5*e3)^up(-e1 -e2 + 5*e3)^up(-e1 + e3 + 5*e3)).normal()
        S = unsign_sphere(C1.join(C2)(4)).normal()
        Cin = up(down((C1 + C2)*einf*(C1 + C2))+e1)
        L = (up(origin)^Cin^einf).normal()

        surf = Interp_Surface(C1, C2, np.array([0., 0., 1.]), 1., 100., .5, 1., 0.)

        alpha_list = np.linspace(0, 1, 100)
        interp_list = [interp_objects_root(C1,C2,alpha) for alpha in alpha_list]
        meet_list = []
        for C in interp_list:
            meet_list.append((meet(L, C) ** 2)[0])

        alpha_probe = np.linspace(0, 1, 20)
        probe_list = [interp_objects_root(C1, C2, alpha) for alpha in alpha_probe]
        probe_meet = []
        for C in probe_list:
            probe_meet.append((meet(L, C) ** 2)[0])

        alpha_spline = scipy.interpolate.Akima1DInterpolator(alpha_probe, probe_meet)
        alpha_vals = alpha_spline.roots()
        alpha_vals = [a for a in alpha_vals if a < 1 and a > 0]
        if len(alpha_vals) == 1:
            print('Single Hit')
            print(alpha_vals)
        if len(alpha_vals) == 2:
            print('Double Hit')
            print(alpha_vals)
        if len(alpha_vals) > 2:
            print('OOPS')
            print(alpha_vals)


        plt.plot(alpha_list, meet_list)
        plt.plot(alpha_probe, probe_meet)
        plt.plot(alpha_list, alpha_spline(alpha_list))
        plt.show()

        trace_check = surf.probe_func(L.value)
        plt.plot(alpha_list, trace_check)
        plt.show()

        draw([up(origin), L, C1, C2] + interp_list)



if __name__ == '__main__':
    unittest.main()
