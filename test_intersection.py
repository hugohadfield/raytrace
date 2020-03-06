
from clifftrace import *
from clifford.g3c import *
from clifford.tools.g3c import *
import matplotlib.pyplot as plt
import scipy.interpolate

import unittest

from scene_objects import *


class TestCombo(unittest.TestCase):

    def test_circle_surface_point_hit(self):
        """
        Make a load of rays and intersect them
        """
        n_rays = 4000

        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
        k = 1.0
        lights_list = []
        colour_light = np.ones(3)
        L = 40. * e3 - 20 * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1
        image_height = 240
        image_width = 320
        f = 1.
        centre3d = 3 * e3 + 1 * e2 + 1 * e1
        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Make the objects
        C1 = (up(e1 + e2) ^ up(e1 - e2) ^ up(e1 + e3)).normal()
        C2 = (up(-e1 + e2) ^ up(-e1 - e2) ^ up(-e1 + e3)).normal()
        S = unsign_sphere(C1.join(C2)(4)).normal()

        surf = CircleSurface(C2, C1, np.array([1., 0., 0.]), k * 1., 100., k * .5, k * 1., k * 0.3)
        object_list = [surf]

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             max_bounces=5,
                             shading_options=shading_options)

        gs = new_scene.as_scene()

        # Get the distance between the planes
        plane1 = (C1^einf).normal()
        plane2 = (C2^einf).normal()
        comp = math.sqrt(-2*(normalise_n_minus_1((C1 * einf * C1)(1))|normalise_n_minus_1((C2 * einf * C2)(1))).value[0])

        hit_error = 0
        ref_error = 0
        total_hits = 0
        double_error = 0
        point_error = 0
        for i in range(n_rays):
            pe = False
            L = (up(centre3d)^up(0.1*random_euc_mv())^einf).normal()
            if (meet(S,L)**2)[0] > 0:

                # First get the hit point with the sphere
                P_tru = point_pair_to_end_points(meet(L, S))[0]
                P_tru = normalise_n_minus_1((P_tru * einf * P_tru)(1))

                # Chuck out any hit points that are not within the plane limits
                a = 0.5*math.sqrt(-2.0*(P_tru|normalise_n_minus_1((plane1*P_tru*plane1)(1))).value[0])
                b = 0.5*math.sqrt(-2.0*(P_tru|normalise_n_minus_1((plane2*P_tru*plane2)(1))).value[0])
                if a < comp and b < comp:
                    total_hits += 1

                    # Get the true reflection
                    Reftru = reflect_in_sphere(L, S, P_tru)

                    # Try and intersect
                    Pval, alpha = surf.intersection_point(L, up(centre3d))
                    if Pval[0] == -1:
                        hit_error += 1
                        gs.add_objects([L], color=Color.BLACK)
                        gs.add_objects([P_tru], color=Color.BLUE)
                        continue
                    else:
                        P = layout.MultiVector(value=Pval)

                    # Try and calculate the reflections
                    Ref = surf.reflect_line(L, P, alpha)

                    # See if we have been successful
                    if np.sum(np.abs(P_tru.value - Pval)) > 1E-3:
                        point_error += 1
                        pe = True
                        gs.add_objects([L], color=Color.BLACK)
                        gs.add_objects([Reftru], color=Color.BLUE)
                        gs.add_objects([Ref], color=Color.RED)
                        gs.add_objects([P_tru], color=Color.BLUE)
                        gs.add_objects([P], color=Color.RED)
                    if np.sum(np.abs(Ref.value - Reftru.value)) > 1E-3:
                        ref_error += 1
                        if pe:
                            double_error += 1
                        else:
                            gs.add_objects([L], color=Color.BLACK)
                            gs.add_objects([Reftru], color=Color.BLUE)
                            gs.add_objects([Ref], color=Color.RED)
                            gs.add_objects([P_tru], color=Color.BLUE)
                            gs.add_objects([P], color=Color.RED)

        print('\n')
        print('Hit errors: ', hit_error, 'of ', total_hits, 'ie. ', 100 * hit_error / total_hits, '%')
        print('Point errors: ', point_error, 'of ', total_hits, 'ie. ', 100 * point_error / total_hits, '%')
        print('Reflection errors: ', ref_error, 'of ', total_hits, 'ie. ', 100 * ref_error / total_hits, '%')
        print('Double errors: ', double_error, 'of ', total_hits, 'ie. ', 100 * double_error / total_hits, '%')
        print('\n', flush=True)
        draw(gs, scale=0.5)



    def test_point_pair_surface_point_hit(self):
        """
        Make a load of rays and intersect them
        """
        n_rays = 4000

        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
        k = 1.0
        lights_list = []
        colour_light = np.ones(3)
        L = 40. * e3 - 20 * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1
        image_height = 240
        image_width = 320
        f = 1.
        centre3d = 3 * e3 + 1 * e2 + 1 * e1
        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Make the objects
        C1 = (up(e1 + e2) ^ up(e1 - e2) ).normal()
        C2 = (up(-e1 + e2) ^ up(-e1 - e2) ).normal()

        surf = PointPairSurface(C2, C1, np.array([1., 0., 0.]), k * 1., 100., k * .5, k * 1., k * 0.3)
        object_list = [surf]

        circle = Circle(e1 + e2, e1 - e2,-e1 + e2, np.array([1., 0., 0.]), k * 1., 100., k * .5, k * 1., k * 0.3)

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             max_bounces=5,
                             shading_options=shading_options)

        gs = new_scene.as_scene()

        # Get the distance between the lines
        line1 = (C1^einf).normal()
        line2 = (C2^einf).normal()
        comp = math.sqrt(-2*(normalise_n_minus_1((C1 * einf * C1)(1))|normalise_n_minus_1((C2 * einf * C2)(1))).value[0])

        hit_error = 0
        point_error = 0
        ref_error = 0
        total_hits = 0
        double_error = 0
        for i in range(n_rays):
            pe = False
            L = (up(centre3d)^up(0.1*random_euc_mv())^einf).normal()

            # First get the hit point with the circle
            P_tru_val, _ = circle.intersection_point(L, up(centre3d))
            if P_tru_val[0] != -1:
                P_tru = layout.MultiVector(value=P_tru_val)

                # Chuck out any hit points that are not within the line limits
                a = 0.5*math.sqrt(-2.0*(P_tru|normalise_n_minus_1((line1*P_tru*line1)(1))).value[0])
                b = 0.5*math.sqrt(-2.0*(P_tru|normalise_n_minus_1((line2*P_tru*line2)(1))).value[0])
                if a < comp and b < comp:
                    total_hits += 1

                    # Get the true reflection
                    Reftru = circle.reflect_line(L, P_tru.value, None)

                    # Try and intersect
                    Pval, alpha = surf.intersection_point(L, up(centre3d))

                    if Pval[0] == -1:

                        # gs = new_scene.as_scene()
                        # surf.plot_probe_func(L)
                        #
                        # talphas = surf.intersection_func(L.value)
                        # print(talphas)
                        # print(surf.bound_func(L.value))
                        # print(alpha)
                        # print(surf.intersect_at_alpha(L, up(centre3d), talphas[0]))
                        # print('\n\n\n\n', flush=True)
                        #
                        # hit_error += 1
                        # gs.add_objects([L], color=Color.BLACK)
                        # gs.add_objects([P_tru], color=Color.BLUE)
                        # gs.add_objects([L])
                        # draw(gs)
                        continue
                    else:
                        P = layout.MultiVector(value=Pval)

                    # Try and calculate the reflections
                    Ref = surf.reflect_line(L, P, alpha)

                    # See if we have been successful
                    if np.sum(np.abs(P_tru.value - Pval)) > 1E-3:
                        point_error += 1
                        pe = True
                        gs.add_objects([L], color=Color.BLACK)
                        gs.add_objects([Reftru], color=Color.BLUE)
                        gs.add_objects([Ref], color=Color.RED)
                        gs.add_objects([P_tru], color=Color.BLUE)
                        gs.add_objects([P], color=Color.RED)
                    if np.sum(np.abs(Ref.value - Reftru.value)) > 1E-3:
                        ref_error += 1
                        if pe:
                            double_error += 1
                        else:
                            gs.add_objects([L], color=Color.BLACK)
                            gs.add_objects([Reftru], color=Color.BLUE)
                            gs.add_objects([Ref], color=Color.RED)
                            gs.add_objects([P_tru], color=Color.BLUE)
                            gs.add_objects([P], color=Color.RED)

        print('\n')
        print('Hit errors: ', hit_error, 'of ', total_hits, 'ie. ', 100 * hit_error / total_hits, '%')
        print('Point errors: ', point_error, 'of ', total_hits, 'ie. ', 100 * point_error / total_hits, '%')
        print('Reflection errors: ', ref_error, 'of ', total_hits, 'ie. ', 100 * ref_error / total_hits, '%')
        print('Double errors: ', double_error, 'of ', total_hits, 'ie. ', 100 * double_error / total_hits, '%')
        print('\n', flush=True)
        draw(gs, scale=0.5)

    def test_spline_hit_diff(self):

        origin = (10*e3+5*e2-e1)
        C1 = (up(e1 + e2)^up(e1 -e2)^up(e1 + e3)).normal()
        C2 = (up(-e1 + e2 + 5*e3)^up(-e1 -e2 + 5*e3)^up(-e1 + e3 + 5*e3)).normal()
        S = unsign_sphere(C1.join(C2)(4)).normal()
        Cin = up(down((C1 + C2)*einf*(C1 + C2))+e1)
        L = (up(origin)^Cin^einf).normal()

        surf = CircleSurface(C1, C2, np.array([0., 0., 1.]), 1., 100., .5, 1., 0.)

        pointofXsurface(L, surf, origin)

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
        plt.plot(surf.probe_alphas, trace_check)
        plt.show()

        #draw([up(origin), L, C1, C2] + interp_list)



if __name__ == '__main__':
    unittest.main()
