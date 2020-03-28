import unittest
from clifftrace import *


class MyTestCase(unittest.TestCase):

    def test_render_random_circle_scene(self):
        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}

        k = 1.  # Magic constant to scale everything by the same amount!
        background_color = np.zeros(3)  # [66./520., 185./510., 244./510.]

        # Light position and color.
        lights_list = []
        L = -30. * e1 + 5. * e3 - 30. * e2
        colour_light = np.ones(3)
        lights_list.append(Light(L, colour_light))
        L = 30. * e1 + 5. * e3 - 30. * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1 + 5.5 * e3
        image_height = 160
        image_width = 200
        f = 1.
        centre3d = -25. * e2 + 1. * e1 + 5.5 * e3

        # Construct objects to render:
        object_list = []
        D1 = generate_dilation_rotor(0.5)
        C1 = normalised((D1 * random_circle() * ~D1)(3))
        C2 = normalised((D1 * random_circle() * ~D1)(3))
        object_list.append(
            CircleSurface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
        )

        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             background_color=background_color,
                             max_bounces=5,
                             shading_options=shading_options)

        # Have a look at what we are rendering
        gs = new_scene.as_scene()
        gs += object_list[0].as_mesh_scene()
        draw(gs, scale=0.1, browser_window=True)

        # Render it all
        imrendered = new_scene.render()

        # Save and show the image
        plt.imsave('Circle.png', imrendered.astype(np.uint8))
        plt.imshow(imrendered.astype(np.uint8))
        plt.show()

        print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
        print('MIN PIX: ', np.min(np.min(np.min(imrendered))))

    def test_render_random_point_pair_scene(self):
        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}

        k = 1.  # Magic constant to scale everything by the same amount!
        background_color = np.zeros(3)  # [66./520., 185./510., 244./510.]

        # Light position and color.
        lights_list = []
        L = -30. * e1 + 5. * e3 - 30. * e2
        colour_light = np.ones(3)
        lights_list.append(Light(L, colour_light))
        L = 30. * e1 + 5. * e3 - 30. * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1 + 5.5 * e3
        image_height = 80
        image_width = 100
        f = 1.
        centre3d = -25. * e2 + 1. * e1 + 5.5 * e3

        # Construct objects to render:
        object_list = []
        D1 = generate_dilation_rotor(0.5)
        C1 = normalised((D1 * random_point_pair() * ~D1)(2))
        C2 = normalised((D1 * random_point_pair() * ~D1)(2))
        object_list.append(
            PointPairSurface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
        )

        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             background_color=background_color,
                             max_bounces=2,
                             shading_options=shading_options)

        # Have a look at what we are rendering
        new_scene.draw()

        # Render it all
        imrendered = new_scene.render()

        # Save and show the image
        plt.imsave('PointPair.png', imrendered.astype(np.uint8))
        plt.imshow(imrendered)
        plt.show()

        print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
        print('MIN PIX: ', np.min(np.min(np.min(imrendered))))

    def test_render_standard_point_pair_scene(self):
        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
        k = 1.0

        lights_list = []
        L = -20. * e1 + 5. * e3 - 10. * e2
        colour_light = np.ones(3)
        lights_list.append(Light(L, colour_light))
        L = 20. * e1 + 5. * e3 - 10. * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1
        image_height = 480
        image_width = 600
        f = 1.
        centre3d = - 10. * e2 + 1. * e1
        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        C1 = normalised(up(5 * e2 + -10 * e1 - 4 * e3) ^ up(5 * e2 + -10 * e1 + 4 * e2))
        C2 = normalised(up(4 * e2 + 10 * e1 - 3 * e3) ^ up(5 * e2 + 10 * e1 + 5 * e3))

        object_list = []
        object_list.append(
            PointPairSurface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
        )

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             max_bounces=2,
                             shading_options=shading_options)

        # Have a look at what we are rendering
        new_scene.draw()

        # Render it all
        imrendered = new_scene.render()

        # Save and show the image
        plt.imsave('PointPairStandard.png', imrendered.astype(np.uint8))
        plt.imshow(imrendered)
        plt.show()

        print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
        print('MIN PIX: ', np.min(np.min(np.min(imrendered))))

    def test_render_triangle_facet(self):
        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
        k = 1.0

        lights_list = []
        L = -20. * e1 + 5. * e3 - 10. * e2
        colour_light = np.ones(3)
        lights_list.append(Light(L, colour_light))
        L = 20. * e1 + 5. * e3 - 10. * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1
        image_height = 80
        image_width = 100
        f = 1.
        centre3d = - 10. * e2 + 1. * e1
        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Make the facet
        p1 = 5 * e2 + -10 * e1 - 4 * e3
        p2 = 5 * e2 + -10 * e1 + 4 * e2
        p3 = 4 * e2 + 10 * e1 - 3 * e3
        object_list = []
        object_list.append(
            TriangularFacet(p1, p2, p3, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
        )

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             max_bounces=2,
                             shading_options=shading_options)

        # Have a look at what we are rendering
        new_scene.draw()

        # Render it all
        imrendered = new_scene.render()

        # Save and show the image
        plt.imsave('Triangle.png', imrendered.astype(np.uint8))
        plt.imshow(imrendered)
        plt.show()

        print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
        print('MIN PIX: ', np.min(np.min(np.min(imrendered))))

    def test_render_combined_scene(self):
        shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                           'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
        k = 1.0

        lights_list = []
        colour_light = np.ones(3)
        L = 40. * e3 - 20 * e2
        lights_list.append(Light(L, colour_light))

        # Construct the camera
        camera_lookat = e1
        image_height = 960
        image_width = 1280
        f = 1.
        centre3d = - 30. * e2 + 1. * e1
        scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

        # Make the objects
        object_list = [
            TriangularFacet(5 * e2 + -10 * e1 - 4 * e3,
                            5 * e2 + -10 * e1 + 4 * e2,
                            4 * e2 + 5 * e1 + 10 * e3,
                            np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.1),
            Plane(-10 * e3, -10 * e3 + e2, -10 * e3 + e1, np.array([0., 1., 1.]), k * 1., 100., k * .5, k * 1.,
                  k * 0.3),
            Sphere(-20 * e1 + 10 * e3 + 15 * e2, 10, np.array([0, 1, 0]), k * 1., 100., k * .5, k * 1., k * 0.8),
        ]

        C1 = - (0.16887 ^ e123) - (0.75896 ^ e124) - (0.91709 ^ e125) - (0.92729 ^ e134) - (0.97853 ^ e135) + (
                    0.63805 ^ e145) - (0.0462 ^ e234) - (0.12142 ^ e235) - (0.29479 ^ e245) - (0.39902 ^ e345)
        C2 = (0.36389 ^ e123) - (1.66662 ^ e124) - (1.81194 ^ e125) - (1.36982 ^ e134) - (1.52625 ^ e135) + (
                    0.1694 ^ e145) + (0.84272 ^ e234) + (0.87727 ^ e235) + (0.17834 ^ e245) + (0.23224 ^ e345)
        C1 = average_objects([C1]).normal()
        C2 = average_objects([C2]).normal()

        object_list.append(
            CircleSurface(C2, C1, np.array([1., 0., 0.]), k * 1., 100., k * .5, k * 1., k * 0.3)
        )
        print('C1', C1)
        print('C2', C2)

        # Construct the scene
        new_scene = RayScene(camera=scene_camera,
                             light_list=lights_list,
                             object_list=object_list,
                             max_bounces=5,
                             shading_options=shading_options)

        # Have a look at what we are rendering
        new_scene.draw()

        # Render it all
        imrendered = new_scene.render()

        # Save and show the image
        plt.imsave('Combined.png', imrendered.astype(np.uint8))
        plt.imshow(imrendered.astype(np.uint8))
        plt.show()

        print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
        print('MIN PIX: ', np.min(np.min(np.min(imrendered))))


if __name__ == '__main__':
    unittest.main()
