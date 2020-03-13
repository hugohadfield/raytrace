from clifford.tools.g3c import *
from scene_objects import *

import matplotlib.pyplot as plt
import matplotlib

from pyganja import *

import numpy as np
import time
import cProfile


red = 'rgb(255, 0 , 0)'
blue = 'rgb(0, 0, 255)'
green = 'rgb(0,255, 0)'
yellow = 'rgb(255, 255, 0)'
magenta = 'rgb(255, 0, 255)'
cyan = 'rgb(0,255,255)'
black = 'rgb(0,0,0)'
dark_blue = 'rgb(8, 0, 84)'
db = [0.033, 0., 0.33]


class Light:
    """ A point light source """
    def __init__(self, position, colour):
        self.position = position
        self.colour = colour

    def as_scene(self):
        gs = GanjaScene()
        gs.add_object((up(self.position) - 0.5*einf)*I5, color=Color.YELLOW)
        print((up(self.position) - 0.5*einf)*I5)
        return gs


class Camera:
    """ A perspective pin hole camera """
    def __init__(self, centre3d, lookat, f, height, width):

        self.upcam = up(centre3d)
        self.lookat = lookat
        self.xmax = 1.0
        self.ymax = self.xmax * (height * 1.0 / width)
        self.Ptl = f * 1.0 * e2 - e1 * self.xmax + e3 * self.ymax
        self.width = width
        self.height = height

        # Get all of the required initial transformations
        optic_axis = new_line(centre3d, lookat)
        self.original = new_line(eo, e2)
        MVR = generate_translation_rotor(centre3d - lookat) * rotor_between_lines(self.original, optic_axis)
        self.MVR = MVR
        dTx = MVR * generate_translation_rotor((2 * self.xmax / (width - 1)) * e1) * ~MVR
        dTy = MVR * generate_translation_rotor(-(2 * self.ymax / (height - 1)) * e3) * ~MVR
        self.dTx = dTx
        self.dTy = dTy


class RayScene:
    """ A ray traced scene """
    def __init__(self, camera=None,
                 object_list=[],
                 light_list=[],
                 max_bounces=2,
                 background_color=np.zeros(3),
                 shading_options={}):

        self.obj_list = object_list
        self.light_list = light_list
        self.camera = camera
        self.background_color = background_color
        self.max_bounces = max_bounces
        self.shading_options = shading_options

    def as_scene(self):
        """ Generates a GanjaScene """
        Ptr = self.camera.Ptl + 2 * e1 * self.camera.xmax
        Pbl = self.camera.Ptl - 2 * e3 * self.camera.ymax
        Pbr = Ptr - 2 * e3 * self.camera.ymax
        rect = [self.camera.Ptl, Ptr, Pbr, Pbl]
        scorners = [self.RMVR(up(pnt)) for pnt in rect]
        cam_c_line = (self.camera.MVR * self.camera.original * ~self.camera.MVR).normal()
        tl = new_line(eo, self.camera.Ptl)
        tr = new_line(eo, Ptr)
        bl = new_line(eo, Pbl)
        br = new_line(eo, Pbr)
        lines = [tl, tr, br, bl]

        gs = GanjaScene()
        for s in self.obj_list:
            gs += s.as_scene()
        for l in self.light_list:
            gs += l.as_scene()
        gs.add_object(self.camera.upcam, color=Color.BLACK)
        gs.add_object(cam_c_line, color=Color.CYAN)
        gs.add_objects([self.RMVR(l) for l in lines], color=Color.BLUE)
        gs.add_objects(scorners, color=Color.BLACK)
        return gs

    def draw(self, scale=0.05, browser_window=True):
        """
        Draws the scene in pyganja
        """
        gs = self.as_scene()
        draw(gs, scale=scale, browser_window=browser_window)
        return gs

    def intersects(self, ray, obj_list, origin):
        """
        Intersects the ray with the object list
        """
        dist = -np.finfo(float).max
        index = None
        pXfin = None
        alphaFin = None
        for idx, obj in enumerate(obj_list):
            pX, alpha = obj.intersection_point(ray, origin)
            if pX[0] < -0.5:
                continue
            if idx == 0:
                dist, index, pXfin, alphaFin = imt_func(pX, origin.value)[0], idx, layout.MultiVector(value=pX), alpha
                continue
            t = imt_func(pX, origin.value)[0]
            if(t > dist):
                dist, index, pXfin, alphaFin = t, idx, layout.MultiVector(value=pX), alpha
        return pXfin, index, alphaFin


    def trace_ray(self, ray, obj_list, origin, depth):
        """ Trace a ray """
        # Initialise the pixel color
        pixel_col = np.zeros(3)

        # Check for intersections with the scene
        pX, index, alpha = self.intersects(ray, obj_list, origin)

        # If there is no intersection return the background color
        if index is None:
            return self.background_color
        # Otherwise get the object we have hit
        obj = obj_list[index]

        # Reflect the line in the object
        reflected = obj.reflect_line(ray, pX, alpha)
        # Get the normal
        norm = normalised(reflected - ray)

        # Iterate over the lights
        for light in self.light_list:

            # Calculate the lighting model
            upl_val = val_up(light.position.value)
            toL = layout.MultiVector(value=val_normalised(omt_func(omt_func(pX.value, upl_val), einf.value)))
            d = layout.MultiVector(value=imt_func(pX.value, upl_val))[0]

            if self.shading_options['ambient'] > 0:
                pixel_col += self.shading_options['ambient'] * obj.ambient * obj.colour

            # Check for shadows
            Satt = 1.
            if self.intersects(toL, obj_list[:index] + obj_list[index + 1:], pX)[0] is not None:
                Satt *= 0.8

            fatt = getfattconf(d, self.shading_options['a1'],
                               self.shading_options['a2'],
                               self.shading_options['a3'])

            if self.shading_options['specular']:
                pixel_col += Satt * fatt * obj.specular * \
                             max(cosangle_between_lines(norm, normalised(toL-ray)), 0) ** obj.spec_k * light.colour

            if self.shading_options['diffuse']:
                pixel_col += Satt * fatt * obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour * light.colour

        if depth >= self.max_bounces:
            return pixel_col
        pixel_col += obj.reflection * self.trace_ray(reflected, obj_list, pX, depth + 1) #/ ((depth + 1) ** 2)
        return pixel_col

    def RMVR(self, mv):
        """
        Apply the camera model view rotor
        """
        return apply_rotor(mv, self.camera.MVR)

    def render(self):
        """
        Ray trace the scene
        """
        img = np.zeros((self.camera.height, self.camera.width, 3))
        initial = self.RMVR(up(self.camera.Ptl))
        clipped = 0
        start_time = time.time()
        for i in range(0, self.camera.width):
            if i % 1 == 0:
                if i != 0:
                    t_current = time.time() - start_time
                    if t_current is not 0:
                        current_percent = (i/self.camera.width * 100)
                        percent_per_second = current_percent/t_current
                        t_est_total = 100/percent_per_second
                        print(i/self.camera.width * 100, "% complete",
                            t_current/60 , 'mins elapsed',
                            (t_est_total - t_current)/60, ' mins remaining')
            point = initial
            line = normalised(self.camera.upcam ^ initial ^ einf)
            for j in range(0, self.camera.height):
                # print("Pixel coords are; %d, %d" % (j, i))
                value = self.trace_ray(line, self.obj_list, self.camera.upcam, 0)
                new_value = np.clip(value, 0, 1)
                if np.any(value > 1.) or np.any(value < 0.):
                    clipped += 1
                img[j, i, :] = new_value * 255.
                point = apply_rotor(point, self.camera.dTy)
                line = normalised(self.camera.upcam ^ point ^ einf)

            initial = apply_rotor(initial, self.camera.dTx)
        # print("Total number of pixels clipped = %d" % clipped)
        print("--- %s seconds ---" % (time.time() - start_time))
        return img




def test_render_random_circle_scene():

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
    C1 = normalised((D1*random_circle()*~D1)(3))
    C2 = normalised((D1*random_circle()*~D1)(3))
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




def test_render_random_point_pair_scene():

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
    C1 = normalised((D1*random_point_pair()*~D1)(2))
    C2 = normalised((D1*random_point_pair()*~D1)(2))
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



def test_render_standard_point_pair_scene():

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

    C1 = normalised(up(5*e2 + -10 *e1 - 4 * e3) ^ up(5*e2 + -10 *e1 + 4 * e2))
    C2 = normalised(up(4*e2 + 10 * e1 - 3 * e3) ^ up(5*e2 + 10 * e1 + 5 * e3))

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



def test_render_triangle_facet():

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


def test_render_combined_scene():

    shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                       'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
    k = 1.0

    lights_list = []
    colour_light = np.ones(3)
    L = 40. * e3 - 20*e2
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
                        4 * e2 + 5 * e1 + 10* e3,
                        np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.1),
        Plane(-10*e3,-10*e3+e2,-10*e3+e1, np.array([0., 1., 1.]), k * 1., 100., k * .5, k * 1., k * 0.3),
        Sphere(-20*e1+10*e3+15*e2, 10, np.array([0,1,0]), k * 1., 100., k * .5, k * 1., k * 0.8),
    ]

    C1 = - (0.16887 ^ e123) - (0.75896 ^ e124) - (0.91709 ^ e125) - (0.92729 ^ e134) - (0.97853 ^ e135) + (0.63805 ^ e145) - (0.0462 ^ e234) - (0.12142 ^ e235) - (0.29479 ^ e245) - (0.39902 ^ e345)
    C2 = (0.36389 ^ e123) - (1.66662 ^ e124) - (1.81194 ^ e125) - (1.36982 ^ e134) - (1.52625 ^ e135) + (0.1694 ^ e145) + (0.84272 ^ e234) + (0.87727 ^ e235) + (0.17834 ^ e245) + (0.23224 ^ e345)
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


def benchmark():
    shading_options = {'ambient': 0.3, 'specular': True, 'diffuse': True,
                       'a1': 0.02, 'a2': 0.0, 'a3': 0.002}
    k = 1.0

    lights_list = []
    colour_light = np.ones(3)
    L = 40. * e3 - 20*e2
    lights_list.append(Light(L, colour_light))


    # Construct the camera
    camera_lookat = e1
    image_height = 240
    image_width = 320
    f = 1.
    centre3d = - 30. * e2 + 1. * e1
    scene_camera = Camera(centre3d, camera_lookat, f, image_height, image_width)

    # Make the objects
    object_list = [
        TriangularFacet(5 * e2 + -10 * e1 - 4 * e3,
                        5 * e2 + -10 * e1 + 4 * e2,
                        4 * e2 + 5 * e1 + 10* e3,
                        np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.1),
        Plane(-10*e3,-10*e3+e2,-10*e3+e1, np.array([0., 1., 1.]), k * 1., 100., k * .5, k * 1., k * 0.3),
        Sphere(-20*e1+10*e3+15*e2, 10, np.array([0,1,0]), k * 1., 100., k * .5, k * 1., k * 0.8),
    ]

    C1 = - (0.16887 ^ e123) - (0.75896 ^ e124) - (0.91709 ^ e125) - (0.92729 ^ e134) - (0.97853 ^ e135) + (0.63805 ^ e145) - (0.0462 ^ e234) - (0.12142 ^ e235) - (0.29479 ^ e245) - (0.39902 ^ e345)
    C2 = (0.36389 ^ e123) - (1.66662 ^ e124) - (1.81194 ^ e125) - (1.36982 ^ e134) - (1.52625 ^ e135) + (0.1694 ^ e145) + (0.84272 ^ e234) + (0.87727 ^ e235) + (0.17834 ^ e245) + (0.23224 ^ e345)
    C1 = average_objects([C1]).normal()
    C2 = average_objects([C2]).normal()

    object_list.append(
        CircleSurface(C2, C1, np.array([1., 0., 0.]), k * 1., 100., k * .5, k * 1., k * 0.3)
    )

    # Construct the scene
    new_scene = RayScene(camera=scene_camera,
                         light_list=lights_list,
                         object_list=object_list,
                         max_bounces=5,
                         shading_options=shading_options)

    # Render it all
    imrendered = new_scene.render()

    # Save and show the image
    plt.imsave('benchmark.png', imrendered.astype(np.uint8))
    # plt.imshow(imrendered.astype(np.uint8))
    # plt.show()

    print('MAX PIX: ', np.max(np.max(np.max(imrendered))))
    print('MIN PIX: ', np.min(np.min(np.min(imrendered))))


def test_benchmark():
    for i in range(5):
        benchmark()


if __name__ == "__main__":
    test_render_random_circle_scene()
