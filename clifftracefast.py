from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
from PIL import Image
import time
from numba import jitclass
from numba import double
from numba import prange

red = 'rgb(255, 0 , 0)'
blue = 'rgb(0, 0, 255)'
green = 'rgb(0,255, 0)'
yellow = 'rgb(255, 255, 0)'
magenta = 'rgb(255, 0, 255)'
cyan = 'rgb(0,255,255)'
black = 'rgb(0,0,0)'
dark_blue = 'rgb(8, 0, 84)'
db = [0.033, 0., 0.33]

ORIENTED_PLANE_CULLING = True


class Sphere:
    def __init__(self, c, r, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = new_sphere(c + r*e1, c + r*e2, c + r*e3, c - r*e1)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Sphere"

    def getColour(self):
        return "rgb(%d, %d, %d)"% (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))

    def getjitSphere(self):
        return jitSphere(self.object.value, self.colour, self.specular, self.spec_k, self.ambient, self.diffuse, self.reflection)

    def getnparray(self):
        out = np.empty((1, 32+5+3), dtype = np.double)
        out[0, 0:32] = self.object.value[:]
        out[0, -3:] = self.colour[:]
        out[0, 32] = self.specular
        out[0, 33] = self.spec_k
        out[0, 34] = self.ambient
        out[0, 35] = self.diffuse
        out[0, 36] = self.reflection
        return out



spec = [
    ('colour', double[:]),
    ('specular', double),
    ('spec_k', double),
    ('ambient', double),
    ('diffuse', double),
    ('reflection', double),
    ('object', double[:]),
]
@jitclass(spec)
class jitSphere:
    def __init__(self, object_val, colour, specular, spec_k, ambient, diffuse, reflection):
        self.object = object_val
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = ambient
        self.diffuse = diffuse
        self.reflection = reflection


    def intersect(self, ray_val, origin_val):
        B = meet_val(ray_val, self.object)
        if gmt_func(B, B)[0] > 0.000001:
            point_vals = val_point_pair_to_end_points(B)
            if imt_func(point_vals[0, :], origin_val)[0] > imt_func(point_vals[1, :], origin_val)[0]:
                return point_vals[0, :]
        output = np.zeros(32)
        output[0] = -1
        return output


    def reflect(self, ray_val, point_val):
        # return normalised((pX | (sphere * ray * sphere)) ^ einf)
        return val_normalised(omt_func(imt_func(point_val, gmt_func(self.object, gmt_func(ray_val, self.object))),
                                       ninf_val))

class Plane:
    def __init__(self, p1, p2, p3, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = new_plane(p1, p2, p3)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Plane"

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))

    def getjitPlane(self):
        return jitPlane(self.object.value, self.colour, self.specular, self.spec_k, self.ambient, self.diffuse,
                         self.reflection)

    def getnparray(self):
        out = np.empty((1, 32 + 5 + 3), dtype=np.double)
        out[0, 0:32] = self.object.value[:]
        out[0, -3:] = self.colour[:]
        out[0, 32] = self.specular
        out[0, 33] = self.spec_k
        out[0, 34] = self.ambient
        out[0, 35] = self.diffuse
        out[0, 36] = self.reflection
        return out

spec = [
    ('colour', double[:]),
    ('specular', double),
    ('spec_k', double),
    ('ambient', double),
    ('diffuse', double),
    ('reflection', double),
    ('object', double[:]),
]
@jitclass(spec)
class jitPlane:
    def __init__(self, object_val, colour, specular, spec_k, ambient, diffuse, reflection):
        self.object = object_val
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = ambient
        self.diffuse = diffuse
        self.reflection = reflection


    def intersect(self, ray_val, point_val):
        pX = val_intersect_line_and_plane_to_point(ray_val, point_val)
        if pX[0] == -1.:
            return pX
        new_line = omt_func(point_val, omt_func(pX, ninf_val))
        if abs((gmt_func(new_line, new_line))[0]) < 0.00001:
            return np.array([-1.])
        if imt_func(ray_val, val_normalised(new_line))[0] > 0:
            return pX
        return np.array([-1.])

    def reflect(self, ray_val, point_val = np.array([0.])):
        gmt_func(gmt_func(self.object.value, ray_val), self.object.value)


class Light:
    def __init__(self, position, colour):
        self.position = position
        self.colour = colour


def drawScene():
    Ptr = Ptl + 2*e1*xmax
    Pbl = Ptl - 2*e3*ymax
    Pbr = Ptr - 2*e3*ymax
    rect = [Ptl, Ptr, Pbr, Pbl]

    sc = GAScene()

    #Draw Camera transformation
    sc.add_line(original, red)
    sc.add_line((MVR*original*~MVR).normal(), red)
    sc.add_euc_point(up(cam), blue)
    sc.add_euc_point(up(lookat), blue)

    #Draw screen corners
    for points in rect:
        sc.add_euc_point(RMVR(up(points)), cyan)

    #Draw screen rectangle

    top = new_point_pair(Ptl, Ptr)
    right = new_point_pair(Ptr, Pbr)
    bottom = new_point_pair(Pbr, Pbl)
    left = new_point_pair(Pbl, Ptl)
    diag = new_point_pair(Ptl, Pbr)
    sides = [top, right, bottom, left, diag]
    for side in sides:
        sc.add_point_pair(RMVR(side), dark_blue)

    tl = new_line(eo, Ptl)
    tr = new_line(eo, Ptr)
    bl = new_line(eo, Pbl)
    br = new_line(eo, Pbr)

    lines = [tl, tr, br, bl]
    for line in lines:
        sc.add_line(RMVR(line).normal(), dark_blue)
    for objects in scene:
        if objects.type == "Sphere":
            sc.add_sphere(objects.object, objects.getColour())
        else:
            sc.add_plane(objects.object, objects.getColour())
    for light in Ls:
        l = light.position
        sc.add_euc_point(up(l), yellow)
        sc.add_sphere(new_sphere(l + e1, l+e2, l+e3, l-e1), yellow)

    print(sc)


def new_sphere(p1, p2, p3, p4):
    return unsign_sphere(normalised(up(p1) ^ up(p2) ^ up(p3) ^ up(p4)))


@numba.njit
def val_new_sphere(p1_val, p2_val, p3_val, p4_val):
    return val_unsign_sphere(val_normalised(omt_func(omt_func(omt_func(p1_val, p2_val),p3_val),p4_val)))

def new_plane(p1, p2, p3):
    return normalised(up(p1) ^ up(p2) ^ up(p3) ^ einf)


def new_line(p1, p2):
    return normalised(up(p1) ^ up(p2) ^ einf)


def new_point_pair(p1, p2):
    return normalised(up(p1) ^ up(p2))


@numba.njit
def val_unsign_sphere(Sval):
    return val_normalised(Sval/imt_func(dual_func(Sval), einf)[0])


def unsign_sphere(S):
    return normalised(S/(S.dual()|einf)[0])


@numba.njit
def val_pointofXSphere(ray_val, sphere_val, origin_val):
    B = meet_val(ray_val, sphere_val)
    if gmt_func(B,B)[0] > 0.000001:
        point_vals = val_point_pair_to_end_points(B)
        if imt_func(point_vals[0,:],origin_val)[0] > imt_func(point_vals[1,:],origin_val)[0]:
            return point_vals[0,:]
    output = np.zeros(32)
    output[0] = -1
    return output


def pointofXsphere(ray, sphere, origin):
    B = meet(ray, sphere)
    if (B**2)[0] > 0.000001:
        points = PointsFromPP(B)
        if(points[0] | origin)[0] > (points[1] | origin)[0]:
            return points[0]
    return None


@numba.njit
def val_pointofXplane(ray_val, plane_val, origin_val):
    pX = val_intersect_line_and_plane_to_point(ray_val, plane_val)
    if pX[0] == -1.:
        return pX
    new_line = omt_func(origin_val, omt_func(pX, ninf_val))
    if abs((gmt_func(new_line, new_line))[0]) < 0.00001:
        return np.array([-1.])
    if imt_func(ray_val, val_normalised(new_line))[0] > 0:
        return pX
    return np.array([-1.])


def pointofXplane(ray, plane, origin):
    p = val_pointofXplane(ray.value, plane.value, origin.value)
    if p[0] == -1.:
        return None
    return layout.MultiVector(value=p)


def cosangle_between_lines(l1, l2):
    return (l1 | l2)[0]

@numba.njit
def val_cosangle_between_lines(l1_val, l2_val):
    return imt_func(l1_val, l2_val)[0]


@numba.njit
def getfattconf(inner_prod, a1, a2, a3):
    return min(1./(a1 + a2 * np.sqrt(-inner_prod) - a3*inner_prod), 1.)


def getfatt(d, a1, a2, a3):
    return min(1./(a1 + a2*d + a3*d*d), 1.)


def PointsFromPP(mv):
    return point_pair_to_end_points(mv)


def reflect_in_sphere(ray, sphere, pX):
        return normalised((pX|(sphere*ray*sphere))^einf)


@numba.njit
def val_reflect_in_sphere(ray_val, sphere_val, point_val):
    return val_normalised(omt_func(imt_func(point_val, gmt_func(sphere_val, gmt_func(ray_val, sphere_val))),
                                   ninf_val))


@numba.njit
def intersects(ray, scene_type_array, scene_array, origin):
    dist = -np.inf
    index = None
    pXfin = None
    for idx in range(scene_array.shape[0]):
        if scene_type_array[idx] :
            pX = val_pointofXSphere(ray, scene_array[idx, :32], origin)
        else:
            if ORIENTED_PLANE_CULLING:
                if meet_val(ray,scene_array[idx, :32])[15] > 0:
                    continue
            pX = val_pointofXplane(ray, scene_array[idx, :32], origin)
        if pX[0] == -1.: continue
        if idx == 0:
            dist, index, pXfin = imt_func(pX, origin)[0] , idx , pX
            continue
        t = imt_func(pX, origin)[0]
        if(t > dist):
            dist, index, pXfin = t, idx, pX
    return pXfin, index



@numba.njit
def trace_ray(ray, scene_type_array, scene_array, origin, depth):
    pixel_col = np.zeros(3)
    pX, index = intersects(ray, scene_type_array, scene_array, origin)
    if index is None:
        return background
    obj = scene_array[int(index), :]

    '''
    NOTE:
    out[0, 0:32] = self.object.value[:]
    out[0, -3:] = self.colour[:]
    out[0, 32] = self.specular
    out[0, 33] = self.spec_k
    out[0, 34] = self.ambient
    out[0, 35] = self.diffuse
    out[0, 36] = self.reflection
    '''

    for i in range(0, lights.shape[0]):
        Satt = 1.
        upl_val = val_up(lights[i, :32])
        toL = val_normalised(omt_func(omt_func(pX, upl_val), ninf_val))
        d = imt_func(pX, upl_val)[0]

        if options_val[0] :
            pixel_col += ambient * obj[34] * obj[-3:]

        if intersects(toL, np.concatenate((scene_type_array[:index], scene_type_array[index+1:])),
                      np.concatenate((scene_array[:index, :], scene_array[index + 1:, :]), axis=0),
                      pX)[1] is not None:
            Satt *= 0.4

        if scene_type_array[int(index)]:
            reflected = -1. * val_reflect_in_sphere(ray, obj[:32], pX)
        else:
            reflected = gmt_func(gmt_func(obj[:32], ray), obj[:32])

        norm = val_normalised(reflected - ray)

        fatt = getfattconf(d, a1, a2, a3)

        if options_val[1] :
            pixel_col += Satt * fatt * obj[32] * \
                         max(val_cosangle_between_lines(norm, val_normalised(toL-ray)), 0) ** obj[33] \
                         * lights[i, -3:]

        if options_val[2] :
            pixel_col += Satt * fatt * obj[35] * max(val_cosangle_between_lines(norm, toL), 0) * obj[-3:]

    if depth >= max_depth:
        return pixel_col
    pixel_col += obj[36] * trace_ray(reflected, scene_type_array, scene_array, pX, depth + 1) \
                 / ((depth + 1) ** 2)
    return pixel_col


def RMVR(mv):
    return apply_rotor(mv, MVR)


@numba.njit
def my_clip(val, min_val, max_val):
    out = np.empty_like(val)
    for i in range(len(val)):
        if val[i] < min_val:
            out[i] = min_val
        elif val[i] > max_val:
            out[i] = max_val
        else:
            out[i] = val[i]
    return out

@numba.njit()
def generate_ray_array():
    initial = val_apply_rotor(val_up(Ptl_val), MVR_val)
    point_array = np.zeros((32,w,h))
    line_array = np.zeros((32,w,h))
    point_array[:,0,0] = initial
    for i in range(1,w):
        point_array[:,i,0] = val_apply_rotor(point_array[:,i-1,0], dTx_val)
        for j in range(1,h):
            point_array[:,i,j] = val_apply_rotor(point_array[:,i,j-1], dTy_val)
            line_array[:,i,j] = val_normalised(omt_func(upcam_val,omt_func(point_array[:,i,j], ninf_val)))
    return line_array

@numba.njit(parallel = True, nogil=True)
def render(line_array):
    img = np.zeros((h, w, 3))
    for i in prange(w):
        for j in prange(h):
            line = line_array[:,i,j]
            value = trace_ray(line, scene_type_val, scene_val, upcam_val, 0)
            value = my_clip(value, 0., 1.)
            img[j, i, :] = value * 255.
    return img


if __name__ == "__main__":
    # Light position and color.
    Ls =[]
    lights = np.empty((1, 32+3), dtype=np.double)
    L = -20.*e1 + 60.*e3 + 4.*e2
    colour_light = np.ones(3)
    light = np.append(L.value, colour_light)
    Ls.append(Light(L, colour_light))
    lights[0, :] = light
    L = 20.*e1 + 60.*e3 + 4.*e2
    lights = np.append(lights, np.append(L.value, colour_light).reshape(1, 35), axis = 0)
    Ls.append(Light(L, colour_light))

    # Shading options
    a1 = 0.02
    a2 = 0.0
    a3 = 0.002
    w = 400
    h = 300
    options = {'ambient': True, 'specular': True, 'diffuse': True}
    options_val = np.array([True, True , True])
    ambient = 0.3
    k = 1.  # Magic constant to scale everything by the same amount!
    max_depth = 2
    background = np.zeros(3) # [66./520., 185./510., 244./510.]

    # Add objects to the scene:
    scene = []
    scene.append(Sphere(-2.*e1 - 7.2*e2 + 4.*e3, 4., np.array([1., 0., 0.]), k*1., 100., k*1., k*1., k*0.1))
    scene.append(Sphere(6.*e1 - 2.0*e2 + 4.*e3, 4., np.array([0.2, 0.2, 0.2]), k*1., 100., k*0.2, k*1.0, k*1.))
    scene.append(Plane(20.*e2+ e1, 20.*e2, 21.*e2, np.array([0.8, 0.8, 0.8]), k*1., 100., k*1., k*1., k*0.1))

    scene_val = np.empty((len(scene), 32+5+3), dtype = np.double)
    scene_type_val = np.empty(len(scene), dtype = np.bool_)
    for idx, objects in enumerate(scene):
        scene_val[idx, :] = objects.getnparray()
        scene_type_val[idx] = 1 if objects.type == 'Sphere' else 0


    # Camera definitions
    cam = 4.*e3 - 20.*e2
    lookat = eo
    upcam = up(cam)
    upcam_val = upcam.value
    f = 1.
    xmax = 1.0
    ymax = xmax*(h*1.0/w)

    # No need to define the up vector since we're assuming it's e3 pre-transform.



    # Get all of the required initial transformations
    optic_axis = new_line(cam, lookat)
    original = new_line(eo, e2)
    MVR = generate_translation_rotor(cam-lookat)*rotor_between_lines(original, optic_axis)
    MVR_val = MVR.value
    dTx = MVR*generate_translation_rotor((2*xmax/(w-1))*e1)*~MVR
    dTx_val = dTx.value
    dTy = MVR*generate_translation_rotor(-(2*ymax/(h-1))*e3)*~MVR
    dTy_val = dTy.value

    Ptl = f*1.0*e2 - e1*xmax + e3*ymax
    Ptl_val = Ptl.value

    drawScene()

    print("Start Ray generation")
    start_time = time.time()
    ray_array = generate_ray_array()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Start Render")

    ORIENTED_PLANE_CULLING = True

    start_time = time.time()

    im1 = Image.fromarray(render(ray_array).astype('uint8'), 'RGB')
    im1.save('fig1.png')
    print("--- %s seconds ---" % (time.time() - start_time))
