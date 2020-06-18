
import numba
import numpy as np
from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
from meshing import *
from pyganja import *
from math_utils import *
from derivatives import *
from polynomial_root import val_jitted_potential_roots_circles, val_jitted_potential_roots_point_pairs

ninf_val = einf.value

import matplotlib.pyplot as plt


def new_sphere(p1, p2, p3, p4):
    """ Make a sphere from 4 3d points """
    return unsign_sphere(normalised(up(p1) ^ up(p2) ^ up(p3) ^ up(p4)))


def new_circle(p1, p2, p3):
    """ Make a circle from 3 3d points """
    return normalised(up(p1) ^ up(p2) ^ up(p3))


def new_plane(p1, p2, p3):
    """ Make a plane from 3 3d points """
    return normalised(up(p1) ^ up(p2) ^ up(p3) ^ einf)


def new_line(p1, p2):
    """ Make a line from 2 3d points """
    return normalised(up(p1) ^ up(p2) ^ einf)


def new_point_pair(p1, p2):
    """ Make a point pair from 2 3d points """
    return normalised(up(p1) ^ up(p2))


@numba.njit
def val_pointofXSphere(ray_val, sphere_val, origin_val):
    """
    Intersection of a ray and a sphere
    """
    B = meet_val(ray_val, sphere_val)
    if gmt_func(B,B)[0] > 0.000001:
        point_vals = val_point_pair_to_end_points(B)
        if imt_func(point_vals[0,:],origin_val)[0] > imt_func(point_vals[1,:],origin_val)[0]:
            return point_vals[0,:]
    output = np.zeros(32)
    output[0] = -1
    return output


def pointofXsphere(ray, sphere, origin):
    """
    Intersection of a ray and a sphere
    """
    return val_pointofXSphere(ray.value, sphere.value, origin.value)


@numba.njit
def val_pointofXplane(ray_val, plane_val, origin_val):
    """
    Intersection of a ray and a plane
    """
    pX = val_intersect_line_and_plane_to_point(ray_val, plane_val)
    if pX[0] == -1.:
        return pX
    new_line1 = omt_func(origin_val, omt_func(pX, ninf_val))
    if abs((gmt_func(new_line1, new_line1))[0]) < 0.00001:
        return np.array([-1.])
    if imt_func(ray_val, val_normalised(new_line1))[0] > 0:
        return pX
    return np.array([-1.])


def pointofXplane(ray, plane, origin):
    """
    Intersection of a ray and a plane
    """
    return val_pointofXplane(ray.value, plane.value, origin.value)


def val_pointofXcircle(ray_val, circle_val, origin_val):
    """
    Intersection of a ray and a circle
    """
    m = meet_val(ray_val, circle_val)
    if (np.abs(m) <= 0.000001).all():
        return np.array([-1.])
    elif gmt_func(m, m)[0] <= 0.00001:
        return val_pointofXplane(ray_val, omt_func(circle_val, einf.value), origin_val)
    else:
        return np.array([-1.])


def pointofXcircle(ray, circle, origin):
    """
    Intersection of a ray and a circle
    """
    return val_pointofXcircle(ray.value, circle.value, origin.value)


def pointofXsurface(L, surf, origin):
    """
    Intersection of a ray and a surface object
    """
    return surf.intersection_point(L, origin)


def cosangle_between_lines(l1, l2):
    """
    Calculate the cosign between the lines
    """
    return (l1 | l2)[0]


def getfattconf(inner_prod, a1, a2, a3):
    return min(1./(a1 + a2 * np.sqrt(-inner_prod) - a3*inner_prod), 1.)


def getfatt(d, a1, a2, a3):
    return min(1./(a1 + a2*d + a3*d*d), 1.)


def reflect_in_sphere(ray, sphere, pX):
    """ Reflects a ray in a sphere """
    pln = (pX|sphere) ^ einf
    return normalised((pln*ray*pln)(3))


@numba.njit
def val_interp_objects_root(C1_val, C2_val, alpha):
    """
    Interpolates C1 and C2 with alpha, note alpha is 1-alpha compared with cliford.tools.g3c
    """
    C_temp = (1-alpha) * C1_val + alpha * C2_val
    return val_normalised(neg_twiddle_root_val(C_temp)[0])


def alt_interp_objects_root(C1, C2, alpha):
    """
    Interpolates C1 and C2 with alpha, note alpha is 1-alpha compared with cliford.tools.g3c
    """
    return layout.MultiVector(value=val_interp_objects_root(C1.value, C2.value, alpha))



class Sphere:
    def __init__(self, c, r, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = new_sphere(c + r * e1, c + r * e2, c + r * e3, c - r * e1)
        self.colour = np.array(colour)
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Sphere"

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0] * 255), int(self.colour[1] * 255), int(self.colour[2] * 255))

    def intersection_point(self, L, origin):
        """
        Given there is an intersection this returns the point of intersection
        """
        return pointofXsphere(L, self.object, origin), None

    def reflect_line(self, L, pX, alpha):
        """
        Given there is an intersection this reflects the line off the object
        """
        return 1. * reflect_in_sphere(L, self.object, pX)

    def as_scene(self):
        gs = GanjaScene()
        gs.add_object(self.object, color=rgb2hex((self.colour * 255).astype(int)))
        return gs


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
        return "rgb(%d, %d, %d)" % (int(self.colour[0] * 255), int(self.colour[1] * 255), int(self.colour[2] * 255))

    def intersection_point(self, L, origin):
        """
        Given there is an intersection this returns the point of intersection
        """
        return pointofXplane(L, self.object, origin), None

    def reflect_line(self, L, pX, alpha):
        """
        Given there is an intersection this reflects the line off the object
        """
        return layout.MultiVector(value=(gmt_func(gmt_func(self.object.value, L.value), self.object.value)))

    def as_scene(self):
        gs = GanjaScene()
        gs.add_object(self.object, color=rgb2hex((self.colour * 255).astype(int)))
        return gs


class TriangularFacet(Plane):
    def __init__(self, p1, p2, p3, *args, **kwargs):
        self.A = up(p1)
        self.B = up(p2)
        self.C = up(p3)
        self.p1 = (self.A ^ self.C).normal()
        self.p2 = (self.A ^ self.B).normal()
        self.p3 = (self.C ^ self.B).normal()
        super().__init__(p1, p2, p3, *args, **kwargs)
        self.type = "Triangle"

    def does_line_hit(self, L):
        p1l = (self.p1 ^ L)[31]
        p2l = (self.p2 ^ L)[31]
        p3l = (self.p3 ^ L)[31]
        alpha = p2l / (p2l - p1l)
        beta = p3l / (p3l - p2l)
        #
        return alpha <= 1 and alpha >= 0 and beta <= 1 and beta >= 0

    def intersection_point(self, L, origin):
        if self.does_line_hit(L):
            return pointofXplane(L, self.object, origin), None
        else:
            return np.array([-1.]), None

    def as_scene(self):
        gs = GanjaScene()
        gs.add_facet([self.A, self.B, self.C], color=rgb2hex((self.colour * 255).astype(int)))
        return gs


class Circle:
    def __init__(self, p1, p2, p3, colour, specular, spec_k, amb, diffuse, reflection):
        self.object = -new_circle(p1, p2, p3)
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Circle"

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0] * 255), int(self.colour[1] * 255), int(self.colour[2] * 255))

    def intersection_point(self, L, origin):
        """
        Given there is an intersection this returns the point of intersection
        """
        return pointofXcircle(L, self.object, origin), None

    def reflect_line(self, L, pX, alpha):
        """
        Given there is an intersection this reflects the line off the object
        """
        return layout.MultiVector(value=(gmt_func(gmt_func(val_normalised(
            omt_func(self.object.value, einf.value)), L.value),
            val_normalised(omt_func(self.object.value, einf.value)))))

    def as_scene(self):
        gs = GanjaScene()
        gs.add_object(self.object, color=rgb2hex((self.colour * 255).astype(int)))
        return gs


class InterpSurface:
    def __init__(self, C1, C2, colour=[0,0,0], specular=0, spec_k=0, 
                 amb=0, diffuse=0, reflection=0, nprobe_alphas=1000):
        self.first = C1
        self.second = C2
        self.colour = colour
        self.specular = specular
        self.spec_k = spec_k
        self.ambient = amb
        self.diffuse = diffuse
        self.reflection = reflection
        self.type = "Surface"
        self._probes = None
        self._probe_func = None
        self._intersection_func = None
        self._bounding_sphere = None
        self.probe_alphas = np.linspace(0, 1, nprobe_alphas)

        self.grade = grade_obj(C1)

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0] * 255), int(self.colour[1] * 255), int(self.colour[2] * 255))

    @property
    def bounding_sphere(self):
        """
        Gets an approximate bounding sphere around the surface to accelerate rendering

        This is specific per type of evolution object and hence needs overwriting
        """
        raise NotImplementedError('bounding_sphere has not been defined in the child class')

    @property
    def probe_func(self):
        """
        Gets the evolution paramater at points of intersection

        This is specific per type of evolution object and hence needs overwriting
        """
        raise NotImplementedError('probe_func has not been defined in the child class')

    def intersect_at_alpha(self, L, origin, alpha):
        """
        Given an intersection, return the point that is the intersection between
        the ray L and the surface at evolution parameter alpha

        This is specific per type of evolution object and hence needs overwriting
        """
        raise NotImplementedError('probe_func has not been defined in the child class')

    def reflect_line(self, L, pX, alpha):
        """
        Given there is an intersection this reflects the line off the object

        This is specific per type of evolution object and hence needs overwriting
        """
        pass

    @property
    def bound_func(self):
        sphere_val = self.bounding_sphere.value

        @numba.njit
        def bound_hit(ray_val):
            B = meet_val(ray_val, sphere_val)
            if gmt_func(B, B)[0] > 0.000001:
                return True
            else:
                return False

        return bound_hit

    @property
    def probes(self):
        if self._probes is None:
            self._probes = [interp_objects_root(self.first, self.second, alpha)(self.grade) for alpha in self.probe_alphas]
        return self._probes

    @property
    def intersection_func(self):
        if self._intersection_func is None:
            pfunc = self.probe_func
            palphas = self.probe_alphas
            bfunc = self.bound_func

            @numba.njit
            def intersect_line(Lval):
                """
                Evaluate the probes and get (up to 6) crossing points
                """
                alphas = -np.ones(6)
                if bfunc(Lval):
                    res = pfunc(Lval)
                    n = 0
                    m1 = np.sign(res[0])
                    for i in range(1, len(res)):
                        m2 = np.sign(res[i])
                        if m2 != m1:  # The normal case is a change of sign
                            if i > 1:
                                alphas[n] = get_root(palphas[i - 2:i + 1], res[i - 2:i + 1])
                            else:
                                alphas[n] = get_root(palphas[i - 1:i + 2], res[i - 1:i + 2])
                            n = n + 1
                            if n > 5:
                                break
                        m1 = m2
                return alphas

            self._intersection_func = intersect_line
        return self._intersection_func

    def as_scene(self):
        gs = GanjaScene()
        nprobes = len(self.probes)
        for i in range(20):
            p = self.probes[int(i * nprobes / 20)]
            gs.add_object(p.normal(), color=rgb2hex((self.colour * 255).astype(int)))
        return gs

    def intersection_point(self, L, origin):
        """
        Given there is an intersection this returns the point of intersection and the
        evolution parameter at that point
        """

        # Find the intersection and select the ones within valid range
        alpha_vals = self.intersection_func(L.value)
        alpha_in_vals = [a for a in alpha_vals if a < 1 and a > 0]

        # Check if it misses entirely
        if len(alpha_in_vals) < 1:
            return np.array([-1.]), None

        # Calc the intersection points
        intersection_points = np.zeros((len(alpha_in_vals), 32))
        nskip = 0
        for i, alp in enumerate(alpha_in_vals):
            ptest = self.intersect_at_alpha(L, origin, alp)
            if self.test_point(ptest):
                intersection_points[i - nskip, :] = ptest
            else:
                nskip += 1
        intersection_points = intersection_points[0:len(alpha_in_vals) - nskip, :]
        if len(alpha_in_vals) - nskip == 0:
            return np.array([-1.]), None

        # Calc the closest intersection point to the origin
        closest_ind = int(np.argmax([imt_func(p, origin.value)[0] for p in intersection_points]))
        return intersection_points[closest_ind, :], alpha_in_vals[closest_ind]

    def test_point(self, ptest):
        return True

    def plot_probe_func(self, L):
        """
        Plots the intersection function
        """
        res = self.probe_func(L.value)
        plt.plot(self.probe_alphas, res)
        plt.plot(self.probe_alphas, self.probe_alphas*0, 'r')
        plt.show()


class CircleSurface(InterpSurface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_intersection_func_to_polynomial(self):
        def intersect_line(Lval):
            return val_jitted_potential_roots_circles(self.first.value, self.second.value, Lval)
        self._intersection_func = intersect_line

    @property
    def probe_func(self):
        """
        This generates a function that takes the meet squared and takes the scalar value of it
        """
        if self._probe_func is None:
            ntcmats = len(self.probes)
            pms = np.array([dual_func(p.value) for p in self.probes])

            @numba.njit
            def tcf(L):
                output = np.zeros(ntcmats)
                Lval = dual_func(L)
                for i in range(ntcmats):
                    val = omt_func(pms[i, :], Lval)
                    output[i] = imt_func(val, val)[0]
                return output

            self._probe_func = tcf
        return self._probe_func

    @property
    def bounding_sphere(self):
        """
        Finds an approximate bounding sphere for a set of circles
        """
        if self._bounding_sphere is None:
            self._bounding_sphere = enclosing_sphere([circle_to_sphere(C) for C in self.probes])
        return self._bounding_sphere

    def intersect_at_alpha(self, L, origin, alpha):
        """
        Given an intersection, return the point that is the intersection between
        the ray L and the surface at evolution parameter alpha

        This is specific per type of surface and hence needs overwriting
        """
        # For each alpha val make the plane associated with it
        interp_circle = alt_interp_objects_root(self.first, self.second, alpha)
        plane1_val = val_normalised(omt_func(interp_circle.value, einf.value))

        # Check if the line lies in this plane
        if np.sum(np.abs(meet(interp_circle, L).value)) < 1E-3:
            # Intersect as it it were a sphere
            S = circle_to_sphere(interp_circle)
            return val_pointofXSphere(L.value, unsign_sphere(S).value, origin.value)
        else:
            return val_pointofXplane(L.value, plane1_val, origin.value)

    def get_analytic_normal(self, alpha, P):
        """
        Get the normal at of the surface at the point P that corresponds to alpha
        Via a closed form expression
        """
        dotC = val_differentiateLinearCircle(alpha, self.second.value, self.first.value)
        dotC = layout.MultiVector(value=dotC)
        C = alt_interp_objects_root(self.first, self.second, alpha)
        omegaC = C * dotC
        dotP = P | omegaC
        LT = (dotP ^ P ^ einf)
        LC = ((C | P) ^ einf)
        normal = (LT * LC * I5)(3).normal()
        return normal

    def get_numerical_normal(self, alpha, P):
        """
        Get the normal at of the surface at the point P that corresponds to alpha
        Via numerical techniques
        """
        Aplus = alt_interp_objects_root(self.first, self.second, alpha + 0.001)
        Aminus = alt_interp_objects_root(self.first, self.second, alpha - 0.001)
        A = alt_interp_objects_root(self.first, self.second, alpha)
        Pplus = project_points_to_circle([P], Aplus)[0]
        Pminus = project_points_to_circle([P], Aminus)[0]
        CA = (Pminus ^ P ^ Pplus)
        Tangent_CA = ((CA | P) ^ einf)
        Tangent_A = ((A | P) ^ einf)
        return -normalised((Tangent_A * Tangent_CA * I5)(3))

    def reflect_line(self, L, pX, alpha):
        """
        Reflects a line in the surface
        """
        normal = normalised(self.get_analytic_normal(alpha, pX))
        return normalised((-normal * L * normal)(3))

    def mesh(self, n_points=21, n_alpha=21):
        """
        Meshes the surface
        """
        vertex_list, face_list, texture_coords = mesh_circle_surface(self.first, self.second, n_points=n_points, n_alpha=n_alpha)
        return vertex_list, face_list, texture_coords

    def vertex_normal_lines(self, ga_vertices, alpha_list):
        """
        For a list of surface vertices ga_vertices this returns the 
        normal to the surface at that point
        """
        npoints = len(ga_vertices)/len(alpha_list)
        output_lines = []
        for i, P in enumerate(ga_vertices):
            alpha = alpha_list[int(i/npoints)]
            output_lines.append( self.get_analytic_normal(alpha, P) )
        return output_lines

    def as_mesh_scene(self, n_points=21, n_alpha=21):
        """
        Meshes the surface and adds it to a GanjaScene
        """
        ga_vertices, face_list, texture_coords = self.mesh(n_points=n_points, n_alpha=n_alpha)
        return get_facet_scene(ga_vertices, face_list)


class PointPairSurface(InterpSurface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diff_probes = None

    def set_intersection_func_to_polynomial(self):
        def intersect_line(Lval):
            potential_roots = val_jitted_potential_roots_point_pairs(self.first.value, self.second.value, Lval)
            return potential_roots
        self._intersection_func = intersect_line

    @property
    def diff_probes(self):
        if self._diff_probes is None:
            self._diff_probes = [project_val(val_differentiateLinearPointPair(alpha, self.first.value, self.second.value), self.grade) for alpha in
                            self.probe_alphas]
        return self._diff_probes

    @property
    def probe_func(self):
        """
        This generates a function that takes the meet and takes the scalar value of it
        """
        if self._probe_func is None:
            ntcmats = len(self.probes)
            pms = np.array([p.value for p in self.probes])

            @numba.njit
            def tcf(L):
                output = np.zeros(ntcmats)
                for i in range(ntcmats):
                    output[i] = omt_func(pms[i, :], L)[31]
                return output

            self._probe_func = tcf
        return self._probe_func

    @property
    def bounding_sphere(self):
        """
        Finds a bounding sphere for two point pairs
        """
        if self._bounding_sphere is None:
            if np.sum(np.abs(self.first ^ self.second)) < 1E-6:
                Sim = (point_pair_to_end_points(self.first)[0] + point_pair_to_end_points(self.second)[1])
                r = np.abs(get_radius_from_sphere((I5*Sim).normal()))
                S = (normalise_n_minus_1((Sim*einf*Sim)(1)) - 0.5*r*r*einf)*I5
                self._bounding_sphere = unsign_sphere(S)
            else:
                self._bounding_sphere = unsign_sphere(self.first ^ self.second)
        return self._bounding_sphere

    def intersect_at_alpha(self, L, origin, alpha):
        """
        Given an intersection, return the point that is the intersection between
        the ray L and the surface at evolution parameter alpha

        This is specific per type of surface and hence needs overwriting
        """
        # For each alpha val make the plane associated with it
        interp_pp = alt_interp_objects_root(self.first, self.second, alpha)
        ppl = normalised(interp_pp ^ einf)

        # Get the point
        point_val = midpoint_between_lines(L, ppl).value

        return point_val

    def get_numerical_normal(self, alpha, P):
        """
        Get the normal at of the surface at the point P that corresponds to alpha
        Via numerical techniques
        """
        Aplus = normalised(alt_interp_objects_root(self.first, self.second, alpha + 0.001) ^ einf)
        Aminus = normalised(alt_interp_objects_root(self.first, self.second, alpha - 0.001) ^ einf)
        A = alt_interp_objects_root(self.first, self.second, alpha)
        Pplus = project_points_to_line([P], Aplus)[0]
        Pminus = project_points_to_line([P], Aminus)[0]
        CA = (Pminus ^ P ^ Pplus)
        Tangent_CA = ((CA | P) ^ einf)
        Tangent_A = normalised(A ^ einf)
        return -normalised((Tangent_A * Tangent_CA * I5)(3))

    def get_analytic_normal(self, alpha, P):
        """
        Get the normal at of the surface at the point P that corresponds to alpha
        Via a closed form expression
        """
        dotC = val_differentiateLinearPointPair(alpha, self.second.value, self.first.value)
        dotC = layout.MultiVector(value=dotC)
        C = alt_interp_objects_root(self.first, self.second, alpha)
        omegaC = C * dotC
        dotP = P | omegaC
        LT = (dotP ^ P ^ einf)
        LC = (C ^ einf).normal()
        normal = (LT * LC * I5)(3).normal()
        return normal

    def reflect_line(self, L, pX, alpha):
        """
        Reflects a line in the surface
        """
        normal = normalised(self.get_analytic_normal(alpha, pX))
        return normalised((-normal * L * normal)(3))

    def test_point(self, ptest):
        """
        Tests a point to see if it is inside the bounding sphere
        """
        if imt_func(ptest, dual_func(self.bounding_sphere.value))[0] >= 0:
            return True
        else:
            return False


def generate_obj_circles():
    n_alpha = 101
    n_points = 71

    for i in range(100):
        print(i)
        C1 = random_circle()
        C2 = random_circle()

        csurf = CircleSurface(C1, C2)
        vertex_list, face_list, texture_coords = csurf.mesh(n_points=n_points, n_alpha=n_alpha)

        alpha_list = [tc[1] for tc in texture_coords]
        normal_lines = csurf.vertex_normal_lines(vertex_list, alpha_list)
        threedns = [-((n*I5)(e123)*I3).normal().value[1:4] for n in normal_lines]

        threedvertexlist = [down(v).value[1:4] for v in vertex_list]
        write_obj_file("surfaces/test{:02d}.obj".format(i), 
            threedvertexlist, face_list, threedns, texture_coords, use_mtl=True)



if __name__ == '__main__':
    generate_obj_circles()