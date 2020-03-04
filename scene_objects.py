
import numba
import numpy as np
from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
from pyganja import *
from math_utils import *


def new_sphere(p1, p2, p3, p4):
    return unsign_sphere(normalised(up(p1) ^ up(p2) ^ up(p3) ^ up(p4)))


def new_circle(p1, p2, p3):
    return normalised(up(p1) ^ up(p2) ^ up(p3))


def new_plane(p1, p2, p3):
    return normalised(up(p1) ^ up(p2) ^ up(p3) ^ einf)


def new_line(p1, p2):
    return normalised(up(p1) ^ up(p2) ^ einf)


def new_point_pair(p1, p2):
    return normalised(up(p1) ^ up(p2))


def unsign_sphere(S):
    return normalised(S/(S.dual()|einf)[0])

@numba.njit
def val_unsign_sphere(S_val):
    return val_normalised(S_val / imt_func(dual_func(S_val), einf.value)[0])




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
    return val_pointofXSphere(ray.value, sphere.value, origin.value)


@numba.njit
def val_pointofXplane(ray_val, plane_val, origin_val):
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
    return val_pointofXplane(ray.value, plane.value, origin.value)


def val_pointofXcircle(ray_val, circle_val, origin_val):
    m = meet_val(ray_val, circle_val)
    if (np.abs(m) <= 0.000001).all():
        return np.array([-1.])
    elif gmt_func(m, m)[0] <= 0.00001:
        return val_pointofXplane(ray_val, omt_func(circle_val, einf.value), origin_val)
    else:
        return np.array([-1.])


def pointofXcircle(ray, circle, origin):
    return val_pointofXcircle(ray.value, circle.value, origin.value)


def pointofXsurface(L, surf, origin):
    return surf.intersection_point(L, origin)


def cosangle_between_lines(l1, l2):
    return (l1 | l2)[0]


def getfattconf(inner_prod, a1, a2, a3):
    return min(1./(a1 + a2 * np.sqrt(-inner_prod) - a3*inner_prod), 1.)


def getfatt(d, a1, a2, a3):
    return min(1./(a1 + a2*d + a3*d*d), 1.)


def reflect_in_sphere(ray, sphere, pX):
    return normalised((pX|(sphere*ray*sphere))^einf)


@numba.njit
def val_interp_objects_root(C1_val, C2_val, alpha):
    C_temp = (1-alpha) * C1_val + alpha * C2_val
    return val_normalised(neg_twiddle_root_val(C_temp)[0])


def my_interp_objects_root(C1, C2, alpha):
    return layout.MultiVector(value=val_interp_objects_root(C1.value, C2.value, alpha))




@numba.njit
def val_differentiateLinearCircle(alpha, C1_val, C2_val):
    X_val = alpha*C1_val + (1-alpha) * C2_val

    phiSquared = -gmt_func(X_val, adjoint_func(X_val))
    phiSq0 = phiSquared[0]
    phiSq4 = project_val(phiSquared,4)

    dotz = C1_val - C2_val
    dotphiSq0 = 2*alpha*gmt_func(C1_val,C1_val)[0] - 2*(1-alpha)*gmt_func(C2_val,C2_val)[0] + (1-2*alpha)*(gmt_func(C1_val,C2_val)+gmt_func(C2_val,C1_val))[0]
    dotphiSq4 = (1-2*alpha) * project_val(gmt_func(C1_val,C2_val)+gmt_func(C2_val,C1_val), 4)

    tempsqrt = np.sqrt(phiSq0**2 -  gmt_func(phiSq4, phiSq4)[0])
    dott = (dotphiSq0 + ((phiSq0*dotphiSq0) - gmt_func(phiSq4, dotphiSq4) -gmt_func(dotphiSq4, phiSq4))/tempsqrt)[0]

    t = phiSq0 + tempsqrt
    sqrt2t = np.sqrt(2*t)

    f = t/(sqrt2t)
    dotf = (3*dott)/(2*sqrt2t)

    g = phiSq4/(sqrt2t)
    dotg = (4*t*dotphiSq4 - dott *phiSq4)/(2*t*sqrt2t)

    k = (f*f - gmt_func(g,g)[0])

    dotk = (2*f*dotf - gmt_func(g,dotg) - gmt_func(dotg, g))[0]

    fminusg = -g
    fminusg[0] += f
    dotfminusdotg = -dotg
    dotfminusdotg[0] += dotf
    term1 = k*gmt_func(dotz, fminusg)
    term2 = k*gmt_func(dotfminusdotg, X_val)
    term3 = -dotk*gmt_func(fminusg, X_val)
    Calphadot = val_normalised(project_val(term1 + term2 + term3, 3))

    return Calphadot


@numba.njit
def val_differentiateLinearPointPair(alpha, C1_val, C2_val):
    X_val = alpha*C1_val + (1-alpha) * C2_val

    phiSquared = -gmt_func(X_val, adjoint_func(X_val))
    phiSq0 = phiSquared[0]
    phiSq4 = project_val(phiSquared,4)

    dotz = C1_val - C2_val
    dotphiSq0 = 2*alpha*gmt_func(C1_val,C1_val)[0] - 2*(1-alpha)*gmt_func(C2_val,C2_val)[0] + (1-2*alpha)*(gmt_func(C1_val,C2_val)+gmt_func(C2_val,C1_val))[0]
    dotphiSq4 = (1-2*alpha) * project_val(gmt_func(C1_val,C2_val)+gmt_func(C2_val,C1_val), 4)

    tempsqrt = np.sqrt(phiSq0**2 -  gmt_func(phiSq4, phiSq4)[0])
    dott = (dotphiSq0 + ((phiSq0*dotphiSq0) - gmt_func(phiSq4, dotphiSq4) -gmt_func(dotphiSq4, phiSq4))/tempsqrt)[0]

    t = phiSq0 + tempsqrt
    sqrt2t = np.sqrt(2*t)

    f = t/(sqrt2t)
    dotf = (3*dott)/(2*sqrt2t)

    g = phiSq4/(sqrt2t)
    dotg = (4*t*dotphiSq4 - dott *phiSq4)/(2*t*sqrt2t)

    k = (f*f - gmt_func(g,g)[0])

    dotk = (2*f*dotf - gmt_func(g,dotg) - gmt_func(dotg, g))[0]

    fminusg = -g
    fminusg[0] += f
    dotfminusdotg = -dotg
    dotfminusdotg[0] += dotf
    term1 = k*gmt_func(dotz, fminusg)
    term2 = k*gmt_func(dotfminusdotg, X_val)
    term3 = -dotk*gmt_func(fminusg, X_val)
    Calphadot = val_normalised(project_val(term1 + term2 + term3, 2))

    return Calphadot














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
        return -1. * reflect_in_sphere(L, self.object, pX)

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
    def __init__(self, C1, C2, colour, specular, spec_k, amb, diffuse, reflection):
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
        self.probe_alphas = np.linspace(0, 1, 1000)

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
                Evaluate the probes and get (up to 4) crossing points
                """
                alphas = -np.ones(4)
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
                            if n > 3:
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


class CircleSurface(InterpSurface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        interp_circle = my_interp_objects_root(self.first, self.second, alpha)
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
        C = my_interp_objects_root(self.first, self.second, alpha)
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
        Aplus = my_interp_objects_root(self.first, self.second, alpha + 0.001)
        Aminus = my_interp_objects_root(self.first, self.second, alpha - 0.001)
        A = my_interp_objects_root(self.first, self.second, alpha)
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


class PointPairSurface(InterpSurface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                # Lval = dual_func(L)
                for i in range(ntcmats):
                    output[i] = omt_func(pms[i, :], L)[31]
                return output

            self._probe_func = tcf
        return self._probe_func

    @property
    def bounding_sphere(self):
        """
        Finds an approximate bounding sphere for a set of circles
        """
        if self._bounding_sphere is None:
            self._bounding_sphere = unsign_sphere(self.first ^ self.second)
        return self._bounding_sphere

    def intersect_at_alpha(self, L, origin, alpha):
        """
        Given an intersection, return the point that is the intersection between
        the ray L and the surface at evolution parameter alpha

        This is specific per type of surface and hence needs overwriting
        """
        # For each alpha val make the plane associated with it
        interp_pp = my_interp_objects_root(self.first, self.second, alpha)
        ppl = normalised(interp_pp ^ einf)

        # Get the point
        point_val = midpoint_between_lines(L, ppl).value

        return point_val

    def get_numerical_normal(self, alpha, P):
        """
        Get the normal at of the surface at the point P that corresponds to alpha
        Via numerical techniques
        """
        Aplus = normalised(my_interp_objects_root(self.first, self.second, alpha + 0.001) ^ einf)
        Aminus = normalised(my_interp_objects_root(self.first, self.second, alpha - 0.001) ^ einf)
        A = my_interp_objects_root(self.first, self.second, alpha)
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
        C = my_interp_objects_root(self.first, self.second, alpha)
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
        if imt_func(ptest, dual_func(self.bounding_sphere.value))[0] > 0:
            return True
        else:
            return False

