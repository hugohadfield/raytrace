from clifford.tools.g3c import *
from clifford.tools.g3c.GAOnline import *
import numpy as np
from PIL import Image
import time
from scipy.optimize import fsolve
import  scipy.interpolate

red = 'rgb(255, 0 , 0)'
blue = 'rgb(0, 0, 255)'
green = 'rgb(0,255, 0)'
yellow = 'rgb(255, 255, 0)'
magenta = 'rgb(255, 0, 255)'
cyan = 'rgb(0,255,255)'
black = 'rgb(0,0,0)'
dark_blue = 'rgb(8, 0, 84)'
db = [0.033, 0., 0.33]


@numba.njit
def nth_polynomial_fit(x, y, n):
    xmat = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            xmat[i,j] = x[i]**((n-j))
    return np.linalg.solve(xmat, y)

@numba.njit
def quad(x, p):
    return p[0]*x**2 + p[1]*x + p[2]

@numba.njit
def bisection(p, start, stop):
    fstart = quad(start, p)
    for __ in range(1000):
        half = start + (stop - start)/2
        fhalf = quad(half, p)
        if abs(fhalf) < 1e-12:
            return half 
        if fhalf * fstart > 0:
            start = half
            fstart = quad(start, p)
        else:
            stop = half
    return half

@numba.njit
def get_root(x, y):
    poly = nth_polynomial_fit(x, y, 2)
    return bisection(poly, x[0], x[2])

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
        return "rgb(%d, %d, %d)" % (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))

class Interp_Surface:
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
        self._probe_mats = None
        self._probe_func = None
        self._intersection_func = None
        self.probe_alphas = np.linspace(0,1,1000)

    def getColour(self):
        return "rgb(%d, %d, %d)" % (int(self.colour[0]*255), int(self.colour[1]*255), int(self.colour[2]*255))

    @property
    def probes(self):
        if self._probes is None:
            self._probes = [interp_objects_root(self.first, self.second, alpha) for alpha in self.probe_alphas]
        return self._probes

    @property
    def probe_mats(self):
        if self._probe_mats is None:
            dual_probes = [p*I5 for p in self.probes]
            self._probe_mats = np.array([mask4@get_left_gmt_matrix(dp.value) for dp in dual_probes])
        return self._probe_mats

    @property
    def probe_func(self):
        if self._probe_func is None:
            ntcmats = len( self.probes )
            pms = np.array([dual_func(p.value) for p in self.probes])
            @numba.njit
            def tcf(L):
                output = np.zeros(ntcmats)
                Lval = dual_func(L)
                for i in range(ntcmats):
                    val = omt_func(pms[i,:],Lval)
                    output[i] = imt_func(val,val)[0]
                return output
            self._probe_func = tcf
        return self._probe_func

    @property
    def intersection_func(self):
        if self._intersection_func is None:
            pfunc = self.probe_func
            palphas = self.probe_alphas
            @numba.njit
            def intersect_line(Lval):
                alphas = -np.ones(2)
                res = pfunc(Lval)
                n = 0
                m1 = np.sign(res[0])
                for i in range(1,len(res)):
                    m2 = np.sign(res[i])
                    if m2 != m1:
                        d0 = abs(res[i-1])
                        d1 = abs(res[i])
                        #print(res[i-2:i+1])
                        if i > 1:
                            alphas[n] = get_root(palphas[i-2:i+1],res[i-2:i+1])
                        else:
                            alphas[n] = get_root(palphas[i-1:i+2],res[i-1:i+2])
                        #alphas[n] = (d1*palphas[i-1] + d0*palphas[i])/(d0+d1)
                        n = n + 1
                        if n > 1:
                            break
                    m1 = m2
                return alphas
            self._intersection_func = intersect_line
        return self._intersection_func


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
    # sc.add_line(original, red)
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
        elif objects.type == "Plane":
            sc.add_plane(objects.object, objects.getColour())
        elif objects.type == "Circle":
            sc.add_circle(objects.object, objects.getColour())
        else:
            col = objects.getColour()
            sc.add_circle(objects.first, col)
            sc.add_circle(objects.second, col)
            for circles in [interp_objects_root(objects.first, objects.second, alpha/100) for alpha in range(1,100,20)]:
                sc.add_circle(circles, col)

    for light in lights:
        l = light.position
        sc.add_euc_point(up(l), yellow)
        sc.add_sphere(new_sphere(l + e1, l+e2, l+e3, l-e1), yellow)

    print(sc)


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
    new_line1 = omt_func(origin_val, omt_func(pX, ninf_val))
    if abs((gmt_func(new_line1, new_line1))[0]) < 0.00001:
        return np.array([-1.])
    if imt_func(ray_val, val_normalised(new_line1))[0] > 0:
        return pX
    return np.array([-1.])


def pointofXplane(ray, plane, origin):
    p = val_pointofXplane(ray.value, plane.value, origin.value)
    if p[0] == -1.:
        return None
    return layout.MultiVector(value=p)


def cosangle_between_lines(l1, l2):
    return (l1 | l2)[0]


def getfattconf(inner_prod, a1, a2, a3):
    return min(1./(a1 + a2 * np.sqrt(-inner_prod) - a3*inner_prod), 1.)


def getfatt(d, a1, a2, a3):
    return min(1./(a1 + a2*d + a3*d*d), 1.)


def PointsFromPP(mv):
    return point_pair_to_end_points(mv)


def reflect_in_sphere(ray, sphere, pX):
        return normalised((pX|(sphere*ray*sphere))^einf)


def pointofXcircle(ray_val, circle_val, origin_val):
    m = meet_val(ray_val, circle_val)
    if (np.abs(m) <= 0.000001).all():
        return np.array([-1.])
    elif gmt_func(m, m)[0] <= 0.00001:
        return val_pointofXplane(ray_val, omt_func(circle_val, einf.value), origin_val)
    else:
        return np.array([-1.])

@numba.njit
def val_interp_objects_root(C1_val, C2_val, alpha):
    C_temp = (1-alpha) * C1_val + alpha * C2_val
    return val_normalised(neg_twiddle_root_val(C_temp)[0])

def my_interp_objects_root(C1, C2, alpha):
    return layout.MultiVector(value = val_interp_objects_root(C1.value, C2.value, alpha))


def pointofXsurface(L, surf, origin):
    C1 = surf.first
    C2 = surf.second

    # Check if the ray hits the endpoints
    hit1 = (meet(L,C1)**2)[0] < 0
    hit2 = (meet(L,C2)**2)[0] < 0
    hiteither = hit1 or hit2

    # Check each
    zeros_crossing = [0, 1]

    probe_meet = surf.probe_func(L.value)
    alpha_vals = surf.intersection_func(L.value)
    # alpha_spline = scipy.interpolate.Akima1DInterpolator(alpha_probe, probe_meet)
    # alpha_vals = alpha_spline.roots()
    alpha_in_vals = [a for a in alpha_vals if a < 1 and a > 0]
    # print(alpha_in_vals, alpha_vals_t)
    success = 0

    if len(alpha_in_vals) == 1:
        zeros_crossing = [alpha_in_vals[0], alpha_in_vals[0]]
        success = 1
    elif len(alpha_in_vals) == 2:
        zeros_crossing = [alpha_in_vals[0], alpha_in_vals[1]]
        success = 1
    elif len(alpha_in_vals) > 2:
        success = 0
    elif len(alpha_in_vals) < 1:
        success = 0

    # Check if it misses entirely
    if success != 1:
        return np.array([-1.]), None

    if (zeros_crossing[0] < 0 or zeros_crossing[0] > 1) and  (zeros_crossing[1] < 0 or zeros_crossing[1] > 1):
        return np.array([-1.]), None

    # Check if it is in plane
    if np.abs(zeros_crossing[0] - zeros_crossing[1]) < 0.0000001:
        # Intersect as it it were a sphere
        C = interp_objects_root(C1, C2, zeros_crossing[0])
        S = (C * (C ^ einf).normal() * I5).normal()
        return val_pointofXSphere(L.value, unsign_sphere(S).value, origin.value), zeros_crossing[0]

    # Get intersection points
    plane1_val = val_normalised(omt_func(interp_objects_root(C1, C2, zeros_crossing[0]).value, einf.value))
    plane2_val = val_normalised(omt_func(interp_objects_root(C1, C2, zeros_crossing[1]).value, einf.value))

    p1_val = val_pointofXplane(L.value, plane1_val, origin.value)
    p2_val = val_pointofXplane(L.value, plane2_val, origin.value)

    if p1_val[0] == -1. and p2_val[0] == -1.:
        return np.array([-1.]), None
    if p2_val[0] == -1.:
        return p1_val, zeros_crossing[0]
    if p1_val[0] == -1.:
        return p2_val, zeros_crossing[1]
    if imt_func(p1_val, origin.value)[0] > imt_func(p2_val, origin.value)[0]:
        return p1_val, zeros_crossing[0]
    else:
        return p2_val, zeros_crossing[1]


def project_points_to_circle(point_list, circle):
    """
    Takes a load of point and projects them onto a circle
    """
    circle_plane = (circle^einf).normal()
    planar_points = project_points_to_plane(point_list,circle_plane)
    circle_points = project_points_to_sphere(planar_points, -circle*circle_plane*I5)
    return circle_points

@numba.njit
def val_differentiateLinearCircle(alpha, C1_val, C2_val):
    X_val = alpha*C1_val + (1-alpha) * C2_val
    phiSquared = -gmt_func(X_val, adjoint_func(X_val))
    phiSq0 = phiSquared[0]
    phiSq4 = project_val(phiSquared,4)
    dotz = C1_val - C2_val
    dotphiSq0 = 2*alpha*gmt_func(C1_val,C1_val)[0] - 2*(1-alpha)*gmt_func(C2_val,C2_val)[0] + (1-2*alpha)*(gmt_func(C1_val,C2_val)+gmt_func(C1_val,C1_val))[0]
    dotphiSq4 = (1-2*alpha) * project_val(gmt_func(C1_val,C2_val)+gmt_func(C1_val,C1_val), 4)
    tempsqrt = np.sqrt(phiSq0 -  gmt_func(phiSq4, phiSq4)[0])
    dott = (dotphiSq0 + (2*(phiSq0*dotphiSq0) - gmt_func(phiSq4, dotphiSq4) -gmt_func(dotphiSq4, phiSq4))/tempsqrt)[0]
    t = dotphiSq0 + tempsqrt
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
    dotfminusdotg[0] += f
    term1 = k*gmt_func(fminusg, dotz)
    term2 = k*gmt_func(dotfminusdotg, X_val)
    term3 = dotk*gmt_func(fminusg, X_val)
    return project_val(val_normalised(term1 + term2 + term3), 3)


def get_analytic_normal(C1,C2,alpha,P):
    dotC = val_differentiateLinearCircle(alpha, C2.value, C1.value)
    dotC = layout.MultiVector(value=dotC)
    C = interp_objects_root(C1, C2, alpha)
    omegaC = C*dotC
    dotP = P|omegaC
    LT = (dotP ^ P ^ einf).normal()
    LC = ((C|P)^einf).normal()
    normal = (LT*LC*I5)(3).normal()
    return normal


def get_numerical_normal(C1, C2, alpha, P):
    Aplus = interp_objects_root(C1,C2,alpha+0.001)
    Aminus = interp_objects_root(C1,C2,alpha-0.001)
    A = interp_objects_root(C1,C2,alpha)
    Pplus = project_points_to_circle([P], Aplus)[0]
    Pminus = project_points_to_circle([P], Aminus)[0]
    CA = (Pminus ^ P ^ Pplus).normal()
    Tangent_CA = ((CA | P) ^ einf).normal()
    Tangent_A = ((A | P) ^ einf).normal()
    return -((Tangent_A*Tangent_CA*I5)(3)).normal()

def reflect_in_surface(ray, object, pX, alpha):
    sc = GAScene()
    sc.add_euc_point(pX, blue)
    file.write(str(sc) + "\n")
    normal = get_analytic_normal(object.first, object.second, alpha, pX)
    return normal + ray


def intersects(ray, scene, origin):
    dist = -np.finfo(float).max
    index = None
    pXfin = None
    alpha = None
    alphaFin = None
    for idx, obj in enumerate(scene):
        if obj.type == "Sphere":
            pX = val_pointofXSphere(ray.value, obj.object.value, origin.value)
        if obj.type == "Plane":
            pX = val_pointofXplane(ray.value, obj.object.value, origin.value)
        if obj.type == "Circle":
            pX = pointofXcircle(ray.value, obj.object.value, origin.value)
        if obj.type == "Surface":
            pX, alpha = pointofXsurface(ray, obj, origin)

        if pX[0] == -1.: continue
        if idx == 0:
            dist, index, pXfin, alphaFin = imt_func(pX, origin.value)[0] , idx , layout.MultiVector(value=pX), alpha
            continue
        t = imt_func(pX, origin.value)[0]
        if(t > dist):
            dist, index, pXfin, alphaFin = t, idx, layout.MultiVector(value=pX), alpha
    return pXfin, index, alphaFin

file = open("intersection_points.txt", "a+")
def trace_ray(ray, scene, origin, depth):
    pixel_col = np.zeros(3)
    pX, index, alpha = intersects(ray, scene, origin)
    if index is None:
        return background
    obj = scene[index]
    sc = GAScene()
    if obj.type == "Sphere":
        sc.add_euc_point(pX)
    else:
        sc.add_euc_point(pX, green)
    file.write(str(sc) + "\n")
    for light in lights:
        Satt = 1.
        upl_val = val_up(light.position.value)
        toL = layout.MultiVector(value=val_normalised(omt_func(omt_func(pX.value, upl_val), einf.value)))
        d = layout.MultiVector(value=imt_func(pX.value, upl_val))[0]

        if options['ambient']:
            pixel_col += ambient * obj.ambient * obj.colour

        if intersects(toL, scene[:index] + scene[index+1:], pX)[0] is not None:
            Satt *= 0.8

        if obj.type == "Sphere":
            reflected = -1.*reflect_in_sphere(ray, obj.object, pX)
        elif obj.type == "Plane":
            reflected = layout.MultiVector(value=(gmt_func(gmt_func(obj.object.value, ray.value), obj.object.value)))
        elif obj.type == "Circle":
            reflected = layout.MultiVector(value=(gmt_func(gmt_func(val_normalised(
                omt_func(obj.object.value, einf.value)), ray.value), val_normalised(omt_func(obj.object.value,einf.value)))))
        else:
            reflected = reflect_in_surface(ray, obj, pX, alpha)


        norm = normalised(reflected - ray)

        # tmp_scene = GAScene()
        # tmp_scene.add_line(ray, red)
        # tmp_scene.add_line(norm, green)
        # tmp_scene.add_line(reflected, green)
        # print(tmp_scene)

        fatt = getfattconf(d, a1, a2, a3)

        if options['specular']:
            pixel_col += Satt * fatt * obj.specular * \
                         max(cosangle_between_lines(norm, normalised(toL-ray)), 0) ** obj.spec_k * light.colour

        if options['diffuse']:
            pixel_col += Satt * fatt * obj.diffuse * max(cosangle_between_lines(norm, toL), 0) * obj.colour * light.colour

    if depth >= max_depth:
        return pixel_col
    pixel_col += obj.reflection * trace_ray(reflected, scene, pX, depth + 1) #/ ((depth + 1) ** 2)
    return pixel_col


def RMVR(mv):
    return apply_rotor(mv, MVR)

def render():
    img = np.zeros((h, w, 3))
    initial = RMVR(up(Ptl))
    clipped = 0
    start_time = time.time()
    for i in range(0, w):
        if i % 1 == 0:
            if i != 0:
                t_current = time.time() - start_time
                current_percent = (i/w * 100)
                percent_per_second = current_percent/t_current
                t_est_total = 100/percent_per_second
                print(i/w * 100, "% complete", 
                    t_current/60 , 'mins elapsed',
                    (t_est_total - t_current)/60, ' mins remaining')
        point = initial
        line = normalised(upcam ^ initial ^ einf)
        for j in range(0, h):
            # print("Pixel coords are; %d, %d" % (j, i))
            value = trace_ray(line, scene, upcam, 0)
            new_value = np.clip(value, 0, 1)
            if np.any(value > 1.) or np.any(value < 0.):
                clipped += 1
            img[j, i, :] = new_value * 255.
            point = apply_rotor(point, dTy)
            line = normalised(upcam ^ point ^ einf)

        initial = apply_rotor(initial, dTx)
    # print("Total number of pixels clipped = %d" % clipped)
    return img

if __name__ == "__main__":
    # Light position and color.
    lights = []
    L = -30. * e1 + 5. * e3 - 30. * e2
    colour_light = np.ones(3)
    lights.append(Light(L, colour_light))
    L = 30. * e1 + 5. * e3 - 30. * e2
    lights.append(Light(L, colour_light))

    # Shading options
    a1 = 0.02
    a2 = 0.0
    a3 = 0.002
    w = 100
    h = 75
    options = {'ambient': True, 'specular': True, 'diffuse': True}
    ambient = 0.3
    k = 1.  # Magic constant to scale everything by the same amount!
    max_depth = 0
    background = np.zeros(3)  # [66./520., 185./510., 244./510.]

    # Add objects to the scene:
    scene = []
    rotorR1 = generate_rotation_rotor(np.pi / 6, e1, e3)
    rotorR2 = generate_rotation_rotor(-np.pi / 8, e1, e3)
    rotorR3 = generate_rotation_rotor(-np.pi / 4, e1, e2)
    rotorT2 = generate_translation_rotor(10 * e1 + 3 * e3 - 3 * e2)
    rotorT1 = generate_translation_rotor(-7 * e1)

    C1 = normalised(up(-4 * e3) ^ up(4 * e3) ^ up(4 * e2))
    C2 = normalised(up(-4 * e3) ^ up(4 * e3) ^ up(4 * e2))
    C1 = apply_rotor(C1, rotorR1)
    C2 = apply_rotor(C2, rotorR2)
    C2 = apply_rotor(C2, rotorR3)
    C1 = apply_rotor(C1, rotorT1)
    C2 = apply_rotor(C2, rotorT2)

    scene.append(
        Interp_Surface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
    )
    scene.append(
        Circle(e1, -e1, e3, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
    )
    scene.append(
        Circle(e1, -e1, e3, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
    )
    scene[1].object = C1
    scene[2].object = C2

    # Camera definitions
    cam = -25. * e2 + 1. * e1 + 5.5 * e3
    lookat = e1 + 5.5 * e3
    upcam = up(cam)
    f = 1.
    xmax = 1.0
    ymax = xmax * (h * 1.0 / w)
    # No need to define the up vector since we're assuming it's e3 pre-transform.
    # TODO: Need to encorporate the "up" vector into this model

    start_time = time.time()

    # Get all of the required initial transformations
    optic_axis = new_line(cam, lookat)
    original = new_line(eo, e2)
    MVR = generate_translation_rotor(cam - lookat) * rotor_between_lines(original, optic_axis)
    dTx = MVR * generate_translation_rotor((2 * xmax / (w - 1)) * e1) * ~MVR
    dTy = MVR * generate_translation_rotor(-(2 * ymax / (h - 1)) * e3) * ~MVR

    Ptl = f * 1.0 * e2 - e1 * xmax + e3 * ymax

    drawScene()

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestLatest.png')

    scene = [scene[0]]

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestLatestNoCaps.png')

    """
    Now render a sphere!
    """

    lights = []
    L = -20. * e1 + 5. * e3 - 10. * e2
    colour_light = np.ones(3)
    lights.append(Light(L, colour_light))
    L = 20. * e1 + 5. * e3 - 10. * e2
    lights.append(Light(L, colour_light))

    cam = - 10. * e2 + 1. * e1
    lookat = e1
    upcam = up(cam)

    optic_axis = new_line(cam, lookat)
    original = new_line(eo, e2)
    MVR = generate_translation_rotor(cam - lookat) * rotor_between_lines(original, optic_axis)
    dTx = MVR * generate_translation_rotor((2 * xmax / (w - 1)) * e1) * ~MVR
    dTy = MVR * generate_translation_rotor(-(2 * ymax / (h - 1)) * e3) * ~MVR

    Ptl = f * 1.0 * e2 - e1 * xmax + e3 * ymax

    # Used to generate sphere
    C1 = normalised(up(-4 * e3) ^ up(4 * e3) ^ up(4 * e2))

    C2 = normalised(up(5 * e1 - 4 * e3) ^ up(5 * e1 + 4 * e3) ^ up(5 * e1 + 4 * e2))

    scene = []
    scene.append(
        Interp_Surface(C2, C1, np.array([0., 0., 1.]), k * 1., 100., k * .5, k * 1., k * 0.)
    )

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestLatestSphere.png')
    drawScene()

    equator_circle = (C1 + C2).normal()

    interp_sphere = (equator_circle * (equator_circle ^ einf).normal() * I5).normal()

    scene = [Sphere(0, 0, np.array([0., 0., 1.]), k * 1., 100., k * 0.5, k * 1., k * 0.)]
    scene[0].object = unsign_sphere(interp_sphere)

    print("\n\nNow drawing Sphere:\n\n")
    drawScene()

    im1 = Image.fromarray(render().astype('uint8'), 'RGB')
    im1.save('figtestSphere.png')

    print("\n\n")
    print("--- %s seconds ---" % (time.time() - start_time))

