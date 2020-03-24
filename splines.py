
from clifford.tools.g3c import *
from clifford.g3c import *
from pyganja import *
from meshing import *
from derivatives import *


def reject_Cdot(C, Cdot):
    """
    Returns Cdot stuff orthogonal to C
    """
    return Cdot - (Cdot|C)*C


def test_reject_Cdot():
    for i in range(1000):
        C = random_circle()
        Cdot = random_circle()
        res = reject_Cdot(C, Cdot)
        assert res != 0
        np.testing.assert_allclose((C|res).value, 0, 0, 1E-6)


def KBtangents(start_point, end_point, 
                previous_point, next_point,
                tension=0, bias=0, continuity=0):
    """
    Calculates the tangents for a KB curve
    """
    di = 0.5*(1.0 - tension)*(1.0 + bias)*(1.0 + continuity)*(start_point - previous_point) + \
        0.5*(1.0 - tension)*(1.0 - bias)*(1.0 - continuity)*(end_point - start_point)
    dip = 0.5*(1.0 - tension)*(1.0 + bias)*(1.0 - continuity)*(end_point - start_point) + \
        0.5*(1.0 - tension)*(1.0 - bias)*(1.0 + continuity)*(next_point - end_point)
    return di, dip


def hermite_curve(X0, V0, X1, V1, alpha):
    """
    Hermite curve equation
    """
    return (2*alpha**3 - 3*alpha**2 + 1)*X0 + \
    (alpha**3 - 2*alpha**2 + alpha)*V0 + \
    (-2*alpha**3 + 3*alpha**2)*X1 + \
    (alpha**3 - alpha**2)* V1


def derivative_hermite_curve(X0, V0, X1, V1, alpha):
    """
    Derivative of the Hermite curve equation
    """
    return (6*alpha**2 - 6*alpha)*X0 + \
    (3*alpha**2 - 4*alpha + 1.0)*V0 + \
    (-6*alpha**2 + 6*alpha)*X1 + \
    (3*alpha**2 - 2*alpha)* V1


def test_derivative_hermite_curve():
    for i in range(10):
        X0 = np.random.randn(3)
        V0 = np.random.randn(3)
        X1 = np.random.randn(3)
        V1 = np.random.randn(3)
        npts = 1000
        alpha_list = np.linspace(0,1,npts)
        delta_alpha = alpha_list[1] - alpha_list[0]
        spline = [hermite_curve(X0, V0, X1, V1, alpha) for alpha in alpha_list]
        dspline = np.array([derivative_hermite_curve(X0, V0, X1, V1, alpha) for alpha in alpha_list])
        mdiff = np.gradient(spline, delta_alpha, axis=0)

        fin = len(dspline) - 2
        np.testing.assert_allclose(dspline[1:fin], mdiff[1:fin], 1E-2, 1E-6)


def derivative_projected_KB_curve(X0, V0, X1, V1, alpha):
    """
    The derivative of the KB curve on the blade manifold at
    evolution parameter value alpha
    """
    Xdash = hermite_curve(X0, V0, X1, V1, alpha)
    dXdashdalpha = derivative_hermite_curve(X0, V0, X1, V1, alpha)
    dXda = differential_manifold_projection(Xdash, dXdashdalpha)(3).normal()
    return dXda


def evolved_circle_normal(C, Cdot, X):
    """
    Get the normal to an evolved circle at X given C and Cdot
    """
    LC = (X|C)^einf
    LT = (X|(C*Cdot))^X^einf
    return normalised((LT*LC*I5)(3))


def projected_KB_spline(control_list,
                        tension=0, 
                        bias=0, 
                        continuity=0,
                        nalpha=20,
                        reject_Cdot_flag=False):
    """
    Generates a KB curve projected onto the object manifold
    """
    output_objects = []
    derivative_objects = []
    npoints = len(control_list)
    for i in range(1,npoints-2):
        previous_point = control_list[i-1]
        start_point = control_list[i]
        end_point = control_list[i+1]
        next_point = control_list[i+2]

        di, dip = KBtangents(start_point, end_point, 
                previous_point, next_point,
                tension=tension, 
                bias=bias, 
                continuity=continuity)

        # Now make the curve
        X0 = start_point
        X1 = end_point
        if reject_Cdot_flag:
            V0 = reject_Cdot(X0, di)
            V1 = reject_Cdot(X1, dip)
        else:
            V0 = di
            V1 = dip

        for alpha in np.linspace(0, 1, nalpha):
            Xdash = hermite_curve(X0, V0, X1, V1, alpha)
            deriv = derivative_projected_KB_curve(X0, V0, X1, V1, alpha)
            C = average_objects([Xdash])(3).normal()
            output_objects.append(C)
            derivative_objects.append(deriv)
    return output_objects, derivative_objects


def test_projected_KB_spline():
    # control_list = [random_circle() for i in range(6)]
    size = 0.1
    t = 1.0*e1
    nobjs = 6
    Cstart = (up(size*1.0*e1)^up(size*1.0*e2)^up(-size*1.0*e1)).normal()
    Tstart = 1 - 0.5*t*einf
    D = generate_dilation_rotor(0.8)
    T = ((1 - 0.5*(0.1*e2)*einf)*generate_rotation_rotor(np.pi/6, e1, e3)*D).normal()
    control_list = [apply_rotor(Cstart, ((T**n)*Tstart)(0,2,4).normal())(3).normal() for n in range(nobjs)]
    spline_objs, derivative_objects = projected_KB_spline(control_list)
    gs = GanjaScene()
    gs.add_objects(spline_objs, color=Color.BLACK)
    gs.add_objects(control_list, color=Color.RED)
    draw(gs, scale=1.0, browser_window=True)


def test_mesh_projected_KB_spline():

    # Generate the circle spline
    size = 0.1
    t = 1.0*e1
    nobjs = 6
    Cstart = (up(size*1.0*e1)^up(size*1.0*e2)^up(-size*1.0*e1)).normal()
    Tstart = 1 - 0.5*t*einf
    D = generate_dilation_rotor(0.8)
    T = ((1 - 0.5*(0.1*e2)*einf)*generate_rotation_rotor(np.pi/6, e1, e3)*D).normal()
    control_list = [apply_rotor(Cstart, ((T**n)*Tstart)(0,2,4).normal())(3).normal() for n in range(nobjs)]

    spline_objs, derivative_objects = projected_KB_spline(control_list)

    # Mesh the spline
    n_points = 21
    n_alpha = len(spline_objs)
    vertex_list, texture_coords = vertex_circles(spline_objs, n_points)
    face_list = mesh_grid(n_alpha, n_points, mask=None, loopx=True)

    # Get the vetex normals
    vertex_normals = []
    for i in range(len(vertex_list)):
        X = vertex_list[i]
        cind = int(i/(n_points+1))
        C = spline_objs[cind]
        Cdot = derivative_objects[cind]
        norm_line = evolved_circle_normal(C, Cdot, X)
        norm_val = -((norm_line*I5)(e123)*I3).normal().value[1:4]
        vertex_normals.append(norm_val)

    # Write the file
    threedvertexlist = [down(v).value[1:4] for v in vertex_list]
    write_obj_file('surfaces/KBSpline.obj', threedvertexlist, face_list, 
                    vertex_normals=vertex_normals, 
                    texture_coords=texture_coords,
                    use_mtl=True)


if __name__ == '__main__':
    # test_reject_Cdot()
    # test_derivative_hermite_spline()
    # test_projected_KB_spline()
    test_mesh_projected_KB_spline()
    pass
