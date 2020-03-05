
from clifford import MVArray
from clifford.g3c import *
from clifford.tools.g3 import *
from clifford.tools.g3c import *
from pyganja import *


def mesh_grid(nypoints, nxpoints, mask=None, loopx=False):
    """
    Meshes a grid of points
    """
    if loopx:
        if mask is not None:
            mask2 = np.zeros((mask.shape[0],mask.shape[1]+1))
            mask2[:mask.shape[0],:mask.shape[1]] = mask
            mask2[:,-1] = mask[:,0]
        else:
            mask2 = None
        return mesh_grid(nypoints, nxpoints+1, mask=mask2, loopx=False)
    else:
        if mask is None:
            mask = np.ones((nypoints,nxpoints))
        else:
            if mask.shape[0] != nypoints:
                raise ValueError('The binary mask must be the same shape as the required mesh')
            if mask.shape[1] != nxpoints:
                raise ValueError('The binary mask must be the same shape as the required mesh')
        face_list = []
        n = 0
        for i in range(nypoints):
            for j in range(nxpoints):
                if j < nxpoints - 1:
                    if i < nypoints - 1:
                        if mask[i,j]:
                            if mask[i+1,j]:
                                if mask[i,j+1]: 
                                    face_list.append([n, n + 1, n + nxpoints])
                                else:
                                    if mask[i+1,j+1]:
                                        face_list.append([n, n + nxpoints+1, n + nxpoints])
                            else:
                                if mask[i+1,j+1]:
                                    face_list.append([n, n + 1, n + nxpoints + 1])
                if j > 0:
                    if i > 0:
                        if mask[i,j]:
                            if mask[i-1,j]:
                                if mask[i,j-1]:
                                    face_list.append([n, n - 1, n - nxpoints])
                n = n + 1
    return face_list


def get_facet_scene(ga_vertices, face_list):
    """
    Converts a set of vertices and face lists into a pyganja scene
    """
    gs = GanjaScene()
    for f in face_list:
        #plane = (MVArray([layout.MultiVector(value=ga_vertices[i,:]) for i in f]).op()^einf).normal()
        # if point_beyond_plane(up(e3), plane):
        #     c = Color.RED
        # else:
        c = int('AA000000',16)
        gs.add_facet([ga_vertices[i] for i in f], color=c)
    return gs


def vertex_circles(Clist, n_points):
    """
    Generates vertices on the circles
    """
    C_old = normalised(up(e1)^up(e2)^up(-e1))
    R = generate_rotation_rotor(2*np.pi/n_points, e1, e2)
    pc = [normalise_n_minus_1((R**n)*up(e1)*~(R**n)) for n in range(n_points+1)]
    vertex_list = []
    for C in Clist:
        Rtrans = TRS_between_rounds(C_old, C)
        C_old = C
        pc = [normalise_n_minus_1((Rtrans*p*~Rtrans)(1)) for p in pc]
        vertex_list += pc
    return vertex_list


def mesh_circle_surface(C1, C2, n_points=21, n_alpha=21):
    """
    Generates the circle list
    """
    Clist = [interp_objects_root(C1,C2,alp) for alp in np.linspace(0,1,n_alpha)]
    vertex_list = vertex_circles(Clist, n_points)
    face_list = mesh_grid(n_alpha, n_points, mask=None, loopx=True)
    return vertex_list, face_list


n_alpha = 21
n_points = 21

# C1 = normalised(up(5*e3+e1)^up(5*e3+e2)^up(5*e3-e1))
# C2 = normalised(up(e1)^up(e2)^up(-e1))
C1 = random_circle()
C2 = random_circle()

vertex_list, face_list = mesh_circle_surface(C1, C2, n_points=n_points, n_alpha=n_alpha)

gs = get_facet_scene(vertex_list, face_list)
gs.add_objects(vertex_list, static=True)
gs.add_objects([interp_objects_root(C1,C2,alp) for alp in np.linspace(0,1,n_alpha)],color=Color.RED)
draw(gs,scale=0.1)