
from clifford import MVArray
from clifford.g3c import *
from clifford.tools.g3 import *
from clifford.tools.g3c import *
from pyganja import *


def write_obj_file(filename, vertices, faces, vertex_normals=None):
    """ Writes a .obj file """
    with open(filename, 'w') as fobj:
        for v in vertices:
            print("v %f %f %f"%(v[0],v[1],v[2]), file=fobj)
        if vertex_normals is not None:
            for vn in vertex_normals:
                print("vn %f %f %f"%(vn[0],vn[1],vn[2]), file=fobj)
            for f in faces:
                print("f %d//%d %d//%d %d//%d"%(f[0]+1,f[0]+1,f[1]+1,f[1]+1,f[2]+1,f[2]+1), file=fobj)
        else:
            for f in faces:
                print("f %d %d %d"%(f[0]+1,f[1]+1,f[2]+1), file=fobj)


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
    gs.add_facets([[ga_vertices[i] for i in f] for f in face_list], color=int('AA000000',16), static=True)
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
    alpha_list = np.linspace(0,1,n_alpha)
    Clist = [interp_objects_root(C1,C2,alp) for alp in alpha_list]
    vertex_list = vertex_circles(Clist, n_points)
    face_list = mesh_grid(n_alpha, n_points, mask=None, loopx=True)
    return vertex_list, face_list, alpha_list


def test_mesh_circles():
    n_alpha = 21
    n_points = 21

    # C1 = normalised(up(5*e3+e1)^up(5*e3+e2)^up(5*e3-e1))
    # C2 = normalised(up(e1)^up(e2)^up(-e1))
    C1 = random_circle()
    C2 = random_circle()

    vertex_list, face_list, alpha_list = mesh_circle_surface(C1, C2, n_points=n_points, n_alpha=n_alpha)

    gs = get_facet_scene(vertex_list, face_list)
    gs.add_objects(vertex_list, static=True)
    gs.add_objects([interp_objects_root(C1,C2,alp) for alp in np.linspace(0,1,n_alpha)],color=Color.RED)
    draw(gs,scale=0.1)


if __name__ == '__main__':
    test_mesh_circles()
