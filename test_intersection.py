
from clifftrace import *
from clifford.g3c import *

def get_intersect_surf(C1,C2,L,origin):
    P_val,alpha = pointofXsurface(L, C1, C2, origin)
    if alpha is None:
        return None,None
    P = layout.MultiVector(value=P_val)
    N = get_normal(C1,C2,alpha,P)
    return P,N

def test_surface_point_hit(n_rays=1500):
    origin = 3*e3+1*e2+1*e1
    C1 = (up(e1 + e2)^up(e1 -e2)^up(e1 + e3)).normal()
    C2 = (up(-e1 + e2)^up(-e1 -e2)^up(-e1 + e3)).normal()
    S = unsign_sphere(C1.join(C2)(4)).normal()
    
    hit_error = 0
    normal_error = 0
    total_hits = 0
    double_error = 0 
    for i in range(n_rays):
        L = (up(origin)^up(0.1*random_euc_mv())^einf).normal()
        if (meet(S,L)**2)[0] > 0:
            total_hits += 1
            P,N = get_intersect_surf(C1,C2,L,origin)
            if N is not None:
                P_tru = point_pair_to_end_points(meet(L,S))[0]
                P_tru = P_tru*einf*P_tru
                truN = ((S*I5)^P_tru^einf).normal()
                if (not np.max(np.abs((truN-N).value)) < 10**-6) and (not np.max(np.abs((P_tru|P).value)) < 10**-6):
                    double_error += 1
                elif not np.max(np.abs((P_tru|P).value)) < 10**-6:
                    hit_error += 1
                elif not np.max(np.abs((truN-N).value)) < 10**-6:
                    normal_error += 1
            else:
                hit_error += 1
    return hit_error, normal_error, double_error, total_hits

hit_error, normal_error, double_error, total_hits = test_surface_point_hit()
print('\n\n')
print('Hit errors: ', 100*hit_error/total_hits)
print('Normal errors: ', 100*normal_error/total_hits)
print('Double errors: ', 100*double_error/total_hits)
