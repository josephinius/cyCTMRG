import cytnx
import numpy as np
import math


"""

C_ab:

      a
     |
     |____ b
  
T_aib:

    a
    |
    |_____ i
    |
    |
    b

W_ijkl:

          i
          |
    l ____|____ j
          |
          |
          k

"""


def initialize_local_tensor(temperature, field, *j):
    """
    w_{ijkl}:

            i
            |
        l --o-- j
            |
            k

    """

    weights = []
    for coupling in j:
        b = coupling / temperature
        c = math.sqrt(math.cosh(b))
        s = math.sqrt(math.sinh(b))
        w = [[c, s], [c, -s]]
        weights.append(np.array(w))
    wf = np.array([math.exp(field / temperature), math.exp(-field / temperature)])
    # tensor_{i, j, k, l} = sum_{a} w[a][i] * w[a][j] * w[a][k] * w[a][l] * wf[a]
    tensor = np.einsum('ai,aj,ak,al,a->ijkl', *weights, wf)
    tensor = cytnx.from_numpy(tensor)
    w = cytnx.UniTensor(tensor, labels=['i', 'j', 'k', 'l'], rowrank=0)
    return w


def initialize_imp_tensor(temperature, field, *j):
    """
    imp_{ijkl}:

            i
            |
        l --o-- j
            |
            k

    """

    weights = []
    for coupling in j:
        b = coupling / temperature
        c = math.sqrt(math.cosh(b))
        s = math.sqrt(math.sinh(b))
        w = [[c, s], [c, -s]]
        weights.append(np.array(w))
    wf = np.array([math.exp(field / temperature), math.exp(-field / temperature)])
    ss = np.array([1., -1.])
    # tensor_{i, j, k, l} = sum_{a} w[a][i] * w[a][j] * w[a][k] * w[a][l] * wf[a] * ss[a]
    tensor = np.einsum('ai,aj,ak,al,a,a->ijkl', *weights, wf, ss)
    tensor = cytnx.from_numpy(tensor)
    w = cytnx.UniTensor(tensor, labels=['i', 'j', 'k', 'l'], rowrank=0)
    return w


def initialize_tm(temperature, field, *j):
    """
    tm_{aib}:

        a
        |___ i
        |
        b

    """

    weights = []
    for coupling in j:
        b = coupling / temperature
        c = math.sqrt(math.cosh(b))
        s = math.sqrt(math.sinh(b))
        w = [[c, s], [c, -s]]
        weights.append(np.array(w))
    wf = np.array([math.exp(field / temperature), math.exp(-field / temperature)])
    # tensor_{a, i, b} = sum_{s} w[s][a] * w[s][i] * w[s][b] * wf[s]
    tensor = np.einsum('sa,si,sb,s->aib', *weights, wf)
    tensor = cytnx.from_numpy(tensor)
    tm = cytnx.UniTensor(tensor, labels=['a', 'i', 'b'], rowrank=0)
    return tm


def initialize_corner(temperature, field, *j):
    """
    corner_{ab}:

        a
        |__ b

    """

    weights = []
    for coupling in j:
        b = coupling / temperature
        c = math.sqrt(math.cosh(b))
        s = math.sqrt(math.sinh(b))
        w = [[c, s], [c, -s]]
        weights.append(np.array(w))
    wf = np.array([math.exp(field / temperature), math.exp(-field / temperature)])
    # tensor_{a, b} = sum_{s} w[s][a] * w[s][b] * wf[s]
    tensor = np.einsum('sa,sb,s->ab', *weights, wf)
    tensor = cytnx.from_numpy(tensor)
    corner = cytnx.UniTensor(tensor, labels=['a', 'b'], rowrank=0)
    return corner


def measurement(corners, tms, w, imp):
    c1, c2, c3, c4 = corners 
    t1, t2, t3, t4 = tms 
    net = cytnx.Network("measurement.net")
    net.PutUniTensor("w", w)
    net.PutUniTensor("c1", c1)
    net.PutUniTensor("c2", c2)
    net.PutUniTensor("c3", c3)
    net.PutUniTensor("c4", c4)
    net.PutUniTensor("t1", t1)
    net.PutUniTensor("t2", t2)
    net.PutUniTensor("t3", t3)
    net.PutUniTensor("t4", t4)
    # print(net)
    norm = net.Launch()
    net.PutUniTensor("w", imp)
    res = net.Launch()
    return res / norm


def weight_rotate(weight):
    """Returns weight rotated anti-clockwise."""
        
    weight = weight.permute([1, 2, 3, 0])
    return weight


def corner_extension(c, t1, t2, w):
    """
    Returns extended corner according to following schema:

       a    i
       |    |  W
    T1 |____|____ j
       |    |
       |____|____ b
     C      T2

    """
    
    net = cytnx.Network()
    net.FromString(["c: t1-c, t2-c",
                    "t1: a, w-t1, t1-c",
                    "t2: t2-c, w-t2, b",
                    "w: i, j, w-t2, w-t1",
                    "TOUT: a,i;b,j"])
    net.setOrder(contract_order = "(w,(t2,(t1, c)))")
    net.PutUniTensor("w", w)
    net.PutUniTensor("c", c)
    net.PutUniTensor("t1", t1)
    net.PutUniTensor("t2", t2)
    return net.Launch()


def create_upper_half(c1, c2):
    """

        C1  ____ __......__ ____  C2
           |    |          |    |
           |____|__......__|____|
           |    |          |    |
           |    |          |    |
           a    i          j    b

    """

    # uh = np.tensordot(c2, c1, axes=([2, 3], [0, 1]))  # uh_{b j a i} = c2_{b j c k} * c1_{c k a i}
    # return uh.reshape((np.prod(uh.shape[:2]), -1))  # uh_{(b j), (a i)}

    c1 = c1.relabels(["c", "k", "a", "i"])
    c2 = c2.relabels(["b", "j", "c", "k"])
    res = cytnx.Contract(c1, c2)
    return res


def create_projectors(c1, c2, c3, c4, dim_cut):
    upper_half = create_upper_half(c1, c2)
    upper_half.print_diagram()
    return None, None

    # q, r_up = linalg.qr(upper_half)
    _, r_up = linalg.qr(upper_half)
    # _, s1, vt1 = linalg.svd(upper_half, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'
    # print('s1.shape', s1.shape)
    # print('v1.shape', vt1.shape)
    # r_up = np.tensordot(np.diag(s1), vt1[:s1.shape[0], :], axes=(0, 0))

    lower_half = create_lower_half(c3, c4)
    _, r_down = linalg.qr(lower_half)
    # _, s2, vt2 = linalg.svd(lower_half, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'
    # r_down = np.tensordot(np.diag(s2), vt2[:s2.shape[0], :], axes=(0, 0))

    # print('r_up.shape', r_up.shape)
    # print('r_down.shape', r_down.shape)

    rr = np.tensordot(r_up, r_down, axes=(1, 1))
    u, s, vt = linalg.svd(rr, lapack_driver='gesvd')  # use 'gesvd' or 'gesdd'

    # print('u', u.shape)
    # print('s', s.shape)
    # print('vt', vt.shape)
    # print('s', s)

    dim_new = min(s.shape[0], dim_cut)
    lambda_new = []

    for i, x in enumerate(s[:dim_new]):
        if (x / s[0]) < EPS:
            print(f's[{i}] too small: ', x)
            # print('s too small')
            # exit()
            break
        lambda_new.append(x)

    dim_new = len(lambda_new)
    lambda_new = np.array(lambda_new)

    # print('np.diag(1 / np.sqrt(lambda_new))', np.diag(1 / np.sqrt(lambda_new)))

    # u = np.conj(u[:, :dim_new]) / np.sqrt(lambda_new)[None, :]
    u = np.tensordot(np.conj(u[:, :dim_new]), np.diag(1 / np.sqrt(lambda_new)), axes=(1, 0))

    # vt = np.conj(vt[:dim_new, :]) / np.sqrt(lambda_new)[:, None]
    vt = np.tensordot(np.diag(1 / np.sqrt(lambda_new)), np.conj(vt[:dim_new, :]), axes=(0, 0))

    upper_projector = np.tensordot(r_down, vt, axes=(0, 1))
    lower_projector = np.tensordot(r_up, u, axes=(0, 0))

    # print('upper_projector', upper_projector)
    # TODO: check Refs. [7] and [8] !
    # a = np.tensordot(upper_projector, lower_projector, axes=(1, 1))
    # print('a.shape', a.shape)
    # print(a)  # should be approximately an identity

    return upper_projector, lower_projector


def extension_and_renormalization(dim, weight, corners, transfer_matrices):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step. Here, one step of
    CTMRG consists of repeating four times following two steps:
        (1) introducing additional column to system;
        (2) 90-degrees rotation of whole system.
    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    corners_extended = []

    for i in range(4):
        weight = weight_rotate(weight)
        c = corners[i]
        tm1 = transfer_matrices[i]
        tm2 = transfer_matrices[(i + 3) % 4]
        # print(f"corner c[{i+1}] extension...")
        # print(f"tm1 - {i + 1}")
        # print(f"tm2 - {(i + 3) % 4 + 1}")
        # weight.print_diagram()
        corners_extended.append(corner_extension(c, tm1, tm2, weight))

    p_up, p_down = create_projectors(*corners_extended, dim)




if __name__ == '__main__':
    dim = 7
    temperature = 1.
    field_global = 0.
    field_boundary = 1.e-14
    field_corner = 1.e-14
    j_x, j_y = 1., 1. 
    jw = (j_y, j_x, j_y, j_x)
    w = initialize_local_tensor(temperature, field_global, *jw)
    imp = initialize_imp_tensor(temperature, field_global, *jw)
    del jw
    # w.print_diagram()
    # imp.print_diagram()
    jt1 = (j_x, j_y, j_x)
    jt2 = (j_y, j_x, j_y)
    t1 = initialize_tm(temperature, field_boundary, *jt1)
    t2 = initialize_tm(temperature, field_boundary, *jt2)
    t3 = initialize_tm(temperature, field_boundary, *jt1)
    t4 = initialize_tm(temperature, field_boundary, *jt2)
    del jt1, jt2
    # t1.print_diagram()
    jc1 = (j_x, j_y)
    jc2 = (j_y, j_x)
    c1 = initialize_corner(temperature, field_corner, *jc1)
    c2 = initialize_corner(temperature, field_corner, *jc2)
    c3 = initialize_corner(temperature, field_corner, *jc1)
    c4 = initialize_corner(temperature, field_corner, *jc2)
    # c1.print_diagram()
    del jc1, jc2

    corners = (c1, c2, c3, c4)
    tms = (t1, t2, t3, t4)

    mag = measurement(corners, tms, w, imp)
    # print("mag: ", mag)

    extension_and_renormalization(dim, w, corners, tms)


