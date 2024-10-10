import cytnx
import numpy as np
import math

EPS = 1.e-32


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

    c1 = c1.relabels(["c", "k", "a", "i"])
    c2 = c2.relabels(["b", "j", "c", "k"])
    res = cytnx.Contract(c2, c1)  # res_{bj;ai}
    return res


def create_lower_half(c3, c4):
    """

           a    i              j    b
           |    |              |    |
           |____|____......____|____|
           |    |              |    |
           |____|____......____|____|
        C4                            C3

    """

    c3 = c3.relabels(["c", "k", "b", "j"])
    c4 = c4.relabels(["a", "i", "c", "k"])
    res = cytnx.Contract(c3, c4)  # res_{bj;ai}
    return res


def extend_corner1(c1, t1, p_up):
    """
    Returns extended and renormalized corner matrix c1.
    """

    net = cytnx.Network()
    net.FromString(["c1: t1-c1, p-c1",
                    "t1: a, p-t1, t1-c1",
                    "p_up: p-c1, p-t1, b",
                    "TOUT: a;b"])
    net.setOrder(contract_order = "((c1, t1),p_up)")
    net.PutUniTensor("c1", c1)
    net.PutUniTensor("t1", t1)
    net.PutUniTensor("p_up", p_up)
    return net.Launch()


def extend_corner4(c4, t3, p_down):
    """
    Returns extended and renormalized corner matrix c4.
    """

    net = cytnx.Network()
    net.FromString(["c4: p-c4, t3-c4",
                    "t3: t3-c4, p-t3, b",
                    "p_down: p-c4, p-t3, a",
                    "TOUT: a;b"])
    net.setOrder(contract_order = "((c4,t3),p_down)")
    net.PutUniTensor("c4", c4)
    net.PutUniTensor("t3", t3)
    net.PutUniTensor("p_down", p_down)
    return net.Launch()


def extend_tm(t4, weight, p_down, p_up):
    """
    Returns extended and renormalized transfer matrix t4. 
    """

    net = cytnx.Network()
    net.FromString(["t4: pd-t, w-t, pu-t",
                    "w: pd-w, j, pu-w, w-t",
                    "p_down: pd-t, pd-w, a",
                    "p_up: pu-t, pu-w, b",
                    "TOUT: a, j, b"])
    net.setOrder(contract_order = "(((t4,p_down),w),p_up)")
    net.PutUniTensor("t4", t4)
    net.PutUniTensor("w", weight)
    net.PutUniTensor("p_down", p_down)
    net.PutUniTensor("p_up", p_up)
    return net.Launch()


def create_projectors(c1, c2, c3, c4, dim_cut):
    upper_half = create_upper_half(c1, c2)
    # upper_half.print_diagram()
    _, r_up = cytnx.linalg.Qr(upper_half)

    lower_half = create_lower_half(c3, c4)
    # lower_half.print_diagram()
    _, r_down = cytnx.linalg.Qr(lower_half)

    r_up.relabel_(0, "r1")
    r_down.relabel_(0, "r2")

    rr = cytnx.Contract(r_up, r_down).set_rowrank_(1)  # rr_{r1;r2}
    # s, u, vt = cytnx.linalg.Svd(rr)
    # print(s)
    s, u, vt, s_err = cytnx.linalg.Svd_truncate(rr, keepdim=dim_cut, err=EPS, return_err=1)
    # print(s)
    # print("s_err: ", s_err)

    cytnx.Pow_(s, -1/2)
    cytnx.Conj_(u)

    u = cytnx.Contract(u, s)

    cytnx.Conj_(vt)
    vt = cytnx.Contract(s, vt)

    upper_projector = cytnx.Contract(r_down, vt).set_rowrank_(2)  # _{a,i;_aux_L}
    lower_projector = cytnx.Contract(r_up, u).set_rowrank_(2)  # _{a,i;_aux_R}

    upper_projector.relabel_("_aux_L", "_aux_")
    lower_projector.relabel_("_aux_R", "_aux_")

    # lower_projector.relabel_("a", "b")
    # lower_projector.relabel_("i", "j")
    # test = cytnx.Contract(upper_projector, lower_projector)
    # test.print_diagram()
    # test = test.reshape(4, 4).set_rowrank(1)
    # print(test)  # should be close to identity 

    return upper_projector, lower_projector


def tuple_rotation(x1, x2, x3, x4):
    """Returns new tuple shifted to left by one place."""

    return x2, x3, x4, x1


def extension_and_renormalization(dim, weight, corners, transfer_matrices, rotate=True):
    """Returns corners and transfer matrices extended (and projected) by one iterative CTMRG step. Here, one step of
    CTMRG consists of repeating four times following two steps:
        (1) introducing additional column to system;
        (2) 90-degrees rotation of whole system.
    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = transfer_matrices

    for _ in range(4):

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

        c1 = extend_corner1(c1, t1, p_up)
        c4 = extend_corner4(c4, t3, p_down)
        t4 = extend_tm(t4, weight, p_down, p_up)

        if rotate:
            c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
            t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
            transfer_matrices = t1, t2, t3, t4
            corners = c1, c2, c3, c4
            weight = weight_rotate(weight)

    return [c1, c2, c3, c4], [t1, t2, t3, t4]


def ctmrg_iteration(corners, tms, weight, imp, num_of_steps="inf"):
    
    mag = 0
    mag_new = -1

    i = 0 

    while (i < 10 or abs(mag - mag_new) > 1.e-12) if (num_of_steps == "inf") else (i < num_of_steps):
        corners, tms = extension_and_renormalization(dim, weight, corners, tms)

        for j in range(4):
            c = corners[j]
            abs_tensor = cytnx.linalg.Abs(c.get_block())
            norm = cytnx.linalg.Max(abs_tensor)
            corners[j] = c / norm.item()
            t = tms[j]
            abs_tensor = cytnx.linalg.Abs(t.get_block())
            norm = cytnx.linalg.Max(abs_tensor)
            tms[j] = t / norm.item()

        mag_new = mag
        mag = measurement(corners, tms, weight, imp).item()
        print(i, mag)
        i += 1

    return mag_new, i


def initialize_tensors(temperature, field_global, field_boundary, field_corner, j_x, j_y):
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

    return corners, tms, w, imp


if __name__ == '__main__':
    dim = 24
    field_global = 1.e-14
    field_boundary = 1.e-1
    field_corner = 1.e-0
    j_x, j_y = 1., 1. 

    start_val = 2.00
    end_val = 2.27
    num = 28

    file_name = f"data_{dim}.txt"

    with open(file_name, 'w') as f:
        f.write(f'# Ising model\n')
        f.write('# D=%d, h_global=%E, h_boundary=%E, h_corner=%E, jx=%E, jy=%E\n' % (dim, field_global, field_boundary, field_corner, j_x, j_y))
        f.write('# temp\t\t\t\tmag\t\t\t\t\titer\n')

    for temperature in np.linspace(start_val, end_val, num=num, endpoint=True):
        corners, tms, w, imp = initialize_tensors(temperature, field_global, field_boundary, field_corner, j_x, j_y)
        mag, iter_count = ctmrg_iteration(corners, tms, w, imp)
        with open(file_name, 'a') as f:
            f.write('%.15f\t%.15f\t%d\n' % (temperature, mag, iter_count))
