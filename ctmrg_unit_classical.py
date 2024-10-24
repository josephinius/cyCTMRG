import cytnx
import numpy as np
import math

EPS = 1.e-32


"""

name=f"ten_x{x}_y{y}"

Index ordering:

ten_{a b c d}

          d
          |
    a ____|____ c
          |
          |
          b

Labels:

a: f"bond_x{x}_y{y}_x" (ket)
b: f"bond_x{x}_y{y}_y" (ket)
c: f"bond_x{x+1}_y{y}_x" (bra)
d: f"bond_x{x}_y{y+1}_y" (bra)

"""

def create_ten(temperature, field, x, Lx, y, Ly, *j):

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

    linktypes = [cytnx.BD_KET, cytnx.BD_KET, cytnx.BD_BRA, cytnx.BD_BRA]
    links = [cytnx.Bond(bond_type=linktype, dim=2) for linktype in linktypes]

    ten = cytnx.UniTensor(bonds=links, 
                          labels=[f"bond_x{x}_y{y}_x", f"bond_x{x}_y{y}_y", f"bond_x{(x+1) % Lx}_y{y}_x", f"bond_x{x}_y{(y+1) % Ly}_y"], 
                          name=f"ten_x{x}_y{y}", 
                          rowrank=2)
    
    ten.get_block_()[:] = tensor

    return ten


def corner_extension(c, t1, t2, w):
    """
    Returns extended corner according to following schema:

       
       |    |  W
    T1 |____|____
       |    |
       |____|____
     C      T2

    """

    cp = cytnx.Contracts([t1, c])
    cp = cytnx.Contracts([cp, t2])

    alist = c.name().split("_")
    alist[0] += "p"
    name_new = "_".join(alist)

    cp = cytnx.Contracts([cp, w]).set_name(name_new)
    return cp


def create_upper_half(c1, c2):
    """

        C1  ____ __......__ ____  C2
           |    |          |    |
           |____|__......__|____|
           |    |          |    |
           |    |          |    |

    """
    res = cytnx.Contract(c2, c1).set_rowrank_(2) 
    return res


def create_lower_half(c3, c4):
    """

           |    |              |    |
           |____|____......____|____|
           |    |              |    |
           |____|____......____|____|
        C4                            C3

    """

    res = cytnx.Contract(c3, c4).set_rowrank_(2)
    return res


def redirect_bond(uten, label):
    bonds = uten.bonds()
    for i, bond in enumerate(bonds):
        if uten.labels()[i] == label:
            bond.redirect_()
            break


def set_bond(uten, label, bond_type):
    print(uten.name())
    bonds = uten.bonds()
    for i, bond in enumerate(bonds):
        if uten.labels()[i] == label:
            if bond.type() != bond_type:
                bond.redirect_()
            break


def create_projectors(c1, c2, c3, c4, chi):
    upper_half = create_upper_half(c1, c2)
    _, r_up = cytnx.linalg.Qr(upper_half)

    lower_half = create_lower_half(c3, c4)
    _, r_down = cytnx.linalg.Qr(lower_half)

    r_up.relabel_(0, "r1")
    r_down.relabel_(0, "r2")

    rr = cytnx.Contract(r_up, r_down).set_rowrank_(1)
    s, u, vt, s_err = cytnx.linalg.Svd_truncate(rr, keepdim=chi, err=EPS, return_err=1)
    print("s_err: ", s_err.item())

    cytnx.Pow_(s, -1/2)
    cytnx.Conj_(u)

    u = cytnx.Contract(u, s)

    cytnx.Conj_(vt)
    vt = cytnx.Contract(s, vt)

    redirect_bond(vt, "r2")
    redirect_bond(u, "r1")

    upper_projector = cytnx.Contract(r_down, vt).set_rowrank_(2)
    lower_projector = cytnx.Contract(r_up, u).set_rowrank_(2)

    upper_ctmrg_type = None 
    for i, l in enumerate(upper_projector.labels()):
        if l.startswith("ctmrg"):
            upper_ctmrg_type = upper_projector.bonds()[i].type()

    lower_ctmrg_type = None 
    for i, l in enumerate(lower_projector.labels()):
        if l.startswith("ctmrg"):
            lower_ctmrg_type = lower_projector.bonds()[i].type()

    new_coordinates = None
    old_coordinates = None
    ctrmg_dir = None

    for l in upper_projector.labels():
        if l.startswith("bond"):
            alist = l.split("_")
            new_coordinates = alist[1:3]
        if l.startswith("ctmrg"):
            alist = l.split("_")
            old_coordinates = alist[1:3]
            ctrmg_dir = alist[-1]

    new_lab = ["ctrmg"] + new_coordinates + [ctrmg_dir]
    new_lab = "_".join(new_lab) 

    upper_projector.relabel_("_aux_L", new_lab)
    lower_projector.relabel_("_aux_R", new_lab)

    set_bond(lower_projector, new_lab, upper_ctmrg_type)
    set_bond(upper_projector, new_lab, lower_ctmrg_type)

    upper_projector.set_name("proj_upper_" + "_".join(old_coordinates))
    lower_projector.set_name("proj_lower_" + "_".join(old_coordinates))

    return upper_projector, lower_projector


def create_projectors_from_weights(chi, corners, edges, weights):

    c1, c2, c3, c4 = corners
    t11, t12, t21, t22, t31, t32, t41, t42 = edges
    w1, w2, w3, w4 = weights

    c1p = corner_extension(c1, t11, t42, w1)
    c2p = corner_extension(c2, t21, t12, w2)
    c3p = corner_extension(c3, t31, t22, w3)
    c4p = corner_extension(c4, t41, t32, w4)

    return create_projectors(c1p, c2p, c3p, c4p, chi)


class CTMRG(object):
    """An implementation of Corner Transfer Matrix Renormalisation Group (CTMRG) algorithm."""

    def __init__(self, chi, weights):
        self.chi = chi
        self.D = weights[0, 0].shape()[0]  # Assuming dimensions for each weigth tensor to be same
        self.Lx = weights.shape[0]
        self.Ly = weights.shape[1]
        self.weights = weights

        corner1 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        corner2 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        corner3 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        corner4 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)

        edge1 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        edge2 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        edge3 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)
        edge4 = np.empty([self.Lx, self.Ly], dtype=cytnx.UniTensor)

        xi_start = 1

        corner_links = [cytnx.Bond(bond_type=cytnx.BD_BRA, dim=xi_start), cytnx.Bond(bond_type=cytnx.BD_KET, dim=xi_start)]

        edge1_links = [cytnx.Bond(bond_type=cytnx.BD_BRA, dim=xi_start), cytnx.Bond(bond_type=cytnx.BD_KET, dim=D), cytnx.Bond(bond_type=cytnx.BD_KET, dim=xi_start)] 
        edge2_links = [cytnx.Bond(bond_type=cytnx.BD_BRA, dim=xi_start), cytnx.Bond(bond_type=cytnx.BD_KET, dim=D), cytnx.Bond(bond_type=cytnx.BD_KET, dim=xi_start)] 
        edge3_links = [cytnx.Bond(bond_type=cytnx.BD_BRA, dim=xi_start), cytnx.Bond(bond_type=cytnx.BD_BRA, dim=D), cytnx.Bond(bond_type=cytnx.BD_KET, dim=xi_start)] 
        edge4_links = [cytnx.Bond(bond_type=cytnx.BD_BRA, dim=xi_start), cytnx.Bond(bond_type=cytnx.BD_BRA, dim=D), cytnx.Bond(bond_type=cytnx.BD_KET, dim=xi_start)] 

        for x in range(Lx):
            for y in range(Ly):

                corner1[x, y] = cytnx.UniTensor(
                    bonds=corner_links,
                    labels=[f"ctmrg_x{(x+1) % Lx}_y{y}_1", f"ctmrg_x{x}_y{y}_4"], 
                    name=f"c1_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(corner1[x, y],mean=0,std=1)

                corner2[x, y] = cytnx.UniTensor(
                    bonds=corner_links,
                    labels=[f"ctmrg_x{x}_y{y}_2", f"ctmrg_x{x}_y{y}_1"], 
                    name=f"c2_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(corner2[x, y],mean=0,std=1)

                corner3[x, y] = cytnx.UniTensor(
                    bonds=corner_links,
                    labels=[f"ctmrg_x{x}_y{y}_3", f"ctmrg_x{x}_y{(y+1) % Ly}_2"], 
                    name=f"c3_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(corner3[x, y],mean=0,std=1)

                corner4[x, y] = cytnx.UniTensor(
                    bonds=corner_links,
                    labels=[f"ctmrg_x{x}_y{(y+1) % Ly}_4", f"ctmrg_x{(x+1) % Lx}_y{y}_3"], 
                    name=f"c4_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(corner4[x, y],mean=0,std=1)

                edge1[x, y] = cytnx.UniTensor(
                    bonds=edge1_links,
                    labels=[f"ctmrg_x{(x+1) % Lx}_y{y}_1", f"bond_x{x}_y{y}_y", f"ctmrg_x{x}_y{y}_1"],
                    name=f"t1_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge1[x, y],mean=0,std=1)

                edge2[x, y] = cytnx.UniTensor(
                    bonds=edge2_links,
                    labels=[f"ctmrg_x{x}_y{y}_2", f"bond_x{x}_y{y}_x", f"ctmrg_x{x}_y{(y+1) % Ly}_2"],
                    name=f"t2_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge2[x, y],mean=0,std=1)

                edge3[x, y] = cytnx.UniTensor(
                    bonds=edge3_links,
                    labels=[f"ctmrg_x{x}_y{y}_3", f"bond_x{x}_y{(y+1) % Ly}_y", f"ctmrg_x{(x+1) % Lx}_y{y}_3"],
                    name=f"t3_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge3[x, y],mean=0,std=1)

                edge4[x, y] = cytnx.UniTensor(
                    bonds=edge4_links,
                    labels=[f"ctmrg_x{x}_y{(y+1) % Ly}_4", f"bond_x{(x+1) % Lx}_y{y}_x", f"ctmrg_x{x}_y{y}_4"],
                    name=f"t4_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge4[x, y],mean=0,std=1)

        self.corners = [corner1, corner2, corner3, corner4]
        self.edges = [edge1, edge2, edge3, edge4]


    def select_corners_and_edges(self, x, y):
        
        lx, ly = self.Lx, self.Ly

        c1 = self.corners[0][x, (y + 3) % ly]
        c2 = self.corners[1][(x + 3) % lx][(y + 3) % ly]
        c3 = self.corners[2][(x + 3) % lx, y]
        c4 = self.corners[3][x, y]

        t11 = self.edges[0][(x + 1) % lx, (y + 3) % ly]
        t12 = self.edges[0][(x + 2) % lx, (y + 3) % ly]

        t21 = self.edges[1][(x + 3) % lx, (y + 2) % ly]
        t22 = self.edges[1][(x + 3) % lx, (y + 1) % ly]

        t31 = self.edges[2][(x + 2) % lx, y]
        t32 = self.edges[2][(x + 1) % lx, y] 

        t41 = self.edges[3][x, (y + 1) % ly]
        t42 = self.edges[3][x, (y + 2) % ly]

        return (c1, c2, c3, c4), (t11, t12, t21, t22, t31, t32, t41, t42)


    def run_ctmrg_num_of_steps(self, num_of_steps):
        
        lx, ly = self.Lx, self.Ly

        x, y = 0, 0

        cc, tt = self.select_corners_and_edges(x, y)

        ww = (self.weights[(x + 1) % lx][(y + 2) % ly], self.weights[(x + 2) % lx][(y + 2) % ly], 
              self.weights[(x + 2) % lx][(y + 1) % ly], self.weights[(x + 1) % lx][(y + 1) % ly])
        
        p_up, p_down = create_projectors_from_weights(self.chi, cc, tt, ww)


def contract_small_window(corners, tms, w):
    
    """
    Returns contraction of network.

                    T1
        C1  ____ _______ ____ C2
           |        |        |
           |        | w      |
      T4   |________o________|   T2
           |        |        |
           |        |        |
           |________|________|
        C4                    C3
                    T3

    """

    c1, c2, c3, c4 = corners
    t1, t2, t3, t4 = tms
    res = cytnx.Contracts([c4, t3])
    res = cytnx.Contracts([res, c3])
    res = cytnx.Contracts([res, t4])
    res = cytnx.Contracts([res, w])
    res = cytnx.Contracts([res, t2])
    res = cytnx.Contracts([res, c1])
    res = cytnx.Contracts([res, t1])
    res = cytnx.Contracts([res, c2])
    return res.item()


if __name__ == '__main__':

    chi = 7  # CTMRG max bond dimension
    D = 2  # "physical" dimension (PEPS)

    t = 1.
    h = 0.

    Lx = 5
    Ly = 5

    # random couplings 
    jx = np.random.rand(Lx, Ly)
    jy = np.random.rand(Lx, Ly)

    ten = np.empty([Lx, Ly], dtype=cytnx.UniTensor)

    for x in range(Lx):
        for y in range(Ly):
            j = (jx[x][y], jy[x][y], jx[(x + 1) % Lx][y], jy[x][(y + 1) % Ly])
            ten[x, y] = create_ten(t, h, x, Lx, y, Ly, *j)
    
    ctm = CTMRG(chi, ten)

    print("ctm.chi", ctm.chi)
    print("ctm.D", ctm.D)
    print("ctm.Lx", ctm.Lx)
    print("ctm.Ly", ctm.Ly)

    # test 1 - lower left corner of the lattice 
    c1 = ctm.corners[0][0, 1]
    c2 = ctm.corners[1][1, 1]
    c3 = ctm.corners[2][1, 0]
    c4 = ctm.corners[3][0, 0]
    res = cytnx.Contracts(TNs = [c1, c2, c3, c4])
    print(res)

    labels = ctm.corners[3][0, 0].labels()
    print(labels)

    # test 2 - small window moving over whole lattice 
    for x in range(Lx):
        for y in range(Ly):

            c1 = ctm.corners[0][x][(y + 2) % Ly]
            c2 = ctm.corners[1][(x + 2) % Lx][(y + 2) % Ly]
            c3 = ctm.corners[2][(x + 2) % Lx][y]
            c4 = ctm.corners[3][x][y]
            
            t1 = ctm.edges[0][(x + 1) % Lx][(y + 2) % Ly]
            t2 = ctm.edges[1][(x + 2) % Lx][(y + 1) % Ly]
            t3 = ctm.edges[2][(x + 1) % Lx][y]
            t4 = ctm.edges[3][x][(y + 1) % Ly]
            
            w = ten[(x + 1) % Lx][(y + 1) % Ly]
            
            corners = (c1, c2, c3, c4)
            tms = (t1, t2, t3, t4)
            res = contract_small_window(corners, tms, w)
            print("x:", x, "y: ", y, "res: ", res)

    ctm.run_ctmrg_num_of_steps(0)
