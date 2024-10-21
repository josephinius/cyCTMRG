import cytnx
import numpy as np
import math


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

a: f"bond_x{x}_y{y}_x"
b: f"bond_x{x}_y{y}_y"
c: f"bond_x{x+1}_y{y}_x"
d: f"bond_x{x}_y{y+1}_y"

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
    ten = cytnx.UniTensor(tensor, 
                        labels=[f"bond_x{x}_y{y}_x", f"bond_x{x}_y{y}_y", f"bond_x{(x+1) % Lx}_y{y}_x", f"bond_x{x}_y{(y+1) % Ly}_y"], 
                        name=f"ten_x{x}_y{y}",
                        rowrank=2)
    return ten


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

        corner_links = [cytnx.Bond(dim=xi_start), cytnx.Bond(dim=xi_start)]
        edge_links = [cytnx.Bond(dim=xi_start), cytnx.Bond(dim=D), cytnx.Bond(dim=xi_start)] 
        
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
                    bonds=edge_links,
                    labels=[f"ctmrg_x{(x+1) % Lx}_y{y}_1", f"bond_x{x}_y{y}_y", f"ctmrg_x{x}_y{y}_1"],
                    name=f"t1_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge1[x, y],mean=0,std=1)

                edge2[x, y] = cytnx.UniTensor(
                    bonds=edge_links,
                    labels=[f"ctmrg_x{x}_y{y}_2", f"bond_x{x}_y{y}_x", f"ctmrg_x{x}_y{(y+1) % Ly}_2"],
                    name=f"t2_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge2[x, y],mean=0,std=1)

                edge3[x, y] = cytnx.UniTensor(
                    bonds=edge_links,
                    labels=[f"ctmrg_x{x}_y{y}_3", f"bond_x{x}_y{(y+1) % Ly}_y", f"ctmrg_x{(x+1) % Lx}_y{y}_3"],
                    name=f"t3_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge3[x, y],mean=0,std=1)

                edge4[x, y] = cytnx.UniTensor(
                    bonds=edge_links,
                    labels=[f"ctmrg_x{x}_y{(y+1) % Ly}_4", f"bond_x{(x+1) % Lx}_y{y}_x", f"ctmrg_x{x}_y{y}_4"],
                    name=f"t4_x{x}_y{y}",
                    rowrank=1)
                cytnx.random.normal_(edge4[x, y],mean=0,std=1)

        self.corners = [corner1, corner2, corner3, corner4]
        self.edges = [edge1, edge2, edge3, edge4]


    def run_ctmrg_num_of_steps(self, num_of_steps):
        # TODO: implement
        pass


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

    Lx = 4
    Ly = 3

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
