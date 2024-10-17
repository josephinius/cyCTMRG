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


# TODO: implement CTMRG class and move the initialization of CTM tensors inside 


if __name__ == '__main__':

    xi = 7  # CTMRG max bond dimension
    D = 2  # "physical" dimension (PEPS)

    t = 1.
    h = 0.

    Lx = 3
    Ly = 3

    # couplings 
    jx = np.random.rand(Lx, Ly)
    jy = np.random.rand(Lx, Ly)

    ten = np.empty([Lx, Ly], dtype=cytnx.UniTensor)

    for x in range(Lx):
        for y in range(Ly):
            j = (jx[x][y], jy[x][y], jx[(x + 1) % Lx][y], jy[x][(y + 1) % Ly])
            ten[x, y] = create_ten(t, h, x, Lx, y, Ly, *j)
    
    corner1 = np.empty([Lx, Ly], dtype=cytnx.UniTensor)
    corner2 = np.empty([Lx, Ly], dtype=cytnx.UniTensor)
    corner3 = np.empty([Lx, Ly], dtype=cytnx.UniTensor)
    corner4 = np.empty([Lx, Ly], dtype=cytnx.UniTensor)

    xi_start = 1

    corner_links=[cytnx.Bond(dim=xi_start), cytnx.Bond(dim=xi_start)];
    
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

    """
    # test - lower left corner of the lattice 
    c1 = corner1[0, 1]
    c2 = corner2[1, 1]
    c3 = corner3[1, 0]
    c4 = corner4[0, 0]
    res = cytnx.Contracts(TNs = [c1, c2, c3, c4])
    print(res)
    """

    # TODO: initialize edges...
    