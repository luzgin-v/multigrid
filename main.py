import numpy as np

import gauss
import jacobi
import multigrid

if __name__ == "__main__":
    table_jacobi = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            table_jacobi[i][j] = jacobi.jacobi(10 ** -(i + 1), 2 ** (j + 5))[0]

    table_gauss = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            table_gauss[i][j] = gauss.gauss(10 ** -(i + 1), 2 ** (j + 5))[0]

    table_multigrid = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            table_multigrid[i][j] = multigrid.multigrid(10 ** -(i + 1), 2 ** (j + 5))[0]

    print(table_jacobi)
    print(table_gauss)
    print(table_multigrid)
