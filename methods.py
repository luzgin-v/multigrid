import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


def f(kh, mh):
    return -6 * (kh * kh * mh * (kh * kh + 2 * mh * mh))


def g(kh, mh):
    return kh ** 4 * mh ** 3


def p(pq, kh, mh):
    return math.sin(math.pi * pq * kh) * math.sin(math.pi * pq * mh)


def jacobi(eps, size_grid):
    """
    Метод Якоби
    :param eps: точность решения
    :param size_grid: размер сетки
    """
    # размер сетки
    h = 1 / size_grid
    # номер последнего элемента в таблице
    last_elem = size_grid - 1
    # предварительное вычисление h^2
    h_2 = h ** 2
    # задаем функцию на 0-ой итерации
    u = np.zeros((size_grid, size_grid))
    # граничные значения
    for index in range(size_grid):
        u[0][index] = g(0, index)
        u[index][0] = g(index, 0)
        u[last_elem][index] = g(last_elem, index * h)
        u[index][last_elem] = g(index * h, last_elem)
    # матрица невязок
    r = np.zeros((size_grid, size_grid))
    # предварительное вычисление f
    func = np.zeros((size_grid, size_grid))
    for k in range(size_grid):
        for m in range(size_grid):
            func[k][m] = f(k * h, m * h)
    iteration = 0
    while True:
        iteration += 1
        # вычисляется сетка на текущей итерации
        last_u = u.copy()
        for k in range(1, size_grid - 1):
            for m in range(1, size_grid - 1):
                u[k][m] = 0.25 * (
                    last_u[k + 1][m] + last_u[k - 1][m] + last_u[k][m + 1] + last_u[k][m - 1] + h_2 * func[k][m])
        # вычисляем невязку
        for k in range(1, size_grid - 1):
            for m in range(1, size_grid - 1):
                r[k][m] = - func[k][m] - (
                    (u[k + 1][m] - 2 * u[k][m] + u[k - 1][m]) / h_2 + (u[k][m + 1] - 2 * u[k][m] + u[k][m - 1]) / h_2)
        if np.max(r) < eps:
            break
    print('Норма невязки при eps = {0} и размером сетки {1}:  {2}'.format(str(eps), str(size_grid), str(np.max(r))))
    return iteration, u


def seidel(eps, size_grid):
    """
    Метод Гаусса-Зейделя
    :param eps: точность решения
    :param size_grid: размер сетки
    """
    # размер сетки
    h = 1 / size_grid
    # номер последнего элемента в таблице
    last_elem = size_grid - 1
    # предварительное вычисление h^2
    h_2 = h ** 2
    # задаем функцию на 0-ой итерации
    u = np.zeros((size_grid, size_grid))
    # граничные значения
    for index in range(size_grid):
        u[0][index] = g(0, index)
        u[index][0] = g(index, 0)
        u[last_elem][index] = g(last_elem, index * h)
        u[index][last_elem] = g(index * h, last_elem)
    # задаем матрицу невязок
    r = np.zeros((size_grid, size_grid))
    # предварительное вычисление f
    func = np.zeros((size_grid, size_grid))
    for k in range(size_grid):
        for m in range(size_grid):
            func[k][m] = f(k * h, m * h)
    iteration = 0
    while True:
        iteration += 1
        # вычисляется сетка на текущей итерации
        last_u = u.copy()
        for k in range(1, size_grid - 1):
            for m in range(1, size_grid - 1):
                u[k][m] = 0.25 * (last_u[k + 1][m] + u[k - 1][m] + last_u[k][m + 1] + u[k][m - 1] + h_2 * func[k][m])
        # вычисляем невязку
        for k in range(1, size_grid - 1):
            for m in range(1, size_grid - 1):
                r[k][m] = - func[k][m] - ((u[k + 1][m] - 2 * u[k][m] + u[k - 1][m]) / h_2 + (u[k][m + 1] - 2 * u[k][m] +
                                                                                             u[k][m - 1]) / h_2)
        if np.max(r) < eps:
            break
    print('Норма невязки при eps = {0} и размером сетки {1}:  {2}'.format(str(eps), str(size_grid), str(np.max(r))))
    return iteration, u


def showPlot(inputArray, size_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, 1, 1 / size_grid)
    y = np.arange(0, 1, 1 / size_grid)
    xgrid, ygrid = np.meshgrid(x, y)
    ax.plot_surface(xgrid, ygrid, inputArray, rstride=1, cstride=1)
    plt.show()


result = jacobi(0.01, 64)[1]
print(result)
showPlot(result, 64)
# table_jacobi = np.zeros((3, 3))
# for i in range(3):
#     for j in range(3):
#         table_jacobi[i][j] = jacobi(10 ** -(i + 1), 2 ** (j + 5))[0]
# print(table_jacobi)
#
# table_seidel = np.zeros((3, 3))
# for i in range(3):
#     for j in range(3):
#         table_seidel[i][j] = seidel(10 ** -(i + 1), 2 ** (j + 5))[0]
# print(table_seidel)
# f = seidel(0.001,128)[1]
# showPlot(f, 128)
