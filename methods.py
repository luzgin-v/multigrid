import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def f(kh, mh):
    return -6 * (kh * kh * mh * (kh * kh + 2 * mh * mh))


def g(kh, mh):
    return kh ** 4 * mh ** 3


def p(pq, kh, mh):
    return math.sin(math.pi * pq * kh) * math.sin(math.pi * pq * mh)


def solve_double_diff_by_jacobi(size_grid, h_2_func, iter=1, initArray=None):
    """
    Решение уравнения Пуассона методом Якоби
    :param h_2_func:
    :param iter: Количество итераций
    :param size_grid: Размер сетки
    :param initArray: Входная матрица (если нет, то будет сгенерирована из 0)
    :return: Результат вычисления в виде матрицы
    """
    outArray = initArray.copy()
    for i in range(iter):
        for k, m in itertools.product(range(1, size_grid - 1), range(1, size_grid - 1)):
            outArray[k][m] = 0.25 * (
                initArray[k + 1][m] + initArray[k - 1][m] + initArray[k][m + 1] + initArray[k][m - 1] +
                h_2_func[k][m])
        if i != iter - 1:
            initArray = outArray.copy()
    return outArray


def solve_double_diff_by_gauss(size_grid, h_2_func, iter=1, initArray=None):
    """
    Решение уравнения Пуассона методом Зайделя
    :param h_2_func:
    :param iter: Количество итераций
    :param size_grid: Размер сетки
    :param initArray: Входная матрица (если нет, то будет сгенерирована из 0)
    :return: Результат вычисления в виде матрицы
    """
    for i in range(iter):
        for k, m in itertools.product(range(1, size_grid - 1), range(1, size_grid - 1)):
            initArray[k][m] = 0.25 * (
                initArray[k + 1][m] + initArray[k - 1][m] + initArray[k][m + 1] + initArray[k][m - 1] + h_2_func[k][m])
    return initArray


def solve_double_diff_by_gauss_full(size_grid, iter=1, initArray=None, funcArray=None):
    """
    Решение уравнения Пуассона методом Зайделя
    :param iter: Количество итераций
    :param size_grid: Размер сетки
    :param funcArray: Матрица ограничения
    :param initArray: Входная матрица (если нет, то будет сгенерирована из 0)
    :return: Результат вычисления в виде матрицы
    """
    h = 1 / size_grid
    square_h = h ** 2
    h_2_func = np.zeros((size_grid, size_grid))
    if funcArray == None:
        for k, m in itertools.product(range(size_grid), range(size_grid)):
            h_2_func[k][m] = f(k * h, m * h) * square_h
    else:
        for k, m in itertools.product(range(size_grid), range(size_grid)):
            h_2_func[k][m] = funcArray[k][m] * square_h
    if initArray == None:
        initArray = np.zeros((size_grid, size_grid))
    for i in range(iter):
        for k, m in itertools.product(range(size_grid - 2, 0, -1), range(size_grid - 2, 0, -1)):
            initArray[k][m] = 0.25 * (
                initArray[k + 1][m] + initArray[k - 1][m] + initArray[k][m + 1] + initArray[k][m - 1] + h_2_func[k][m])
    return initArray


def compute_discrepancy(u, func, h_2, size_grid):
    r = np.zeros((size_grid, size_grid))
    for k, m in itertools.product(range(1, size_grid - 1), range(1, size_grid - 1)):
        r[k][m] = - func[k][m] - (
            (u[k + 1][m] - 2 * u[k][m] + u[k - 1][m]) / h_2 + (u[k][m + 1] - 2 * u[k][m] + u[k][m - 1]) / h_2)
    return r


def compute_func(size_grid):
    h = 1 / size_grid
    func = np.zeros((size_grid, size_grid))
    for k, m in itertools.product(range(size_grid), range(size_grid)):
        func[k][m] = f(k * h, m * h)
    return func


def compute_h_2_func(size_grid):
    h = 1 / size_grid
    h_2 = h ** 2
    func = np.zeros((size_grid, size_grid))
    h_2_func = func.copy()
    for k, m in itertools.product(range(size_grid), range(size_grid)):
        func[k][m] = f(k * h, m * h)
        h_2_func[k][m] = func[k][m] * h_2
    return h_2_func


def compute_boundary_values(size_grid):
    last_elem = size_grid - 1
    h = 1 / size_grid
    u = np.zeros((size_grid, size_grid))
    for index in range(size_grid):
        u[0][index] = g(0, index * h)
        u[index][0] = g(index * h, 0)
        u[last_elem][index] = g(last_elem*h, index * h)
        u[index][last_elem] = g(index * h, last_elem*h)
    return u


def set_boundary_as_zero(inputArray, size_grid):
    last_elem = size_grid - 1
    for index in range(size_grid):
        inputArray[0][index] = 0
        inputArray[index][0] = 0
        inputArray[last_elem][index] = 0
        inputArray[index][last_elem] = 0
    return inputArray


def double_diff(inputArray, size_grid):
    square_h = (1 / size_grid) ** 2
    outArray = np.zeros((size_grid, size_grid))
    for k, m in itertools.product(range(1, size_grid - 1), range(1, size_grid - 1)):
        outArray[k][m] = (((inputArray[k + 1][m] - 2 * inputArray[k][m] + inputArray[k - 1][m]) / square_h) + (
            (inputArray[k][m + 1] - 2 * inputArray[k][m] + inputArray[k][m - 1]) / square_h))
    return outArray


def conversionToRoughGrid(inputArray, prev_size_grid):
    """
    Оператор перехода на грубую сетку
    :param inputArray: Исходная матрица
    :param prev_size_grid: Исходный размер сетки
    :return: Матрица на грубой сетке и новый размер сетки
    """
    new_size_grid = prev_size_grid // 2
    outArray = np.zeros((new_size_grid + 1, new_size_grid + 1))
    for i in range(new_size_grid + 1):
        for j in range(new_size_grid + 1):
            outArray[i][j] = inputArray[i * 2][j * 2]
    return outArray, new_size_grid + 1


def conversionToDetailedGrid(inputArray, prev_size_grid):
    """
    Оператор перехода на подробную сетку
    :param inputArray: Исходная матрица
    :param prev_size_grid: Исходный размер сетки
    :return: Матрица на подробной сетке и новый размер сетки
    """
    new_size_grid = (prev_size_grid - 1) * 2 + 1
    outArray = np.zeros((new_size_grid, new_size_grid))
    for i in range(new_size_grid):
        for j in range(new_size_grid):
            i_odd = i % 2 == 0
            j_odd = j % 2 == 0
            if i_odd and j_odd:
                outArray[i][j] = inputArray[i // 2][j // 2]
            elif not i_odd and j_odd:
                outArray[i][j] = 0.5 * (inputArray[i // 2][j // 2] + inputArray[(i + 2) // 2][j // 2])
            elif i_odd and not j_odd:
                outArray[i][j] = 0.5 * (inputArray[i // 2][j // 2] + inputArray[i // 2][(j + 2) // 2])
            else:
                outArray[i][j] = 0.25 * (
                    inputArray[i // 2][j // 2] + inputArray[i // 2][(j + 2) // 2] + inputArray[(i + 2) // 2][
                        j // 2] +
                    inputArray[(i + 2) // 2][(j + 2) // 2])
    return outArray, new_size_grid


def showPlot(inputArray, size_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, 1, 1 / size_grid)
    y = np.arange(0, 1, 1 / size_grid)
    xgrid, ygrid = np.meshgrid(x, y)
    ax.plot_surface(xgrid, ygrid, inputArray, rstride=1, cstride=1)
    plt.show()


if __name__ == "__main__":
    pass

# p = np.random.randint(0, 100, (9, 9))
# p = set_boundary_as_zero(p, 9)
# print(p)
# new_p, n = conversionToRoughGrid(p, 8)
# print(n)
# print(new_p)
# new_p, n = conversionToRoughGrid(new_p, n)
# print(n)
# print(new_p)
# n_p, n = conversionToDetailedGrid(new_p, n)
# print(n_p)
# print(n_p.shape)

# print(compute_boundary_values(8))
# showPlot(compute_boundary_values(8), 8)
