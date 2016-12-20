from methods import *


def gauss(eps, size_grid, absolute=False):
    """
    Метод Гаусса-Зейделя
    :param absolute: абсолютное значение ошибки для выхода из цикла
    :param eps: точность решения
    :param size_grid: размер сетки
    """
    size_grid += 1
    func = compute_func(size_grid)
    h = 1 / size_grid
    h_2 = h ** 2
    u = compute_boundary_values(size_grid)
    h_2_func = compute_h_2_func(size_grid)
    iteration = 0
    r = None
    while True:
        iteration += 1
        u = solve_double_diff_by_gauss(h_2_func=h_2_func, initArray=u, iter=1, size_grid=size_grid)
        r = compute_discrepancy(u, func, h_2, size_grid)
        if absolute and np.max(np.abs(r)) < eps:
            break
        elif not absolute and np.max(r) < eps:
            break
    print('Норма невязки при eps = {0} и размером сетки {1}:  {2}'.format(str(eps), str(size_grid),
                                                                          str(np.max(np.abs(r))) if absolute else str(
                                                                              np.max(r))))
    return iteration, u


if __name__ == "__main__":
    iterations, result = gauss(0.1, 64)
    print(iterations)
    showPlot(result, 65)

    # table_gauss = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         table_gauss[i][j] = gauss(10 ** -(i + 1), 2 ** (j + 5))[0]
    # print(table_gauss)
