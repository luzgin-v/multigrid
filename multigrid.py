from methods import *


def multigrid(eps, size_grid, deep_v_cycle=None, absolute=False):
    """
    Многосеточный метод
    :param absolute: абсолютное значение ошибки для выхода из цикла
    :param deep_v_cycle: глубина V цикла
    :param eps: точность решения
    :param size_grid: размер сетки
    """
    if deep_v_cycle is None:
        deep_v_cycle = int(math.log(size_grid, 2) - 3)
    print("Глубина спуска по V-циклу: " + str(deep_v_cycle))
    size_grid += 1
    iteration = 0
    u = compute_boundary_values(size_grid)
    h_2_func = compute_h_2_func(size_grid)
    func = compute_func(size_grid)
    # showPlot(h_2_func, size_grid)
    h_2 = (1 / size_grid) ** 2
    r = None
    while True:
        iteration += 1
        u = solve_double_diff_by_gauss(size_grid=size_grid, h_2_func=h_2_func, iter=1, initArray=u)
        # u = solve_double_diff_by_gauss_full(size_grid=size_grid, initArray=u)
        r = compute_discrepancy(u, func, h_2, size_grid)
        if absolute and np.max(np.abs(r)) < eps:
            break
        elif not absolute and np.max(r) < eps:
            break
        elif iteration > 100:
            break
        new_size = size_grid
        r_arr = []
        error_arr = []
        # ограничение невязки на грубую сетку | спуск по v-циклу
        new_r = r.copy()
        for i in range(deep_v_cycle - 1):
            r_rough, new_size = conversionToRoughGrid(new_r, new_size)
            new_error = solve_double_diff_by_gauss_full(size_grid=new_size, funcArray=-1 * r_rough)
            error_arr.append(new_error)
            new_r = r_rough - double_diff(new_error, new_size)
            r_arr.append(new_r)
        # точное решение | низшая точка v-цикла
        r_rough, new_size = conversionToRoughGrid(new_r, new_size)
        new_error = solve_double_diff_by_gauss_full(size_grid=new_size, funcArray=-1 * r_rough)
        tmp_error = new_error.copy()
        # уточнение поправки на подробную сетку | подъем по v-циклу
        for i in range(1, len(error_arr) + 1):
            detailed_error, new_size = conversionToDetailedGrid(tmp_error, new_size)
            error_with_star = error_arr[-i] + detailed_error
            tmp_error = solve_double_diff_by_gauss_full(size_grid=new_size, initArray=error_with_star,
                                                        funcArray=-1 * r_arr[-i])
        detailed_error, new_size = conversionToDetailedGrid(tmp_error, new_size)
        error_with_star = u + detailed_error
        u = solve_double_diff_by_gauss(size_grid=size_grid, h_2_func=h_2_func, initArray=error_with_star)
    print('Норма невязки при eps = {0} и размером сетки {1}:  {2}'.format(str(eps), str(size_grid),
                                                                          str(np.max(np.abs(r))) if absolute else str(
                                                                              np.max(r))))
    print('Количество итераций: ' + str(iteration))
    return iteration, u


if __name__ == "__main__":
    iter, res = multigrid(0.1, 32)
    print(iter)
    showPlot(res, 33)

    # table_multigrid = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         table_multigrid[i][j] = multigrid(10 ** -(i + 1), 2 ** (j + 5))[0]
    # print(table_multigrid)
