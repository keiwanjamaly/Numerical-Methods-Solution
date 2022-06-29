import numpy as np

from System import System


def initial_condition(x, x_0, sigma):
    return np.array([np.exp(-((x - x_0) ** 2) / sigma ** 2)])


def initial_condition_2(x, x_0, sigma):
    simple_initial_condition = initial_condition(
        x, x_0, sigma)[0]
    dx = x[1] - x[0]
    return np.array(
        [simple_initial_condition, diff(simple_initial_condition, dx), np.zeros(len(x))])


def diff(u, dx):
    return (np.roll(u, 1) - np.roll(u, -1)) / (2 * dx)


def L_1(u, delta_x):
    return np.array(diff(u[0], delta_x))


def L_2(u, delta_x):
    return np.array([u[2], diff(u[2], delta_x), diff(u[1], delta_x)])


def main():
    methods = ["FTCS", "LAX", "Leapfrog", "Lax–Wendroff", "analytic"]
    systems = [(L_1, initial_condition), (L_2, initial_condition_2)]
    for (system, ic) in systems:
        for method in methods:
            x = System(system, lambda x: ic(
                x, 0.5, 0.1), method, 0.001, 0.01)
            x.solve()
            x.plot()


if __name__ == "__main__":
    main()
