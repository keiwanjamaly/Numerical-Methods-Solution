import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect


from numpy import poly1d as Polynom

from polynomial import poly_interp
from cubic_spline import cubic_spline
from bilinear import bilinear


def exercise_i():
    # plot interval
    y = Polynom([-1, 4, -30, 200, 3])

    # calculate roots
    roots = np.zeros(2)
    roots[0] = bisect(y, -10, 3)
    roots[1] = bisect(y, 3, 10)

    # plot function
    x = np.linspace(-10, 10, 100)
    plt.plot(x, y(x))
    plt.scatter(roots, y(roots))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title(
        f'Roots are at $x_0 = {roots[0]:.3f}$ and $x_1 = {roots[1]:.3f}$')
    plt.show()

    # interpolate polynom
    for i in range(1, 4):
        # points, at which the points are calculated
        interp_x = np.linspace(-10, 10, i + 1)
        y_x = y(interp_x)
        # generate data points
        points = list(zip(interp_x, y_x))
        p_1 = poly_interp(points)

        # calculate error
        x_err = np.linspace(-5, 5, 2)
        error = np.abs(p_1(x_err) - y(x_err))

        # print at x=5 and x=-5
        print(
            f"for order {i}, the value of p(x_1)={p_1(x_err[0])} ans p(x_2)={p_1(x_err[1])}")
        print(
            f'interpolations of y(x) at order {i} with error = {(error)}')

        # plot the result
        plt.plot(x, p_1(x), label=f'interpolated polynomial')
        plt.scatter(interp_x, y_x)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.plot(x, y(x), label="original")
        plt.scatter(x_err, error, label="error")
        plt.legend()
        plt.show()


def exercise_ii():
    def y(x):
        A = [1, 0.5, -0.5]
        m = [8, 24, 8/3]
        L = 64
        result = 0
        result += A[0] * np.cos(2*np.pi * m[0]/L * x)
        result += A[1] * np.cos(2*np.pi * m[1]/L * x)
        result += A[2] * np.cos(2*np.pi * m[2]/L * x)

        return result

    # error for cubic splines and polynomials
    error_cp = []
    error_p = []
    spline_order = list(range(2, 20))
    print(f"{y(12)=}")
    for i in spline_order:
        x_interp = np.linspace(0, 20, i)
        print(f"spline/polynomial order = {i}")

        y_x = y(x_interp)
        points = list(zip(x_interp, y_x))
        p_1 = poly_interp(points)
        print(f"{p_1(12)=}")

        if i != 2:
            p_2 = cubic_spline(points)
            print(f"{p_2(12)=}")

        error_p.append(np.abs(y(12) - p_1(12)))
        if i != 2:
            error_cp.append(np.abs(y(12) - p_2(12)))

    # plot error
    plt.plot(spline_order, error_p, label="polynomial")
    plt.plot(spline_order[1:], error_cp, label="cubic splines")
    plt.xlabel("number of data points")
    plt.ylabel("error")
    plt.title("error at x = 12")
    plt.legend()
    plt.show()

    # # plot everything
    x = np.linspace(0, 20, 1000)
    plt.plot(x, y(x), label="original")
    plt.plot(x, p_1(x), label="polynomial interpolation")
    plt.plot(x, p_2(x), label="cubic spline")
    plt.scatter(x_interp, y_x)
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.show()


def exercise_iii():
    def z(x, y):
        return x**2 - y**2 + 1

    # generate data points
    error = []
    grid_sizes = np.arange(3, 31, 1, dtype=int)
    for size in grid_sizes:
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        x, y = np.meshgrid(x, y)
        zc = z(x, y)
        points = (x, y, zc)

        z_interp = bilinear(points)
        error.append(np.abs(z_interp(5, 0) - z(5, 0)))

    # plot error
    plt.plot(grid_sizes, error)
    plt.xlabel("gird sizes")
    plt.ylabel("error")
    plt.show()

    # plot interpolation
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z(x, y)")
    ax.plot_surface(x, y, z_interp(x, y), color="red")
    ax.set_title("interpolation")

    # plot original
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z(x, y)")
    ax.set_title("original")
    ax.plot_surface(x, y, z(x, y), color="red")
    plt.show()


if __name__ == "__main__":
    # exercise_i()
    # exercise_ii()
    exercise_iii()
    # pass
