import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def exercise_1():
    m = 2
    n = 4
    # import the necessary functions
    from closed_quadratures import tapezoidal, simpson

    # define the function to be integrated
    def f(x, m, n):
        return np.sin(x)**(2*m-1) * np.cos(x)**(2*n-1)

    result_tapezoidal = []
    N_conv = np.arange(5, 100, 2, dtype=int)
    for N in N_conv:
        result = tapezoidal(lambda x: f(x, m, n), 0, np.pi/2, N)
        result_tapezoidal.append(result)

    # evaluate integral at different sampling rates
    result_simpson = []
    for N in N_conv:
        result = simpson(lambda x: f(x, m, n), 0, np.pi/2, N)
        result_simpson.append(result)

    print(
        f"for N={N_conv[-1]}: I_trapezoid = {result_tapezoidal[-1]}, I_simpson={result_simpson[-1]}")

    # plot two solutions
    plt.plot(N_conv, result_tapezoidal, label="tapezoidal")
    plt.plot(N_conv, result_simpson, label="simpson")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("$I_2$")
    plt.show()


def exercise_2():
    # import the necessary functions
    from open_quadratures import tapezoidal, simpson

    # define the function to be integrated
    def f(y, p):
        return (y/(1-y))**(p-1) / ((1.0 + y/(1-y)) * (1-y)**2)

    # def f(y, p):
    #     return (np.tan(np.pi/2*y))**(p-1) / (1 + np.tan(np.pi/2*y)) * np.pi/(2 * np.cos(np.pi/2*y)**2)

    # evaluate integral at different sampling rates
    result_tapezoidal = []
    N_conv = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for N in N_conv:
        result = tapezoidal(lambda pos: f(pos, 0.5), 0, 1, N)
        result_tapezoidal.append(result - np.pi)

    result_simpson = []
    for N in N_conv:
        result = simpson(lambda x: f(x, 0.5), 0, 1, N)
        result_simpson.append(result - np.pi)

    # plot two solutions
    plt.plot(N_conv, result_tapezoidal, label="tapezoidal")
    plt.plot(N_conv, result_simpson, label="simpson")
    plt.legend()
    plt.xlabel("N")
    plt.xscale("log")
    plt.yscale("symlog")
    plt.ylabel("$I_1$")
    plt.show()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
