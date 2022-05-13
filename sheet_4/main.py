import numpy as np
import matplotlib.pyplot as plt


def exercise_1():
    # import the necessary functions
    from closed_quadratures import tapezoidal, simpson

    # define the function to be integrated
    def f(x, m, n):
        return np.sin(x)**(2*m-1) * np.cos(x)**(2*n-1)

    result_tapezoidal = []
    N_conv = np.arange(3, 30, 2, dtype=int)
    for N in N_conv:
        result = tapezoidal(lambda x: f(x, 2, 1), 0, np.pi/2, N)
        result_tapezoidal.append(result)

    # evaluate integral at different sampling rates
    result_simpson = []
    for N in N_conv:
        result = simpson(lambda x: f(x, 2, 1), 0, np.pi/2, N)
        result_simpson.append(result)

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
    def f(x, p):
        return x**(p-1)/(1+x)

    x = 1000
    print(tapezoidal(lambda x: f(x, 0.5), 0, x, x * 100),
          tapezoidal(lambda x: f(x, 0.5), 0, 2*x, 2*x * 100))

    # evaluate integral at different sampling rates
    result_tapezoidal = []
    N_conv = np.arange(9, 30000, 100, dtype=int)
    for N in N_conv:
        result = tapezoidal(lambda x: f(x, 0.5), 0, x, N)
        result_tapezoidal.append(result)

    result_simpson = []
    for N in N_conv:
        result = simpson(lambda x: f(x, 0.5), 0, x, N)
        result_simpson.append(result)

    # plot two solutions
    plt.plot(N_conv, result_tapezoidal, label="tapezoidal")
    plt.plot(N_conv, result_simpson, label="simpson")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("$I_1$")
    plt.show()


if __name__ == "__main__":
    # exercise_1()
    exercise_2()
