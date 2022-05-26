from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


def logarithmic(x):
    return -np.log(x)


def logarithmic_inverse(x):
    return -np.log(1-x)


def tangential(x):
    return np.tan(np.pi/2 * x)


def quotential(x):
    return x/(1-x)


def function(x):
    p = 0.5
    return x**(p-1)/(1+x)


if __name__ == "__main__":
    x = np.linspace(0, 1, 10000000, endpoint=False)

    # plt.plot(x, logarithmic(x), label="$-ln(x)$")
    plt.plot(x, logarithmic_inverse(x), label="$-ln(1-x)$")
    plt.plot(x, tangential(x), label="$\\tan(\pi/2x)$")
    plt.plot(x, quotential(x), label="$x/(1-x)$")
    # plt.plot(x, function(x), label="$f(x)$")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y(x)")

    plt.xscale("log")
    plt.yscale("log")

    plt.show()
