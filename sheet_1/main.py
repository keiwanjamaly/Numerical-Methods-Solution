import matplotlib.pyplot as plt
import numpy as np


def bisection(f, a, b, error_x):
    f_a = f(a)
    f_b = f(b)

    # check if the interval is bracketed
    if f_a * f_b > 0:
        raise RuntimeError("The interval is not bracketed")

    n = 0  # used for counting the bisection iterations
    while abs(b-a) > error_x:
        n += 1

        # take the middle of the interval
        x = (a+b)/2
        f_x = f(x)

        if f_x * f_a > 0:  # f(x) and f(a) have same sign, so root is between (x, b)
            a = x
            f_a = f_x
        else:  # f(x) and f(a) have different signs, so root is between (a, x)
            b = x
            f_b = f_x

    print(f'Bisection took {n} iterations.')

    return (b+a)/2


def newton_raphson(f, df, error_x, x=0):
    n = 0
    while abs(f(x)/df(x)) > error_x:
        n += 1

        # calculate the next point of the root chain
        x = x - f(x)/df(x)

    print(f'Newton-Raphson took {n} iterations.')

    return x


if __name__ == "__main__":

    # set the absolute precision of the root on the x-Axis
    precision = 1e-7

    # the followinf two functions are test cases with known roots, to verify, that the code works.
    # in practice, you should test a function more rigorosly, but it this should be enough to convince ourself, that the
    # functions work properly.
    def f_1(x):
        return x**2 - 10*x**3 + 10

    def df_1(x):
        return 2*x - 30*x**2

    # plot function
    x = np.linspace(-3, 3, 100)
    plt.plot(x, f_1(x))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel(r'$f(x) = - 10x^3 + x^2 + 10$')
    plt.show()

    print("\n")
    print("===================================================================")
    print("Running tests! Root should be at x≈1.03446910730266")
    print("Bisection: x=", bisection(f_1, -5, 5, precision))
    print("Newton-Raphson: x=", newton_raphson(f_1, df_1, precision, 4))

    # define the energy balance equation and it's derivative
    def f_2(T_e, config):
        """
        alpha = U/(p*c**2)
        """
        T_p = config["T_p"]
        T_gamma = config["T_gamma"]
        alpha = config["alpha"]
        Q = config["Q"]
        Lambda = config["Lambda"]
        c = config["c"]

        return Q * alpha * (T_e - T_gamma) - Lambda / c**3 * (T_p/T_e - 1) * 1/np.sqrt(T_e)

    def df_2(T_e, config):
        """
        alpha = U/(p*c**2)
        """
        T_p = config["T_p"]
        T_gamma = config["T_gamma"]
        alpha = config["alpha"]
        Q = config["Q"]
        Lambda = config["Lambda"]
        c = config["c"]

        return Q * alpha - Lambda / c**3 * (- 3*T_p/(2*T_e**(5/2)) + 1/(2*T_e**(3/2)))

    # all function parameters are stored in this config dictionary, which reduces the number of parameters, which needs
    # to be passed to the function.
    config = {
        "Q": 2.7e-10,
        "Lambda": 4.4e30,
        "c": 2.99e10
    }
    # i)
    # define function parameters
    config_i = config.copy()
    config_i["T_p"] = 1e9
    config_i["T_gamma"] = 1e7
    config_i["alpha"] = 1

    # plot function
    x = np.logspace(6, 8, 100)
    plt.loglog(x, np.abs(f_2(x, config_i)))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel(r'$|f(x)|$')
    plt.show()

    # calculate and print the root
    print("\n")
    print("===================================================================")
    print("Exercise i)")
    print("Bisection: x=", bisection(lambda x: f_2(
        x, config_i), 1e6, 1e8, precision))
    print("Newton-Raphson: x=", newton_raphson(lambda x: f_2(x, config_i),
          lambda x: df_2(x, config_i), precision, 2e7))

    # ii)
    # define function parameters
    config_ii = config.copy()
    config_ii["T_p"] = 1e7
    config_ii["T_gamma"] = 1e9
    config_ii["alpha"] = 8e-5

    # plot function
    x = np.logspace(6, 9, 100)
    plt.loglog(x, np.abs(f_2(x, config_ii)))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel(r'$|f(x)|$')
    plt.show()

    # calculate and print the root
    print("\n")
    print("===================================================================")
    print("Exercise ii)")
    # The Bisection method does not work, because the wanted absolute precision is smaller than the numerical precision
    # of the root.
    # print("Bisection: x=", bisection(lambda x: f_2(x, config_ii), 1e6, 1e9, precision))
    print("Newton-Raphson: x=", newton_raphson(lambda x: f_2(x, config_ii),
          lambda x: df_2(x, config_ii), precision, 5e8))
