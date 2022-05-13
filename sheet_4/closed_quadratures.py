import numpy as np


def tapezoidal(f, x_0, x_1, N):

    integral = 0

    # define sampling positions
    x = np.linspace(x_0, x_1, N)
    h = (x_1 - x_0)/N

    # calculate integral
    integral += f(x[0])/2
    for i in range(1, N - 1):
        integral += f(x[i])
    integral += f(x[-1])/2

    integral *= h

    return integral


def simpson(f, x_0, x_1, N):

    # handle uneven sampling rate
    if N % 2 != 1:
        print(f"{N=} must be uneven! N={N+1} is choosen!")
        N += 1

    integral = 0

    # define sampling positions
    x = np.linspace(x_0, x_1, N)
    h = (x_1 - x_0)/N

    # calculate integral
    integral += f(x[0])/3
    for i in range(1, N - 1):
        if i % 2:
            integral += 2/3*f(x[i])
        else:
            integral += 4/3*f(x[i])
    integral += f(x[-1])/3

    integral *= h

    return integral
