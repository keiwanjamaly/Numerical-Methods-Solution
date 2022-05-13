import numpy as np


def tapezoidal(f, x_0, x_1, N):
    integral = 0

    # define sampling positions
    x = np.linspace(x_0, x_1, N)
    h = (x_1 - x_0)/N

    # calculate integral
    integral += f(x[1]) * 3/2
    for i in range(2, N - 2):
        integral += f(x[i])
    integral += f(x[-2]) * 3/2

    integral *= h

    return integral


def simpson(f, x_0, x_1, N):

    # handle out uneven sampling rate
    if N % 2 != 1:
        print(f"{N=} must be uneven! N={N+1} is choosen!")
        N += 1
    integral = 0

    # define sampling positions
    x = np.linspace(x_0, x_1, N)
    h = (x_1 - x_0)/N

    # calculate integral
    integral += f(x[1]) * 27/12
    # for f(x[2]) the factor is 0
    integral += f(x[3]) * 13/12
    for i in range(4, N - 4):
        if i % 2:
            integral += 4/3*f(x[i])
        else:
            integral += 2/3*f(x[i])
    integral += f(x[-4]) * 13/12
    # for f(x[-3]) the factor is 0
    integral += f(x[-2]) * 27/12

    integral *= h

    return integral
