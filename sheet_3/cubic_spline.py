import numpy as np
from numpy import poly1d as Polynom
from typing import List, Tuple
from scipy.linalg import lu_solve, lu_factor
from scipy.sparse import diags


def cubic_spline(points: List[Tuple[float]]):
    x, y = list(zip(*points))
    N = len(points)

    a = np.zeros(N-3)
    b = np.zeros(N-2)
    c = np.zeros(N-3)
    d = np.zeros(N-2)

    # construct a_ij
    for i in range(2, N-1):
        a[i - 2] = (x[i] - x[i-1])/6

    # construct b_ij
    for i in range(1, N-1):
        b[i-1] = (x[i+1] - x[i-1])/3

    # construct c_ij
    for i in range(1, N-2):
        c[i - 2] = (x[i+1] - x[i])/6

    # construct d_i
    for i in range(1, N-1):
        d[i-1] = (y[i+1] - y[i])/(x[i+1] - x[i]) - \
            (y[i] - y[i-1])/(x[i] - x[i-1])

    # construct tridiagonal matrix
    tridiagonal_matrix = diags([a, b, c], [-1, 0, 1]).toarray()

    # solve tridiagonal matrix
    y_pp = lu_solve(lu_factor(tridiagonal_matrix), d)
    y_pp = np.append(y_pp, 0)
    y_pp = np.insert(y_pp, 0, 0)

    # construct polynomial
    A = np.zeros(N-1, dtype=Polynom)
    B = np.zeros(N-1, dtype=Polynom)
    C = np.zeros(N-1, dtype=Polynom)
    D = np.zeros(N-1, dtype=Polynom)
    valid_interval = np.zeros((N-1, 2))
    for j in range(N-1):
        A[j] = Polynom([-1, x[j+1]]) / (x[j+1] - x[j])
        B[j] = 1 - A[j]
        C[j] = 1/6 * (A[j]**3 - A[j]) * (x[j+1] - x[j])**2
        D[j] = 1/6 * (B[j]**3 - B[j]) * (x[j+1] - x[j])**2
        valid_interval[j][0], valid_interval[j][1] = x[j], x[j+1]

    # construct resulting function
    def result(x):
        for j in range(N-1):
            # get interval, in where x lies
            if x > valid_interval[j][0] and x <= valid_interval[j][1]:
                return (A[j]*y[j] + B[j]*y[j+1] + C[j]*y_pp[j] + D[j]*y_pp[j+1])(x)

    return np.vectorize(result)
