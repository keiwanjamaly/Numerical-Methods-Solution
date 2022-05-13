import numpy as np
from numpy import poly1d as Polynom


def bilinear(points):
    x1, x2, z = points
    nx2, nx1 = x1.shape
    t = np.zeros((nx2-1, nx1-1), dtype=Polynom)
    u = np.zeros((nx2-1, nx1-1), dtype=Polynom)
    y = np.zeros((nx2-1, nx1-1, 4))
    for i in range(nx1-1):
        for j in range(nx2-1):
            # treat x1[j,i], x2[j,i]
            t[j, i] = Polynom([x1[j, i]], True)/(x1[j, i+1] - x1[j, i])
            u[j, i] = Polynom([x2[j, i]], True)/(x2[j+1, i] - x2[j, i])
            y[j, i, 0], y[j, i, 1], y[j, i, 2], y[j, i,
                                                  3] = z[j, i], z[j, i+1], z[j+1, i+1], z[j+1, i]

    def result(x1c, x2c):
        for i in range(nx1-1):
            for j in range(nx2-1):
                # treat x1[j,i], x2[j,i]
                if x1[j, i] <= x1c < x1[j, i+1] and x2[j, i] <= x2c < x2[j+1, i]:
                    tc = t[j, i](x1c)
                    uc = u[j, i](x2c)
                    y_0, y_1, y_2, y_3 = y[j, i]
                    return (1-tc)*(1-uc)*y_0 + tc*(1-uc)*y_1 + tc*uc*y_2 + (1-tc)*uc*y_3

    return np.vectorize(result)
