from numpy import poly1d as Polynom
from typing import List, Tuple


def poly_interp(points: List[Tuple[float]]):

    if len(points) == 1:  # model break condition
        return Polynom([points[0][1]])
    else:
        # (x - x_i) * P_(i+1, ..., j)(x)
        lhs = Polynom([points[0][0]], True) * poly_interp(points[1:])
        # (x - x_j) * P_(i, ..., j-1)(x)
        rhs = Polynom([points[-1][0]], True) * poly_interp(points[:-1])

    # (lhs - rhs) / (x_i - x_j)
    return (lhs - rhs) / (points[-1][0] - points[0][0])
