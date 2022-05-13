import numpy as np
import copy


def carmers_rule(A, b):
    n = 0
    # determinant of matrix A
    det_A = (A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2])
             - A[1][0] * (A[0][1] * A[2][2] - A[2][1] * A[0][2])
             + A[2][0] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]))

    n += 14

    result = [0, 0, 0]
    # fill coefficients
    for i in range(3):
        for j in range(3):

            # remove the row and line for the subdeterminant.
            x = [0, 1, 2]
            x.remove(j)
            y = [0, 1, 2]
            y.remove(i)

            # use the indices of the submatrix to calculate the subdeterminant
            # and it's coefficients
            c_ji = (-1)**(i+j) * (A[x[0]][y[0]] * A[x[1]]
                                  [y[1]] - A[x[1]][y[0]] * A[x[0]][y[1]])
            n += 5

            result[i] += c_ji * b[j] / det_A

            n += 2

    return result, n


def LU_decomp(A, b):
    n = 0
    # perform LU decomposition
    N = len(A)
    # copy matrix for later comparison, since we will reuse memory of the
    # decomposed matrix
    A_copy = copy.deepcopy(A)
    for j in range(N):
        for i in range(0, j+1):
            # do \beta decomposition part
            A_copy[i][j] = A_copy[i][j] - \
                sum([A_copy[i][k]*A_copy[k][j] for k in range(0, i)])
            n += 1  # minus sum
            n += i  # multiplications in sum
            n += i - 1  # summations in sum
        for i in range(j+1, N):
            # do \alpha decomposition part
            A_copy[i][j] = (A_copy[i][j] - sum([A_copy[i][k]*A_copy[k][j]
                                                for k in range(0, j)]))/A_copy[j][j]
            n += 2  # minus sum and division
            n += j  # multiplications in sum
            n += j  # summations in sum

    # debug code for LU decomposition check
    # L = np.zeros((N, N))
    # U = np.zeros((N, N))
    # for i in range(0, N):
    #     for j in range(0, N):
    #         if i == j:
    #             L[i][j] = 1
    #             U[i][j] = A_copy[i][j]
    #         if i < j:
    #             U[i][j] = A_copy[i][j]
    #         if i > j:
    #             L[i][j] = A_copy[i][j]

    # functions which will transform from A_copy to L and U
    def alpha(i, j):
        if i == j:
            return 1
        else:
            return A_copy[i][j]

    def beta(i, j):
        return A_copy[i][j]

    # perform forward substitution
    y = [0 for _ in range(N)]
    y[0] = b[0]/alpha(0, 0)
    for i in range(1, N):
        y[i] = (b[i] - sum([alpha(i, j) * y[j]
                for j in range(0, i)]))/alpha(i, i)
        n += 2  # minus sum and division
        n += i  # multiplications in sum
        n += i - 1  # summations in sum

    # perform backward substitution
    x = [0 for _ in range(N)]
    x[-1] = y[N-1]/beta(N-1, N-1)
    for i in range(N-2, -1, -1):
        x[i] = (y[i] - sum([beta(i, j) * x[j]
                for j in range(i+1, N)]))/beta(i, i)
        n += 2  # minus sum and division
        n += (N - (i+1))  # multiplications in sum
        n += (N - (i+1)) - 1  # summations in sum

    return x, n


def calculate_residuals(A, b, x):
    N = len(A)
    r = [0 for _ in range(N)]
    for i in range(N):
        r[i] = sum([A[i][j]*x[j] for j in range(N)]) - b[i]

    return r


def generate_matrix(n):
    a = [[0 for _ in range(n)] for _ in range(n)]
    eps = 1e-5
    for i in range(n):
        for j in range(n):
            a[i][j] = 1/(np.cos((i+1) + eps) + np.sin((j+1) + eps))

    return a


def generate_vector(n):
    b = [0 for _ in range(n)]
    eps = 1e-5
    for i in range(n):
        b[i] = 1/(i+1) + eps

    return b


if __name__ == "__main__":
    # exercise i)
    A_i = generate_matrix(3)
    b_i = generate_vector(3)
    x_i, n = carmers_rule(A_i, b_i)

    print("===================================================================")
    print("Exercise i)")
    print("x =", x_i)
    print(f"it took {n} operations!")

    # exercise ii)
    # generate random 10x10 problems and hope, that they are linearly independent
    A_ii = generate_matrix(10)
    b_ii = generate_vector(10)
    x_ii, n = LU_decomp(A_ii, b_ii)

    print("\n")
    print("===================================================================")
    print("Exercise ii)")
    print("x =", x_ii)
    print(f"it took {n} operations!")

    # exercise iii)
    r_i = calculate_residuals(A_i, b_i, x_i)
    r_ii = calculate_residuals(A_ii, b_ii, x_ii)

    print("\n")
    print("===================================================================")
    print("Exercise iii)")
    print("r_i =", r_i)
    print("error of i)", np.linalg.norm(r_i, ord=np.inf))
    print("r_ii =", r_ii)
    print("error of ii)", np.linalg.norm(r_ii, ord=np.inf))
