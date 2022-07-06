import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class Solver:
    def __init__(self, _solver):
        self.D = 1
        self.L = 1
        self.x = np.linspace(0, self.L, 101)
        self.delta_x = self.x[1] - self.x[0]
        self.delta_t = 0.1 * self.delta_x ** 2 / (2 * self.D)
        self.t = 0
        self.t_end = 0.1
        self.gamma = 2 * self.D * self.delta_t / self.delta_x ** 2
        self.u = self.initial_condition(self.x)

        self.solution = [self.u]

        self.method = _solver
        if _solver == "FTCS":
            self.solver = self.__FTCS
        elif _solver == "Dufort_Frankel":
            self.solver = self.__Dufort_Frankel
        elif _solver == "analytic":
            self.solver = self.__analytic
        elif _solver == "BTCS":
            self.solver = self.__BTCS
        elif _solver == "Crank_Nicholson":
            self.solver = self.__Crank_Nicholson
        else:
            raise RuntimeError(f'Solver {_solver} does not exist!')

    def solve(self):
        while self.t < self.t_end:
            self.u = self.solver()
            self.solution.append(self.u)
            self.t += self.delta_t

    def initial_condition(self, x):
        return np.where(x <= self.L / 2, 2 * x / self.L, - 2 * x / self.L + 2)

    def __FTCS(self):
        res = np.zeros(self.u.shape)
        res[1:-1] = self.gamma / 2 * (self.u[:-2] - 2 * self.u[1:-1] + self.u[2:])
        return self.u + res

    def __Dufort_Frankel(self):
        if self.t == 0.0:
            self.u_minus = self.u
            return self.__FTCS()
        else:
            avg = np.zeros(self.u.shape)
            avg[1:-1] = self.u[:-2] + self.u[2:]
            self.u_minus, res = self.u, (1 - self.gamma) / (1 + self.gamma) * self.u_minus + self.gamma / (
                    1 + self.gamma) * avg
            return res

    def __BTCS(self):
        n = self.u.shape[0]
        s = diags([-self.gamma, 2 * (1 + self.gamma), -self.gamma], [-1, 0, 1], shape=(n, n), format="csc")
        x = spsolve(s, 2 * self.u)
        return x

    def __Crank_Nicholson(self):
        n = self.u.shape[0]
        s1 = diags([-self.gamma / 4, 1 + self.gamma / 2, -self.gamma / 4], [-1, 0, 1], shape=(n, n), format="csc")
        s2 = diags([self.gamma / 4, 1 - self.gamma / 2, self.gamma / 4], [-1, 0, 1], shape=(n, n), format="csc")
        x = spsolve(s1, s2.dot(self.u))
        return x

    def __analytic(self):
        def u(x, t):
            return sum(
                8 * np.sin(m * np.pi / 2) / (m * np.pi) ** 2 * np.sin(m * np.pi / self.L * x) * np.exp(
                    - self.D * (m * np.pi / self.L) ** 2 * t)
                for m in range(1, 100))

        return u(self.x, self.t)

    def plot(self):
        plt.rcParams['figure.dpi'] = 300
        number_of_plots = 4
        n = len(self.solution)
        fig = plt.figure()

        for i, t_pos in enumerate(np.linspace(0, n - 1, number_of_plots, dtype=int)):
            plt.plot(self.x, self.solution[t_pos], label=f't={(t_pos * self.delta_t):.4f}')
            plt.title(f'Method: {self.method}')

        plt.legend()
        plt.grid()
        plt.show()


def main():
    for method in ["analytic", "FTCS", "Dufort_Frankel", "BTCS", "Crank_Nicholson"][-1:]:
        x = Solver(method)
        x.solve()
        x.plot()


if __name__ == "__main__":
    main()
