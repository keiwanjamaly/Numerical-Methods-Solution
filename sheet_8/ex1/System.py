import matplotlib.pyplot as plt
import numpy as np


class System:
    def __init__(self, L, inital_conditions, solver, delta_t, delta_x, t=0, t_end=1, x_0=0,
                 x_end=1):

        self.method = solver
        if solver == "FTCS":
            self.solver = self.__FTCS
        elif solver == "LAX":
            self.solver = self.__LAX
        elif solver == "Leapfrog":
            self.solver = self.__Leapfrog
        elif solver == "Lax–Wendroff":
            self.solver = self.__Lax_Wendroff
        elif solver == "analytic":
            self.solver = self.__analytical
        else:
            raise ValueError("Unknown solver")

        self.L = L
        self.delta_t = delta_t
        self.delta_x = delta_x
        self.x = np.arange(x_0, x_end, delta_x)
        self.t = np.arange(t, t_end, delta_t)
        ic = self.__set_initial_conditions(inital_conditions)
        self.y = np.empty((self.t.shape[0], len(
            ic), self.x.shape[0]), dtype=np.ndarray)
        self.y[0] = ic
        self.t_end = t_end
        self.initial_condition = inital_conditions

    def __set_initial_conditions(self, inital_conditions):
        return inital_conditions(self.x)

    def solve(self):
        for iter in range(len(self.t) - 1):
            self.y[iter + 1] = self.solver(iter, self.delta_t, self.y)

        return self.y, self.t, self.x

    def __FTCS(self, i, delta_t, y):
        return y[i] - delta_t * self.L(y[i], self.delta_x)

    def __LAX(self, i, delta_t, y):
        return (np.roll(y[i], 1, axis=1) + np.roll(y[i], -1, axis=1)) / 2 - delta_t * self.L(y[i], self.delta_x)

    def __Leapfrog(self, i, delta_t, y):
        if i == 0:
            y[1] = y[0]
            delta_t2 = delta_t ** 2
            for _ in np.arange(self.t[0], delta_t, delta_t2):
                y[1] = self.__FTCS(1, delta_t2, y)

            return y[1]

        return y[i - 1] - 2 * delta_t * self.L(y[i], self.delta_x)

    def __Lax_Wendroff(self, i, delta_t, y):
        factor = (delta_t / self.delta_x) ** 2 / 2
        correction = factor * (np.roll(y[i], 1, axis=1) - 2 * y[i] + np.roll(y[i], -1, axis=1))
        return y[i] - delta_t * self.L(y[i], self.delta_x) + correction

    def __analytical(self, i, delta_t, y):
        return self.initial_condition((self.x - delta_t * i) % 1)

    def plot(self):
        # TODO: make compacring plots
        number_of_plots = 4
        n = len(self.t)

        for i, t_pos in enumerate(np.linspace(0, n - 1, number_of_plots, dtype=int)):
            ax = plt.subplot(number_of_plots, 1, i + 1)
            ax.plot(self.x, self.y[t_pos, 0], label=f't={self.t[t_pos]:.3f}')
            ax.legend()
            ax.set_ylim(-0.2, 1.2)
            ax.yaxis.grid()
            if t_pos != n - 1:
                ax.set_xticks([])
            if i == 0:
                ax.set_title(f'Method: {self.method}')

        plt.show()
