import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class System:
    def __init__(self, L, inital_conditions, solver, delta_t, delta_x, t=0, t_end=10, x_0=0, x_end=1):
        self.method = solver
        if solver == "FTCS":
            self.solver = self.__FTCS
        elif solver == "upwind":
            self.solver = self.__upwind
        elif solver == "analytical":
            self.solver = self.__analytical
        else:
            raise ValueError("Unknown solver")

        self.L = L
        self.x = np.arange(x_0, x_end, delta_x)
        self.t = np.arange(t, t_end, delta_t)
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.y = np.empty((self.t.shape[0], self.x.shape[0]), dtype=np.ndarray)
        self.y[0] = inital_conditions(self.x)
        self.initial_conditions = inital_conditions

    def solve(self):
        for iter in range(len(self.t)-1):
            # self.y.append(self.solver())
            self.y[iter + 1, :] = self.solver(iter)

        return self.y, self.t, self.x

    def __diff_symmetric(self, i):
        res = np.empty(self.x.shape[0] - 2)
        y = self.y[i]
        res = (y[2:] - y[:-2])/(self.delta_x*2)
        return res

    def __diff_forward(self, i):
        res = np.empty(self.x.shape[0] - 2)
        y = self.y[i]
        res = (y[1:-1] - y[:-2])/self.delta_x
        return res

    def __FTCS(self, i):
        res = np.empty(self.y[i].shape)
        res[1:-1] = self.y[i][1:-1] - self.delta_t * \
            self.L(i, self.__diff_symmetric)
        Q = (self.delta_x - self.delta_t) / (self.delta_x + self.delta_t)
        y = self.y[i]
        res[-1] = y[-2] - Q * res[-2] + Q * y[-1]
        res[0] = 0
        return res

    def __upwind(self, i):
        res = np.empty(self.y[i].shape)
        res[1:-1] = self.y[i][1:-1] - self.delta_t * \
            self.L(i, self.__diff_forward)
        Q = (self.delta_x - self.delta_t) / (self.delta_x + self.delta_t)
        y = self.y[i]
        res[-1] = y[-2] - Q * res[-2] + Q * y[-1]
        res[0] = 0
        return res

    def __analytical(self, i):
        return self.initial_conditions(self.x - self.delta_t * i)

    def plot(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        line, = ax.plot(self.x, self.y[0, :])
        ax.set_title(
            f"method = {self.method}")

        def update(i):
            line.set_ydata(self.y[i, :])

        ani = FuncAnimation(fig, update, frames=range(
            len(self.t)), interval=1)
        plt.show()
