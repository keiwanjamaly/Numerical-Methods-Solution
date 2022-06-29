import matplotlib.pyplot as plt
import numpy as np


class Direction:
    x = 0
    y = 1


class System:
    def __init__(self, solver):
        x_0 = 0.5
        y_0 = 0.5
        sigma = 0.1
        self.delta_x = 0.01
        self.delta_t = self.delta_x / np.sqrt(2)
        self.x_grid = np.arange(0, 1, self.delta_x)
        self.y_grid = np.arange(0, 1, self.delta_x)
        self.x, self.y = np.meshgrid(self.x_grid, self.y_grid)
        self.t = 0
        self.t_end = 1.0
        self.u = np.exp(-((self.x - x_0) ** 2 + (self.y - y_0) ** 2) / sigma ** 2)
        self.r = self.__diff(self.u, Direction.x)
        self.l = self.__diff(self.u, Direction.y)
        self.s = np.zeros(self.u.shape)

        self.solution = [self.u]

        if solver == "Lax-Friedrich":
            self.__solver = self.__Lax_Friedrichs
        elif solver == "Leapfrog":
            self.__solver = self.__Leapfrog

    def solve(self):
        while self.t < self.t_end:
            self.r, self.l, self.s, self.u = self.__solver()
            self.solution.append(self.u)
            self.t += self.delta_t

    def nearest_neighbour_sum(self, y):
        res = np.roll(y, -1, axis=Direction.x)
        res += np.roll(y, 1, axis=Direction.x)
        res += np.roll(y, -1, axis=Direction.y)
        res += np.roll(y, 1, axis=Direction.y)
        return res

    def __diff(self, y, direction):
        return (np.roll(y, 1, axis=direction) - np.roll(y, -1, axis=direction)) / (2 * self.delta_x)

    def __Lax_Friedrichs(self):
        r_new = self.nearest_neighbour_sum(self.r) / 4 + self.delta_t * self.__diff(self.s, Direction.x)
        l_new = self.nearest_neighbour_sum(self.l) / 4 + self.delta_t * self.__diff(self.s, Direction.y)
        s_new = self.nearest_neighbour_sum(self.s) / 4 + self.delta_t * (
                self.__diff(self.r, Direction.x) + self.__diff(self.l, Direction.y))
        u_new = self.nearest_neighbour_sum(self.u) / 4 + self.delta_t * self.s
        return r_new, l_new, s_new, u_new

    def __Leapfrog(self):
        alpha = self.delta_t / self.delta_x
        if self.t == 0.0:
            self.u_minus = self.u
            return self.__Lax_Friedrichs()
        else:
            self.u_minus, res = self.u, - self.u_minus + alpha ** 2 * self.nearest_neighbour_sum(
                self.u) + 2 * self.u * (1 - 2 * alpha ** 2)
            return None, None, None, res

    def plot(self):
        plt.rcParams['figure.dpi'] = 300
        number_of_plots = 9
        n = len(self.solution)
        fig = plt.figure()
        grid_size = int(np.ceil(np.sqrt(number_of_plots)))

        for i, t_pos in enumerate(np.linspace(0, n - 1, number_of_plots, dtype=int)):
            ax = fig.add_subplot(grid_size, grid_size, i + 1, projection='3d')
            ax.plot_surface(
                self.x, self.y, self.solution[t_pos], cmap='viridis', edgecolor='none')
            ax.set_title(f't={(t_pos * self.delta_t):.3f}', y=1)

        plt.show()

        return


def main():
    for system in ["Lax-Friedrich", "Leapfrog"]:
        x = System(system)
        x.solve()
        x.plot()


if __name__ == "__main__":
    main()
