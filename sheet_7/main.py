import numpy as np
from System import System
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def initial_condition(x):
    x_0 = 0.5
    sigma = 0.1
    return np.exp(-((x - x_0)**2)/sigma**2)


def L(i, diff_x):
    return diff_x(i)


def main():
    methods = ["upwind", "FTCS", "analytical"]
    solutions = []
    lines = []
    fig = plt.figure()
    ax = plt.subplot(111)
    time_step = 0.001

    for method in methods:
        x = System(L, initial_condition, method, time_step, 0.01, t_end=1)
        y_sol, t_sol, x_sol = x.solve()

        line, = ax.plot(x_sol, y_sol[0], label=f"method = {method}")
        lines.append(line)
        solutions.append(y_sol)

    def update(t):
        for i in range(len(methods)):
            lines[i].set_ydata(solutions[i][t])

    ani = FuncAnimation(fig, update, frames=range(
        len(t_sol)), interval=10)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
