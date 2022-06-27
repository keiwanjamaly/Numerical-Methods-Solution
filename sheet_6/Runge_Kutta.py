import numpy as np
from numpy.linalg import norm

# base class for Runge-Kutta solver


class Runge_Kutta():
    def __init__(self, f, x0, y0, h, x_end, step_size_adjustment=False):
        self.f = f
        self.x = [x0]
        self.y = [np.array(y0)]
        self.h = h
        self.x_end = x_end
        self.step_size_adjustment = step_size_adjustment

        # solve ode problem
        self.rk_solve()

        # convert solution to numpy arrays
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def rk_step(self, h):
        # the specific definition of the Runge-Kutta step, which should be
        # implemented in a child class
        raise NotImplementedError()

    def rk_solve(self):
        # the runge-kutta solver
        epsilon = 1e-7
        N = 0
        while self.x[-1] <= self.x_end:
            x_new, y_new = self.rk_step(self.h, self.x[-1], self.y[-1])

            # additional steps for stepsize adjustment
            if self.step_size_adjustment:
                # perform additional h step so that h + h = 2h
                x_new, y_new = self.rk_step(self.h, x_new, y_new)
                # perform 2*h step
                x_new_2h, y_new_2h = self.rk_step(
                    self.h*2, self.x[-1], self.y[-1])

                # calculate error
                delta = norm(y_new - y_new_2h)
                delta_ref = epsilon * norm(y_new + self.h*self.f(x_new, y_new))

                # adjust stepsize
                self.h = self.h * (delta_ref / delta) ** (1/5)

                N += 1

            self.x.append(x_new)
            self.y.append(y_new)

        # print out the advantage for stepsize adjustment
        if self.step_size_adjustment:
            print(
                f"Used {N=} iterations instead of {((self.x[-1] - self.x[0])/self.h):.0f}")


class Runge_Kutta_2(Runge_Kutta):
    # define rk2 step
    def rk_step(self, h, x, y):
        k_1 = self.f(x, y)
        k_2 = self.f(x + h/2, y + k_1/2 * h)
        x_new = x + h
        y_new = y + k_2 * h

        return x_new, y_new


class Runge_Kutta_4(Runge_Kutta):
    # define rk4 step
    def rk_step(self, h, x, y):
        k_1 = self.f(x, y)
        k_2 = self.f(x + h/2, y + k_1/2 * h)
        k_3 = self.f(x + h/2, y + k_2/2 * h)
        k_4 = self.f(x + h, y + k_3 * h)
        x_new = x + h
        y_new = y + (k_1 + 2*k_2 + 2*k_3 + k_4) / 6 * h

        return x_new, y_new
