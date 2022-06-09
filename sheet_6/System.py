import numpy as np
from Runge_Kutta import Runge_Kutta_2, Runge_Kutta_4
from plot import plot
from scipy.integrate import solve_ivp


class System:
    def __init__(self, r_1_div_M, e, M=5, RK_order=2, step_size_ajustment=False):
        self.M = M
        self.r_1 = r_1_div_M * M
        self.e = e
        self.p = (1 + e)*self.r_1/M

        self.chi_0 = 0
        self.chi_end = 2*np.pi

        # select Runge-Kutta method
        if RK_order == 2:
            self.RK = Runge_Kutta_2
        elif RK_order == 4:
            self.RK = Runge_Kutta_4

        # calculate Runge-Kutta solution
        solution = self.RK(
            self.__f, self.chi_0, [0, 0], 0.01, self.chi_end, step_size_ajustment)

        # format Runge-Kutta solution for plotting
        self.t_solution = solution.y[:, 0]
        self.phi_solution = solution.y[:, 1]
        self.chi_solution = solution.x
        self.r_solution = self.__r(self.chi_solution)

        # solution2 = solve_ivp(self.__f, [0, 2*np.pi], [0, 0])

        pass

    def plot(self):
        # plot solution
        plot(self.r_solution, self.t_solution,
             self.phi_solution, self.chi_solution)

    def verify_delta_phi(self):
        # check the \delta \phi = 4\pi/p condition
        print(f'1/p = {1/self.p}')
        phi_analytically = 6*np.pi/self.p + 2*np.pi
        phi_numerically = self.phi_solution[-1]
        print(f'phi numerically =', phi_numerically)
        print(f'phi analytically =', phi_analytically)

    def __f(self, chi, y):
        # ode equation for the problem
        p = self.p
        e = self.e
        M = self.M
        # dt/dchi and dphi/dchi
        df_dchi = [p**2*M/((p - 2 - 2*e*np.cos(chi)) * (1 + e*np.cos(chi))**2) * np.sqrt((p - 2 - 2*e)*(p - 2 + 2*e)/(p - 6 - 2*e*np.cos(chi))),
                   np.sqrt(p/(p - 6 - 2*e*np.cos(chi)))]
        return np.array(df_dchi)

    def __r(self, chi):
        # calculate r
        p = self.p
        M = self.M
        e = self.e
        return p*M/(1 + e*np.cos(chi))
