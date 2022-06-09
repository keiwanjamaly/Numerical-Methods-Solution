import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot(r_solution, t_solution, phi_solution, chi_solution):
    fig = plt.figure()
    margin = 0.5
    plt.subplots_adjust(wspace=margin, hspace=margin)
    ax = plt.subplot(3, 3, 1, polar=True)
    ax.plot(phi_solution, r_solution)
    line_polar, = ax.plot([], [], 'ro')

    # plot r vs t
    ax_2 = plt.subplot(3, 3, 2)
    ax_2.plot(t_solution, r_solution)
    line_r_t, = ax_2.plot([], [], 'ro')
    ax_2.set_xlabel(r'$t$')
    ax_2.set_ylabel(r'$r$')

    # plot r vs chi
    ax_3 = plt.subplot(3, 3, 3)
    ax_3.plot(chi_solution, r_solution)
    line_r_chi, = ax_3.plot([], [], 'ro')
    ax_3.set_xlabel(r'$\chi$')
    ax_3.set_ylabel(r'$r$')

    # plot phi vs chi
    ax_4 = plt.subplot(3, 3, 4)
    ax_4.plot(chi_solution, phi_solution)
    line_phi_chi, = ax_4.plot([], [], 'ro')
    ax_4.axhline(y=2*np.pi, color='r', linestyle='-')
    ax_4.set_xlabel(r'$\chi$')
    ax_4.set_ylabel(r'$\phi$')

    # plot r*cos(phi) vs chi
    ax_5 = plt.subplot(3, 3, 5)
    ax_5.plot(chi_solution, r_solution*np.cos(phi_solution))
    line_r_cos_phi, = ax_5.plot([], [], 'ro')
    ax_5.set_xlabel(r'$\chi$')
    ax_5.set_ylabel(r'$r*\cos(\phi)$')

    # plot r*sin(phi) vs chi
    ax_6 = plt.subplot(3, 3, 6)
    ax_6.plot(chi_solution, r_solution*np.sin(phi_solution))
    line_r_sin_phi, = ax_6.plot([], [], 'ro')
    ax_6.set_xlabel(r'$\chi$')
    ax_6.set_ylabel(r'$r*\sin(\phi)$')

    # plot r*cos(phi) vs r*sin(phi)
    ax_7 = plt.subplot(3, 3, 7)
    ax_7.plot(r_solution*np.sin(phi_solution), r_solution*np.cos(phi_solution))
    line_r_cos_phi_vs_r_sin_phi, = ax_7.plot([], [], 'ro')
    ax_7.set_xlabel(r'$r*\sin(\phi)$')
    ax_7.set_ylabel(r'$r*\cos(\phi)$')

    def update(t):
        i = t
        line_polar.set_data(phi_solution[i], r_solution[i])
        line_r_t.set_data(t_solution[i], r_solution[i])
        line_r_chi.set_data(chi_solution[i], r_solution[i])
        line_phi_chi.set_data(chi_solution[i], phi_solution[i])
        line_r_cos_phi.set_data(
            chi_solution[i], r_solution[i]*np.cos(phi_solution[i]))
        line_r_sin_phi.set_data(
            chi_solution[i], r_solution[i]*np.sin(phi_solution[i]))
        line_r_cos_phi_vs_r_sin_phi.set_data(
            r_solution[i]*np.sin(phi_solution[i]), r_solution[i]*np.cos(phi_solution[i]))

    ani = FuncAnimation(fig, update, frames=range(
        len(chi_solution)), interval=10)
    plt.show()
