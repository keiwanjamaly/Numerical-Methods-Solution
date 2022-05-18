from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def exercise_1():
    # define function variables
    v_max = 5e4
    v_min = 0
    T = 6e3
    N = 10000

    # function value of the comparison function
    ramdom_photon_probability = T**2

    # function for calculating probability of a givin photon
    def rho(v):
        return v**2 * 1/(np.exp(v/T) - 1)

    # create a uniform distribution [0, 1] -> [0, v_max]
    def f_inv(x):
        return x*v_max

    # list of all accepted photons
    photons = []

    # calculate loops for acceptance rate
    n = 0
    while len(photons) < N:
        n += 1
        # generate random photon
        random_photon = f_inv(np.random.random(1)[0])
        # calculate probability of randon photon rho(v)
        random_photon_probability_of_density = rho(random_photon)
        # draw random number from uniform distribution [0, f(v)]
        x = np.random.random(1)[0] * ramdom_photon_probability
        # accept if x < rho(v)
        if x < random_photon_probability_of_density:
            photons.append(random_photon)

    # plot everything
    plt.hist(photons, bins=100, density=True, label="random photons")

    frequencies = np.linspace(v_max, v_min, 10000, endpoint=False)
    # calculate normalization for truncated number dencity
    normalization = quad(rho, v_min, v_max)[0]
    plt.plot(frequencies, rho(frequencies)/normalization, label="probability")

    plt.xlabel(r"$\nu$")
    plt.title(f"acceptance rate: {(N/n):.2f}")
    plt.show()


def exercise_2():
    # define functions for integration
    def f(x):
        return 1/(1+x**2)

    def w(x):
        return (4 - 2*x)/3

    number_of_random_points = [10, 20, 50,
                               100, 200, 500, 1000, 2000, 5000]

    I = []
    S = []

    I_precise = []
    S_precise = []

    for N in number_of_random_points:
        rnd_points = np.random.random(N)

        # calculate f(x)/w(x) point samples
        f_x = f(rnd_points)/w(rnd_points)

        # calculate average and variance
        f_avg = np.average(f_x)
        f_var = np.var(f_x)

        I_precise.append(f_avg)
        S_precise.append(f_var)

        # calculate f(x) point samples
        f_x = f(rnd_points)

        # calculate average and variance
        f_avg = np.average(f_x)
        f_var = np.var(f_x)

        I.append(f_avg)
        S.append(f_var)

    # plot f(x)
    figure, axis = plt.subplots(1, 2)
    axis[0].errorbar(number_of_random_points, I, yerr=S, label="f(x)")
    axis[0].set_xscale("log")
    axis[0].set_xlabel("Number of random points")
    axis[0].set_ylabel("f(x)")

    # plot f(x)/w(x)
    axis[1].errorbar(number_of_random_points, I_precise,
                     yerr=S_precise, label="f(x)/w(x)")
    axis[1].set_xscale("log")
    axis[1].set_xlabel("Number of random points")
    axis[1].set_ylabel("f(x)/w(x)")

    plt.show()

    # explain what happens
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x), label="f(x)")
    plt.plot(x, w(x), label="w(x)")
    plt.plot(x, f(x)/w(x), label="f(x)/w(x)")
    plt.legend()
    plt.xlabel("x")
    plt.show()


def exercise_3():
    # specify number of random points
    N = 1000000
    # generate random points
    points = np.random.random((N, 3))
    # distribute random points accordingly
    points[:, 0] = 1 + 3*points[:, 0]  # 1 <= x <= 4
    points[:, 1] = -3 + 7*points[:, 1]  # -3 <= x <= 7
    points[:, 2] = -1 + 2*points[:, 2]  # 1 <= x <= 4

    def density(point):
        # calculate density of torus, which is either 0 (outside) or 1 (inside)
        x, y, z = point

        return 1 if z**2 + (np.sqrt(x**2 + y**2) - 3)**2 <= 1 else 0

    # calculate the denominator
    rho_x = np.array([density(point) for point in points])
    mass = np.average(rho_x)
    S_mass = np.var(rho_x)

    # calculate the nominator
    x_i_x = np.array([rho_x[i]*points[i] for i in range(N)])
    x_i = np.zeros(3)
    S_x_i = np.zeros(3)
    for i in range(3):
        x_i[i] = np.average(x_i_x[:, i])/mass
        S_x_i[i] = np.var(x_i_x[:, i])

    # calculate error propagation
    S = np.zeros(3)
    for i in range(3):
        S[i] = S_x_i[i]/mass + x_i[i]/mass**2 * S_mass

    print(
        f"x = ({x_i[0]:.2f}, {x_i[1]:.2f}, {x_i[2]:.2f}) ± ({S[0]:.2f}, {S[1]:.2f}, {S[2]:.2f})")


if __name__ == "__main__":
    # exercise_1()
    # exercise_2()
    exercise_3()
