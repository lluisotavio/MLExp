import sys, os
sys.path.insert(0, '.')

import numpy as np
from MLExp.tests.problem_classes import VanDerPolSystem
from MLExp.numerics.timeint import RK4
import matplotlib.pyplot as plt
from argparse import ArgumentParser


# Testing to solve a nonlinear oscillator problem using
# a 4th order and 4 steps Runge-Kutta

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--mu', type=float)
    parser.add_argument('--A', type=float)
    parser.add_argument('--omega', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    T = args.time
    dt = args.dt
    mu = args.mu
    A = args.A
    omega = eval(args.omega)

    initial_state = np.array([.5, 0, 0])

    problem = VanDerPolSystem(mu, A, omega)

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    iter = 0

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step(current_state, dt)

        variables_timesteps.append(variables_state[:, None])
        derivatives_timesteps.append(derivatives_state[:, None])
        current_state = variables_state

        sys.stdout.write("\rIteration {}".format(iter))
        sys.stdout.flush()
        iter += 1

    variables_matrix = np.hstack(variables_timesteps)[:-1, :]
    derivatives_matrix = np.hstack(derivatives_timesteps)[:-1, :]

    # Just for experimentation
    variables_matrix = variables_matrix# / np.abs(variables_matrix.max(1)[:, None])
    derivatives_matrix = derivatives_matrix# / np.abs(variables_matrix.max(1)[:, None])

    plt.plot(time, variables_matrix[0, :], label="x")
    plt.plot(time, variables_matrix[1, :], label="y")

    label_string = "mu_{}_A_{}_omega_{}_T_{}_dt_{}/".format(mu, A, omega, T, dt)

    if not os.path.isdir(data_path + label_string):
        os.mkdir(data_path + label_string)

    np.save(data_path + "{}VanDerPol_variables.npy".format(label_string), variables_matrix)
    np.save(data_path + "{}VanDerPol_derivatives.npy".format(label_string), derivatives_matrix)

    plt.xlabel("Time(s)")
    plt.title("Lorenz System")
    plt.legend()

    plt.grid(True)

    plt.show()
