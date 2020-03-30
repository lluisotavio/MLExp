import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from MLExp.core.tf_applications.neural_net_classes import DenseNetwork
from MLExp.numerics.timeint import RK4, FunctionWrapper

from argparse import ArgumentParser

if __name__ == "__main__":


    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--case', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)

    args = parser.parse_args()

    data_path = args.data_path
    case = args.case
    T_max = args.time
    dt = args.dt

    variables_file = data_path + case + '_variables.npy'
    derivatives_file = data_path + case + '_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0] / 2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    Time = np.arange(training_dim*dt, (training_dim + test_input_cube.shape[0])*dt, dt)

    model_name = case + "_tf_surrogate"
    log_file = data_path + 'log.out'

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    test_setup = {
        'layers_cells_list': [input_dim, 100, 100, 100, output_dim],
        'dropouts_rates_list': [0, 0, 0, 0],
        'learning_rate': 1e-05,
        'l2_reg': 1e-06,
        'activation_function': 'elu',
        'loss_function': 'mse',
        'optimizer': 'adam',
        'n_epochs': 20000,
        'outputpath': data_path,
        'model_name': model_name,
        'input_dim': input_dim,
        'output_dim': output_dim
    }

    neural_net = DenseNetwork(test_setup)
    neural_net.restore()

    # Post-processing
    print("Estimating the test output using the trained model.")

    estimated_cube = neural_net.predict(test_input_cube)
    estimated_cube_noise1 = neural_net.predict(1.01 * test_input_cube)
    estimated_cube_noise5 = neural_net.predict(1.05 * test_input_cube)
    estimated_cube_noise10 = neural_net.predict(1.10 * test_input_cube)

    fp = open(log_file, 'w')
    variables_names = {0:'x', 1:'y', 2:'z'}
    #fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for ss in range(estimated_cube.shape[1]):

        error = np.linalg.norm(estimated_cube[ss, :] - test_output_cube[ss, :], 2)
        error_noise1 = np.linalg.norm(estimated_cube_noise1[ss, :] - test_output_cube[ss, :], 2)
        error_noise5 = np.linalg.norm(estimated_cube_noise5[ss, :] - test_output_cube[ss, :], 2)
        error_noise10 = np.linalg.norm(estimated_cube_noise10[ss, :] - test_output_cube[ss, :], 2)

        relative_error = 100 * error / np.linalg.norm(test_output_cube, 2)
        relative_error_noise1 = 100 * error_noise1 / np.linalg.norm(test_output_cube, 2)
        relative_error_noise5 = 100 * error_noise5 / np.linalg.norm(test_output_cube, 2)
        relative_error_noise10 = 100 * error_noise10 / np.linalg.norm(test_output_cube, 2)

        if (relative_error > 100
            or relative_error_noise1 > 100
            or relative_error_noise5 > 100
            or relative_error_noise10 > 100):

            relative_error = error
            relative_error_noise1 = error_noise1
            relative_error_noise5 = error_noise5
            relative_error_noise10 = error_noise10

        name = variables_names.get(ss)

        log_string_1 = "Derivative series {}, L2 error evaluation: {}".format(name, relative_error)
        print(log_string_1)
        fp.writelines(log_string_1 + '\n')

        log_string_2 = "Derivative series {}, noise at 1%, L2 error evaluation: {}".format(name, relative_error_noise1)
        print(log_string_2)
        fp.writelines(log_string_2 + '\n')

        log_string_3 = "Derivative series {}, noise at 5%, L2 error evaluation: {}".format(name, relative_error_noise5)
        print(log_string_3)
        fp.writelines(log_string_3 + '\n')

        log_string_4 = "Derivative series {}, noise at 10%, L2 error evaluation: {}".format(name, relative_error_noise10)
        print(log_string_4)
        fp.writelines(log_string_4 + '\n')

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.plot(Time, test_output_cube[:, ss], label="Target")
        plt.plot(Time, estimated_cube[:, ss], label="Estimated")

        plt.legend()
        plt.grid(True)
        plt.ylabel(name)
        plt.xlabel("t(s)")
        plt.title("Derivative series of the variable {}".format(name))

        plt.savefig(data_path + '_' + case + "_derivative_series_{}".format(name))
        plt.show()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.plot(Time, test_output_cube[:, ss] - estimated_cube[:, ss], label="Target")

        plt.grid(True)
        plt.ylabel(name)
        plt.xlabel("t(s)")
        plt.title("Pointwise error, derivative series of the variable {}".format(name))

        plt.savefig(data_path + '_' + case + "_derivative_series_{}_pointwise".format(name))
        plt.show()

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(neural_net.predict)

    solver = RK4(right_operator)

    initial_state = input_cube[-1, :]

    time = 0
    estimated_variables = list()

    N_steps = int(T_max/dt)
    n_steps = test_input_cube.shape[0]
    interval = int(N_steps/n_steps)

    ii = 0
    # Approach based on Lui & Wolf (https://arxiv.org/abs/1903.05206)
    while time < T_max:

        state, derivative_state = solver.step(initial_state, dt)
        estimated_variables.append(state)
        initial_state = state
        sys.stdout.write("\rIteration {}".format(ii))
        sys.stdout.flush()
        time += dt
        ii += 1

    estimated_variables = np.vstack(estimated_variables)

    print("Extrapolation concluded.")

    for ss in range(estimated_variables.shape[1]):

        name = variables_names.get(ss)

        error = np.linalg.norm(estimated_variables[ss, :] - test_input_cube[ss, :], 2)
        relative_error = 100 * error / np.linalg.norm(test_input_cube, 2)

        log_string = "Variable series {}, L2 error evaluation: {}".format(name, relative_error)
        print(log_string)
        fp.writelines(log_string + '\n')

        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.autoscale(enable='False')
        plt.plot(Time, test_input_cube[:, ss], label="Target")
        plt.plot(Time, estimated_variables[:, ss], label="Estimated")
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
        plt.legend()
        plt.grid(True)
        plt.title("Variable {}".format(name))
        plt.ylabel(name)
        plt.xlabel("t(s)")
        plt.savefig(data_path + '_' + case + '_variable_series_{}.png'.format(name))
        plt.show()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.plot(Time, test_input_cube[:, ss]-estimated_variables[::interval, ss])
        plt.grid(True)
        plt.title("Pointwise error, variable {}".format(name))
        plt.ylabel("Target - Estimated")
        plt.xlabel("t(s)")
        plt.savefig(data_path + '_' + case + '_variable_series_pointwise{}.png'.format(name))
        plt.show()

    fp.close()

    print('Model restored.')
