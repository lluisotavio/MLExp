import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork
from MLExp.numerics.timeint import RK4, FunctionWrapper
from MLExp.core.heuristics import TabuSearch
from argparse import ArgumentParser

import json

def prediction(neural_net, test_input_cube, choices, initial_state, log_path):

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(neural_net.predict)

    solver = RK4(right_operator)

    time = choices['time']
    T_max = choices['T_max']
    dt = choices['dt']

    fp = open(log_path, 'a+')

    estimated_variables = list()

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

    estimated_variables = np.vstack(estimated_variables)[:-1, :]

    print("Extrapolation concluded.")

    for ss in range(estimated_variables.shape[1]):

        error = np.linalg.norm(estimated_variables[:, ss] - test_input_cube[:, ss], 2)
        relative_error = 100 * error / np.linalg.norm(test_input_cube[:, ss], 2)
        log_string = "Variable series {}, L2 error evaluation: {}".format(ss, relative_error)
        print(log_string)
        fp.writelines(log_string + '\n')

    fp.writelines("\n")

    fp.close()

    return relative_error

def exec_setups(setups, input_dim, output_dim, test_input_cube, choices, initial_state, log_path, iter):

    errors_dict = dict()
    fp = open(log_path, 'a+')
    fp.writelines("Iteration {} \n".format(iter))

    for setup_key, test_setup in setups.items():

        model_name = "Oscillator_tf_surrogate" + '_' + setup_key

        test_setup['outputpath'] = data_path
        test_setup['model_name'] = model_name
        test_setup['input_dim'] = input_dim
        test_setup['output_dim'] = output_dim

        neural_net = DenseNetwork(test_setup)

        neural_net.fit(input_cube, output_cube)

        relative_error = prediction(neural_net, test_input_cube, choices, initial_state, log_path)

        errors_dict[setup_key] = relative_error
        print("Model constructed.")

    return errors_dict

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--case', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)

    args = parser.parse_args()

    data_path = args.data_path
    case = args.case
    T = args.time
    dt = args.dt

    variables_file = data_path + case + '_variables.npy'
    derivatives_file = data_path + case + '_derivatives.npy'
    setups_file = data_path + "setups.json"

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0]/2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    fp = open(setups_file, "r")
    setups = json.load(fp)
    initial_state = input_cube[-1, :]

    choices = {
                'time': 0,
                'T_max': T,
                'dt': dt
              }

    log_path = data_path + 'log.out'
    fp = open(log_path, 'w')
    fp.writelines("Execution log\n")
    fp.close()
    iter = 0
    error_dict = exec_setups(setups, input_dim, output_dim,
                             test_input_cube, choices, initial_state, log_path, iter)
    iter += 1

    key_min = min(error_dict, key=error_dict.get)
    error_min = error_dict[key_min]
    origin_setup = setups[key_min]

    iter_max = 5
    tol = 2.0

    tabu_search_config = {
                            'n_disturbances': 5,
                            'disturbance_list': {'layers_cells_list': 2}
                         }

    while (error_min >= tol) and (iter <= iter_max):

        tabu_search = TabuSearch(tabu_search_config)
        new_setups = tabu_search(origin_setup, key_min)

        error_dict = exec_setups(new_setups, input_dim, output_dim,
                                 test_input_cube, choices, initial_state, log_path, iter)

        key_min = min(error_dict, key=error_dict.get)
        error_min_current = error_dict[key_min]

        if error_min_current < error_min:

            origin_setup = new_setups[key_min]
            error_min = error_min_current

        iter += 1

        print("Iteration {} executed".format(iter))
        print("Error: {}".format(error_min))

    jf = open(data_path + "chosen_setup.json", "w")
    json.dump(origin_setup, jf)
    jf.close()

    print("Execution concluded.")
