import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork

from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--case', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    case = args.case

    # Standard nomenclature
    variables_file = data_path + case + '_variables.npy'
    derivatives_file = data_path + case + '_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    # We are using the first half part of the data set
    # for the training process
    training_dim = int(variables.shape[0]/2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    model_name = case + "_tf_surrogate"

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    # This test setup (or multiple setups) can be stored in JSON files and
    # read at run time
    test_setup = {
                  'layers_cells_list': [input_dim, 100, 100, 100, output_dim],
                  'dropouts_rates_list': [0, 0, 0, 0],
                  'learning_rate': 1e-05,
                  'l2_reg': 1e-07, #1e-05,
                  'activation_function': 'elu',
                  'loss_function': 'mse',
                  'optimizer': 'adam',
                  'n_epochs': 2000,
                  'outputpath': data_path,
                  'model_name': model_name,
                  'input_dim': input_dim,
                  'output_dim': output_dim
                 }

    # It constructs the neural net
    neural_net = DenseNetwork(test_setup)
    # It executes the training process and saves the model
    # in data_path
    neural_net.fit(input_cube, output_cube)

    print("Model constructed.")

