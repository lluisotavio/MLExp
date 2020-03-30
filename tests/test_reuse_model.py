import sys

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from core.keras_applications.neural_net_classes import DenseNetwork
from core.keras_applications.initilization import DenseGraphDisturbance
from numerics.timeint import RK4, FunctionWrapper

if __name__ == "__main__":

    data_path = 'MLExp/data/'

    variables_file = data_path + 'Oscillator_variables.npy'
    derivatives_file = data_path + 'Oscillator_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0] / 2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    model_name = "Oscillator_surrogate.h5"

    model_previous = load_model(data_path + model_name)
    weights_previous = model_previous.get_weights()

    test_setup_disturbed = {
        'layers_cells_list': [50, 50],  # [50, 50] + [1, 0]
        'dropouts_rates_list': [0, 0],
        'learning_rate': 1e-04,
        'l2_reg': 1e-06,
        'activation_function': 'elu',
        'loss_function': 'mse',
        'optimizer': 'adam',
        'n_epochs': 20000
    }

    neural_net_disturbed = DenseNetwork(test_setup_disturbed)
    model_disturbed = neural_net_disturbed.construct(2, 2)

    graph_disturbing = DenseGraphDisturbance(weights_previous)
    #graph_disturbing.addition([0], [1])

    weights_current = graph_disturbing.weights_list
    model_disturbed.set_weights(weights_current)

    neural_net_disturbed.fit(input_cube, output_cube, model=model_disturbed)
    print("Saving the trained model")
    neural_net_disturbed.save(data_path + model_name)
