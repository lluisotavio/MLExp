import numpy as np
import matplotlib.pyplot as plt

from core.keras_applications.neural_net_classes import DenseNetwork

if __name__ == "__main__":

    data_path = 'MLExp/data/'

    variables_file = data_path + 'Oscillator_variables.npy'
    derivatives_file = data_path + 'Oscillator_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0]/2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    model_name = "Oscillator_surrogate.h5"

    test_setup = {
                  'layers_cells_list': [50, 50],
                  'dropouts_rates_list': [0, 0],
                  'learning_rate': 1e-05,
                  'l2_reg': 1e-06,
                  'activation_function': 'elu',
                  'loss_function': 'mse',
                  'optimizer': 'adam',
                  'n_epochs': 20000
                  }

    neural_net = DenseNetwork(test_setup)
    neural_net.fit(input_cube, output_cube)
    print("Saving the trained model")
    neural_net.save(data_path+model_name)

