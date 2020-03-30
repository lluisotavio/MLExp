import numpy as np

class DenseGraphDisturbance:

    def __init__(self, weigths_list):

        self.weight_disturbance = 1
        self.bias_disturbance = 0
        self.weights_list = weigths_list

    def addition(self, indices_list, disturbances_list):

        for index, disturbance in zip(indices_list, disturbances_list):

            weights = self.weights_list[index]
            biases = self.weights_list[index + 1]

            input_dim_l = weights.shape[0]
            output_dim_l = weights.shape[1]

            # Adding a cell to the layer l implies in adding a new
            # column in the weights matrix W_l

            added_weights_column = self.weight_disturbance*np.ones((input_dim_l, disturbance))
            added_biases_rows = self.bias_disturbance*np.ones(disturbance)

            weights_new = np.hstack([weights, added_weights_column])
            biases_new = np.hstack([biases, added_biases_rows])

            self.weights_list[index] = weights_new
            self.weights_list[index + 1] = biases_new

            weights_next = self.weights_list[index + 2]
            output_dim = weights_next.shape[1]
            added_weights_rows = self.weight_disturbance*np.ones([disturbance, output_dim])
            weights_next_new = np.vstack([weights_next, added_weights_rows])

            self.weights_list[index + 2] = weights_next_new

