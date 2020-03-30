import numpy as np
    
class ExplicitIntegrator:

    def __init__(self, coeffs, weights, right_operator):

        self.coeffs = coeffs
        self.weights = weights
        self.right_operator = right_operator
        self.n_stages = len(self.coeffs)

    def step(self, variables_state_initial, dt):

        variables_state = variables_state_initial
        residuals_list = np.zeros((self.n_stages,) + variables_state.shape)

        for stage in range(self.n_stages):

            k = self.right_operator(variables_state)
            residuals_list[stage, :] = k
            k_weighted = self.weights[stage].dot(residuals_list)
            variables_state = variables_state_initial + self.coeffs[stage] * dt * k_weighted

        return variables_state, k_weighted

class RK4(ExplicitIntegrator):

    def __init__(self, right_operator):

        weights = np.array(
                           [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [1/6, 2/6, 2/6, 1/6]]
                          )

        coeffs = np.array([0.5, 0.5, 1, 1])

        ExplicitIntegrator.__init__(self, coeffs, weights, right_operator)

class FunctionWrapper:

    def __init__(self, function):

        self.function = function

    def __call__(self, input_data):

        input_data = input_data[None, :]

        return self.function(input_data)[0, :]

