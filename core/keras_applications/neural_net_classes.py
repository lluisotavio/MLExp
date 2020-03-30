
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from keras import regularizers

class DenseNetwork:

    def __init__(self, setup):

        self.layers_cells_list = setup['layers_cells_list']
        self.dropouts_rates_list = setup['dropouts_rates_list']
        self.learning_rate = setup['learning_rate']
        self.l2_reg = setup['l2_reg']
        self.activation_function = setup['activation_function']
        self.optimizer = setup['optimizer']
        self.loss_function = setup['loss_function']
        self.n_epochs = setup['n_epochs']
        self.model = None

    def construct(self, input_dim, output_dim):

        input_tensor = Input(shape=(input_dim,))
        layer_input = input_tensor

        for layer_cells, dropout_rate in \
            zip(self.layers_cells_list, self.dropouts_rates_list):

            dense_output = Dense(layer_cells,
                                 activation=self.activation_function,
                                 kernel_regularizer=regularizers.l2(self.l2_reg))(layer_input)

            layer_output = Dropout(dropout_rate)(dense_output)

            layer_input = layer_output

        output_tensor = Dense(output_dim, activation='linear')(layer_output)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function)

        return model

    def fit(self, input_data, output_data, model=None):

        if not model:
            model = self.construct(input_data.shape[1], output_data.shape[1])

        model.fit(input_data, output_data,
                  batch_size = input_data.shape[0],
                  epochs=self.n_epochs)

        self.model = model

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = load_model(path)
