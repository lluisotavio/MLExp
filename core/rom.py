import numpy as np
from keras.layers import (
    Input, Dense, Dropout,
    Conv2D, Conv2DTranspose, Flatten,
    Reshape, MaxPooling2D
 )
from keras.models import Model
from keras.models import load_model
from keras import regularizers

class AutoEncoder:

    def __init__(self, layers_configuration, setup):

        self.layers_configuration = layers_configuration
        self.learning_rate = setup['learning_rate']
        self.l2_reg = setup['l2_reg']
        self.optimizer = setup['optimizer']
        self.loss_function = setup['loss_function']
        self.n_epochs = setup['n_epochs']
        self.model = None
        self.conv_types = {'encoder': Conv2D, 'decoder': Conv2DTranspose}

    def layerwise_construct(self, layer, layer_input, stage=None, input_shape=None):

        filters_layer = layer.get('filters')
        kernel_size_layer = layer.get('kernel_size')
        strides_layer = layer.get('strides')
        padding_layer = layer.get('padding')
        activation_layer = layer.get('activation')
        pool_size_layer = layer.get('pool_size')
        pool_strides_layer = layer.get('pool_strides')
        pool_padding_layer = layer.get('pool_padding')

        kwargs = {'strides' : strides_layer,
                  'padding' : padding_layer,
                  'activation' : activation_layer}

        convLayer = self.conv_types.get(stage)

        if input_shape:
            kwargs['input_shape'] = input_shape

        conv_op = convLayer(filters_layer, kernel_size_layer, **kwargs)

        layer_output_conv = conv_op(layer_input)

        if pool_size_layer:
            maxpool_op = MaxPooling2D(pool_size=pool_size_layer,
                                      strides=pool_strides_layer,
                                      padding=pool_padding_layer)

            layer_output = maxpool_op(layer_output_conv)
        else:
            layer_output = layer_output_conv

        return  layer_output

    def construct(self, n_channels, n_rows, n_columns):

        input_tensor = Input(shape=(n_rows, n_columns, n_channels))

        # Beginning of the Encoder transformation
        encoder_layers = self.layers_configuration.get('encoder')
        decoder_layers = self.layers_configuration.get('decoder')

        layer_input = input_tensor

        for layer_key, layer in encoder_layers.items():

            layer_output = self.layerwise_construct(layer, layer_input, stage='encoder')
            layer_input = layer_output

        # "Bottleneck" operations. Here should be the central dense layers operations

        output_dims = layer_output.shape
        total_dimension = output_dims[1].value*output_dims[2].value*output_dims[3].value
        layer_output = Reshape([total_dimension])(layer_output)
        # Here the dense operations for dimensionality reduction must be performed
        output_dims = layer_output.shape
        layer_output = Reshape([1, 1, output_dims[1].value])(layer_output)
        encoder_output_tensor = layer_output
        # End of the encoder transformation

        layer_input = encoder_output_tensor
        input_shape = [shape.value for shape in layer_input.shape]

        # Getting out the first layer
        first_key, first_layer = encoder_layers.popitem(last=False)

        layer_output = self.layerwise_construct(first_layer,
                                                layer_input,
                                                stage='decoder',
                                                input_shape=input_shape)
        layer_input = layer_output

        for layer_key, layer in decoder_layers.items():

            layer_output = self.layerwise_construct(layer, layer_input, stage='decoder')
            layer_input = layer_output

        output_tensor = layer_output

        print("Encoder constructed")

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function)

        return model

    def fit(self, input_data, output_data, model=None):

        n_rows, n_columns, n_channels = input_data.shape[1:]

        if not model:
            model = self.construct(n_channels, n_rows, n_columns)

        model.fit(input_data, output_data,
                  batch_size=input_data.shape[0],
                  epochs=self.n_epochs)

        self.model = model

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = load_model(path)
