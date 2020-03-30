import numpy as np
from argparse import ArgumentParser
from core.rom import AutoEncoder
from collections import OrderedDict

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    data_path = args.data_path

    # Loading the training and testing data
    # It considers that the input format is Numpy.
    # TODO Generalize it for other formats, such as HDF5

    data = np.load(data_path)

    print("Input data loaded.")

    batch_size = data.shape[0]

    training_data = data[:batch_size, :, :, :]
    testing_data = data[batch_size:, :, :, :]

    # The number of channels is equivalent to the number of variables
    n_channels = data.shape[3]
    n_rows = data.shape[1]
    n_columns = data.shape[2]

    encoder_keys = ['conv1', 'conv2', 'conv3']
    decoder_keys = ['dconv1', 'dconv2', 'dconv3']
    
    encoder_dict = OrderedDict()
    decoder_dict = OrderedDict()
    
    conv1 = {
                        'filters': 2*n_channels,
                        'kernel_size': (5, 10),
                        'strides': (5, 5),
                        'padding': "valid",
                        'activation': "relu",
                        'pool_size': None,
                        'pool_strides': None,
                        'pool_padding': None
            }

    conv2 = {
                        'filters': 4 * n_channels,
                        'kernel_size': (5, 10),
                        'strides': (3, 5),
                        'padding': "valid",
                        'activation': 'relu',
                        'pool_size': None,
                        'pool_strides': None,
                        'pool_padding': None
            }

    conv3 = {

                        'filters': 8 * n_channels,
                        'kernel_size': (3, 3),
                        'strides': (3, 4),
                        'padding': "valid",
                        'activation': 'relu',
                        'pool_size': None,
                        'pool_strides': None,
                        'pool_padding': None
            }

    dconv1 = {
                        'filters': 8 * n_channels,
                        'kernel_size': (2, 2),
                        'strides': (7, 7),
                        'padding': "valid",
                        'activation': "relu",
                        'pool_size': None,
                        'pool_strides': None,
                        'pool_padding': None
              }

    dconv2 =  {
                        'filters': 4 * n_channels,
                        'kernel_size': (2, 2),
                        'strides': (6, 6),
                        'padding': "same",
                        'activation': 'relu',
                        'pool_size': (2, 2),
                        'pool_strides': (2, 2),
                        'pool_padding': "same"
              }

    dconv3 = {
                        'filters': n_channels,
                        'kernel_size': (2, 2),
                        'strides': (1, 1),
                        'padding': "same",
                        'activation': 'relu',
                        'pool_size': (2, 2),
                        'pool_strides': (2, 2),
                        'pool_padding': "same"
              }


    encoder_list = [conv1, conv2, conv3]
    decoder_list = [dconv1, dconv2, dconv3]

    for key, item in zip(encoder_keys, encoder_list):
        encoder_dict[key] = item

    for key, item in zip(decoder_keys, decoder_list):
        decoder_dict[key] = item

    layers_configuration = {'encoder': encoder_dict, 'decoder': decoder_dict}

    setup = {
                'learning_rate': 1e-5,
                'optimizer': 'adam',
                'loss_function': 'mse',
                'l2_reg': 1e-05,
                'n_epochs' : 1000
            }

    autoencoder_rom = AutoEncoder(layers_configuration, setup)

    autoencoder_rom.fit(training_data, training_data)