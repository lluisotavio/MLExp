import numpy as np
import tensorflow as tf
import os
from MLExp.core.losses import loss_switcher

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
        self.outputpath = setup['outputpath']
        self.model_name = setup['model_name']
        self.savepath = self.outputpath + self.model_name + '/'
        self.input_dim = setup['input_dim']
        self.output_dim = setup['output_dim']

    def construct(self, input_dim, output_dim):

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        self.input_data_ph = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.output_data_ph = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.output_data_pred = self.network(self.input_data_ph, self.weights, self.biases)


        self.loss = loss_switcher(self.loss_function)(self.output_data_ph,
                                                      self.output_data_pred,
                                                      regularization_penalty=self.l2_reg,
                                                      weights=self.weights)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                beta1=0.9,
                                                                beta2=0.999,
                                                                epsilon=1e-08)

        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    # Based on https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
    def initialize_neural_net(self, layers):

        weights = list()
        biases = list()
        num_layers = len(layers)

        for l in range(0, num_layers - 1):

            W = self.xavier_init(size=[layers[l], layers[l + 1]], index=l)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                            dtype=tf.float32,
                            name='biases_{}'.format(l))
            weights.append(W)
            biases.append(b)

        return weights, biases

    # Based on https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
    def xavier_init(self, size, index):

        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev),
                                               dtype=tf.float32,
                                               name='weights_{}'.format(index))

    def network(self, input_data, weights, biases):

        H = input_data

        for ll, layer in enumerate(self.layers_cells_list[:-2]):

            W = weights[ll]
            b = biases[ll]
            H = tf.nn.elu(tf.add(tf.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y

    def callback(self, loss):

        print('Loss: %.3e' % loss)

    def fit(self, input_data, output_data):

        self.weights, self.biases = self.initialize_neural_net(self.layers_cells_list)
        self.saver = tf.train.Saver()
        self.construct(self.input_dim, self.output_dim)
        self.output_data_expr = self.network(self.input_data_ph, self.weights, self.biases)

        var_map = {self.input_data_ph: input_data, self.output_data_ph: output_data}

        for it in range(self.n_epochs):

            self.sess.run(self.train_op_Adam, var_map)

            if it % 10 == 0:

                loss_value = self.sess.run(self.loss, var_map)
                print('It: %d, Loss: %.3e' % (it, loss_value))

        self.optimizer.minimize(self.sess,
                                feed_dict=var_map,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)

        self.saver.save(self.sess, self.savepath + self.model_name)

    def restore_coeffs(self, layers):

        weights = list()
        biases = list()
        num_layers = len(layers)

        for l in range(0, num_layers - 1):

            W_array = self.sess.run('weights_{}:0'.format(l))
            W = tf.Variable(tf.zeros(W_array.shape, dtype=tf.float32),
                        dtype=tf.float32,
                        name='biases_{}'.format(l))
            W.load(W_array, self.sess)

            b_array = self.sess.run('biases_{}:0'.format(l))
            b = tf.Variable(tf.zeros(b_array.shape, dtype=tf.float32),
                        dtype=tf.float32,
                        name='biases_{}'.format(l))
            b.load(b_array, self.sess)

            weights.append(W)
            biases.append(b)

        return weights, biases

    def restore(self):

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        init = tf.global_variables_initializer()

        self.sess.run(init)

        metafile = self.savepath + self.model_name + ".meta"

        saver = tf.train.import_meta_graph(metafile)

        saver.restore(self.sess, tf.train.latest_checkpoint(self.savepath))

        self.weights, self.biases = self.restore_coeffs(self.layers_cells_list)

        self.input_data_ph = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        self.output_data_expr = self.network(self.input_data_ph, self.weights, self.biases)

    def get_weights(self, layer_index=None):

        if layer_index:

            weights = self.sess.run(self.weights[layer_index])

        else:

            weights = list()
            for weight_tensor in self.weights:
                weight = self.sess.run(weight_tensor)
                weights.append(weight)

        return weights

    def get_biases(self, layer_index=None):

        if layer_index:

            biases = self.sess.run(self.biases[layer_index])

        else:

            biases = list()
            for bias_tensor in self.biases:
                bias = self.sess.run(bias_tensor)
                biases.append(bias)

        return biases

    def predict(self, input_cube):

        map_dict = {self.input_data_ph: input_cube}

        output_data_estimated = self.sess.run(self.output_data_expr, map_dict)

        return output_data_estimated

