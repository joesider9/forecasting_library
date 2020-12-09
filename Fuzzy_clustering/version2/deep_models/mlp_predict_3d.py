import numpy as np
import tensorflow as tf


class MLP_predict():
    def __init__(self, static_data, trial=0, probabilistc=False):
        self.static_data = static_data
        self.probabilistic = probabilistc
        self.trial = trial

    def init_weights(self, shape):
        init_random_dist = tf.random.truncated_normal(shape, stddev=0.001)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.001, shape=shape)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self, input_layer, init_w, init_b):
        W = self.init_weights(init_w)
        b = self.init_bias(init_b)
        return tf.add(tf.matmul(input_layer, W), b, name='prediction'), W, b

    def huber(self, y_true, y_pred, eps=0.001):
        error = y_true - y_pred
        cond = tf.abs(error) < eps

        squared_loss = tf.square(error) / (2 * eps)
        linear_loss = tf.abs(error) - 0.5 * eps

        return tf.where(cond, squared_loss, linear_loss)

    def build_graph(self, x1, best_weights, units, hold_prob, act_func):
        if act_func == 'elu':
            act_func = tf.nn.elu
        elif act_func == 'tanh':
            act_func = tf.nn.sigmoid
        with tf.name_scope("build_mlp") as scope:
            if self.trial == 0:
                mlp_1 = tf.keras.layers.Dense(
                    units[0],
                    name='dense1',
                    activation=act_func)
                full_out_dropout = tf.nn.dropout(mlp_1(x1), rate=1 - hold_prob)

            elif self.trial == 1:
                mlp_1 = tf.keras.layers.Dense(
                    units[0],
                    name='dense1',
                    activation=act_func)
                full_one_dropout = tf.nn.dropout(mlp_1(x1), rate=1 - hold_prob)
                full_layer_one = tf.keras.layers.Dense(units=units[1], activation=act_func, name='dense2')
                full_out_dropout = tf.nn.dropout(full_layer_one(full_one_dropout), rate=1 - hold_prob)

            elif self.trial == 2:
                mlp_1 = tf.keras.layers.Dense(
                    units[0],
                    name='dense1',
                    activation=act_func)
                full_one_dropout = tf.nn.dropout(mlp_1(x1), rate=1 - hold_prob)

                full_layer_two = tf.keras.layers.Dense(units=units[1], activation=act_func, name='dense2')
                full_two_dropout = tf.nn.dropout(full_layer_two(full_one_dropout), rate=1 - hold_prob)

                full_layer_three = tf.keras.layers.Dense(units=units[1], activation=act_func, name='dense3')
                full_out_dropout = tf.nn.dropout(full_layer_three(full_two_dropout), rate=1 - hold_prob)

            if self.probabilistic:
                quantiles = np.linspace(0.1, 0.9, 9)
                outputs = []
                for i, q in enumerate(quantiles):
                    # Get output layers
                    output = tf.layers.dense(full_out_dropout, 1,
                                             name="{}_q{}".format(i, int(q * 100)))
                    outputs.append(output)

            else:
                y_pred, W, b = self.normal_full_layer(full_out_dropout, best_weights['build_mlp/Variable:0'],
                                                      best_weights['build_mlp/Variable_1:0'])

        if self.trial == 0:
            weights = mlp_1.trainable_weights
            if self.probabilistic:
                for output in outputs:
                    weights += output.trainable_weights
                return y_pred, weights, mlp_1, outputs
            else:
                weights += [W, b]
                return y_pred, weights, mlp_1

        elif self.trial == 1:
            weights = mlp_1.trainable_weights + full_layer_one.trainable_weights
            if self.probabilistic:
                for output in outputs:
                    weights += output.trainable_weights
                return y_pred, weights, mlp_1, full_layer_one, outputs
            else:
                weights += [W, b]
                return y_pred, weights, mlp_1, full_layer_one

        elif self.trial == 2:
            weights = mlp_1.trainable_weights + full_layer_two.trainable_weights + full_layer_three.trainable_weights
            if self.probabilistic:
                for output in outputs:
                    weights += output.trainable_weights
                return y_pred, weights, mlp_1, full_layer_two, full_layer_three, outputs
            else:
                weights += [W, b]
                return y_pred, weights, mlp_1, full_layer_two, full_layer_three

    def predict(self, X):
        units = self.model['units']
        act_func = self.model['act_func']
        best_weights = self.model['best_weights']

        H = X

        tf.compat.v1.reset_default_graph()
        graph_mlp = tf.Graph()
        with graph_mlp.as_default():
            with tf.device("/cpu:0"):
                x1 = tf.compat.v1.placeholder('float', shape=[None, H.shape[1]], name='input_data')
            with tf.device("/cpu:0"):
                if self.trial == 0:
                    if self.probabilistic:
                        y_pred_, weights, mlp_1, outputs = self.build_graph(x1, best_weights, units, 1, act_func)
                    else:
                        y_pred_, weights, mlp_1 = self.build_graph(x1, best_weights, units, 1, act_func)

                elif self.trial == 1:
                    if self.probabilistic:
                        y_pred_, weights, mlp_1, full_layer_one, outputs = self.build_graph(x1, best_weights, units, 1,
                                                                                            act_func)
                    else:
                        y_pred_, weights, mlp_1, full_layer_one = self.build_graph(x1, best_weights, units, 1, act_func)

                elif self.trial == 2:
                    if self.probabilistic:
                        y_pred_, weights, mlp_1, full_layer_two, full_layer_three, outputs = self.build_graph(x1,
                                                                                                              best_weights,
                                                                                                              units, 1,
                                                                                                              act_func)
                    else:
                        y_pred_, weights, mlp_1, full_layer_two, full_layer_three = self.build_graph(x1, best_weights,
                                                                                                     units, 1, act_func)

        config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config_tf.gpu_options.allow_growth = True
        quantiles = np.linspace(0.1, 0.9, 9)
        with tf.compat.v1.Session(graph=graph_mlp, config=config_tf) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            mlp_1.set_weights([best_weights['build_mlp/dense1/kernel:0'], best_weights['build_mlp/dense1/bias:0']])
            if self.trial == 0:
                if self.probabilistic:
                    mlp_1.set_weights([best_weights['build_mlp/dense1/kernel:0'],
                                       best_weights['build_mlp/dense1/bias:0']])

                    for i, q in enumerate(quantiles):
                        outputs[i].set_weights(
                            [best_weights['build_mlp/{}_q{}/kernel:0'.format(i, int(q * 100))],
                             best_weights['build_mlp/{}_q{}/bias:0'.format(i, int(q * 100))]])
                else:
                    mlp_1.set_weights(
                        [best_weights['build_mlp/dense1/kernel:0'], best_weights['build_mlp/dense1/recurrent_kernel:0'],
                         best_weights['build_mlp/dense1/bias:0']])

            elif self.trial == 1:
                if self.probabilistic:
                    mlp_1.set_weights([best_weights['build_mlp/dense1/kernel:0'],
                                       best_weights['build_mlp/dense1/bias:0']])
                    full_layer_one.set_weights(
                        [best_weights['build_mlp/dense2/kernel:0'], best_weights['build_mlp/dense2/bias:0']])
                    for i, q in enumerate(quantiles):
                        outputs[i].set_weights(
                            [best_weights['build_mlp/{}_q{}/kernel:0'.format(i, int(q * 100))],
                             best_weights['build_mlp/{}_q{}/bias:0'.format(i, int(q * 100))]])
                else:
                    mlp_1.set_weights([best_weights['build_mlp/dense1/kernel:0'],
                                       best_weights['build_mlp/dense1/bias:0']])
                    full_layer_one.set_weights(
                        [best_weights['build_mlp/dense2/kernel:0'], best_weights['build_mlp/dense2/bias:0']])

            elif self.trial == 2:
                if self.probabilistic:
                    mlp_1.set_weights([best_weights['build_mlp/dense1/kernel:0'],
                                       best_weights['build_mlp/dense1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_mlp/dense2/kernel:0'], best_weights['build_mlp/dense2/bias:0']])
                    full_layer_three.set_weights([best_weights['build_mlp/dense3/kernel:0'],
                                                  best_weights['build_mlp/dense3/bias:0']])
                    for i, q in enumerate(quantiles):
                        outputs[i].set_weights(
                            [best_weights['build_mlp/{}_q{}/kernel:0'.format(i, int(q * 100))],
                             best_weights['build_mlp/{}_q{}/bias:0'.format(i, int(q * 100))]])
                else:
                    mlp_1.set_weights([best_weights['build_mlp/mlp1/kernel:0'],
                                       best_weights['build_mlp/mlp1/recurrent_kernel:0'],
                                       best_weights['build_mlp/mlp1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_mlp/dense1/kernel:0'], best_weights['build_mlp/dense1/bias:0']])

                    full_layer_three.set_weights([best_weights['build_mlp/dense3/kernel:0'],
                                                  best_weights['build_mlp/dense3/bias:0']])

            if self.probabilistic:
                results, weights_run = sess.run([outputs, weights],
                                                feed_dict={x1: H})
                y_pred = np.array([item for sublist in results
                                   for item in sublist])
            else:
                y_pred, weights_run = sess.run([y_pred_, weights],
                                               feed_dict={x1: H})

            sess.close()

        return y_pred
