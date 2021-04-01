import tensorflow as tf


class Lstm3dPredict:
    def __init__(self, model, scale_lstm, trial, probabilistic):
        self.scale_lstm = scale_lstm
        self.trial = trial
        self.probabilistic = probabilistic
        self.model = model

    def create_inputs(self, X_train):
        self.N, self.D1, self.D2 = X_train.shape

        H = X_train
        H = H.reshape(-1, self.D1 * self.D2)
        sc = self.scale_lstm
        H = sc.transform(H.reshape(-1, self.D1 * self.D2))
        H = H.reshape(-1, self.D1, self.D2)

        return H

    def init_weights(self, init_w):
        init_random_dist = tf.convert_to_tensor(init_w)
        return tf.Variable(init_random_dist)

    def init_bias(self, init_b):
        init_bias_vals = tf.convert_to_tensor(init_b)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self, input_layer, init_w, init_b):

        W = self.init_weights(init_w)
        b = self.init_bias(init_b)
        return tf.add(tf.matmul(input_layer, W), b, name='prediction'), W, b

    def build_graph(self, x1, best_weights, units, hold_prob):
        with tf.name_scope("build_lstm") as scope:
            if self.trial == 0:
                lstm_1 = tf.keras.layers.LSTM(
                    units[0],
                    name='lstm1',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_out_dropout = tf.nn.dropout(lstm_1(x1), rate=1 - hold_prob)
                shape = full_out_dropout.get_shape().as_list()
                full_out_dropout = tf.reshape(full_out_dropout, [-1, shape[1] * shape[2]])

            elif self.trial == 1:
                lstm_1 = tf.keras.layers.LSTM(
                    units[0],
                    name='lstm1',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_one_dropout = tf.nn.dropout(lstm_1(x1), rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                lstm_1_flat = tf.reshape(full_one_dropout, [-1, shape[1] * shape[2]])
                full_layer_one = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu,
                                                       name='dense1')
                full_out_dropout = tf.nn.dropout(full_layer_one(lstm_1_flat), rate=1 - hold_prob)

            elif self.trial == 2:
                lstm_1 = tf.keras.layers.LSTM(
                    units[0],
                    name='lstm1',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_one_dropout = tf.nn.dropout(lstm_1(x1), rate=1 - hold_prob)

                shape = full_one_dropout.get_shape().as_list()
                lstm_2_flat = tf.reshape(full_one_dropout, [-1, shape[1] * shape[2]])
                full_layer_two = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu,
                                                       name='dense1')
                full_two_dropout = tf.nn.dropout(full_layer_two(lstm_2_flat), rate=1 - hold_prob)
                full_two_dropout = tf.reshape(full_two_dropout, [-1, shape[1], shape[2]])

                lstm_2 = tf.keras.layers.LSTM(
                    units[2],
                    name='lstm2',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_out_dropout = tf.nn.dropout(lstm_2(full_two_dropout), rate=1 - hold_prob)
                shape = full_out_dropout.get_shape().as_list()
                full_out_dropout = tf.reshape(full_out_dropout, [-1, shape[1] * shape[2]])
            elif self.trial == 3:
                lstm_1 = tf.keras.layers.LSTM(
                    units[0],
                    name='lstm1',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_one_dropout = tf.nn.dropout(lstm_1(x1), rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                lstm_2_flat = tf.reshape(full_one_dropout, [-1, shape[1] * shape[2]])

                full_layer_two = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu,
                                                       name='dense1')
                full_two_dropout = tf.nn.dropout(full_layer_two(lstm_2_flat), rate=1 - hold_prob)
                full_two_dropout = tf.reshape(full_two_dropout, [-1, shape[1], shape[2]])
                lstm_2 = tf.keras.layers.LSTM(
                    units[2],
                    name='lstm2',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_three_dropout = tf.nn.dropout(lstm_2(full_two_dropout), rate=1 - hold_prob)
                shape = full_three_dropout.get_shape().as_list()
                lstm_2_flat = tf.reshape(full_three_dropout, [-1, shape[1] * shape[2]])
                full_layer_three = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu,
                                                         name='dense2')
                full_three_dropout = tf.nn.dropout(full_layer_three(lstm_2_flat), rate=1 - hold_prob)
                full_three_dropout = tf.reshape(full_three_dropout, [-1, shape[1], shape[2]])
                lstm_3 = tf.keras.layers.LSTM(
                    units[2],
                    name='lstm3',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_out_dropout = tf.nn.dropout(lstm_3(full_three_dropout), rate=1 - hold_prob)
                shape = full_out_dropout.get_shape().as_list()
                full_out_dropout = tf.reshape(full_out_dropout, [-1, shape[1] * shape[2]])
            if self.probabilistic:
                prob_layer = tf.keras.layers.Dense(100, activation=tf.nn.softmax, name='dense_prob')
                y_pred = prob_layer(full_out_dropout)
            else:
                y_pred, W, b = self.normal_full_layer(full_out_dropout, best_weights['build_lstm/Variable:0'],
                                                      best_weights['build_lstm/Variable_1:0'])

            if self.trial == 0:
                weights = lstm_1.trainable_weights
                if self.probabilistic:
                    weights += prob_layer.trainable_weights
                    return y_pred, weights, lstm_1, prob_layer
                else:
                    weights += [W, b]
                    return y_pred, weights, lstm_1

            elif self.trial == 1:
                weights = lstm_1.trainable_weights + full_layer_one.trainable_weights
                if self.probabilistic:
                    weights += prob_layer.trainable_weights
                    return y_pred, weights, lstm_1, full_layer_one, prob_layer
                else:
                    weights += [W, b]
                    return y_pred, weights, lstm_1, full_layer_one

            elif self.trial == 2:
                weights = lstm_1.trainable_weights + full_layer_two.trainable_weights + lstm_2.trainable_weights
                if self.probabilistic:
                    weights += prob_layer.trainable_weights
                    return y_pred, weights, lstm_1, full_layer_two, lstm_2, prob_layer
                else:
                    weights += [W, b]
                    return y_pred, weights, lstm_1, full_layer_two, lstm_2

            elif self.trial == 3:
                weights = lstm_1.trainable_weights + full_layer_two.trainable_weights + lstm_2.trainable_weights + full_layer_three.trainable_weights + lstm_3.trainable_weights
                if self.probabilistic:
                    weights += prob_layer.trainable_weights
                    return y_pred, weights, lstm_1, full_layer_two, lstm_2, full_layer_three, lstm_3, prob_layer
                else:
                    weights += [W, b]
                    return y_pred, weights, lstm_1, full_layer_two, lstm_2, full_layer_three, lstm_3

    def predict(self, X):
        units = self.model['units']
        best_weights = self.model['best_weights']

        H = self.create_inputs(X)

        tf.compat.v1.reset_default_graph()
        graph_lstm = tf.Graph()
        with graph_lstm.as_default():
            with tf.device("/cpu:0"):
                x1 = tf.compat.v1.placeholder('float', shape=[None, self.D1, self.D2], name='input_data')
            with tf.device("/cpu:0"):
                if self.trial == 0:
                    if self.probabilistic:
                        y_pred_, weights, lstm_1, prob_layer = self.build_graph(x1, best_weights, units, 1)
                    else:
                        y_pred_, weights, lstm_1 = self.build_graph(x1, best_weights, units, 1)

                elif self.trial == 1:
                    if self.probabilistic:
                        y_pred_, weights, lstm_1, full_layer_one, prob_layer = self.build_graph(x1, best_weights, units,
                                                                                                1)
                    else:
                        y_pred_, weights, lstm_1, full_layer_one = self.build_graph(x1, best_weights, units, 1)

                elif self.trial == 2:
                    if self.probabilistic:
                        y_pred_, weights, lstm_1, full_layer_two, lstm_2, prob_layer = self.build_graph(x1,
                                                                                                        best_weights,
                                                                                                        units, 1)
                    else:
                        y_pred_, weights, lstm_1, full_layer_two, lstm_2 = self.build_graph(x1, best_weights, units, 1)

                elif self.trial == 3:
                    if self.probabilistic:
                        y_pred_, weights, lstm_1, full_layer_two, lstm_2, full_layer_three, lstm_3, prob_layer = self.build_graph(
                            x1, best_weights, units, 1)
                    else:
                        y_pred_, weights, lstm_1, full_layer_two, lstm_2, full_layer_three, lstm_3 = self.build_graph(
                            x1, best_weights, units, 1)

        config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config_tf.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=graph_lstm, config=config_tf) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            lstm_1.set_weights(
                [best_weights['build_lstm/lstm1/kernel:0'], best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                 best_weights['build_lstm/lstm1/bias:0']])
            if self.trial == 0:
                if self.probabilistic:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    prob_layer.set_weights(
                        [best_weights['build_lstm/dense_prob/kernel:0'], best_weights['build_lstm/dense_prob/bias:0']])
                else:
                    lstm_1.set_weights(
                        [best_weights['build_lstm/lstm1/kernel:0'], best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                         best_weights['build_lstm/lstm1/bias:0']])

            elif self.trial == 1:
                if self.probabilistic:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_one.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])
                    prob_layer.set_weights(
                        [best_weights['build_lstm/dense_prob/kernel:0'],
                         best_weights['build_lstm/dense_prob/bias:0']])
                else:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_one.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])

            elif self.trial == 2:
                if self.probabilistic:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])
                    lstm_2.set_weights([best_weights['build_lstm/lstm2/kernel:0'],
                                        best_weights['build_lstm/lstm2/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm2/bias:0']])
                    prob_layer.set_weights(
                        [best_weights['build_lstm/dense_prob/kernel:0'],
                         best_weights['build_lstm/dense_prob/bias:0']])
                else:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])
                    lstm_2.set_weights([best_weights['build_lstm/lstm2/kernel:0'],
                                        best_weights['build_lstm/lstm2/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm2/bias:0']])
            elif self.trial == 3:
                if self.probabilistic:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])
                    lstm_2.set_weights([best_weights['build_lstm/lstm2/kernel:0'],
                                        best_weights['build_lstm/lstm2/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm2/bias:0']])
                    full_layer_three.set_weights(
                        [best_weights['build_lstm/dense2/kernel:0'], best_weights['build_lstm/dense2/bias:0']])
                    lstm_3.set_weights([best_weights['build_lstm/lstm3/kernel:0'],
                                        best_weights['build_lstm/lstm3/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm3/bias:0']])
                    prob_layer.set_weights(
                        [best_weights['build_lstm/dense_prob/kernel:0'],
                         best_weights['build_lstm/dense_prob/bias:0']])
                else:
                    lstm_1.set_weights([best_weights['build_lstm/lstm1/kernel:0'],
                                        best_weights['build_lstm/lstm1/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm1/bias:0']])
                    full_layer_two.set_weights(
                        [best_weights['build_lstm/dense1/kernel:0'], best_weights['build_lstm/dense1/bias:0']])
                    lstm_2.set_weights([best_weights['build_lstm/lstm2/kernel:0'],
                                        best_weights['build_lstm/lstm2/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm2/bias:0']])
                    full_layer_three.set_weights(
                        [best_weights['build_lstm/dense2/kernel:0'], best_weights['build_lstm/dense2/bias:0']])
                    lstm_3.set_weights([best_weights['build_lstm/lstm3/kernel:0'],
                                        best_weights['build_lstm/lstm3/recurrent_kernel:0'],
                                        best_weights['build_lstm/lstm3/bias:0']])

            y_pred, weights_run = sess.run([y_pred_, weights],
                                           feed_dict={x1: H})

            sess.close()

        return y_pred
