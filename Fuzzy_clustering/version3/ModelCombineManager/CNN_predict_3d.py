import tensorflow as tf
import numpy as np


class CNN_3d_predict():
    def __init__(self, model, scale_cnn, trial, pool_size):
        self.scale_cnn = scale_cnn
        self.trial = trial
        self.model = model
        self.pool_size = pool_size
    def create_inputs(self, X_train):
        self.N, self.D1, self.D2, self.depth = X_train.shape
        H = []

        for i in range(self.depth):
            H.append(X_train[:, :, :, i])
            H[i] = np.array(H[i])
            H[i] = H[i].reshape(-1, self.D1 * self.D2)
            sc = self.scale_cnn[i]
            H[i] = sc.transform(H[i].reshape(-1, self.D1 * self.D2))
            H[i] = H[i].reshape(-1, self.D1, self.D2)

        H = np.transpose(np.stack(H), [1, 2, 3, 0])

        return H

    def init_weights(self, init_w):
        init_random_dist = tf.convert_to_tensor(init_w)
        return tf.Variable(init_random_dist)

    def init_bias(self, init_b):
        init_bias_vals = tf.convert_to_tensor(init_b)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self,input_layer, init_w, init_b):

        W = self.init_weights(init_w)
        b = self.init_bias(init_b)
        return tf.add(tf.matmul(input_layer, W), b, name='prediction'), W, b

    def build_graph(self, x1, best_weights, kernels, h_size, hold_prob, filters):

        with tf.name_scope("build_cnn") as scope:

            if self.trial == 0:
                convo_1 = tf.keras.layers.Conv2D(filters=int(filters),
                                                 kernel_size=kernels,
                                                 padding="same",
                                                 name='cnn1',
                                                 activation=tf.nn.elu)

                convo_1_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=1,
                                                                name='pool1')
                cnn_output = convo_1_pool(convo_1(x1))
                full_one_dropout = tf.nn.dropout(cnn_output, rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                s = shape[1] * shape[2] * shape[3]
                convo_2_flat = tf.reshape(full_one_dropout, [-1, s])

            elif self.trial == 1:
                convo_1 = tf.keras.layers.Conv3D(filters=int(filters),
                                                 kernel_size=kernels,
                                                 padding="same",
                                                 name='cnn1',
                                                 activation=tf.nn.elu)

                convo_1_pool = tf.keras.layers.AveragePooling3D(pool_size=self.pool_size, strides=1,
                                                                name='pool1')
                cnn_output = convo_1_pool(convo_1(tf.expand_dims(x1, axis=4)))
                full_one_dropout = tf.nn.dropout(cnn_output, rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                s = shape[1] * shape[2] * shape[3] * shape[4]
                convo_2_flat = tf.reshape(full_one_dropout, [-1, s])

            elif self.trial == 2:
                convo_1 = tf.keras.layers.Conv3D(filters=int(filters),
                                                 kernel_size=kernels,
                                                 padding="same",
                                                 name='cnn1',
                                                 activation=tf.nn.elu)

                convo_1_pool = tf.keras.layers.AveragePooling3D(pool_size=self.pool_size, strides=1,
                                                                name='pool1')
                cnn_1 = convo_1_pool(convo_1(tf.expand_dims(x1, axis=4)))
                full_one_dropout = tf.nn.dropout(cnn_1, rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                convo_1_flat = tf.reshape(full_one_dropout, [-1, shape[1], shape[2] * shape[3], shape[4]])

                convo_2 = tf.keras.layers.Conv2D(filters=int(filters),
                                                 kernel_size=kernels[:-1],
                                                 padding="same",
                                                 name='cnn2',
                                                 activation=tf.nn.elu)

                convo_2_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size[:-1], strides=1,
                                                                name='pool2')
                cnn_output = convo_2_pool(convo_2(convo_1_flat))
                full_two_dropout = tf.nn.dropout(cnn_output, rate=1 - hold_prob)
                shape = full_two_dropout.get_shape().as_list()
                s = shape[1] * shape[2] * shape[3]
                convo_2_flat = tf.reshape(full_two_dropout, [-1, s])
            elif self.trial == 3:
                convo_1 = tf.keras.layers.Conv3D(filters=int(filters),
                                                 kernel_size=kernels,
                                                 padding="same",
                                                 name='cnn1',
                                                 activation=tf.nn.elu)

                convo_1_pool = tf.keras.layers.AveragePooling3D(pool_size=self.pool_size, strides=1,
                                                                name='pool1')
                cnn_1 = convo_1_pool(convo_1(tf.expand_dims(x1, axis=4)))
                full_one_dropout = tf.nn.dropout(cnn_1, rate=1 - hold_prob)
                shape = full_one_dropout.get_shape().as_list()
                s = shape[1] * shape[2] * shape[3] * shape[4]
                convo_1_flat = tf.reshape(full_one_dropout, [-1, s], name='reshape1')

                full_layer_middle = tf.keras.layers.Dense(units=2000, activation=tf.nn.elu, name='dense_middle')
                full_middle_dropout = tf.nn.dropout(full_layer_middle(convo_1_flat), rate=1 - hold_prob)
                full_middle_dropout = tf.reshape(full_middle_dropout, [-1, 10, 20, 10], name='reshape2')

                convo_2 = tf.keras.layers.Conv2D(filters=int(filters),
                                                 kernel_size=kernels[:-1],
                                                 padding="same",
                                                 name='cnn2',
                                                 activation=tf.nn.elu)

                convo_2_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size[:-1], strides=1,
                                                                name='pool2')
                cnn_output = convo_2_pool(convo_2(full_middle_dropout))
                full_two_dropout = tf.nn.dropout(cnn_output, rate=1 - hold_prob)
                shape = full_two_dropout.get_shape().as_list()
                s = shape[1] * shape[2] * shape[3]
                convo_2_flat = tf.reshape(full_two_dropout, [-1, s])

            full_layer_one = tf.keras.layers.Dense(units=h_size[0],activation=tf.nn.elu, name='dense1')

            full_layer_two = tf.keras.layers.Dense(units=h_size[1], activation=tf.nn.elu, name='dense2')
            full_two_dropout = tf.nn.dropout(full_layer_one(convo_2_flat), keep_prob=hold_prob)
            dense_output = tf.nn.dropout(full_layer_two(full_two_dropout), keep_prob=hold_prob)

            y_pred, W, b = self.normal_full_layer(dense_output, best_weights['build_cnn/Variable:0'],best_weights['build_cnn/Variable_1:0'] )
            if self.trial == 1 or self.trial == 0:
                weights = convo_1.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
                return y_pred, weights, convo_1, full_layer_one, full_layer_two
            elif self.trial == 2:
                weights = convo_1.trainable_weights + convo_2.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
                return y_pred, weights, convo_1, convo_2, full_layer_one, full_layer_two
            else:
                weights = convo_1.trainable_weights + full_layer_middle.trainable_weights + convo_2.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
                return y_pred, weights, convo_1, convo_2, full_layer_middle, full_layer_one, full_layer_two



    def predict(self, X):

        filters = self.model['filters']
        kernels = self.model['kernels']
        h_size = self.model['h_size']
        best_weights = self.model['best_weights']
        H = self.create_inputs(X)

        tf.compat.v1.reset_default_graph()
        graph_cnn = tf.Graph()
        with graph_cnn.as_default():
            with tf.device("/cpu:0"):
                x1 = tf.compat.v1.placeholder('float', shape=[None, self.D1, self.D2, self.depth], name='input_data')
                hold_prob = tf.compat.v1.placeholder(tf.float32, name='drop')
            with tf.device("/cpu:0"):


                if self.trial == 1 or self.trial == 0:
                    y_pred_, weights, convo_1, full_layer_one, full_layer_two = self.build_graph(x1, best_weights, kernels, h_size, hold_prob,filters)
                elif self.trial == 2:
                    y_pred_, weights, convo_1, convo_2, full_layer_one, full_layer_two = self.build_graph(x1, best_weights, kernels, h_size, hold_prob, filters)
                else:
                    y_pred_, weights, convo_1, convo_2, full_layer_middle, full_layer_one, full_layer_two = self.build_graph(x1, best_weights, kernels, h_size, hold_prob, filters)



        config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config_tf.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=graph_cnn, config=config_tf) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            if self.trial == 1 or self.trial == 0:
                convo_1.set_weights([best_weights['build_cnn/cnn1/kernel:0'], best_weights['build_cnn/cnn1/bias:0']])
            elif self.trial == 2:
                convo_1.set_weights(
                    [best_weights['build_cnn/cnn1/kernel:0'], best_weights['build_cnn/cnn1/bias:0']])
                convo_2.set_weights(
                    [best_weights['build_cnn/cnn2/kernel:0'], best_weights['build_cnn/cnn2/bias:0']])
            else:
                convo_1.set_weights(
                    [best_weights['build_cnn/cnn1/kernel:0'], best_weights['build_cnn/cnn1/bias:0']])
                convo_2.set_weights(
                    [best_weights['build_cnn/cnn2/kernel:0'], best_weights['build_cnn/cnn2/bias:0']])
                full_layer_middle.set_weights(
                    [best_weights['build_cnn/dense_middle/kernel:0'], best_weights['build_cnn/dense_middle/bias:0']])

            full_layer_one.set_weights(
                [best_weights['build_cnn/dense1/kernel:0'], best_weights['build_cnn/dense1/bias:0']])
            full_layer_two.set_weights(
                [best_weights['build_cnn/dense2/kernel:0'], best_weights['build_cnn/dense2/bias:0']])

            y_pred, weights_run= sess.run([y_pred_, weights],
                             feed_dict={x1: H, hold_prob:1})


            sess.close()


        return y_pred
