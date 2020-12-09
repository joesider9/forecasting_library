import joblib
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.interpolate import interp2d
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class CNN():
    def __init__(self, static_data, rated, models, X_train, y_train, X_val, y_val, X_test, y_test, pool_size, trial=0):
        self.static_data = static_data
        self.trial = trial
        self.pool_size = pool_size
        self.rated = rated
        self.rbf_models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def rbf_map(self, X, num_centr, centroids, radius):
        hmap_list = []
        s = X.shape
        d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [num_centr, 1, 1]), [1, 0, 2]) - np.tile(
            np.expand_dims(centroids, axis=0), [s[0], 1, 1])
        d = np.sqrt(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])), 2))
        phi = np.exp((-1) * np.power(d, 2))

        return np.transpose(phi, [1, 0, 2])

    def rescale(self, arr, nrows, ncol):
        W, H = arr.shape
        new_W, new_H = (nrows, ncol)
        xrange = lambda x: np.linspace(0, 1, x)

        f = interp2d(xrange(H), xrange(W), arr, kind="linear")
        new_arr = f(xrange(new_H), xrange(new_W))

        return new_arr

    def create_inputs(self, X_train, X_val, X_test):
        self.N, self.D = X_train.shape

        H_train = []
        H_val = []
        H_test = []
        self.scale_cnn = []
        self.depth = len(self.rbf_models)

        self.num_centr = 0

        for i in range(self.depth):
            if self.rbf_models[i]['centroids'].shape[0] > self.num_centr:
                self.num_centr = self.rbf_models[i]['centroids'].shape[0]
        if self.num_centr > 48:
            self.num_centr = 48

        for i in range(self.depth):
            if len(self.rbf_models[i]['Radius'].shape) == 1:
                self.rbf_models[i]['Radius'] = np.tile(self.rbf_models[i]['Radius'].reshape(1, -1), [self.num_centr, 1])

            if self.rbf_models[i]['centroids'].shape[0] != self.num_centr:
                centroids = self.rescale(self.rbf_models[i]['centroids'], self.num_centr, self.D)
            else:
                centroids = self.rbf_models[i]['centroids']
            if np.isscalar(self.rbf_models[i]['Radius']):
                Radius = self.rbf_models[i]['Radius']
            elif self.rbf_models[i]['Radius'].shape[0] == self.num_centr:
                Radius = self.rbf_models[i]['Radius']
            elif self.rbf_models[i]['Radius'].shape[0] < self.num_centr:
                Radius = self.rescale(self.rbf_models[i]['Radius'], self.num_centr, self.D)
            else:
                raise ValueError('Unkown shape')
            H_train.append(
                np.nan_to_num(np.transpose(self.rbf_map(X_train, self.num_centr, centroids, Radius), [1, 2, 0])))
            H_val.append(np.nan_to_num(np.transpose(self.rbf_map(X_val, self.num_centr, centroids, Radius), [1, 2, 0])))
            H_test.append(
                np.nan_to_num(np.transpose(self.rbf_map(X_test, self.num_centr, centroids, Radius), [1, 2, 0])))
            H = np.vstack((H_train[i], H_val[i], H_test[i]))
            H = H.reshape(-1, self.D * self.num_centr)
            sc = MinMaxScaler()
            sc.fit(H)
            self.scale_cnn.append(sc)
            H_train[i] = sc.transform(H_train[i].reshape(-1, self.D * self.num_centr))
            H_train[i] = np.nan_to_num(H_train[i].reshape(-1, self.D, self.num_centr))
            H_val[i] = sc.transform(H_val[i].reshape(-1, self.D * self.num_centr))
            H_val[i] = np.nan_to_num(H_val[i].reshape(-1, self.D, self.num_centr))
            H_test[i] = sc.transform(H_test[i].reshape(-1, self.D * self.num_centr))
            H_test[i] = np.nan_to_num(H_test[i].reshape(-1, self.D, self.num_centr))

        H_train = np.transpose(np.stack(H_train), [1, 2, 3, 0])
        H_val = np.transpose(np.stack(H_val), [1, 2, 3, 0])
        H_test = np.transpose(np.stack(H_test), [1, 2, 3, 0])

        return H_train, H_val, H_test

    def init_weights(self, shape):
        init_random_dist = tf.random.truncated_normal(shape, stddev=0.001)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.001, shape=shape)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.add(tf.matmul(input_layer, W), b, name='prediction'), W, b

    def build_graph(self, x1, y_pred_, learning_rate, kernels, h_size, hold_prob, filters):
        if not self.rated is None:
            norm_val = tf.constant(1, tf.float32, name='rated')
        else:
            norm_val = y_pred_
        with tf.name_scope("build_rbf_cnn") as scope:

            if self.trial == 0:
                convo_1 = tf.keras.layers.Conv2D(filters=int(filters),
                                                 kernel_size=kernels,
                                                 padding="same",
                                                 name='cnn1',
                                                 activation=tf.nn.elu)

                convo_1_pool = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=1, name='pool1')
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

            full_layer_one = tf.keras.layers.Dense(units=h_size[0], activation=tf.nn.elu, name='dense1')
            full_layer_two = tf.keras.layers.Dense(units=h_size[1], activation=tf.nn.elu,
                                                   name='dense2')
            full_two_dropout = tf.nn.dropout(full_layer_one(convo_2_flat), rate=1 - hold_prob)
            dense_output = tf.nn.dropout(full_layer_two(full_two_dropout), rate=1 - hold_prob)
            y_pred, W, b = self.normal_full_layer(dense_output, 1)

            if self.trial == 1 or self.trial == 0:
                weights = convo_1.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
            elif self.trial == 2:
                weights = convo_1.trainable_weights + convo_2.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
            else:
                weights = convo_1.trainable_weights + full_layer_middle.trainable_weights + convo_2.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [
                    W, b]
        with tf.name_scope("train_rbf_cnn") as scope:
            err = tf.divide(tf.abs(y_pred - y_pred_), norm_val)
            cost_cnn = tf.reduce_mean(tf.square(err))
            optimizer_cnn = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_cnn = optimizer_cnn.minimize(cost_cnn)
            accuracy_cnn = tf.reduce_mean(err)
            sse_cnn = tf.reduce_sum(tf.square(err))
            rse_cnn = tf.sqrt(tf.reduce_mean(tf.square(err)))

        return train_cnn, cost_cnn, accuracy_cnn, sse_cnn, rse_cnn, weights

    def distance(self, obj_new, obj_old, obj_max, obj_min):
        if np.any(np.isinf(obj_old)):
            obj_old = obj_new.copy()
            obj_max = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
            obj_min = obj_new.copy()
        d = 0
        for i in range(obj_new.shape[0]):
            if obj_max[i] < obj_new[i]:
                obj_max[i] = obj_new[i]
            if obj_min[i] > obj_new[i]:
                obj_min[i] = obj_new[i]

            d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        if d < -0.0001:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def train_cnn(self, max_iterations=10000, learning_rate=5e-4, kernels=[2, 8], h_size=[248, 52], gpu_id='/cpu:0',
                  filters=24):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(self.static_data['ngpus'])])
        if learning_rate > 0.9:
            learning_rate = 5e-5
        H_train, H_val, H_test = self.create_inputs(self.X_train, self.X_val, self.X_test)
        y_val = self.y_val
        y_test = self.y_test
        y_train = self.y_train

        batch_size = np.min([100, int(self.N / 5)])
        tf.compat.v1.reset_default_graph()
        graph_cnn = tf.Graph()
        with graph_cnn.as_default():
            with tf.device("/cpu:0"):
                x1 = tf.compat.v1.placeholder('float', shape=[None, self.D, self.num_centr, self.depth],
                                              name='input_data')
                y_pred_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='target_cnn')
                hold_prob = tf.compat.v1.placeholder(tf.float32, name='drop')
            with tf.device(gpu_id):
                train_cnn, cost_cnn, accuracy_cnn, sse_cnn, rse_cnn, weights = self.build_graph(x1, y_pred_,
                                                                                                learning_rate, kernels,
                                                                                                h_size, hold_prob,
                                                                                                filters)

        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        batches = [np.random.choice(self.N, batch_size, replace=False) for _ in range(max_iterations + 1)]
        partitions = 100
        H_val_list = []
        y_val_list = []
        for i in range(0, H_val.shape[0], partitions):
            if (i + partitions + 1) > H_val.shape[0]:
                H_val_list.append(H_val[i:])
                y_val_list.append(y_val[i:])
            else:
                H_val_list.append(H_val[i:i + partitions])
                y_val_list.append(y_val[i:i + partitions])
        H_test_list = []
        y_test_list = []
        for i in range(0, H_test.shape[0], partitions):
            if (i + partitions + 1) > H_test.shape[0]:
                H_test_list.append(H_test[i:])
                y_test_list.append(y_test[i:])
            else:
                H_test_list.append(H_test[i:i + partitions])
                y_test_list.append(y_test[i:i + partitions])

        path_group = self.static_data['path_group']
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        if sys.platform != 'linux':
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                              intra_op_parallelism_threads=self.static_data['intra_op'],
                                              inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
        else:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        res = dict()
        self.best_weights = dict()
        best_iteration = 0
        best_glob_iterations = 0
        ext_iterations = max_iterations
        train_flag = True
        patience = 5000
        wait = 0
        loops = 0

        with tf.compat.v1.Session(graph=graph_cnn, config=config) as sess:
            print('Start session for whole rbf-cnn %s' % self.num_centr)

            sess.run(tf.compat.v1.global_variables_initializer())
            while train_flag:
                for i in tqdm(range(max_iterations)):
                    if i % 500 == 0:

                        sess.run([train_cnn],
                                 feed_dict={x1: H_train[batches[i]], y_pred_: y_train[batches[i]], hold_prob: 1})
                        for hval, yval in zip(H_val_list, y_val_list):
                            acc_new_v, mse_new_v, sse_new_v, rse_new_v, weights_cnn = sess.run(
                                [accuracy_cnn, cost_cnn, sse_cnn, rse_cnn, weights],
                                feed_dict={x1: hval, y_pred_: yval, hold_prob: 1})
                        for htest, ytest in zip(H_test_list, y_test_list):
                            acc_new_t, mse_new_t, sse_new_t, rse_new_t = sess.run(
                                [accuracy_cnn, cost_cnn, sse_cnn, rse_cnn],
                                feed_dict={x1: htest, y_pred_: ytest, hold_prob: 1})

                        acc_new = 0.4 * acc_new_v + 0.6 * acc_new_t
                        mse_new = 0.4 * mse_new_v + 0.6 * mse_new_t
                        sse_new = 0.4 * sse_new_v + 0.6 * sse_new_t
                        rse_new = 0.4 * rse_new_v + 0.6 * rse_new_t

                        obj_new = np.array([acc_new, mse_new, sse_new, rse_new])
                        flag, obj_old, obj_max, obj_min = self.distance(obj_new, obj_old, obj_max, obj_min)
                        if flag:
                            variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
                            for k, v in zip(variables_names, weights_cnn):
                                self.best_weights[k] = v
                            res[str(i)] = obj_old
                            print(acc_new)
                            best_iteration = i
                            wait = 0
                        else:
                            wait += 1
                        if wait > patience:
                            train_flag = False
                            break

                    else:
                        sess.run(train_cnn,
                                 feed_dict={x1: H_train[batches[i]], y_pred_: y_train[batches[i]], hold_prob: 1})
                        wait += 1
                best_glob_iterations = ext_iterations + best_iteration
                if (max_iterations - best_iteration) <= 5000:
                    if loops > 3:
                        best_glob_iterations = ext_iterations + best_iteration
                        train_flag = False
                    else:
                        ext_iterations += 6000
                        max_iterations = 6000
                        best_iteration = 0
                        loops += 1
                else:
                    best_glob_iterations = ext_iterations + best_iteration
                    train_flag = False

            sess.close()

        model_dict = dict()
        model_dict['filters'] = filters
        model_dict['kernels'] = kernels
        model_dict['h_size'] = h_size
        model_dict['best_weights'] = self.best_weights

        model_dict['static_data'] = self.static_data
        model_dict['n_vars'] = self.D
        model_dict['num_centr'] = self.num_centr
        model_dict['depth'] = self.depth
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy cnn: %s" % obj_old[0])

        return obj_old[0], self.scale_cnn, model_dict
