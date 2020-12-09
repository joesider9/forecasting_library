import tensorflow as tf
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp2d
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import os, joblib, re, sys

class LSTM_3d():
    def __init__(self, static_data,rated, X_train, y_train, X_val, y_val, X_test, y_test, trial = 0, probabilistc=False):
        self.static_data=static_data
        self.rated = static_data['rated']
        self.probabilistic = probabilistc
        self.trial =trial
        self.rated = rated
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def create_inputs(self, X_train, X_val, X_test):
        self.N, self.D1, self.D2 = X_train.shape

        H_train = X_train
        H_val = X_val
        H_test = X_test
        H = np.vstack((H_train, H_val, H_test))
        H = H.reshape(-1, self.D1 * self.D2)
        sc = MinMaxScaler()
        sc.fit(H)
        self.scale_lstm = sc
        H_train = sc.transform(H_train.reshape(-1, self.D1 * self.D2))
        H_train = H_train.reshape(-1, self.D1, self.D2)
        H_val = sc.transform(H_val.reshape(-1, self.D1 * self.D2))
        H_val = H_val.reshape(-1, self.D1, self.D2)
        H_test = sc.transform(H_test.reshape(-1, self.D1 * self.D2))
        H_test = H_test.reshape(-1, self.D1, self.D2)

        return H_train ,H_val, H_test

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

    def build_graph(self, x1, y_pred_, learning_rate, units, hold_prob):
        if not self.rated is None:
            norm_val = tf.constant(1, tf.float32, name='rated')
        else:
            norm_val=y_pred_
        with tf.name_scope("build_lstm") as scope:
            if self.trial == 0:
                lstm_1 = tf.keras.layers.LSTM(
                    units[0],
                    name='lstm1',
                    return_sequences=True,
                    activation=tf.nn.elu)
                full_out_dropout =tf.nn.dropout(lstm_1(x1), rate=1-hold_prob)
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
                full_layer_one = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu, name='dense1')
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
                full_layer_two = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu, name='dense1')
                full_two_dropout = tf.nn.dropout(full_layer_two(lstm_2_flat), rate=1 - hold_prob)
                full_two_dropout = tf.reshape(full_two_dropout,[-1, shape[1],  shape[2]])

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

                full_layer_two = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu, name='dense1')
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
                full_layer_three = tf.keras.layers.Dense(units=shape[1] * shape[2], activation=tf.nn.elu, name='dense2')
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
                prob_layer = tf.keras.layers.Dense(y_pred_.shape[1], activation=tf.nn.softmax, name='dense_prob')
                y_pred = prob_layer(full_out_dropout)
            else:
                y_pred, W, b = self.normal_full_layer(full_out_dropout, 1)

            if self.trial == 0:
                weights = lstm_1.trainable_weights
            elif self.trial == 1:
                weights = lstm_1.trainable_weights + full_layer_one.trainable_weights
            elif self.trial == 2:
                weights = lstm_1.trainable_weights + full_layer_two.trainable_weights + lstm_2.trainable_weights
            elif self.trial == 3:
                weights = lstm_1.trainable_weights + full_layer_two.trainable_weights + lstm_2.trainable_weights + full_layer_three.trainable_weights +lstm_3.trainable_weights

            if self.probabilistic:
                weights += prob_layer.trainable_weights
            else:
                weights += [W, b]


        with tf.name_scope("train_lstm") as scope:
            if self.probabilistic:
                cost_lstm = tf.losses.softmax_cross_entropy(y_pred_, y_pred)
                optimizer_lstm = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_lstm = optimizer_lstm.minimize(cost_lstm)
                accuracy_lstm = 1/tf.metrics.accuracy(y_pred - y_pred_)
                sse_lstm = 1/tf.metrics.recall(y_pred - y_pred_)
                rse_lstm = 1/tf.metrics.precision(y_pred - y_pred_)
            else:
                err = tf.divide(tf.abs(y_pred - y_pred_), norm_val)
                cost_lstm = tf.reduce_mean(tf.square(err))
                optimizer_lstm = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_lstm = optimizer_lstm.minimize(cost_lstm)
                accuracy_lstm = tf.reduce_mean(err)
                sse_lstm = tf.reduce_sum(tf.square(err))
                rse_lstm = tf.sqrt(tf.reduce_mean(tf.square(err)))

        return train_lstm, cost_lstm, accuracy_lstm, sse_lstm, rse_lstm, weights

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
            obj_old=obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def train(self, max_iterations=10000, learning_rate=5e-4, units=32, hold_prob=1):
        gpu_device = [gpu for gpu in tf.config.experimental_list_devices() if 'GPU' in gpu][0]
        gpu_id = gpu_device.split()[-1]
        gpu_num = re.findall('\d', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num[-1]
        print('lstm strart')
        H_train, H_val, H_test = self.create_inputs(self.X_train, self.X_val, self.X_test)
        if not self.probabilistic:
            y_val = self.y_val
            y_test = self.y_test
            y_train=self.y_train
        else:
            classes = np.arange(0.1, 20, 0.2)

            y_val = np.digitize(self.y_val, classes, right=True)
            y_test = np.digitize(self.y_test, classes, right=True)
            y_train = np.digitize(self.y_train, classes, right=True)
            binarizer = MultiLabelBinarizer(classes=classes)
            y_train = binarizer.fit_transform(y_train)
            y_val = binarizer.transform(y_val)
            y_test = binarizer.transform(y_test)

        batch_size = np.min([100, int(self.N / 5)])
        tf.compat.v1.reset_default_graph()
        graph_lstm = tf.Graph()
        with graph_lstm.as_default():
            with tf.device(gpu_id):
                x1 = tf.compat.v1.placeholder('float', shape=[None, H_train.shape[1], H_train.shape[2]], name='input_data')
                y_pred_ =tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]], name='target_lstm')

            with tf.device(gpu_id):
                train_lstm, cost_lstm, accuracy_lstm, sse_lstm, rse_lstm, weights= self.build_graph(x1, y_pred_, learning_rate, units, hold_prob)

        obj_old = np.inf*np.ones(4)
        obj_max = np.inf*np.ones(4)
        obj_min = np.inf*np.ones(4)
        batches = [np.random.choice(self.N, batch_size, replace=False) for _ in range(max_iterations+1)]

        path_group = self.static_data['path_group']
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        if sys.platform != 'linux':
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=self.static_data['intra_op'],
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
        patience = 8000
        wait = 0
        loops = 0

        with tf.compat.v1.Session(graph=graph_lstm, config=config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            while train_flag:
                for i in tqdm(range(max_iterations)):
                    if i % 500 == 0:

                        sess.run([train_lstm],
                                 feed_dict={x1: H_train[batches[i]], y_pred_: y_train[batches[i]]})

                        acc_new_v, mse_new_v, sse_new_v, rse_new_v, weights_lstm = sess.run([accuracy_lstm, cost_lstm,
                                                                                                              sse_lstm, rse_lstm, weights],
                                                                                                              feed_dict={x1: H_val, y_pred_: y_val,
                                                                                                                         })
                        acc_new_t, mse_new_t, sse_new_t, rse_new_t= sess.run([accuracy_lstm, cost_lstm, sse_lstm, rse_lstm],
                                                    feed_dict={x1: H_test, y_pred_: y_test})

                        acc_new = 0.4*acc_new_v + 0.6*acc_new_t
                        mse_new = 0.4*mse_new_v + 0.6*mse_new_t
                        sse_new = 0.4*sse_new_v + 0.6*sse_new_t
                        rse_new = 0.4*rse_new_v + 0.6*rse_new_t

                        obj_new = np.array([acc_new, mse_new, sse_new, rse_new])
                        flag, obj_old, obj_max, obj_min= self.distance(obj_new,obj_old,obj_max,obj_min)
                        if flag:
                            variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
                            for k, v in zip(variables_names, weights_lstm):
                                self.best_weights[k]=v
                            res[str(i)]= obj_old
                            print(acc_new)

                            best_iteration = i
                            wait = 0
                        else:
                            wait += 1
                        if wait > patience:
                            train_flag = False
                            break
                    else:
                        sess.run(train_lstm,
                                 feed_dict={x1: H_train[batches[i]], y_pred_: y_train[batches[i]]})
                        wait += 1
                best_glob_iterations = ext_iterations + best_iteration
                if (max_iterations - best_iteration) <= 5000 and max_iterations>2000:
                    if loops > 3:
                        best_glob_iterations = ext_iterations + best_iteration
                        train_flag = False
                    else:
                        ext_iterations += 8000
                        max_iterations = 8000
                        best_iteration = 0
                        loops += 1
                else:
                    best_glob_iterations = ext_iterations + best_iteration
                    train_flag = False

            sess.close()


        model_dict = dict()
        model_dict['units'] = units
        model_dict['hold_prob'] = hold_prob
        model_dict['best_weights'] = self.best_weights
        model_dict['static_data'] = self.static_data
        model_dict['n_vars'] = self.D1
        model_dict['depth'] = self.D2
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy lstm-3d: %s" % obj_old[0])

        return obj_old[0], self.scale_lstm, model_dict