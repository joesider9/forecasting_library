import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os, joblib, re, sys

class MLP():
    def __init__(self, static_data,rated, X_train, y_train, X_val, y_val, X_test, y_test, trial = 0, probabilistc=False):
        self.static_data=static_data
        self.probabilistic = probabilistc
        self.trial =trial
        self.rated = rated
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


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

    def huber(self, y_true, y_pred, eps=0.001):
        error = y_true - y_pred
        cond = tf.abs(error) < eps

        squared_loss = tf.square(error) / (2 * eps)
        linear_loss = tf.abs(error) - 0.5 * eps

        return tf.where(cond, squared_loss, linear_loss)

    def build_graph(self, x1, y_pred_, learning_rate, units, hold_prob, act_func):
        if not self.rated is None:
            norm_val = tf.constant(1, tf.float32, name='rated')
        else:
            norm_val=y_pred_
        if act_func == 'elu':
            act = tf.nn.elu
        elif act_func == 'tanh':
            act_func = tf.nn.sigmoid
        with tf.name_scope("build_mlp") as scope:
            if self.trial == 0:
                mlp_1 = tf.keras.layers.Dense(
                    units[0],
                    name='dense1',
                    activation=act_func)
                full_out_dropout =tf.nn.dropout(mlp_1(x1), rate=1-hold_prob)

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
                losses = []
                for i, q in enumerate(quantiles):
                    # Get output layers
                    output = tf.layers.dense(full_out_dropout, 1,
                                             name="{}_q{}".format(i, int(q * 100)))
                    outputs.append(output)

                    # Create losses
                    error = self.huber(y_pred_, output)
                    loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error),
                                          axis=-1)

                    losses.append(loss)
                if self.trial == 0:
                    weights = mlp_1.trainable_weights
                elif self.trial == 1:
                    weights = mlp_1.trainable_weights + full_layer_one.trainable_weights
                elif self.trial == 2:
                    weights = mlp_1.trainable_weights + full_layer_two.trainable_weights + full_layer_three.trainable_weights

                for output in  outputs:
                    weights += output.trainable_weights

            else:
                y_pred, W, b = self.normal_full_layer(full_out_dropout, 1)

                if self.trial == 0:
                    weights = mlp_1.trainable_weights
                elif self.trial == 1:
                    weights = mlp_1.trainable_weights + full_layer_one.trainable_weights
                elif self.trial == 2:
                    weights = mlp_1.trainable_weights + full_layer_two.trainable_weights + full_layer_three.trainable_weights

                weights += [W, b]


        with tf.name_scope("train_mlp") as scope:
            if self.probabilistic:
                cost_mlp = tf.reduce_mean(tf.add_n(losses))
                optimizer_mlp = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_mlp = optimizer_mlp.minimize(cost_mlp)
                accuracy_mlp = tf.reduce_mean(tf.add_n(losses))
                sse_mlp = tf.reduce_sum(tf.add_n(losses))
                rse_mlp = tf.sqrt(tf.reduce_mean(tf.square(tf.add_n(losses))))
            else:
                err = tf.divide(tf.abs(y_pred - y_pred_), norm_val)
                cost_mlp = tf.reduce_mean(tf.square(err))
                optimizer_mlp = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_mlp = optimizer_mlp.minimize(cost_mlp)
                accuracy_mlp =tf.reduce_mean(err)
                sse_mlp = tf.reduce_sum(tf.square(err))
                rse_mlp = tf.sqrt(tf.reduce_mean(tf.square(err)))

        return train_mlp, cost_mlp, accuracy_mlp, sse_mlp, rse_mlp, weights

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

    def train(self, max_iterations=10000, learning_rate=5e-4, units=32, hold_prob=1, act_func='elu'):
        gpu_device = [gpu for gpu in tf.config.experimental_list_devices() if 'GPU' in gpu][0]
        gpu_id = gpu_device.split()[-1]
        gpu_num = re.findall('\d', gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num[-1]
        print('mlp strart')
        H_train = self.X_train
        H_val = self.X_val
        H_test = self.X_test
        y_val = self.y_val
        y_test = self.y_test
        y_train=self.y_train

        self.N = H_train.shape[1]
        batch_size = np.min([100, int(self.N / 5)])
        tf.compat.v1.reset_default_graph()
        graph_mlp = tf.Graph()
        with graph_mlp.as_default():
            with tf.device(gpu_id):
                x1 = tf.compat.v1.placeholder('float', shape=[None, H_train.shape[1], H_train.shape[2]], name='input_data')
                y_pred_ =tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1]], name='target_mlp')

            with tf.device(gpu_id):
                train_mlp, cost_mlp, accuracy_mlp, sse_mlp, rse_mlp, weights= self.build_graph(x1, y_pred_, learning_rate, units, hold_prob, act_func)

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
        patience = 10000
        wait = 0
        loops = 0

        with tf.compat.v1.Session(graph=graph_mlp, config=config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            while train_flag:
                for i in tqdm(range(max_iterations)):
                    if i % 500 == 0:

                        sess.run([train_mlp],
                                 feed_dict={x1: H_train[batches[i]], y_pred_: y_train[batches[i]]})

                        acc_new_v, mse_new_v, sse_new_v, rse_new_v, weights_mlp = sess.run([accuracy_mlp, cost_mlp,
                                                                                                              sse_mlp, rse_mlp, weights],
                                                                                                              feed_dict={x1: H_val, y_pred_: y_val,
                                                                                                                         })
                        acc_new_t, mse_new_t, sse_new_t, rse_new_t= sess.run([accuracy_mlp, cost_mlp, sse_mlp, rse_mlp],
                                                    feed_dict={x1: H_test, y_pred_: y_test})

                        acc_new = 0.4*acc_new_v + 0.6*acc_new_t
                        mse_new = 0.4*mse_new_v + 0.6*mse_new_t
                        sse_new = 0.4*sse_new_v + 0.6*sse_new_t
                        rse_new = 0.4*rse_new_v + 0.6*rse_new_t

                        obj_new = np.array([acc_new, mse_new, sse_new, rse_new])
                        flag, obj_old, obj_max, obj_min= self.distance(obj_new,obj_old,obj_max,obj_min)
                        if flag:
                            variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
                            for k, v in zip(variables_names, weights_mlp):
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
                        sess.run(train_mlp,
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
        model_dict['best_act_func'] = act_func
        model_dict['static_data'] = self.static_data
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy mlp-3d: %s" % obj_old[0])

        return obj_old[0], model_dict