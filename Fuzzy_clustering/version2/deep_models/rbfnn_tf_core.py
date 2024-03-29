import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm import tqdm

from Fuzzy_clustering.ver_tf2.clustering.algorithms import FCV


class RbfNN:

    def __init__(self, static_data, max_iterations=10000, learning_rate=5e-4, mean_var=0.05, std_var=0.05):
        self.static_data = static_data
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.mean_var = mean_var
        self.std_var = std_var
        self.rated = self.static_data['rated']

    def find_centers(self, X_train, num_centr):
        self.N, self.D = X_train.shape

        self.batch_size = self.N
        try:
            centers = FCV(X_train, n_clusters=num_centr, r=4).optimize()
            c = centers.C
        except:
            c = KMeans(n_clusters=num_centr, random_state=0).fit(X_train)
            c = c.cluster_centers_

        centroids = c.astype(np.float32)
        return centroids

    def build_train_graph(self, x, y_, centroids, num_centr, lr):

        with tf.name_scope("Hidden_layer") as scope:
            cnt = pd.DataFrame(centroids, index=['c' + str(i) for i in range(centroids.shape[0])],
                               columns=['v' + str(i) for i in range(centroids.shape[1])])
            var_init = pd.DataFrame(columns=['v' + str(i) for i in range(centroids.shape[1])])
            for r in cnt.index:
                v = (cnt.loc[r] - cnt.drop(r)).min()
                v[v == 0] = 0.0001
                v.name = r
                var_init = var_init.append(v)
            var_init = tf.convert_to_tensor(var_init.values, dtype=tf.float32, name='var_init')
            var = tf.Variable(var_init,
                              dtype=tf.float32, name='RBF_variance')

            centroids = tf.convert_to_tensor(centroids, dtype=tf.float32, name='centroids')

            s = tf.shape(x)

            d1 = tf.transpose(tf.tile(tf.expand_dims(x, 0), [num_centr, 1, 1]), perm=[1, 0, 2]) - tf.tile(
                tf.expand_dims(centroids, 0), [s[0], 1, 1])
            d = tf.sqrt(
                tf.reduce_sum(tf.pow(tf.multiply(d1, tf.tile(tf.expand_dims(var, 0), [s[0], 1, 1])), 2), axis=2))

            phi = tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d)))

        with tf.name_scope("Output_layer") as scope:
            w = tf.cast(tf.linalg.lstsq(tf.cast(phi, tf.float64), tf.cast(y_, tf.float64),
                                        l2_regularizer=tf.cast(0.0005, tf.float64)), tf.float32)

            h = tf.matmul(phi, w)

        with tf.name_scope("Softmax") as scope:
            if not self.rated is None:
                cost = tf.reduce_mean(tf.square(h - y_))
            else:
                err = tf.divide(tf.abs(h - y_), y_)
                cost = tf.reduce_mean(tf.square(err))

        with tf.name_scope("train") as scope:
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            train_step = optimizer.minimize(cost)

        return train_step, w, centroids, var, cost

    def build_eval_graph(self, x_val, y_val_, x_test, y_test_, num_centr, centroids, w, var):

        with tf.name_scope("Evaluating") as scope:
            s_val = tf.shape(x_val)
            d1_val = tf.transpose(tf.tile(tf.expand_dims(x_val, 0), [num_centr, 1, 1]), perm=[1, 0, 2]) - tf.tile(
                tf.expand_dims(centroids, 0), [s_val[0], 1, 1])
            d_val = tf.sqrt(
                tf.reduce_sum(tf.pow(tf.multiply(d1_val, tf.tile(tf.expand_dims(var, 0), [s_val[0], 1, 1])), 2),
                              axis=2))
            phi_val = tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d_val)))
            h_val = tf.matmul(phi_val, w, name='pred')
            if not self.rated is None:
                err_val = tf.abs(tf.subtract(h_val, y_val_))
            else:
                err_val = tf.divide(tf.abs(tf.subtract(h_val, y_val_)), y_val_)
            accuracyv = tf.reduce_mean(err_val, name='accuracy')
            ssev = tf.reduce_sum(tf.square(err_val), name='ssev')
            msev = tf.reduce_mean(tf.square(err_val), name='msev')
            rsev = tf.sqrt(tf.reduce_mean(tf.square(err_val)), name='rsev')

        with tf.name_scope("Testing") as scope:
            s_test = tf.shape(x_test)
            d1_test = tf.transpose(tf.tile(tf.expand_dims(x_test, 0), [num_centr, 1, 1]), perm=[1, 0, 2]) - tf.tile(
                tf.expand_dims(centroids, 0), [s_test[0], 1, 1])
            d_test = tf.sqrt(
                tf.reduce_sum(tf.pow(tf.multiply(d1_test, tf.tile(tf.expand_dims(var, 0), [s_test[0], 1, 1])), 2),
                              axis=2))
            phi_test = tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d_test)))
            h_test = tf.matmul(phi_test, w, name='pred')
            if not self.rated is None:
                err_test = tf.abs(tf.subtract(h_test, y_test_))
            else:
                err_test = tf.divide(tf.abs(tf.subtract(h_test, y_test_)), y_test_)

            accuracyt = tf.reduce_mean(err_test, name='accuracy')
            sset = tf.reduce_sum(tf.square(err_test), name='sset')
            mset = tf.reduce_mean(tf.square(err_test), name='mset')
            rset = tf.sqrt(tf.reduce_mean(tf.square(err_test)), name='rset')
            accuracy = 0.6 * accuracyt + 0.4 * accuracyv + 2 * tf.abs(accuracyt - accuracyv)
            sse = 0.6 * sset + 0.4 * ssev
            mse = 0.6 * mset + 0.4 * msev
            rse = 0.6 * rsev + 0.4 * rset
        return accuracy, sse, mse, rse

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

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, num_centr, lr, gpu_id='/cpu:0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(self.static_data['ngpus'])])
        centroids = self.find_centers(X_train, num_centr)
        centers = centroids
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                # Placeholders for data
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.D], name='input_data')
                y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='target')
                x_val = tf.compat.v1.placeholder(tf.float32, shape=[None, self.D], name='val_data')
                y_val_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='val_target')
                x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, self.D], name='test_data')
                y_test_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='test_target')
            with tf.device(gpu_id):
                train_step, w, centroids, var, cost = self.build_train_graph(x, y_, centroids, num_centr, lr)
                accuracy, sse, mse, rse = self.build_eval_graph(x_val, y_val_, x_test, y_test_, num_centr, centroids, w,
                                                                var)

        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)

        batches = [np.random.choice(self.N, self.batch_size, replace=False) for _ in range(self.max_iterations + 1)]

        path_group = self.static_data['path_group']
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        if cpu_status != 0:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
        else:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

        lr = self.learning_rate
        res = dict()
        self.best_weights = None
        best_iteration = 0
        best_glob_iterations = 0
        max_iterations = self.max_iterations
        ext_iterations = self.max_iterations
        train_flag = True
        patience = 10000
        wait = 0
        loops = 0

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            print('Start session for %s' % num_centr)
            sess.run(tf.compat.v1.global_variables_initializer())
            while train_flag:
                for i in tqdm(range(max_iterations)):
                    if i % 100 == 0:

                        sess.run([train_step],
                                 feed_dict={x: X_train[batches[i]], y_: y_train[batches[i]], x_val: X_val,
                                            y_val_: y_val, x_test: X_test, y_test_: y_test})
                        radius, wi, acc_new, sse_new, mse_new, rse_new = sess.run([var, w, accuracy, sse, mse, rse],
                                                                                  feed_dict={x: X_train, y_: y_train,
                                                                                             x_val: X_val,
                                                                                             y_val_: y_val,
                                                                                             x_test: X_test,
                                                                                             y_test_: y_test})

                        obj_new = np.array([acc_new, sse_new, mse_new, rse_new])
                        flag, obj_old, obj_max, obj_min = self.distance(obj_new, obj_old, obj_max, obj_min)
                        if flag:
                            Radius = radius
                            W = wi
                            res[str(i)] = obj_old
                            print("RBFNN accuracy : %s" % str(obj_old[0]))
                            best_iteration = i
                            wait = 0
                        else:
                            wait += 1
                        if wait > patience:
                            train_flag = False
                            break

                    else:
                        sess.run(train_step, feed_dict={x: X_train[batches[i]], y_: y_train[batches[i]], x_val: X_val,
                                                        y_val_: y_val, x_test: X_test, y_test_: y_test})
                        wait += 1
                best_glob_iterations = ext_iterations + best_iteration
                if (max_iterations - best_iteration) <= 8000:
                    if loops > 3:
                        best_glob_iterations = ext_iterations + best_iteration
                        train_flag = False
                    else:
                        ext_iterations += 10000
                        max_iterations = 10000
                        best_iteration = 0
                        loops += 1
                else:
                    best_glob_iterations = ext_iterations + best_iteration
                    train_flag = False

            sess.close()

        model_dict = dict()
        model_dict['centroids'] = centers
        model_dict['Radius'] = Radius
        model_dict['n_vars'] = self.D
        model_dict['num_centr'] = num_centr
        model_dict['W'] = W
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy RBFNN: %s" % str(obj_old[0]))

        return obj_old[3], centers, Radius, W, model_dict
