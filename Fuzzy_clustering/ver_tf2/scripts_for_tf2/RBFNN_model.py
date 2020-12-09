import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from clustering.algorithms import FCV
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from Fuzzy_clustering.ver_0.utils_for_forecast import split_continuous
from util_database import write_database
from Fuzzy_clustering.ver_tf2.Forecast_model import forecast_model

class RBF_model(tf.keras.Model):
    def __init__(self, num_centr):
        super(RBF_model, self).__init__()
        self.num_centr = num_centr

    def find_centers(self,X_train):
        self.N, self.D = X_train.shape

        self.batch_size = self.N
        try:
            centers = FCV(X_train, n_clusters=self.num_centr, r=4).optimize()
            c = centers.C
        except:
            c = KMeans(n_clusters=self.num_centr, random_state=0).fit(X_train)
            c = c.cluster_centers_

        centroids = c.astype(np.float32)
        return centroids

    def initialize(self,inputs):
        centroids = self.find_centers(inputs)
        cnt = pd.DataFrame(centroids, index=['c' + str(i) for i in range(centroids.shape[0])],
                           columns=['v' + str(i) for i in range(centroids.shape[1])])
        var_init = pd.DataFrame(columns=['v' + str(i) for i in range(centroids.shape[1])])
        for r in cnt.index:
            v = (cnt.loc[r] - cnt.drop(r)).min()
            v[v == 0] = 0.0001
            v.name = r
            var_init = var_init.append(v)
        var_init = tf.convert_to_tensor(var_init.values, dtype=tf.float32, name='var_init')
        self.var = tf.Variable(var_init,
                          dtype=tf.float32, name='RBF_variance')

        self.centroids = tf.convert_to_tensor(centroids, dtype=tf.float32, name='centroids')
    @tf.function
    def lin_out(self, x, y):
        return tf.linalg.lstsq(x, y, l2_regularizer=0)

    @tf.function
    def rbf_map(self, x):
        s = tf.shape(x)

        d1 = tf.transpose(tf.tile(tf.expand_dims(x, 0), [self.num_centr, 1, 1]), perm=[1, 0, 2]) - tf.tile(
            tf.expand_dims(self.centroids, 0), [s[0], 1, 1])
        d = tf.sqrt(
            tf.reduce_sum(tf.pow(tf.multiply(d1, tf.tile(tf.expand_dims(self.var, 0), [s[0], 1, 1])), 2), axis=2))

        return tf.cast(tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d))), tf.float32)

    def call(self, inputs, training=None, mask=None):
        if training:
            x = inputs[:, :-1]
            y = tf.expand_dims(inputs[:, -1], 1)
        else:
            x=inputs

        phi = self.rbf_map(x)
        if training:
            self.w = self.lin_out(phi, y)
            h = tf.matmul(phi, self.w)
        else:
            h = tf.matmul(phi, self.w)
        return h

class sum_square_loss(tf.keras.losses.Loss):
    def __init__(self, name='SSE', **kwargs):
        super(sum_square_loss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        return math_ops.reduce_sum(math_ops.square(y_true - y_pred))


class RBF_train():
    def __init__(self, path_model, rated=None, max_iterations=10000,):
        self.path_model = path_model
        self.rated = rated
        self.max_iterations = max_iterations

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
        if d < 0:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, num_centr, lr, gpu_id=[-1]):
        tf.config.experimental.set_visible_devices(gpu_id[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_id[0], True)
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(True)
        self.N, self.D = X_train.shape
        X_val = np.vstack((X_val, X_test))
        y_val = np.vstack((y_val, y_test))

        X_train = X_train.astype('float32',casting='same_kind')
        X_val = X_val.astype('float32',casting='same_kind')
        y_train = y_train.astype('float32',casting='same_kind')
        y_val = y_val.astype('float32',casting='same_kind')

        batch_size = self.N
        model = RBF_model(num_centr)
        model.initialize(X_train)
        optimizer = tf.keras.optimizers.Adam(lr)

        batches = [np.random.choice(self.N, batch_size, replace=False) for _ in range(self.max_iterations)]
        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        if self.rated is None:
            loss_fn = sum_square_loss()
            mae = tf.keras.metrics.MeanAbsolutePercentageError(name='mae')
            mse = tf.keras.metrics.MeanSquaredLogarithmicError(name='mse')
            rms = tf.keras.metrics.RootMeanSquaredError(name='rms')
        else:
            loss_fn = sum_square_loss()
            mae = tf.keras.metrics.MeanAbsolutePercentageError(name='mae')
            mse = tf.keras.metrics.MeanSquaredLogarithmicError(name='mse')
            rms = tf.keras.metrics.RootMeanSquaredError(name='rms')

        res = dict()
        self.best_weights = None
        best_iteration = 0
        best_glob_iterations = 0
        max_iterations = self.max_iterations
        ext_iterations = self.max_iterations
        train_flag = True
        patience = 20000
        wait = 0
        while train_flag:
            for i in tqdm(range(max_iterations)):
                if i % 500 == 0:
                    with tf.GradientTape() as tape:
                        predictions = model(np.hstack((X_train,y_train)), training=True)
                        loss = loss_fn(y_train, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    pred_val = model(X_val, training=False)
                    val_mae = mae(y_val, pred_val)
                    val_mse = mse(y_val, pred_val)
                    val_rms = rms(y_val, pred_val)
                    val_sse = loss_fn(y_val, pred_val)
                    obj_new = np.array([val_mae, val_mse, val_sse, val_rms])
                    flag, obj_old, obj_max, obj_min = self.distance(obj_new, obj_old, obj_max, obj_min)
                    if flag:
                        res[str(i)] = obj_old
                        print(val_mae.numpy())
                        self.best_weights = model.get_weights()
                        best_iteration = i
                        wait = 0
                    else:
                        wait += 1
                    if wait > patience:
                        train_flag = False
                        break
                else:
                    with tf.GradientTape() as tape:
                        predictions = model(np.hstack((X_train,y_train)), training=True)
                        loss = loss_fn(y_train, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    wait += 1
            best_glob_iterations = ext_iterations + best_iteration
            if (max_iterations - best_iteration) <= 10000:
                ext_iterations += 20000
                max_iterations = 20000
            else:
                best_glob_iterations = ext_iterations + best_iteration
                train_flag = False

        model.set_weights(self.best_weights)
        model.save_weights(self.path_model + '/rbf_model.h5')

        model_dict = dict()
        model_dict['centroids'] = model.centroids.numpy()
        model_dict['Radius'] = model.var.numpy()
        model_dict['n_vars'] = self.D
        model_dict['num_centr'] = num_centr
        model_dict['W'] = model.w.numpy()
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy cnn: %s" % obj_old[0])

        return obj_old[3], model.centroids.numpy(), model.var.numpy(), model.w.numpy(), model_dict


if __name__ == '__main__':
    cluster_dir = 'D:/APE_net_ver2/Regressor_layer/rule.2'
    data_dir = 'D:/APE_net_ver2/Regressor_layer/rule.2/data'


    rated = None
    X = np.load(os.path.join(data_dir, 'X_train.npy'))
    y = np.load(os.path.join(data_dir, 'y_train.npy'))
    static_data = write_database()
    forecast = forecast_model(static_data, use_db=False)
    forecast.load()
    X = X[:, 0:-1]
    X = forecast.sc.transform(X)
    y = forecast.scale_y.transform(y)
    X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    rbf=RBF_train(cluster_dir+'/RBFNN',rated=rated, max_iterations=1000)
    rbf.train(X_train, y_train, X_val,  y_val, X_test, y_test, 12, 0.0001, gpu_id=tf.config.experimental.list_physical_devices('GPU'))