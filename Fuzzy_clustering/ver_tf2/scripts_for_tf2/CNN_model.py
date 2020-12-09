import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ProgbarLogger, CSVLogger
from tensorflow.python.ops import math_ops
from tqdm import tqdm
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from clustering.algorithms import FCV
import numpy as np
import os
import pandas as pd
from Fuzzy_clustering.ver_0.utils_for_forecast import split_continuous


class CNN_model(tf.keras.Model):
    def __init__(self, static_data, kernels, D, num_centr, depth, hold_prob):
        super(CNN_model, self).__init__()
        self.D =D
        self.num_centr=num_centr
        self.depth=depth
        self.hold_prob=hold_prob
        self.static_data= static_data
        self.convo_1 = tf.keras.layers.Conv2D(filters=int(static_data['filters']),
                                         kernel_size=kernels,

                                         padding="same",
                                         activation=tf.nn.elu)

        self.convo_1_pool = tf.keras.layers.AveragePooling2D(pool_size=static_data['pool_size'], strides=1)
        self.flat=tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(1-self.hold_prob)
        self.full_layer_one = tf.keras.layers.Dense(units=self.static_data['h_size'][0], activation=tf.nn.elu)
        self.full_layer_two = tf.keras.layers.Dense(units=self.static_data['h_size'][1], activation=tf.nn.elu)
        self.output_layer = tf.keras.layers.Dense(1)


    @tf.function
    def call(self, inputs, training=None, mask=None):
        H = tf.reshape(inputs, [-1, self.D, self.num_centr, self.depth])
        cnn_output1=self.convo_1(H)
        cnn_output = self.convo_1_pool(cnn_output1)
        flatten = self.flat(cnn_output)
        full_two_dropout =self.full_layer_one(flatten)
        dense_output = self.full_layer_two(full_two_dropout)
        return self.output_layer(dense_output)


# class CNNCustomCallback(tf.keras.callbacks.Callback):
#     def __init__(self, patience=30000):
#         super(CNNCustomCallback, self).__init__()
#         self.patience = patience
#         self.wait=0
#         self.best_weights = None
#         self.obj_old = np.inf*tf.ones(4)
#         self.obj_max = np.inf*tf.ones(4)
#         self.obj_min = np.inf*tf.ones(4)
#
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % 100 == 0:
#             self.obj_new=np.array([logs['val_loss'], logs['val_mae'],  logs['val_mse'], logs['val_rms']])
#             if np.any(np.isinf(self.obj_old)):
#                 self.obj_old = self.obj_new.copy()
#                 self.obj_max = self.obj_new.copy()
#                 return True
#
#             if np.any(np.isinf(self.obj_min)) and not np.all(self.obj_max == self.obj_new):
#                 self.obj_min = self.obj_new.copy()
#             d = 0
#             for i in range(self.obj_new.shape[0]):
#                 if self.obj_max[i] < self.obj_new[i]:
#                     self.obj_max[i] = self.obj_new[i]
#                 if self.obj_min[i] > self.obj_new[i]:
#                     self.obj_min[i] = self.obj_new[i]
#
#                 d += (self.obj_new[i] - self.obj_old[i]) / (self.obj_max[i] - self.obj_min[i])
#             if d < 0:
#                 self.obj_old=self.obj_new.copy()
#                 self.wait = 0
#                 self.best_weights = self.model.get_weights()
#                 print(epoch)
#                 print(logs)
#             else:
#                 self.wait += 1
#                 if self.wait >= self.patience:
#                     self.stopped_epoch = True
#                     self.model.stop_training = True
#                     print('Restoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)
#
#     def on_train_end(self, logs=None):
#         print('Training ends!! Restoring model weights from the best epoch.')
#         self.model.set_weights(self.best_weights)

class sum_square_loss(tf.keras.losses.Loss):
    def __init__(self, name='SSE', **kwargs):
        super(sum_square_loss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        return math_ops.reduce_sum(math_ops.square(y_true - y_pred))


class CNN():
    def __init__(self, static_data,rated, models, X_train, y_train, X_val, y_val, X_test, y_test):
        self.static_data=static_data
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
        for i in range(num_centr):
            d1 = np.power(X - centroids[i, np.newaxis], 2)
            t1 = np.multiply(d1, radius[i, np.newaxis])
            h_map = np.exp((-1) * t1)
            hmap_list.append(h_map)

        return np.stack(hmap_list)

    def create_inputs(self, X_train, X_val, X_test):
        self.N, self.D = X_train.shape

        X_val = np.vstack((X_val, X_test))

        H_train = []
        H_val = []
        self.scale_cnn = []
        self.depth = len(self.rbf_models)
        self.num_centr = self.rbf_models[0]['centroids'].shape[0]
        for i in range(self.depth):
            centroids = self.rbf_models[i]['centroids']
            Radius = self.rbf_models[i]['Radius']
            H_train.append(np.transpose(self.rbf_map(X_train, self.num_centr, centroids, Radius), [1, 2, 0]))
            H_val.append(np.transpose(self.rbf_map(X_val, self.num_centr, centroids, Radius), [1, 2, 0]))
            H = np.vstack((H_train[i], H_val[i]))
            H = H.reshape(-1, self.D * self.num_centr)
            sc = MinMaxScaler()
            sc.fit(H)
            self.scale_cnn.append(sc)
            H_train[i] = sc.transform(H_train[i].reshape(-1, self.D * self.num_centr))
            H_train[i] = H_train[i].reshape(-1, self.D, self.num_centr)
            H_val[i] = sc.transform(H_val[i].reshape(-1, self.D * self.num_centr))
            H_val[i] = H_val[i].reshape(-1, self.D, self.num_centr)

        H_train = np.transpose(np.stack(H_train), [1, 2, 3, 0])
        H_val = np.transpose(np.stack(H_val), [1, 2, 3, 0])

        return H_train ,H_val


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

    def train_cnn(self, path_model, max_iterations=10000, learning_rate=5e-4, kernels=[2, 8], gpu_id=[-1]):


        tf.config.experimental.set_visible_devices(gpu_id[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_id[0], True)
        tf.config.set_soft_device_placement(True)
        # tf.debugging.set_log_device_placement(True)
        H_train, H_val = self.create_inputs(self.X_train, self.X_val, self.X_test)

        y_val = np.vstack((self.y_val, self.y_test))
        y_train = self.y_train
        batch_size = np.min([100, int(self.N / 5)])
        model=CNN_model(self.static_data,kernels,self.D,self.num_centr, self.depth, 1)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        # if self.rated is None:
        #
        #     model.compile(optimizer,loss=sum_square_loss(),metrics=[tf.metrics.MeanAbsolutePercentageError(name='mae'),
        #                tf.metrics.MeanSquaredLogarithmicError(name='mse'), tf.metrics.RootMeanSquaredError(name='rms')])
        # else:
        #     model.compile(optimizer, loss=sum_square_loss(),
        #                   metrics=[tf.metrics.MeanAbsolutePercentageError(name='mae'), tf.metrics.MeanSquaredLogarithmicError(name='mse'),
        #                tf.metrics.RootMeanSquaredError(name='rms')])
        #
        # prog_bar = ProgbarLogger(count_mode='steps', stateful_metrics=['loss', 'mae', 'sse', 'mse', 'rms'])
        # csv_logger = CSVLogger(path_model + '/log_with_centr_' + str(kernels[1]) + '.csv')
        # model.fit(H_train, y_train, batch_size=batch_size,
        #           epochs=max_iterations, validation_data=(H_val,y_val), verbose=0, callbacks=[csv_logger, CNNCustomCallback()])
        # model.save_weights(path_model + '/cnn_model.h5')

        batches = [np.random.choice(self.N, batch_size, replace=False) for _ in range(max_iterations)]
        obj_old = np.inf*np.ones(4)
        obj_max = np.inf*np.ones(4)
        obj_min = np.inf*np.ones(4)
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
        ext_iterations = max_iterations
        train_flag = True
        patience = 15000
        wait = 0
        while train_flag:
            for i in tqdm(range(max_iterations)):
                if i % 100 == 0:
                    with tf.GradientTape() as tape:
                        predictions = model(H_train[batches[i]], training=True)
                        loss = loss_fn(y_train[batches[i]], predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    pred_val = model(H_val, training=False)
                    val_mae=mae(y_val, pred_val)
                    val_mse=mse(y_val, pred_val)
                    val_rms=rms(y_val, pred_val)
                    val_sse=loss_fn(y_val, pred_val)
                    obj_new = np.array([val_mae, val_mse, val_sse, val_rms])
                    flag, obj_old, obj_max, obj_min = self.distance(obj_new, obj_old, obj_max, obj_min)
                    if flag:
                        res[str(i)] = obj_old
                        print(val_mae.numpy())
                        self.best_weights = model.get_weights()
                        best_iteration = i
                        wait = 0
                    else:
                        wait+=1
                    if wait>patience:
                        train_flag = False
                        break
                else:
                    with tf.GradientTape() as tape:
                        predictions = model(H_train[batches[i]], training=True)
                        loss = loss_fn(y_train[batches[i]], predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    wait+=1
            best_glob_iterations = ext_iterations + best_iteration
            if (max_iterations-best_iteration) <= 5000:
                ext_iterations+= 10000
                max_iterations = 10000
            else:
                best_glob_iterations = ext_iterations + best_iteration
                train_flag = False

        model.set_weights(self.best_weights)
        model.save_weights(path_model + '/cnn_model.h5')

        model_dict =dict()
        model_dict['kernels'] = kernels
        model_dict['static_data'] = self.static_data
        model_dict['n_vars'] = self.D
        model_dict['num_centr'] = self.num_centr
        model_dict['depth'] = self.depth
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy cnn: %s" % obj_old[0])

        return obj_old[3], self.scale_cnn, model_dict