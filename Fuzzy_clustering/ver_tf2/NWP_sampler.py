import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os, joblib, logging, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from sklearn.model_selection import train_test_split

class nwp_sampler():

    def __init__(self, static_data):
        self.static_data = static_data
        self.model_type = self.static_data['type']
        self.data_variables = self.static_data['data_variables']
        self.model_dir = os.path.join(self.static_data['path_model'], 'NWP_sampler')
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def train(self, X_inp, X, gpu_id='/cpu:0'):
        if gpu_id == '/device:GPU:0':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif gpu_id == '/device:GPU:1':
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.gpu_id = gpu_id

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.static_data['path_model'], 'log_model.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print('CNN training...begin')
        logger.info('CNN training...begin')

        i=0
        X1 = []

        self.columns = []
        self.model_name = []
        self.max_val = []
        for var in sorted(self.data_variables):
            if ((var == 'WS') and (self.model_type =='wind')) or ((var == 'Flux') and (self.model_type == 'pv')):

                var_name = 'flux' if var == 'Flux' else 'wind'
                self.columns.extend(['p_' + var_name] + [var_name] + ['n_' + var_name])

                X1.append(X[:, :, :, i])
                model1_name = 'p_' + var_name
                self.model_name.append(model1_name)
                self.max_val.append(1000 if var == 'Flux' else 30)

                X1.append(X[:, :, :, i+1])
                model1_name = var_name
                self.model_name.append(model1_name)
                self.max_val.append(1000 if var == 'Flux' else 30)

                X1.append(X[:, :, :, i + 2])
                model1_name = 'n_' + var_name
                self.model_name.append(model1_name)
                self.max_val.append(1000 if var == 'Flux' else 30)
                i += 3
            elif var in {'WD', 'Cloud'}:
                X1.append(X[:, :, :, i])

                self.columns.append('cloud' if var == 'Cloud' else 'direction')
                model2_name = 'cloud' if var == 'Cloud' else 'direction'
                self.model_name.append(model2_name)
                self.max_val.append(100 if var == 'Cloud' else 360)
                i += 1
            elif (var in {'Temperature'}):
                X1.append(X[:, :, :, i])
                self.columns.append('Temp' if var == 'Temperature' else 'wind')
                model2_name = 'Temp' if var == 'Temperature' else 'wind'
                self.model_name.append(model2_name)
                self.max_val.append(320 if var == 'Temperature' else 30)
                i+=1
            elif ((var == 'WS') and (self.model_type == 'pv')):
                X1.append(X[:, :, :, i])
                self.columns.append('Temp' if var == 'Temperature' else 'wind')
                model2_name = 'Temp' if var == 'Temperature' else 'wind'
                self.model_name.append(model2_name)
                self.max_val.append(320 if var == 'Temperature' else 30)
                i += 1
            else:
                i += 1

        X_inp = X_inp[self.columns].values
        self.models = dict()

        for x, var, max_val in zip(X1, self.model_name,self.max_val):
            self.models[var] = self.train_model(X_inp, (x / max_val)[:, :, :, np.newaxis])


        self.istrained = True
        self.save(self.model_dir)

    def build_graph(self, inp, nwps, learning_rate):


        with tf.name_scope("build_cnn") as scope:
            inp = tf.keras.layers.GaussianNoise(0.05)(inp)
            full_layer_one = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name='dense1')
            full_layer_two = tf.keras.layers.Dense(units=96, activation=tf.nn.elu,
                                             name='dense2')
            full_layer_three = tf.keras.layers.Dense(units=self.D1 * self.D2, activation=tf.nn.elu,
                                                   name='dense3')
            full_two_out = full_layer_one(inp)
            full_three_out = full_layer_two(full_two_out)
            y_pred = full_layer_three(full_three_out)
            y_pred = tf.reshape(y_pred, [-1, self.D1, self.D2, self.depth], name='reshape')
            weights = full_layer_one.trainable_weights + full_layer_two.trainable_weights + full_layer_three.trainable_weights

        with tf.name_scope("train_cnn") as scope:
            cost_cnn = tf.reduce_mean(tf.square(y_pred - nwps))
            optimizer_cnn = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_cnn = optimizer_cnn.minimize(cost_cnn)
            accuracy_cnn = tf.reduce_mean(tf.abs(y_pred - nwps))
            sse_cnn = tf.reduce_sum(tf.square(y_pred - nwps))
            rse_cnn = tf.sqrt(tf.reduce_mean(tf.square(y_pred - nwps)))
        return train_cnn, cost_cnn, accuracy_cnn, sse_cnn, rse_cnn, weights, full_layer_one, full_layer_two, full_layer_three

    def build_graph_predict(self, inp):


        with tf.name_scope("build_cnn") as scope:

            full_layer_one = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name='dense1')
            full_layer_two = tf.keras.layers.Dense(units=96, activation=tf.nn.elu,
                                             name='dense2')
            full_layer_three = tf.keras.layers.Dense(units=self.D1 * self.D2, activation=tf.nn.elu,
                                                   name='dense3')
            full_two_out = full_layer_one(inp)
            full_three_out = full_layer_two(full_two_out)
            y_pred = full_layer_three(full_three_out)
            y_pred = tf.reshape(y_pred, [-1, self.D1, self.D2, self.depth], name='reshape')


        return y_pred, full_layer_one, full_layer_two, full_layer_three

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
        if d < -0.05:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def train_model(self, X_inp, X):
        if len(X.shape)==3:
            X=X[:, :, :, np.newaxis]
        if X_inp.shape[0] != X.shape[0]:
            raise ValueError('dataset_X and dataset_cnn has not the same N samples')
        X_train, X_test, y_train, y_test = split_continuous(X_inp, X, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

        N, self.D1, self.D2, self.depth = y_train.shape

        batch_size = 100
        tf.compat.v1.reset_default_graph()
        graph_cnn = tf.Graph()
        with graph_cnn.as_default():
            with tf.device("/cpu:0"):
                x1 = tf.compat.v1.placeholder('float', shape=[None, X_inp.shape[1]],
                                              name='input_data')
                y_pred_ = tf.compat.v1.placeholder(tf.float32, shape=[None, self.D1, self.D2, self.depth], name='target_cnn')
            with tf.device(self.gpu_id):
                train_cnn, cost_cnn, accuracy_cnn, sse_cnn, rse_cnn, weights, full_layer_one, full_layer_two, full_layer_three = self.build_graph(x1, y_pred_, self.static_data['CNN']['learning_rate'])

        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        batches = [np.random.choice(N, batch_size, replace=False) for _ in range(self.static_data['CNN']['max_iterations'] + 1)]

        config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config_tf.gpu_options.allow_growth = True

        res = dict()
        self.best_weights = dict()
        best_iteration = 0
        best_glob_iterations = 0
        ext_iterations = self.static_data['CNN']['max_iterations']
        train_flag = True
        patience = 10000
        wait = 0
        max_iterations =self.static_data['CNN']['max_iterations']
        with tf.compat.v1.Session(graph=graph_cnn, config=config_tf) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            while train_flag:
                for i in tqdm(range(max_iterations)):
                    if i % 1000 == 0:

                        sess.run([train_cnn],
                                 feed_dict={x1: X_train[batches[i]], y_pred_: y_train[batches[i]]})

                        acc_new_v, mse_new_v, sse_new_v, rse_new_v, weights_cnn = sess.run(
                            [accuracy_cnn, cost_cnn, sse_cnn, rse_cnn, weights],
                            feed_dict={x1: X_val, y_pred_: y_val})
                        acc_new_t, mse_new_t, sse_new_t, rse_new_t = sess.run([accuracy_cnn, cost_cnn, sse_cnn, rse_cnn],
                                                                              feed_dict={x1: X_test, y_pred_: y_test})

                        acc_new = 0.4 * acc_new_v + 0.6 * acc_new_t + 2 * np.abs(acc_new_v - acc_new_t)
                        mse_new = 0.4 * mse_new_v + 0.6 * mse_new_t + 2 * np.abs(mse_new_v - acc_new_t)
                        sse_new = 0.4 * sse_new_v + 0.6 * sse_new_t + 2 * np.abs(sse_new_t - sse_new_v)
                        rse_new = 0.4 * rse_new_v + 0.6 * rse_new_t + 2 * np.abs(rse_new_t - rse_new_v)

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
                                 feed_dict={x1: X_train[batches[i]], y_pred_: y_train[batches[i]]})
                        wait += 1
                best_glob_iterations = ext_iterations + best_iteration
                if (max_iterations - best_iteration) <= 5000 and max_iterations > 2000:
                    ext_iterations += 10000
                    max_iterations = 10000
                    best_iteration = 0
                else:
                    best_glob_iterations = ext_iterations + best_iteration
                    train_flag = False

            sess.close()

        model_dict = dict()
        model_dict['best_weights'] = self.best_weights
        model_dict['static_data'] = self.static_data
        model_dict['n_vars'] = self.D1 * self.D2
        model_dict['depth'] = self.depth
        model_dict['best_iteration'] = best_glob_iterations
        model_dict['metrics'] = obj_old
        model_dict['error_func'] = res
        print("Total accuracy cnn-3d: %s" % obj_old[0])

        return model_dict

    def run_models(self, X_inp, model_name):
        if self.istrained:
            best_weights = self.models[model_name]['best_weights']
            tf.compat.v1.reset_default_graph()
            graph_cnn = tf.Graph()
            with graph_cnn.as_default():
                with tf.device("/cpu:0"):
                    x1 = tf.compat.v1.placeholder('float', shape=[None, X_inp.shape[1]],
                                                  name='input_data')
                with tf.device(self.gpu_id):
                    y_pred_, full_layer_one, full_layer_two, full_layer_three = self.build_graph_predict(x1)

            config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config_tf.gpu_options.allow_growth = True

            with tf.compat.v1.Session(graph=graph_cnn, config=config_tf) as sess:

                sess.run(tf.compat.v1.global_variables_initializer())
                full_layer_one.set_weights(
                    [best_weights['build_cnn/dense1/kernel:0'], best_weights['build_cnn/dense1/bias:0']])
                full_layer_two.set_weights(
                    [best_weights['build_cnn/dense2/kernel:0'], best_weights['build_cnn/dense2/bias:0']])
                full_layer_three.set_weights(
                    [best_weights['build_cnn/dense3/kernel:0'], best_weights['build_cnn/dense3/bias:0']])

                y_pred = sess.run(y_pred_, feed_dict={x1: X_inp})

                sess.close()

        else:
            raise ModuleNotFoundError("Error on prediction of %s cluster. The model nwp_sampler seems not properly trained")

        return y_pred

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'nwp_sampler' + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(cluster_dir, 'nwp_sampler' + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open nwp_sampler model')
        else:
            raise ImportError('Cannot find nwp_sampler model')

    def save(self, pathname):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'model_dir', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(pathname, 'nwp_sampler' + '.pickle'), compress=9)

if __name__ == '__main__':
    if sys.platform == 'linux':
        sys_folder = '/media/smartrue/HHD1/George/models/'
    else:
        sys_folder = 'D:/models/'

    path_project = sys_folder + '/Crossbow/Bulgaria_ver2/pv/Lach/model_ver0'
    static_data = joblib.load(os.path.join(path_project, 'static_data.pickle'))

    data_path = static_data['path_data']
    X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)


    if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
        X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
    else:
        X_cnn = np.array([])

    sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
    X = pd.DataFrame(sc.transform(X.values),columns=X.columns,index=X.index)

    create_nwps = nwp_sampler(static_data)

    create_nwps.train(X, X_cnn, gpu_id=static_data['CNN']['gpus'][0])

    wind_samples = create_nwps.run_models(X[create_nwps.columns].values,'flux')
    dir_samples = create_nwps.run_models(X[create_nwps.columns].values,'cloud')

    print('End')
