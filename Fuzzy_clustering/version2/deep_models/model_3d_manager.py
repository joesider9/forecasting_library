import os

import joblib
import numpy as np

from Fuzzy_clustering.version2.deep_models.cnn_tf_core import CNN
from Fuzzy_clustering.version2.deep_models.cnn_tf_core_3d import CNN_3d
from Fuzzy_clustering.version2.deep_models.lstm_tf_core_3d import LSTM_3d
from Fuzzy_clustering.version2.deep_models.rbfnn_tf_core import RBFNN
from Fuzzy_clustering.version2.deep_models.mlp_tf_core import MLP
from Fuzzy_clustering.version2.model_manager.rbf_cnn_model import RBF_CNN_model


class model3d_manager():
    def __init__(self, static_data, cluster, method, params):
        self.params = params
        self.test = params['test']
        self.method = str.lower(method)
        self.cluster = cluster
        self.istrained = False
        if self.method == 'cnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'CNN_3d')
        elif self.method == 'lstm':
            self.model_dir = os.path.join(cluster.cluster_dir, 'LSTM_3d')
        elif self.method == 'rbfnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'RBFNN')
        elif self.method == 'rbf-cnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'RBF_CNN')
        elif self.method == 'mlp_3d':
            self.model_dir = os.path.join(cluster.cluster_dir, 'MLP_3D')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.test_dir = self.model_dir
        try:
            self.load()
        except:
            pass
        if not self.istrained:
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
            try:
                self.load()
            except:
                if not os.path.exists(self.test_dir):
                    os.makedirs(self.test_dir)
                pass

        self.static_data = static_data
        self.cluster_name = cluster.cluster_name
        self.rated = static_data['rated']
        self.data_dir = cluster.data_dir
        self.probabilistic = False

    def fit(self):
        if self.istrained == False:
            print('Train ', self.method, ' ', self.cluster_name)
            if self.method == 'cnn':
                return self.optimize_cnn()
            elif self.method == 'lstm':
                return self.optimize_lstm()
            elif self.method == 'rbfnn':
                return self.optimize_rbf()
            elif self.method == 'rbf-cnn':
                return self.optimize_rbf_cnn()
            elif self.method == 'mlp_3d':
                return self.optimize_mlp()
        else:
            return self.acc

    def fit_TL(self):
        if self.istrained == False:
            print('Train ', self.method, ' ', self.cluster_name, ' ', self.test)
            if self.method == 'cnn':
                return self.optimize_cnn_TL()
            elif self.method == 'lstm':
                return self.optimize_lstm_TL()
            elif self.method == 'rbfnn':
                return self.optimize_rbf_TL()
            elif self.method == 'rbf-cnn':
                return self.optimize_rbf_cnn_TL()
        else:
            return self.acc

    def load_data(self):
        if self.method == 'cnn':
            if os.path.exists(os.path.join(self.data_dir, 'dataset_cnn.pickle')):
                cvs = joblib.load(os.path.join(self.data_dir, 'cvs_cnn.pickle'))
            else:
                cvs = np.array([])
        elif self.method == 'lstm':
            if os.path.exists(os.path.join(self.data_dir, 'dataset_lstm.pickle')):
                cvs = joblib.load(os.path.join(self.data_dir, 'cvs_lstm.pickle'))
            else:
                cvs = np.array([])
        elif self.method == 'rbfnn':
            if os.path.exists(os.path.join(self.data_dir, 'dataset_X.csv')):
                cvs = joblib.load(os.path.join(self.data_dir, 'cvs.pickle'))
            else:
                cvs = np.array([])
        elif self.method == 'rbf-cnn':
            if os.path.exists(os.path.join(self.data_dir, 'dataset_X.csv')):
                cvs = joblib.load(os.path.join(self.data_dir, 'cvs.pickle'))
            else:
                cvs = np.array([])
        elif self.method == 'mlp_3d':
            if os.path.exists(os.path.join(self.data_dir, 'dataset_X.csv')):
                cvs = joblib.load(os.path.join(self.data_dir, 'cvs.pickle'))
            else:
                cvs = np.array([])
        return cvs

    def optimize_rbf(self):
        self.num_centr = self.params['num_centr']
        self.lr = self.params['lr']
        self.mean_var = self.static_data['RBF']['mean_var']
        self.std_var = self.static_data['RBF']['std_var']
        max_iterations = self.static_data['RBF']['max_iterations']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)
        rbf = RBFNN(self.static_data, max_iterations=max_iterations)
        self.acc, self.centroids, self.radius, self.w, self.model = rbf.train(X_train, y_train, X_val, y_val, X_test,
                                                                              y_test, self.num_centr, self.lr,
                                                                              gpu_id=self.gpu)

        self.istrained = True
        self.save()
        return self.acc

    def load_rbf_models(self):
        model_rbfs = RBF_CNN_model(self.static_data, self.cluster, cnn=False)
        rbf_models = [model_rbfs.model_rbf_ols.models, model_rbfs.model_rbf_ga.models, model_rbfs.model_rbfnn.model]
        return rbf_models

    def optimize_rbf_cnn(self):
        self.trial = self.params['trial']
        self.pool_size = self.params['pool_size']
        self.kernels = self.params['kernels']
        self.lr = self.params['lr']
        self.hsize = self.params['h_size']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = self.static_data['CNN']['filters']
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.static_data_cnn = self.static_data['CNN']
        self.static_data_rbf = self.static_data['RBF']

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        self.rbf_models = self.load_rbf_models()

        cnn = CNN(self.static_data, self.rated, self.rbf_models, X_train, y_train, X_val, y_val, X_test, y_test,
                  self.pool_size, self.trial)

        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                                     learning_rate=self.lr, kernels=self.kernels,
                                                                     h_size=self.hsize, gpu_id=self.gpu,
                                                                     filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
            pass

        if not flag:
            raise MemoryError('GPU low memory for RBF-CNN with filters %s', str(self.filters))
        self.istrained = True
        self.save()
        return self.acc

    def optimize_cnn(self):
        self.trial = self.params['trial']
        self.pool_size = self.params['pool_size']
        self.kernels = self.params['kernels']
        self.lr = self.params['lr']
        self.hsize = self.params['h_size']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = self.static_data['CNN']['filters']
        cvs = self.load_data()
        cnn = CNN_3d(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], self.pool_size,
                     trial=self.trial)
        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                                     learning_rate=self.lr, kernels=self.kernels,
                                                                     h_size=self.hsize, gpu_id=self.gpu,
                                                                     filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
                pass

        if not flag:
            raise MemoryError('GPU low memory for CNN with filters %s', str(self.filters))
        self.istrained = True
        self.save()
        return self.acc

    def optimize_lstm(self):
        self.trial = self.params['trial']
        self.units = self.params['units']
        self.lr = self.params['lr']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        lstm_max_iterations = self.static_data['LSTM']['max_iterations']
        self.hold_prob = self.static_data['LSTM']['hold_prob']
        cvs = self.load_data()
        lstm = LSTM_3d(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], trial=self.trial,
                       probabilistc=self.probabilistic)
        # try:
        self.acc, self.scale_lstm, self.model = lstm.train(max_iterations=lstm_max_iterations,
                                                           learning_rate=self.lr, units=self.units,
                                                           hold_prob=self.hold_prob, gpu_id=self.gpu)
        # except:
        #     acc_old_lstm=np.inf
        #     scale_lstm=None
        #     model_lstm=None
        #     pass
        self.istrained = True
        self.save()
        return self.acc

    def optimize_mlp(self):
        self.trial = self.params['trial']
        self.units = self.params['units']
        self.act_func = self.params['act_func']
        self.lr = self.params['lr']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        lstm_max_iterations = self.static_data['LSTM']['max_iterations']
        self.hold_prob = self.static_data['LSTM']['hold_prob']
        cvs = self.load_data()
        mlp = MLP(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], trial=self.trial,
                  probabilistc=self.probabilistic)
        # try:
        self.acc, self.model = mlp.train(max_iterations=lstm_max_iterations,
                                         learning_rate=self.lr, units=self.units,
                                         hold_prob=self.hold_prob, act_func=self.act_func, gpu_id=self.gpu)
        # except:
        #     acc_old_lstm=np.inf
        #     scale_lstm=None
        #     model_lstm=None
        #     pass
        self.istrained = True
        self.save()
        return self.acc

    def load(self):
        if os.path.exists(os.path.join(self.test_dir, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.test_dir, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.test_dir, self.method + '.pickle'), compress=9)

    def optimize_rbf_TL(self):
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_TL_dir = os.path.join(cluster_dir_tl, 'RBFNN')
        model_TL = joblib.load(os.path.join(model_TL_dir, self.method + '.pickle'))

        self.num_centr = model_TL['num_centr']
        self.lr = model_TL['lr']
        self.mean_var = model_TL['mean_var']
        self.std_var = model_TL['std_var']
        max_iterations = self.static_data['RBF']['max_iterations']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)
        rbf = RBFNN(self.static_data, max_iterations=max_iterations)
        self.acc, self.centroids, self.radius, self.w, self.model = rbf.train(X_train, y_train, X_val, y_val, X_test,
                                                                              y_test, self.num_centr, self.lr,
                                                                              gpu_id=self.gpu)

        self.istrained = True
        self.save()
        return self.acc

    def optimize_rbf_cnn_TL(self):
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_TL_dir = os.path.join(cluster_dir_tl, 'RBF_CNN')
        model_TL = joblib.load(os.path.join(model_TL_dir, self.method + '.pickle'))

        self.trial = model_TL['trial']
        self.pool_size = model_TL['pool_size']
        self.kernels = model_TL['kernels']
        self.lr = model_TL['lr']
        self.hsize = model_TL['h_size']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = model_TL['filters']
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.static_data_cnn = self.static_data['CNN']
        self.static_data_rbf = self.static_data['RBF']

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        self.rbf_models = self.load_rbf_models()

        cnn = CNN(self.static_data, self.rated, self.rbf_models, X_train, y_train, X_val, y_val, X_test, y_test,
                  self.pool_size, self.trial)

        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                                     learning_rate=self.lr, kernels=self.kernels,
                                                                     h_size=self.hsize, gpu_id=self.gpu,
                                                                     filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
            pass

        if not flag:
            self.acc = np.inf
            self.scale_cnn = None
            self.model = None
        self.istrained = True
        self.save()
        return self.acc

    def optimize_cnn_TL(self):
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_TL_dir = os.path.join(cluster_dir_tl, 'CNN_3d')
        model_TL = joblib.load(os.path.join(model_TL_dir, self.method + '.pickle'))

        self.trial = model_TL['trial']
        self.pool_size = model_TL['pool_size']
        self.kernels = model_TL['kernels']
        self.lr = model_TL['lr']
        self.hsize = model_TL['h_size']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = model_TL['filters']
        cvs = self.load_data()
        cnn = CNN_3d(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], self.pool_size,
                     trial=self.trial)
        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                                     learning_rate=self.lr, kernels=self.kernels,
                                                                     h_size=self.hsize, gpu_id=self.gpu,
                                                                     filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
                pass

        if not flag:
            self.acc = np.inf
            self.scale_cnn = None
            self.model = None
        self.istrained = True
        self.save()
        return self.acc

    def optimize_lstm_TL(self):
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_TL_dir = os.path.join(cluster_dir_tl, 'LSTM_3d')
        model_TL = joblib.load(os.path.join(model_TL_dir, self.method + '.pickle'))

        self.trial = model_TL['trial']
        self.units = model_TL['units']
        self.lr = model_TL['lr']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        lstm_max_iterations = self.static_data['LSTM']['max_iterations']
        self.hold_prob = model_TL['hold_prob']
        cvs = self.load_data()
        lstm = LSTM_3d(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], trial=self.trial,
                       probabilistc=self.probabilistic)
        # try:
        self.acc, self.scale_lstm, self.model = lstm.train(max_iterations=lstm_max_iterations,
                                                           learning_rate=self.lr, units=self.units,
                                                           hold_prob=self.hold_prob, gpu_id=self.gpu)
        # except:
        #     acc_old_lstm=np.inf
        #     scale_lstm=None
        #     model_lstm=None
        #     pass
        self.istrained = True
        self.save()
        return self.acc
