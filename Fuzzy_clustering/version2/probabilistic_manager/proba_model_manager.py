import os

import joblib

from Fuzzy_clustering.version2.probabilistic_manager.mlp_predict_3d import MLP_predict
from Fuzzy_clustering.version2.probabilistic_manager.mlp_tf_core import MLP


class proba_model_manager():
    def __init__(self, static_data, params={}):

        self.istrained = False
        self.method = 'mlp'
        self.static_data = static_data
        self.model_dir = os.path.join(static_data['path_model'], 'Probabilistic')
        self.data_dir = self.static_data['path_data']
        if len(params) > 0:
            self.params = params
            self.test = params['test']
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
        if hasattr(self, 'test_dir'):
            try:
                if not os.path.exists(self.test_dir):
                    os.makedirs(self.test_dir)
                self.load(self.test_dir)
            except:
                pass
        else:
            try:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                self.load(self.model_dir)
            except:
                pass

        self.cluster_name = static_data['_id']
        self.rated = static_data['rated']
        self.probabilistic = True



    def fit(self):
        if self.istrained == False:

            return self.optimize_mlp()

        else:
            return self.acc

    def load_data(self):
        if os.path.exists(os.path.join(self.data_dir, 'cvs_proba.pickle')):
            cvs = joblib.load(os.path.join(self.data_dir, 'cvs_proba.pickle'))
        else:
            raise ImportError('Predictions for probabilistic not found ')

        return cvs

    def predict(self, X):
        model = MLP_predict(self.static_data, trial=self.trial,
                            probabilistc=self.probabilistic)
        return model.predict(X)

    def optimize_mlp(self):
        self.trial = self.params['trial']
        self.units = self.params['units']
        self.act_func = self.params['act_func']
        self.lr = self.params['lr']
        self.gpu = '/device:GPU:' + str(self.params['gpu'])
        lstm_max_iterations = self.static_data['MLP']['max_iterations']
        self.hold_prob = self.static_data['MLP']['hold_prob']
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        mlp = MLP(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, trial=self.trial,
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

    def load(self, path):
        if os.path.exists(os.path.join(path, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(path, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'temp_dir', 'model_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.test_dir, self.method + '.pickle'), compress=9)
