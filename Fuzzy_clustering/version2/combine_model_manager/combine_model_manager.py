import logging
import os

import joblib
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split

from Fuzzy_clustering.version2.sklearn_models.sklearn_models_deap import sklearn_model


class CombineModelManager(object):
    def __init__(self, project):
        self.project = project
        self.static_data = project.static_data
        self.istrained = False
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type = self.static_data['type']
        self.combine_methods = self.static_data['combine_methods']
        if 'is_probabilistic' in self.static_data.keys():
            self.is_probabilistic = self.static_data['is_probabilistic']
        else:
            self.is_probabilistic = False
        methods = [method for method in self.static_data['project_methods'].keys() if
                   self.static_data['project_methods'][method] == True]
        self.methods = []
        for method in methods:
            if method == 'RBF_ALL_CNN':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
            elif method == 'RBF_ALL':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
            else:
                self.methods.append(method)
        self.methods += self.combine_methods
        self.weight_size_full = len(self.methods)
        self.weight_size = len(self.combine_methods)
        self.rated = self.static_data['rated']
        self.n_jobs = self.static_data['sklearn']['njobs']
        try:
            self.load(self.model_dir)
        except:
            pass
        self.data_dir = self.static_data['path_data']

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_combine_model.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def bcp_fit(self, X, y):
        sigma = np.nanstd((y - X).astype(float), axis=0).reshape(-1, 1)
        err = []
        preds = []
        w = np.ones([1, X.shape[1]]) / X.shape[1]
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp = inp.reshape(-1, 1)
            mask = ~np.isnan(inp)
            pred = np.matmul(w[mask.T] / np.sum(w[mask.T]), inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p = np.exp(-1 * np.square((targ - inp[mask].T) / (np.sqrt(2 * np.pi) * sigma[mask])))
            w[mask.T] = ((w[mask.T] * p) / np.sum(w[mask.T] * p))

            count += 1
        return w

    def train(self):
        if len(self.combine_methods) > 1:
            pred_cluster, predictions, y_pred = self.project.predict_clusters()

            self.combine_methods = [method for method in self.combine_methods if method in predictions.keys()]
            self.models = dict()
            for method in self.combine_methods:
                pred = predictions[method].values.astype('float')
                pred[np.where(np.isnan(pred))] = 0

                cvs = []
                for _ in range(3):
                    X_train, X_test1, y_train, y_test1 = train_test_split(pred, y_pred.values, test_size=0.15)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
                    cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                mlp_model = sklearn_model(self.static_data, self.model_dir + '/' + method, self.rated, 'mlp',
                                          self.n_jobs, path_group=self.static_data['path_group'])
                if mlp_model.istrained == False:
                    self.models['mlp_' + method] = mlp_model.train(cvs)
                else:
                    self.models['mlp_' + method] = mlp_model.to_dict()
            combine_method = 'bcp'
            for method in self.combine_methods:
                self.models['bcp_' + method] = self.bcp_fit(predictions[method].values.astype('float'), y_pred.values)

        else:
            self.combine_methods = ['average']

        self.istrained = True
        self.save(self.model_dir)
        return 'Done'

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'temp_dir', 'cluster_lstm_dir', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        return dict

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'combine_models.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'combine_models.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RLS model')
        else:
            raise ImportError('Cannot find RLS model')

    def save(self, pathname):
        cluster_dir = pathname
        f = open(os.path.join(cluster_dir, 'combine_models.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
        f.close()
