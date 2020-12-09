import logging
import os

import joblib
import numpy as np

from Fuzzy_clustering.version2.cluster_predict_manager.sklearn_combine_predict import sklearn_model_predict


class ClusterCombiner():
    def __init__(self, static_data, cluster):
        self.istrained = False
        self.cluster = cluster
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.model_dir = os.path.join(self.cluster_dir, 'Combine')
        self.static_data = static_data
        self.model_type = static_data['type']
        self.combine_methods = static_data['combine_methods']
        self.rated = static_data['rated']
        self.n_jobs = static_data['sklearn']['njobs']
        self.resampling = static_data['resampling']
        try:
            self.load(self.model_dir)
        except:
            pass
        self.methods = []
        for method in cluster.methods:
            if method == 'RBF_ALL_CNN':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
            elif method == 'RBF_ALL':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
            else:
                self.methods.append(method)
        self.data_dir = os.path.join(self.cluster_dir, 'data')

        logger = logging.getLogger('combine_' + self.cluster_name + '_' + static_data['_id'])
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_combine.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def predict(self, X):
        if self.istrained:
            pred_combine = dict()
            if len(self.methods) > 1:
                X_pred = np.array([])
                if not hasattr(self, 'best_methods'):
                    self.best_methods = [meth for meth in X.keys()]
                for method in sorted(self.best_methods):
                    if X_pred.shape[0] == 0:
                        X_pred = X[method]
                    else:
                        X_pred = np.hstack((X_pred, X[method]))
                if len(self.combine_methods) > 1:
                    if not hasattr(self, 'model'):
                        raise ValueError('The combine models does not exist')

                for combine_method in self.combine_methods:
                    if X_pred.shape[0] > 0:
                        if combine_method == 'rls':
                            pred = np.matmul(self.model[combine_method]['w'], X_pred.T).T
                        elif combine_method == 'bcp':
                            pred = np.matmul(self.model[combine_method]['w'], X_pred.T).T

                        elif combine_method == 'mlp':
                            self.model[combine_method] = sklearn_model_predict(self.static_data, self.model_dir,
                                                                               self.rated, 'mlp',
                                                                               self.n_jobs)
                            pred = self.model[combine_method].predict(X_pred)

                        elif combine_method == 'bayesian_ridge':
                            pred = self.model[combine_method].predict(X_pred)

                        elif combine_method == 'elastic_net':
                            pred = self.model[combine_method].predict(X_pred)

                        elif combine_method == 'ridge':
                            pred = self.model[combine_method].predict(X_pred)

                        else:
                            pred = np.mean(X_pred, axis=1).reshape(-1, 1)

                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)

                        pred[np.where(pred < 0)] = 0
                        pred_combine[combine_method] = pred
                    else:
                        pred_combine[combine_method] = np.array([])
        else:
            raise ValueError('combine model not trained for %s of %s', self.cluster_name, self.static_data['_id'])
        return pred_combine

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
