import os
import numpy as np
import joblib
from Fuzzy_clustering.ver_tf2.Sklearn_predict import sklearn_model_predict

class combine_model_predict(object):
    def __init__(self, static_data, cluster_dir, is_global=False):
        self.istrained = False
        self.combine_methods = static_data['combine_methods']
        self.cluster_name = os.path.basename(cluster_dir)
        self.cluster_dir = cluster_dir
        self.model_dir = os.path.join(self.cluster_dir, 'Combine')
        try:
            self.load(self.model_dir)
        except:
            pass

        self.static_data = static_data
        self.model_type = static_data['type']
        self.methods = []
        if is_global:
            for method in static_data['project_methods'].keys():
                if self.static_data['project_methods'][method]['Global'] == True:
                    if method == 'ML_RBF_ALL_CNN':
                        self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                    elif method == 'ML_RBF_ALL':
                        self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
                    else:
                        self.methods.append(method)
        else:
            for method in static_data['project_methods'].keys():
                if static_data['project_methods'][method]['status'] == 'train':
                    if method == 'ML_RBF_ALL_CNN':
                        self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                    elif method == 'ML_RBF_ALL':
                        self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
                    else:
                        self.methods.append(method)
        self.rated = static_data['rated']

        self.n_jobs = 2 * static_data['njobs']

        self.data_dir = os.path.join(self.cluster_dir, 'data')
    def averaged(self,X):
        pred_combine = dict()
        X_pred = np.array([])
        self.best_methods = X.keys()
        for method in sorted(self.best_methods):
            if X_pred.shape[0]==0:
                X_pred = X[method]
            else:
                X_pred = np.hstack((X_pred, X[method]))
        if X_pred.shape[0]>0:
            pred = np.mean(X_pred, axis=1).reshape(-1, 1)

            if len(pred.shape)==1:
                pred = pred.reshape(-1,1)

            pred[np.where(pred<0)] = 0
            pred_combine['average'] = pred
        else:
            pred_combine['average'] = np.array([])

        return pred_combine

    def predict(self, X):
        pred_combine = dict()
        if len(self.methods) > 1:
            X_pred = np.array([])
            if not hasattr(self, 'best_methods'):
                self.best_methods = X.keys()
            for method in sorted(self.best_methods):
                if X_pred.shape[0]==0:
                    X_pred = X[method]
                else:
                    X_pred = np.hstack((X_pred, X[method]))
            X_pred /= 20
            if not hasattr(self, 'model'):
                raise ValueError('The combine models does not exist')

            for combine_method in self.combine_methods:
                if X_pred.shape[0]>0:
                    if combine_method == 'rls':
                        pred = np.matmul(self.model[combine_method]['w'], X_pred.T).T
                    elif combine_method == 'bcp':
                        pred =np.matmul(self.model[combine_method]['w'], X_pred.T).T

                    elif combine_method == 'mlp':
                        self.model[combine_method] = sklearn_model_predict(self.model_dir, self.rated, 'mlp', self.n_jobs)
                        pred = self.model[combine_method].predict(X_pred)

                    elif combine_method == 'bayesian_ridge':
                        pred = self.model[combine_method].predict(X_pred)

                    elif combine_method == 'elastic_net':
                        pred = self.model[combine_method].predict(X_pred)

                    elif combine_method == 'ridge':
                        pred = self.model[combine_method].predict(X_pred)

                    else:
                        pred = np.mean(X_pred, axis=1).reshape(-1, 1)

                    if len(pred.shape)==1:
                        pred = pred.reshape(-1,1)

                    pred[np.where(pred<0)] = 0
                    pred_combine[combine_method] = 20 * pred
                else:
                    pred_combine[combine_method] = np.array([])
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

