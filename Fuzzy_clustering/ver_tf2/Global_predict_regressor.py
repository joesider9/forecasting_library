import os
import pickle
import numpy as np
import pandas as pd

from Fuzzy_clustering.ver_tf2.RBFNN_predict import rbf_model_predict
from Fuzzy_clustering.ver_tf2.RBF_ols_predict import rbf_ols_predict
from Fuzzy_clustering.ver_tf2.Sklearn_predict import sklearn_model_predict
from Fuzzy_clustering.ver_tf2.CNN_predict import CNN_predict
from Fuzzy_clustering.ver_tf2.CNN_predict_3d import CNN_3d_predict
from Fuzzy_clustering.ver_tf2.LSTM_predict_3d import LSTM_3d_predict
from Fuzzy_clustering.ver_tf2.Combine_module_predict import combine_model_predict
from joblib import Parallel, delayed
from datetime import datetime
import time, logging



class MultiEvaluator():
    def __init__(self, processes: int = 8):

        self.processes = processes

    def predict(self, i, x, model):
        return i, model.predict(x)

    def evaluate(self, X, model):
        partitions = 3000
        X_list=[]
        for i in range(0, X.shape[0], partitions):
            if (i+partitions+1)>X.shape[0]:
                X_list.append(X[i:])
            else:
                X_list.append(X[i:i+partitions])
        pred =Parallel(self.processes)(delayed(self.predict)(i, x, model) for i, x in enumerate(X_list))
        indices = np.array([p[0] for p in pred])
        predictions = np.array([])
        for ind in indices:
            if len(predictions.shape)==1:
                predictions = pred[ind][1]
            else:
                predictions = np.vstack((predictions, pred[ind][1]))
        return predictions




class global_predict(object):
    def __init__(self, static_data):
        self.istrained = False
        self.cluster_dir = os.path.join(static_data['path_model'], 'Global_regressor')
        try:
            self.load(self.cluster_dir)
        except:
            pass
        self.static_data=static_data
        self.model_type=static_data['type']
        self.methods=static_data['project_methods']
        self.combine_methods=static_data['combine_methods']
        self.rated=static_data['rated']
        self.n_jobs=static_data['njobs']
        self.data_dir = os.path.join(self.cluster_dir, 'data')

    def parallel_pred_model(self, X, method, static_data, rated, cluster_dir, X_cnn=np.array([]), X_lstm=np.array([])):
        parallel = MultiEvaluator(2*self.n_jobs)
        if method == 'ML_RBF_ALL':
            model_rbf = rbf_model_predict(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == True:
                pred1 = parallel.evaluate(X, model_rbf_ols)
            if model_rbf_ga.istrained == True:
                pred2 = parallel.evaluate(X, model_rbf_ga)
            if model_rbf.istrained == True:
                pred3 = parallel.evaluate(X, model_rbf)

            pred1[np.where(pred1 < 0)] = 0
            pred2[np.where(pred2 < 0)] = 0
            pred3[np.where(pred3 < 0)] = 0

            return [pred1, pred2, pred3]


        elif method == 'ML_RBF_ALL_CNN':
            model_rbf = rbf_model_predict(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == True:
                pred1 = parallel.evaluate(X, model_rbf_ols)
            if model_rbf_ga.istrained == True:
                pred2 = parallel.evaluate(X, model_rbf_ga)
            if model_rbf.istrained == True:
                pred3 = parallel.evaluate(X, model_rbf)

            rbf_dir = [model_rbf_ols.cluster_dir, model_rbf_ga.cluster_dir, model_rbf.cluster_dir]

            model_cnn = CNN_predict(static_data, rated, cluster_dir, rbf_dir)
            if model_cnn.istrained == True:
                pred4 = parallel.evaluate(X, model_cnn)
            pred1[np.where(pred1 < 0)] = 0
            pred2[np.where(pred2 < 0)] = 0
            pred3[np.where(pred3 < 0)] = 0
            pred4[np.where(pred4 < 0)] = 0
            return [pred1, pred2, pred3, pred4]

        elif method == 'ML_NUSVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = parallel.evaluate(X, model_sklearn)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_MLP':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = parallel.evaluate(X, model_sklearn)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_SVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = parallel.evaluate(X, model_sklearn)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_RF':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = parallel.evaluate(X, model_sklearn)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_XGB':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = parallel.evaluate(X, model_sklearn)
            pred[np.where(pred < 0)] = 0
            return [pred]
        elif method == 'ML_CNN_3d':
            model_cnn3d = CNN_3d_predict(static_data, rated, cluster_dir)
            if model_cnn3d.istrained == True:
                pred = parallel.evaluate(X_cnn, model_cnn3d)
            pred[np.where(pred < 0)] = 0
            return [pred]
        elif method == 'ML_LSTM_3d':
            model_lstm_3d = LSTM_3d_predict(static_data, rated, cluster_dir)
            if model_lstm_3d.istrained == True:
                pred = parallel.evaluate(X_lstm, model_lstm_3d)
            pred[np.where(pred < 0)] = 0
            return [pred]

        else:
            return [np.nan]

    def spark_predict(self, X, X_cnn=np.array([]), X_lstm=np.array([]), fs_reduced=False):

        if hasattr(self, 'features') and fs_reduced == False:
            X = X[:, self.features]
        methods = []
        for model in self.static_data['project_methods'].keys():
            if self.static_data['project_methods'][model]['Global'] == True:
                methods.append(model)

        predictions = dict()

        for method in methods:
            pred = self.parallel_pred_model(X, method, self.static_data, self.rated, self.cluster_dir, X_cnn=X_cnn,
                                   X_lstm=X_lstm)

            for p in pred:
                if np.any(np.isnan(p)):
                    raise ValueError('There are nans in dataset of Global models of method %s is not trained well', method)

            if method == 'ML_RBF_ALL_CNN':
                predictions['RBF_OLS'] = 20 * pred[0]
                predictions['GA_RBF_OLS'] = 20 * pred[1]
                predictions['RBFNN'] = 20 * pred[2]
                predictions['RBF-CNN'] = 20 * pred[3]
            elif method == 'ML_RBF_ALL':
                predictions['RBF_OLS'] = 20 * pred[0]
                predictions['GA_RBF_OLS'] = 20 * pred[1]
                predictions['RBFNN'] = 20 * pred[2]
            else:
                predictions[method] = 20 * pred[0]

        comb_model = combine_model_predict(self.static_data, self.cluster_dir, is_global=True)
        if comb_model.istrained==True and len(predictions.keys())>1:
            pred_combine = comb_model.predict(predictions)
            predictions.update(pred_combine)

        return predictions

    def pred_model(self, X, method, static_data, rated, cluster_dir, X_cnn=np.array([]), X_lstm=np.array([])):

        if method == 'ML_RBF_ALL':
            model_rbf = rbf_model_predict(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == True:
                pred1 = model_rbf_ols.predict(X)
            if model_rbf_ga.istrained == True:
                pred2 = model_rbf_ga.predict(X)
            if model_rbf.istrained == True:
                pred3 = model_rbf.predict(X)
            pred1[np.where(pred1 < 0)] = 0
            pred2[np.where(pred2 < 0)] = 0
            pred3[np.where(pred3 < 0)] = 0
            return [pred1, pred2, pred3]


        elif method == 'ML_RBF_ALL_CNN':
            model_rbf = rbf_model_predict(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_predict(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == True:
                pred1 = model_rbf_ols.predict(X)
            if model_rbf_ga.istrained == True:
                pred2 = model_rbf_ga.predict(X)
            if model_rbf.istrained == True:
                pred3 = model_rbf.predict(X)

            rbf_dir = [model_rbf_ols.cluster_dir, model_rbf_ga.cluster_dir, model_rbf.cluster_dir]

            model_cnn = CNN_predict(static_data, rated, cluster_dir, rbf_dir)
            if model_cnn.istrained == True:
                pred4 = model_cnn.predict(X)
            pred1[np.where(pred1 < 0)] = 0
            pred2[np.where(pred2 < 0)] = 0
            pred3[np.where(pred3 < 0)] = 0
            pred4[np.where(pred4 < 0)] = 0
            return [pred1, pred2, pred3, pred4]

        elif method == 'ML_NUSVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = model_sklearn.predict(X)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_MLP':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = model_sklearn.predict(X)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_SVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = model_sklearn.predict(X)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_RF':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = model_sklearn.predict(X)
            pred[np.where(pred < 0)] = 0
            return [pred]

        elif method == 'ML_XGB':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_predict(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == True:
                pred = model_sklearn.predict(X)
            pred[np.where(pred < 0)] = 0
            return [pred]
        elif method == 'ML_CNN_3d':
            model_cnn3d = CNN_3d_predict(static_data, rated, cluster_dir)
            if model_cnn3d.istrained == True:
                pred = model_cnn3d.predict(X_cnn)
            pred[np.where(pred < 0)] = 0
            return [pred]
        elif method == 'ML_LSTM_3d':
            model_lstm_3d = LSTM_3d_predict(static_data, rated, cluster_dir)
            if model_lstm_3d.istrained == True:
                pred = model_lstm_3d.predict(X_lstm)
            pred[np.where(pred < 0)] = 0
            return [pred]

        else:
            return [np.nan]


    def predict(self, X, X_cnn=np.array([]), X_lstm=np.array([]), fs_reduced=False):

        self.load(self.cluster_dir)
        if hasattr(self, 'features') and fs_reduced==False:
            X = X[:, self.features]
        methods = []
        for model in self.static_data['project_methods'].keys():
            if self.static_data['project_methods'][model]['Global'] == True:
                methods.append(model)

        predictions=dict()

        for method in methods:
            pred = self.pred_model(X, method, self.static_data,  self.rated, self.cluster_dir, X_cnn=X_cnn, X_lstm=X_lstm)

            for p in pred:
                if np.any(np.isnan(p)):
                    raise ValueError('There are nans in dataset of Global models of method %s is not trained well', method)

            if method == 'ML_RBF_ALL_CNN':
                predictions['RBF_OLS'] = 20 * pred[0]
                predictions['GA_RBF_OLS'] = 20 * pred[1]
                predictions['RBFNN'] = 20 * pred[2]
                predictions['RBF-CNN'] = 20 * pred[3]
            elif method == 'ML_RBF_ALL':
                predictions['RBF_OLS'] = 20 * pred[0]
                predictions['GA_RBF_OLS'] = 20 * pred[1]
                predictions['RBFNN'] = 20 * pred[2]
            else:
                predictions[method] = 20 * pred[0]
        comb_model = combine_model_predict(self.static_data, self.cluster_dir, is_global=True)
        if comb_model.istrained == True and len(predictions.keys())>1:
            pred_combine = comb_model.predict(predictions)
            predictions.update(pred_combine)
        elif len(predictions.keys())>1:
            pred_combine = comb_model.averaged(predictions)
            predictions.update(pred_combine)

        return predictions

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 20
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def evaluate(self, pred_all, y, y_cnn=None):
        result = pd.DataFrame(index=[method for method in pred_all.keys()], columns=['sse', 'rms', 'mae', 'mse'])
        for method, pred in pred_all.items():
            if method == 'ML_CNN_3d' and not y_cnn is None:
                result.loc[method] = self.compute_metrics(pred, y_cnn, self.rated)
            else:
                result.loc[method] = self.compute_metrics(pred, y, self.rated)
        return result

    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'Global_models.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'Global_models.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict = {}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open Global models')
        else:
            raise ImportError('Cannot find Global models')