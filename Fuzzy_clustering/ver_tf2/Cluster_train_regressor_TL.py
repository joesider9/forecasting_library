import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous

from Fuzzy_clustering.ver_tf2.RBFNN_module import rbf_model
from Fuzzy_clustering.ver_tf2.RBF_ols import rbf_ols_module
from Fuzzy_clustering.ver_tf2.CNN_module import cnn_model
from Fuzzy_clustering.ver_tf2.CNN_module_3d import cnn_3d_model
from Fuzzy_clustering.ver_tf2.Cluster_predict_regressors import cluster_predict
from Fuzzy_clustering.ver_tf2.Combine_module_train import combine_model
from Fuzzy_clustering.ver_tf2.LSTM_module_3d import lstm_3d_model
from Fuzzy_clustering.ver_tf2.RBFNN_module import rbf_model
from Fuzzy_clustering.ver_tf2.RBF_ols import rbf_ols_module
from Fuzzy_clustering.ver_tf2.Sklearn_models_TL import sklearn_model_tl
from Fuzzy_clustering.ver_tf2.Combine_module_train import combine_model
from Fuzzy_clustering.ver_tf2.Clusterer import clusterer
from Fuzzy_clustering.ver_tf2.Cluster_predict_regressors import cluster_predict
from sklearn.model_selection import train_test_split
from datetime import datetime
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
import time, logging, warnings, joblib





class cluster_train_tl(object):
    def __init__(self, static_data, clust, x_scaler, gpu):
        self.istrained = False
        self.cluster_dir = os.path.join(static_data['path_model'], 'Regressor_layer/' + clust)
        self.cluster_name = clust
        try:
            self.load(self.cluster_dir)
        except:
            pass
        self.static_data=static_data
        self.x_scaler = x_scaler
        self.model_type=static_data['type']
        self.methods=static_data['project_methods']
        self.combine_methods=static_data['combine_methods']
        self.rated=static_data['rated']
        self.n_jobs=static_data['njobs']
        self.gpu = gpu

        self.data_dir = os.path.join(self.cluster_dir, 'data')

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


    def split_test_data(self, X, y, act, X_cnn=np.array([]), X_lstm=np.array([]), test_indices =None):
        self.N_tot, self.D = X.shape
        if not test_indices is None:
            X_test = X.loc[test_indices['dates_test']]
            y_test = y.loc[test_indices['dates_test']]
            act_test = act.loc[test_indices['dates_test']]

            X = X.loc[test_indices['dates_train']]
            y = y.loc[test_indices['dates_train']]
            act = act.loc[test_indices['dates_train']]

            if len(X_cnn.shape) > 1:
                X_cnn_test = X_cnn[test_indices['indices_test']]
                X_cnn = X_cnn[test_indices['indices_train']]
            else:
                X_cnn_test = np.array([])

            if len(X_lstm.shape) > 1:
                X_lstm_test = X_lstm[test_indices['indices_test']]
                X_lstm = X_lstm[test_indices['indices_train']]
            else:
                X_lstm_test = np.array([])
        else:
            X_test = pd.DataFrame([])
            y_test = pd.DataFrame([])
            act_test = pd.DataFrame([])
            X_cnn_test = np.array([])
            X_lstm_test = np.array([])

        self.N_test = X_test.shape[0]
        return X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test

    def split_dataset(self, X, y, act, X_cnn=np.array([]), X_lstm=np.array([])):
        if len(y.shape)>1:
            y=y.ravel()
        if len(act.shape)>1:
            act=act.ravel()
        self.N_tot, self.D = X.shape

        #TODO in version 3 integrate activation act to self.cvs

        X_train, X_test1, y_train, y_test1, mask_test1 = split_continuous(X, y, test_size=0.15, random_state=42, mask=False)

        cvs = []
        for _ in range(3):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
            cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])

        self.N_train = cvs[0][0].shape[0]
        self.N_val = cvs[0][2].shape[0] + cvs[0][4].shape[0]

        return cvs, mask_test1, X, y, act, X_cnn, X_lstm


    def load_data(self):
        data_path = self.data_dir
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        act = pd.read_csv(os.path.join(data_path, 'dataset_act.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            if X_cnn.shape[1] == 6:
                X_cnn = X_cnn.transpose([0, 2, 3, 1])
        else:
            X_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
        else:
            X_lstm = np.array([])
        if os.path.exists(os.path.join(self.data_dir, 'test_indices.pickle')):
            test_indices = joblib.load(os.path.join(self.data_dir, 'test_indices.pickle'))
        else:
            test_indices = None

        return X, y, act, X_cnn, X_lstm, test_indices

    def fit(self, rule_model=None):
        if not self.istrained:
            X, y, act, X_cnn, X_lstm, test_indices = self.load_data()
            self.variables = X.columns
            indices = X.index
            X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test = self.split_test_data(X, y,
                                                                                                               act,
                                                                                                               X_cnn=X_cnn,
                                                                                                               X_lstm=X_lstm,
                                                                                                               test_indices=test_indices)
            X = X.values
            y = y.values / 20
            act = act.values
            #

            if len(y.shape)==1:
                y = y[:, np.newaxis]
            if len(act.shape)==1:
                act = act[:, np.newaxis]

            if not 'features' in rule_model.keys():
                raise ValueError('the Main rule has not attribute features %s', self.cluster_name)
            self.features = rule_model['features']
            cvs, mask_test1, X, y, act, X_cnn, X_lstm = self.split_dataset(X, y, act, X_cnn, X_lstm)
            self.indices = indices[:X.shape[0]]
            for i in range(3):
                cvs[i][0] = cvs[i][0][:, self.features]
                cvs[i][2] = cvs[i][2][:, self.features]
                cvs[i][4] = cvs[i][4][:, self.features]

            self.models = dict()
            for method in self.static_data['project_methods'].keys():
                if self.static_data['project_methods'][method]['status'] == 'train':

                    self.fit_model(cvs, method, self.static_data, self.cluster_dir, rule_model['models'], self.gpu, X_cnn=X_cnn, X_lstm=X_lstm, y=y, rated=1)

            comb_model = combine_model(self.static_data, self.cluster_dir, x_scaler=self.x_scaler)
            if comb_model.istrained == False and X_test.shape[0] > 0:
                comb_model.train(X_test, y_test, act_test, X_cnn_test, X_lstm_test)

                predict_module = cluster_predict(self.static_data, self.cluster_name)
                predictions = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm= X_lstm_test)
                result = predict_module.evaluate(predictions, y_test.values)
                result.to_csv(os.path.join(self.data_dir, 'result_test.csv'))

            self.istrained = True
            self.save(self.cluster_dir)

        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        return dict

    def fit_model(self, cvs, method, static_data, cluster_dir, models, gpu, X_cnn=np.array([]), X_lstm=np.array([]), y=np.array([]), rated=1):

        if method == 'ML_RBF_ALL':
            model_rbf = rbf_model(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == False or static_data['train_online'] == True:
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf_TL(cvs, models['RBF_OLS'])
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False or static_data['train_online'] == True:
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf_TL(cvs, models['GA_RBF_OLS'])
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            if model_rbf.istrained == False or static_data['train_online'] == True:
                self.models['RBFNN'] = model_rbf.rbf_train_TL(cvs, models['RBFNN'], gpu)
            else:
                self.models['RBFNN'] = model_rbf.to_dict()

        elif method == 'ML_RBF_ALL_CNN':
            model_rbf = rbf_model(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == False or static_data['train_online'] == True:
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf_TL(cvs, models['RBF_OLS'])
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False or static_data['train_online'] == True:
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf_TL(cvs, models['GA_RBF_OLS'])
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            if model_rbf.istrained == False or static_data['train_online'] == True:
                self.models['RBFNN'] = model_rbf.rbf_train_TL(cvs, models['RBFNN'], gpu)
            else:
                self.models['RBFNN'] = model_rbf.to_dict()

            rbf_dir = [model_rbf_ols.cluster_dir, model_rbf_ga.cluster_dir, model_rbf.cluster_dir]

            model_cnn = cnn_model(static_data, rated, cluster_dir, rbf_dir)
            if model_cnn.istrained == False or static_data['train_online'] == True:
                self.models['RBF-CNN'] = model_cnn.train_cnn_TL(cvs, models['RBF-CNN'], gpu)
            else:
                self.models['RBF-CNN'] = model_cnn.to_dict()
        elif method == 'ML_NUSVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_tl(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == False or static_data['train_online'] == True:
                self.models['NUSVM'] = model_sklearn.train(cvs, models['NUSVM']['best_params'])
            else:
                self.models['NUSVM'] = model_sklearn.to_dict()
        elif method == 'ML_MLP':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_tl(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == False or static_data['train_online'] == True:
                self.models['MLP'] = model_sklearn.train(cvs, models['MLP']['best_params'])

            else:
                self.models['MLP'] = model_sklearn.to_dict()
        elif method == 'ML_SVM':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_tl(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == False or static_data['train_online'] == True:
                self.models['SVM'] = model_sklearn.train(cvs, models['SVM']['best_params'])
            else:
                self.models['SVM'] = model_sklearn.to_dict()
        elif method == 'ML_RF':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_tl(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == False or static_data['train_online'] == True:
                self.models['RF'] = model_sklearn.train(cvs, models['RF']['best_params'])
            else:
                self.models['RF'] = model_sklearn.to_dict()
        elif method == 'ML_XGB':
            method = method.replace('ML_', '')
            model_sklearn = sklearn_model_tl(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained == False or static_data['train_online'] == True:
                self.models['XGB'] = model_sklearn.train(cvs, models['XGB']['best_params'])
            else:
                self.models['XGB'] = model_sklearn.to_dict()
        elif method == 'ML_CNN_3d':
            cnn_model_3d = cnn_3d_model(static_data, rated, cluster_dir)
            if cnn_model_3d.istrained == False or static_data['train_online'] == True:
                self.models['CNN_3d'] = cnn_model_3d.train_cnn_TL(X_cnn, y, models['CNN_3d'], gpu)
            else:
                self.models['CNN_3d'] = cnn_model_3d.to_dict()
        elif method == 'ML_LSTM_3d':
            lstm_model_3d = lstm_3d_model(static_data, rated, cluster_dir)
            if lstm_model_3d.istrained == False or static_data['train_online'] == True:
                self.models['LSTM_3d'] = lstm_model_3d.train_lstm_TL(X_lstm, y, models['LSTM_3d'], gpu)
            else:
                self.models['LSTM_3d'] = lstm_model_3d.to_dict()
        self.save(self.cluster_dir)

    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                dict={}
                for k in tmp_dict.keys():
                    if k in ['logger', 'static_data']:
                        dict[k] = tmp_dict[k]
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open rule model %s', self.cluster_name)
        else:
            raise ImportError('Cannot find rule model %s', self.cluster_name)


    def save(self, pathname):
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        f = open(os.path.join(pathname, 'model_' + self.cluster_name +'.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()





