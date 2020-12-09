import os
import pickle, glob, shutil
import numpy as np
import pandas as pd

from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous

from Fuzzy_clustering.ver_tf2.RBFNN_module import rbf_model
from Fuzzy_clustering.ver_tf2.RBF_ols import rbf_ols_module
from Fuzzy_clustering.ver_tf2.CNN_module import cnn_model
from Fuzzy_clustering.ver_tf2.CNN_module_3d import cnn_3d_model
from Fuzzy_clustering.ver_tf2.LSTM_module_3d import lstm_3d_model
from Fuzzy_clustering.ver_tf2.Combine_module_train import combine_model
from Fuzzy_clustering.ver_tf2.Clusterer import clusterer
from Fuzzy_clustering.ver_tf2.Global_predict_regressor import global_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
from Fuzzy_clustering.ver_tf2.imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, SMOTE,ADASYN
import time, logging, warnings, joblib


class global_train(object):
    def __init__(self, static_data, x_scaler):
        self.istrained = False
        self.cluster_dir=os.path.join(static_data['path_model'], 'Global_regressor')
        try:
            self.load(self.cluster_dir)
        except:
            pass
        self.static_data=static_data
        self.model_type=static_data['type']
        self.x_scaler = x_scaler
        self.methods=static_data['project_methods']
        self.combine_methods=static_data['combine_methods']
        self.rated=static_data['rated']
        self.n_jobs=static_data['njobs']
        self.var_lin = static_data['clustering']['var_lin']
        self.cluster_dir=os.path.join(static_data['path_model'], 'Global_regressor')
        self.data_dir = os.path.join(self.cluster_dir, 'data')

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        logger = logging.getLogger('Glob_train_procedure' + '_' +self.model_type)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.cluster_dir, 'log_train_procedure.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def move_files(self, path1, path2):
        for filename in glob.glob(os.path.join(path1, '*.*')):
            shutil.copy(filename, path2)

    def split_dataset(self, X, y, act, X_cnn=np.array([]), X_lstm=np.array([])):
        if len(y.shape)>1:
            y=y.ravel()
        if len(act.shape)>1:
            act=act.ravel()
        self.N_tot, self.D = X.shape


        X_train, X_test1, y_train, y_test1, mask_test1 = split_continuous(X, y, test_size=0.15, random_state=42, mask=False)

        cvs = []
        for _ in range(3):
            X_train1 = np.copy(X_train)
            y_train1 = np.copy(y_train)
            X_train1, X_val, y_train1, y_val = train_test_split(X_train1, y_train1, test_size=0.15)
            cvs.append([X_train1, y_train1, X_val, y_val, X_test1, y_test1])

        self.N_train = cvs[0][0].shape[0]
        self.N_val = cvs[0][2].shape[0] + cvs[0][4].shape[0]

        return cvs, mask_test1, X, y, act, X_cnn, X_lstm


    def find_features(self, cvs, method, njobs):

        if method=='boruta':
            from Fuzzy_clustering.ver_tf2.Feature_selection_boruta import FS
        else:
            from Fuzzy_clustering.ver_tf2.Feature_selection_permutation import FS

        fs=FS(self.cluster_dir, 2*njobs)
        self.features=fs.fit(cvs)

        self.save(self.cluster_dir)

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

    def load_data(self):
        data_path = self.data_dir
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        act = pd.read_csv(os.path.join(data_path, 'dataset_act.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            if X_cnn.shape[1]==6:
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

    def fit(self):

        self.logger.info('Start training Global models')
        self.logger.info('/n')
        X, y, act, X_cnn, X_lstm, test_indices = self.load_data()
        self.variables = X.columns
        indices = X.index
        X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test = self.split_test_data(X, y,
                                                                                                           act,
                                                                                                           X_cnn=X_cnn,
                                                                                                           X_lstm=X_lstm,
                                                                                                           test_indices=test_indices)

        if X_test.shape[0]>0:
            lin_models = LinearRegression().fit(X[self.var_lin].values, y.values.ravel())
            preds = lin_models.predict(X_test[self.var_lin].values).ravel()

            err = (preds - y_test.values.ravel()) / 20

            rms = np.sum(np.square(err))
            mae = np.mean(np.abs(err))
            print('rms = %s', rms)
            print('mae = %s', mae)
            self.logger.info("Objective from linear models: %s", mae)

        X = X.values
        y = y.values / 20
        act = act.values


        if len(y.shape)==1:
            y = y[:, np.newaxis]
        if len(act.shape)==1:
            act = act[:, np.newaxis]


        try:
            self.load(self.cluster_dir)
        except:
            pass



        if hasattr(self, 'features') and self.static_data['train_online'] == False:
            pass
        else:
            if self.static_data['sklearn']['fs_status'] != 'ok':
                X_train, X_test1, y_train, y_test1 = split_continuous(X, y, test_size=0.15, random_state=42)

                cvs = []
                for _ in range(3):
                    X_train1 = np.copy(X_train)
                    y_train1 = np.copy(y_train)
                    X_train1, X_val, y_train1, y_val = train_test_split(X_train1, y_train1, test_size=0.15)
                    cvs.append([X_train1, y_train1, X_val, y_val, X_test1, y_test1])
                self.find_features(cvs, self.static_data['sklearn']['fs_method'], self.static_data['sklearn']['njobs'])

        cvs, mask_test1, X, y, act, X_cnn, X_lstm = self.split_dataset(X, y, act, X_cnn, X_lstm)
        self.indices = indices[:X.shape[0]]
        for i in range(3):
            cvs[i][0] = cvs[i][0][:, self.features]
            cvs[i][2] = cvs[i][2][:, self.features]
            cvs[i][4] = cvs[i][4][:, self.features]

        self.logger.info('Data info for Global models')
        self.logger.info('Number of variables %s', str(self.D))
        self.logger.info('Number of total samples %s', str(self.N_tot))
        self.logger.info('Number of training samples %s', str(self.N_train))
        self.logger.info('Number of validation samples %s', str(self.N_val))
        self.logger.info('Number of testing samples %s', str(self.N_test))
        self.logger.info('/n')

        self.models = dict()
        for method in self.static_data['project_methods'].keys():
            if self.static_data['project_methods'][method]['Global'] == True:
                self.logger.info('Training start of method %s', method)
                self.logger.info('/n')
                if 'sklearn_method' in self.static_data['project_methods'][method].keys():
                    optimize_method = self.static_data['project_methods'][method]['sklearn_method']
                else:
                    optimize_method = []
                self.fit_model(cvs, method, self.static_data, self.cluster_dir, optimize_method, X_cnn=X_cnn, X_lstm=X_lstm, y=y, rated=1)
                self.logger.info('Training end of method %s', method)

        comb_model = combine_model(self.static_data, self.cluster_dir, x_scaler=self.x_scaler,is_global=True)
        if comb_model.istrained == False and X_test.shape[0] > 0:
            comb_model.train(X_test, y_test, act_test, X_cnn_test, X_lstm_test)

            predict_module = global_predict(self.static_data)
            predictions = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm= X_lstm_test)
            result = predict_module.evaluate(predictions, y_test.values)
            result.to_csv(os.path.join(self.data_dir, 'result_test.csv'))

        self.logger.info('Training end for Global models')
        self.logger.info('/n')

        self.istrained = True
        self.save(self.cluster_dir)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        return dict

    def fit_model(self, cvs, method, static_data, cluster_dir, optimize_method, X_cnn=np.array([]), X_lstm=np.array([]), y=np.array([]), rated=1):
        # deap, optuna, skopt, grid_search
        if optimize_method=='deap':
            from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import sklearn_model
        elif optimize_method=='optuna':
            from Fuzzy_clustering.ver_tf2.Sklearn_models_optuna import sklearn_model
        elif optimize_method=='skopt':
            from Fuzzy_clustering.ver_tf2.Sklearn_models_skopt import sklearn_model
        else:
            from Fuzzy_clustering.ver_tf2.SKlearn_models import sklearn_model
        # if (datetime.now().hour>=8 and datetime.now().hour<10):
        #     time.sleep(2*60*60)
        if method == 'ML_RBF_ALL':
            model_rbf = rbf_model(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_ols')
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf(cvs)
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_ga')
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf(cvs)
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            if model_rbf.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_adam')
                self.models['RBFNN'] = model_rbf.rbf_train(cvs)
            else:
                self.models['RBFNN'] = model_rbf.to_dict()

        elif method == 'ML_RBF_ALL_CNN':
            model_rbf = rbf_model(static_data['RBF'], rated, cluster_dir)
            model_rbf_ols = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(cluster_dir, rated, static_data['sklearn']['njobs'], GA=True)

            if model_rbf_ols.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_ols')
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf(cvs)
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_ga')
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf(cvs)
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            if model_rbf.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of model_rbf_adam')
                self.models['RBFNN'] = model_rbf.rbf_train(cvs)
            else:
                self.models['RBFNN'] = model_rbf.to_dict()

            rbf_dir = [model_rbf_ols.cluster_dir, model_rbf_ga.cluster_dir, model_rbf.cluster_dir]

            model_cnn = cnn_model(static_data, rated, cluster_dir, rbf_dir)
            if model_cnn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of model_cnn')
                self.models['RBF-CNN'] = model_cnn.train_cnn(cvs)
            else:
                self.models['RBF-CNN'] = model_cnn.to_dict()

        elif method == 'ML_NUSVM':
            method =method.replace('ML_','')
            model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of NUSVM')
                self.models['NUSVM'] = model_sklearn.train(cvs)
            else:
                self.models['NUSVM'] = model_sklearn.to_dict()
        elif method == 'ML_MLP':
            method = method.replace('ML_','')
            model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of MLP')
                self.models['MLP'] = model_sklearn.train(cvs)
            else:
                self.models['MLP'] = model_sklearn.to_dict()
        elif method == 'ML_SVM':
            method = method.replace('ML_','')
            model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of SVM')
                self.models['SVM'] = model_sklearn.train(cvs)
            else:
                self.models['SVM'] = model_sklearn.to_dict()
        elif method == 'ML_RF':
            method = method.replace('ML_','')
            model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of RF')
                self.models['RF'] = model_sklearn.train(cvs)
            else:
                self.models['RF'] = model_sklearn.to_dict()
        elif method == 'ML_XGB':
            method = method.replace('ML_','')
            model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
            if model_sklearn.istrained==False or static_data['train_online']==True:
                self.logger.info('Start of training of XGB')
                self.models['XGB'] = model_sklearn.train(cvs)
            else:
                self.models['XGB'] = model_sklearn.to_dict()
        elif method == 'ML_CNN_3d':
            cnn_model_3d = cnn_3d_model(static_data, rated, cluster_dir)
            if cnn_model_3d.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of CNN_3d')
                self.models['CNN_3d'] = cnn_model_3d.train_cnn(X_cnn, y)
            else:
                self.models['CNN_3d'] = cnn_model_3d.to_dict()
        elif method == 'ML_LSTM_3d':
            lstm_model_3d = lstm_3d_model(static_data, rated, cluster_dir)
            if lstm_model_3d.istrained == False or static_data['train_online'] == True:
                self.logger.info('Start of training of LSTM_3d')
                self.models['LSTM_3d'] = lstm_model_3d.train_lstm(X_lstm, y)
            else:
                self.models['LSTM_3d'] = lstm_model_3d.to_dict()

        self.save(self.cluster_dir)

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


    def save(self, pathname):
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        f = open(os.path.join(pathname, 'Global_models.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()





