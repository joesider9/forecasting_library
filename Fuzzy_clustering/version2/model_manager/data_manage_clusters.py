import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from Fuzzy_clustering.version2.common_utils.utils_for_forecast import split_continuous


class ClusterObject:
    def __init__(self, static_data, clust):
        self.istrained = False
        self.cluster_name = clust
        self.cluster_dir = os.path.join(static_data['path_model'], 'Regressor_layer/' + clust)
        self.static_data = static_data
        self.model_type = static_data['type']

        self.rated = static_data['rated']
        self.n_jobs = static_data['njobs']
        self.var_lin = static_data['clustering']['var_lin']
        self.data_dir = os.path.join(self.cluster_dir, 'data')
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']

        try:
            self.load(self.cluster_dir)
        except:
            pass
        self.methods = [method for method in static_data['project_methods'].keys() if
                        static_data['project_methods'][method] == True]
        self.combine_methods = static_data['combine_methods']
        if not os.path.exists(self.cluster_dir):
            os.makedirs(self.cluster_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_data(self):
        data_path = self.data_dir
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
        act = pd.read_csv(os.path.join(data_path, 'dataset_act.csv'), index_col=0, header=0, parse_dates=True,
                          dayfirst=True)
        cvs = joblib.load(os.path.join(data_path, 'cvs_full.pickle'))

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            cvs_cnn = joblib.load(os.path.join(self.data_dir, 'cvs_cnn.pickle'))
        else:
            X_cnn = np.array([])
            cvs_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
            cvs_lstm = joblib.load(os.path.join(self.data_dir, 'cvs_lstm.pickle'))
        else:
            X_lstm = np.array([])
            cvs_lstm = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_X_test.csv')):
            X_test = pd.read_csv(os.path.join(data_path, 'dataset_X_test.csv'), index_col=0, header=0, parse_dates=True,
                                 dayfirst=True)
            y_test = pd.read_csv(os.path.join(data_path, 'dataset_y_test.csv'), index_col=0, header=0, parse_dates=True,
                                 dayfirst=True)
            act_test = pd.read_csv(os.path.join(data_path, 'dataset_act_test.csv'), index_col=0, header=0,
                                   parse_dates=True,
                                   dayfirst=True)

            if os.path.exists(os.path.join(data_path, 'dataset_cnn_test.pickle')):
                X_cnn_test = joblib.load(os.path.join(data_path, 'dataset_cnn_test.pickle'))
            else:
                X_cnn_test = np.array([])

            if os.path.exists(os.path.join(data_path, 'dataset_lstm_test.pickle')):
                X_lstm_test = joblib.load(os.path.join(data_path, 'dataset_lstm_test.pickle'))
            else:
                X_lstm_test = np.array([])
        else:
            X_test = pd.DataFrame([])
            y_test = pd.DataFrame([])
            act_test = pd.DataFrame([])
            X_cnn_test = np.array([])
            X_lstm_test = np.array([])

        return cvs, cvs_cnn, cvs_lstm, X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test

    def split_dataset(self, x, y, act):
        x = x.values
        y = y.values
        act = act.values
        if len(y.shape) > 1:
            y = y.ravel()
        if len(act.shape) > 1:
            act = act.ravel()
        self.N_tot, self.D = x.shape

        # TODO in version 4 integrate activation act to self.cvs

        cvs = []
        for _ in range(3):
            X_train, X_test, y_train, y_test = split_continuous(x, y, test_size=0.15,
                                                                random_state=np.random.randint(100))
            X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.15)
            cvs.append([X_train1, y_train1, X_val, y_val, X_test, y_test])

        self.N_train = cvs[0][0].shape[0]
        self.N_val = cvs[0][2].shape[0] + cvs[0][4].shape[0]

        return cvs

    def save_cluster_data(self, X1, y1, X_cnn1, X_lstm1, activations=None, split_test=None):
        self.N_tot, self.D = X1.shape
        if not os.path.exists(os.path.join(self.data_dir, 'dataset_X.csv')) or self.static_data['recreate_datasets']:
            if activations is None:
                nind = np.arange(X1.shape[0])
                act = pd.DataFrame(1, index=X1.index, columns=[self.cluster_name])
            else:
                nind = np.where(activations[self.cluster_name] >= self.thres_act)[0]
                act = activations[self.cluster_name].iloc[nind]

            X = X1.iloc[nind]
            y = y1.iloc[nind]

            if len(X_cnn1.shape) > 1:
                X_cnn = X_cnn1[nind]
            else:
                X_cnn = X_cnn1

            if len(X_lstm1.shape) > 1:
                X_lstm = X_lstm1[nind]
            else:
                X_lstm = X_lstm1

            if not split_test is None:

                test_ind = np.where(X.index >= split_test)[0]
                indices_test = test_ind

                X_test = X.iloc[test_ind]
                y_test = y.iloc[test_ind]
                act_test = act.iloc[test_ind]

                if len(X_cnn.shape) > 1:
                    X_cnn_test = X_cnn[indices_test]
                else:
                    X_cnn_test = np.array([])

                if len(X_lstm.shape) > 1:
                    X_lstm_test = X_lstm[indices_test]
                else:
                    X_lstm_test = np.array([])
            else:
                X_test = pd.DataFrame([])
                y_test = pd.DataFrame([])
                act_test = pd.DataFrame([])
                X_cnn_test = np.array([])
                X_lstm_test = np.array([])

            self.N_test = X_test.shape[0]

            cvs = self.split_dataset(X, y, act)

            X.to_csv(os.path.join(self.data_dir, 'dataset_X.csv'))
            y.to_csv(os.path.join(self.data_dir, 'dataset_y.csv'))
            act.to_csv(os.path.join(self.data_dir, 'dataset_act.csv'))
            joblib.dump(cvs, os.path.join(self.data_dir, 'cvs_full.pickle'))

            if len(X_cnn.shape) > 1:
                joblib.dump(X_cnn, os.path.join(self.data_dir, 'dataset_cnn.pickle'))

                X_cnn_tr, X_cnn_ts, y_cnn_tr, y_cnn_ts = split_continuous(X_cnn, y.values, test_size=0.15,
                                                                          random_state=42)
                X_cnn_tr, X_cnn_val, y_cnn_tr, y_cnn_val = train_test_split(X_cnn_tr, y_cnn_tr, test_size=0.15,
                                                                            random_state=42)
                cvs_cnn = [X_cnn_tr, y_cnn_tr, X_cnn_val, y_cnn_val, X_cnn_ts, y_cnn_ts]
                joblib.dump(cvs_cnn, os.path.join(self.data_dir, 'cvs_cnn.pickle'))

            if len(X_lstm.shape) > 1:
                joblib.dump(X_lstm, os.path.join(self.data_dir, 'dataset_lstm.pickle'))
                X_lstm_tr, X_lstm_ts, y_lstm_tr, y_lstm_ts = split_continuous(X_lstm, y.values, test_size=0.15,
                                                                              random_state=42)
                X_lstm_tr, X_lstm_val, y_lstm_tr, y_lstm_val = train_test_split(X_lstm_tr, y_lstm_tr, test_size=0.15,
                                                                                random_state=42)
                cvs_lstm = [X_lstm_tr, y_lstm_tr, X_lstm_val, y_lstm_val, X_lstm_ts, y_lstm_ts]
                joblib.dump(cvs_lstm, os.path.join(self.data_dir, 'cvs_lstm.pickle'))

            if X_test.shape[0] > 0:
                X_test.to_csv(os.path.join(self.data_dir, 'dataset_X_test.csv'))
                y_test.to_csv(os.path.join(self.data_dir, 'dataset_y_test.csv'))
                act_test.to_csv(os.path.join(self.data_dir, 'dataset_act_test.csv'))
                if len(X_cnn_test.shape) > 1:
                    joblib.dump(X_cnn_test, os.path.join(self.data_dir, 'dataset_cnn_test.pickle'))
                if len(X_lstm_test.shape) > 1:
                    joblib.dump(X_lstm_test, os.path.join(self.data_dir, 'dataset_lstm_test.pickle'))

                lin_models = LinearRegression().fit(X[self.var_lin].values, y.values.ravel())
                preds = lin_models.predict(X_test[self.var_lin].values).ravel()

                err = (preds - y_test.values.ravel())

                self.lin_rms = np.sum(np.square(err))
                self.lin_mae = np.mean(np.abs(err))
                print('rms = %s', self.lin_rms)
                print('mae = %s', self.lin_mae)
        #         self.logger.info("Objective from linear models: %s", mae)
        # self.logger.info('Data info for cluster %s', self.cluster_name)
        # self.logger.info('Number of variables %s', str(self.D))
        # self.logger.info('Number of total samples %s', str(self.N_tot))
        # self.logger.info('Number of training samples %s', str(self.N_train))
        # self.logger.info('Number of validation samples %s', str(self.N_val))
        # self.logger.info('Number of testing samples %s', str(self.N_test))
        # self.logger.info('/n')
        self.save(self.cluster_dir)

        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        return dict

    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'model_' + self.cluster_name + '.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'model_' + self.cluster_name + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict = {}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open rule model %s', self.cluster_name)
        else:
            raise ImportError('Cannot find rule model %s', self.cluster_name)

    def save(self, pathname):
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        f = open(os.path.join(pathname, 'model_' + self.cluster_name + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
class cluster_object():
    def __init__(self, static_data, clust):
        self.istrained = False
        self.cluster_name = clust
        self.cluster_dir = os.path.join(static_data['path_model'], 'Regressor_layer/' + clust)
        self.static_data = static_data
        self.model_type = static_data['type']
        self.methods = [method for method in static_data['project_methods'].keys() if static_data['project_methods'][method]==True]
        self.combine_methods = static_data['combine_methods']
        self.rated = static_data['rated']
        self.n_jobs = static_data['njobs']
        self.var_lin = static_data['clustering']['var_lin']
        self.data_dir = os.path.join(self.cluster_dir, 'data')
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']

        try:
            self.load(self.cluster_dir)
        except:
            raise ImportError('Cannot find cluster ', clust)


    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict={}
                for k in tmp_dict.keys():
                    tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open rule model %s', self.cluster_name)
        else:
            raise ImportError('Cannot find rule model %s', self.cluster_name)


