import os
import pickle
# for timing
from contextlib import contextmanager
from timeit import default_timer

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler

from Fuzzy_clustering.version2.cluster_predict_manager.cluster_predict_manager import ClusterPredict
from Fuzzy_clustering.version2.fuzzy_clustering_manager.clusterer_optimize_deep import Clusterer
from Fuzzy_clustering.version2.model_manager.data_manage_clusters import ClusterObject


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def create_cluster(static_data, cluster_name, activations, X1, y1, X_cnn, X_lstm, split_test):
    cluster = ClusterObject(static_data, cluster_name)
    if cluster.istrained == False:
        cluster.save_cluster_data(X1, y1, X_cnn, X_lstm, activations=activations, split_test=split_test)

    return cluster_name, cluster


class ModelTrainManager(object):

    def __init__(self, path_model):
        self.istrained = False
        self.path_model = path_model
        try:
            self.load()
        except:
            pass
        if hasattr(self, 'is_trained'):
            self.istrained = self.is_trained

    def init(self, static_data, data_variables):
        self.data_variables = data_variables
        self.static_data = static_data
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']

        x, y, x_cnn, x_lstm = self.load_data()
        if y.isna().any().values[0]:
            x = x.drop(y.index[np.where(y.isna())[0]])
            if len(x_cnn.shape) > 1:
                x_cnn = np.delete(x_cnn, np.where(y.isna())[0], axis=0)
            if len(x_lstm.shape) > 1:
                x_lstm = np.delete(x_lstm, np.where(y.isna())[0], axis=0)
            y = y.drop(y.index[np.where(y.isna())[0]])
        if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
            index = np.where(x['flux'] > 1e-8)[0]
            x = x.iloc[index]
            y = y.iloc[index]
            # x_cnn = x_cnn[index]
        sc = MinMaxScaler().fit(x.values)
        joblib.dump(sc, os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
        scale_y = MaxAbsScaler().fit(y.values)
        joblib.dump(scale_y, os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

    def load_data(self):
        data_path = self.static_data['path_data']
        x = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            x_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            x_cnn = x_cnn.transpose([0, 2, 3, 1])
        else:
            x_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            x_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
        else:
            x_lstm = np.array([])

        return x, y, x_cnn, x_lstm

    def split_test_data(self, X1, activations=None):
        if activations is None:
            indices = X1.index
            n_split = int(np.round(len(indices) * 0.7))
            split_test = indices[n_split + 1]
        else:
            split_test = None
            rate = 0.25
            while split_test is None and rate > 0.1:
                split_indices = []
                for clust in activations.columns:
                    indices = activations[clust].index[activations[clust] >= self.thres_act].tolist()
                    if len(indices) > 0:
                        n_split = int(np.round(len(indices) * (1 - rate)))
                        split_indices.append(indices[n_split + 1])
                split_test = pd.Series(split_indices).min()

                X_test = X1.loc[split_test:]
                if X_test.shape[0] > 0.4 * X1.shape[0]:
                    split_test = None
                    rate -= 0.025
        if split_test is None:
            split_test = X1.index[int(0.4 * X1.shape[0])]
        self.split_test = split_test
        self.save()
        return split_test

    def load_test_data(self):
        X, y, X_cnn, X_lstm = self.load_data()

        test_ind = np.where(X.index >= self.split_test)[0]
        indices_test = test_ind

        X_test = X.iloc[test_ind]
        y_test = y.iloc[test_ind]

        if len(X_cnn.shape) > 1:
            X_cnn_test = X_cnn[indices_test]
        else:
            X_cnn_test = np.array([])

        if len(X_lstm.shape) > 1:
            X_lstm_test = X_lstm[indices_test]
        else:
            X_lstm_test = np.array([])
        return X_test, y_test, X_cnn_test, X_lstm_test

    def create_clusters_and_data(self):

        X, y, X_cnn, X_lstm = self.load_data()
        if y.isna().any().values[0]:
            X = X.drop(y.index[np.where(y.isna())[0]])
            if len(X_cnn.shape) > 1:
                X_cnn = np.delete(X_cnn, np.where(y.isna())[0], axis=0)
            if len(X_lstm.shape) > 1:
                X_lstm = np.delete(X_lstm, np.where(y.isna())[0], axis=0)
            y = y.drop(y.index[np.where(y.isna())[0]])
        if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
            index = np.where(X['flux'] > 1e-8)[0]
            X = X.iloc[index]
            y = y.iloc[index]
            X_cnn = X_cnn[index]

        sc = joblib.load(os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
        X1 = pd.DataFrame(sc.transform(X.values), columns=X.columns, index=X.index)

        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

        y1 = pd.DataFrame(scale_y.transform(y.values), columns=y.columns, index=y.index)

        

        if self.static_data['clustering']['is_Fuzzy']:
            self.clusterer = Clusterer(self.static_data)

            activations = self.clusterer.compute_activations(X1)
            self.split_test_data(X1, activations=activations)
        else:
            self.split_test_data(X1)

        self.clusters = {}
        if self.static_data['is_Global']:
            cluster = ClusterObject(self.static_data, 'global')
            if cluster.istrained == False:
                cluster.save_cluster_data(X1, y1, X_cnn, X_lstm, split_test=self.split_test)
            self.clusters['global'] = cluster
        if self.static_data['clustering']['is_Fuzzy']:
            results = Parallel(int(self.static_data['njobs'])) \
                (delayed(create_cluster)(self.static_data, cluster_name, activations, X1, y1, X_cnn, X_lstm,
                                         self.split_test)
                 for cluster_name in activations.columns)

            for res in results:
                self.clusters[res[0]] = res[1]

        self.save()
        return 'Done'

    def check_if_all_nans(self, activations):

        if activations.isna().all(axis=1).any() == True:
            indices = activations.index[activations.isna().all(axis=1).to_numpy().ravel()]
            if indices.shape[0] > 50:
                raise RuntimeError('Too many nans. Please check your model')
            for ind in indices:
                act = activations.loc[ind]
                clust = act.idxmax()
                activations.loc[ind, clust] = 0.1

        return activations

    def predict_clusters(self):
        X_test, y_test, X_cnn_test, X_lstm_test = self.load_test_data()
        sc = joblib.load(os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
        pred_cluster = dict()
        X_test = pd.DataFrame(sc.transform(X_test.values), columns=X_test.columns, index=X_test.index)
        y_test = pd.DataFrame(scale_y.transform(y_test.values), columns=y_test.columns, index=y_test.index)
        if self.static_data['clustering']['is_Fuzzy']:
            if not hasattr(self, 'clusterer'):
                self.clusterer = Clusterer(self.static_data['path_fuzzy_models'])
            act_test = self.clusterer.compute_activations(X_test)
            act_test = self.check_if_all_nans(act_test)
        for clust in self.clusters.keys():
            predict_module = ClusterPredict(self.static_data, self.clusters[clust])
            if clust == 'global':
                if len(self.clusters[clust].methods) > 0:
                    pred_cluster[clust] = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test)
                    pred_cluster[clust]['metrics'] = predict_module.evaluate(pred_cluster['global'], y_test.values)
                    pred_cluster[clust]['dates'] = X_test.index
                    pred_cluster[clust]['index'] = np.arange(0, X_test.shape[0])
            else:
                dates = X_test.index[act_test[clust] >= self.thres_act]
                nind = np.where(act_test[clust] >= self.thres_act)[0]
                nind.sort()

                x = X_test.loc[dates]
                targ = y_test.loc[dates].values
                if len(X_cnn_test.shape) > 1:
                    x_cnn = X_cnn_test[nind]
                else:
                    x_cnn = np.array([])
                if len(X_lstm_test.shape) > 1:
                    x_lstm = X_lstm_test[nind]
                else:
                    x_lstm = np.array([])
                pred_cluster[clust] = predict_module.predict(x.values, X_cnn=x_cnn, X_lstm=x_lstm)
                pred_cluster[clust]['metrics'] = predict_module.evaluate(pred_cluster[clust], targ)
                pred_cluster[clust]['dates'] = dates
                pred_cluster[clust]['index'] = nind
        predictions = dict()
        result_clust = pd.DataFrame()
        for clust in pred_cluster.keys():
            for method in pred_cluster[clust].keys():
                if not method in {'dates', 'index', 'metrics'}:
                    if not method in predictions.keys():
                        predictions[method] = pd.DataFrame(index=X_test.index,
                                                           columns=[cl for cl in pred_cluster.keys()])
                    predictions[method].loc[pred_cluster[clust]['dates'], clust] = pred_cluster[clust][method].ravel()
                elif method in {'metrics'}:
                    result_clust = pd.concat([result_clust, pred_cluster[clust][method]['mae'].rename(clust)], axis=1)
        result_clust.to_csv(os.path.join(self.static_data['path_data'], 'result_of_clusters.csv'))
        return pred_cluster, predictions, y_test

    def load(self):
        if os.path.exists(os.path.join(self.path_model, 'manager' + '.pickle')):
            try:
                f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                if 'path_model' in tmp_dict.keys():
                    del tmp_dict['path_model']
                self.__dict__.update(tmp_dict)
            except:
                raise ValueError('Cannot find model for %s', self.path_model)
        else:
            raise ValueError('Cannot find model for %s', self.path_model)

    def load_to_transfer(self, path_model):
        if os.path.exists(os.path.join(path_model, 'manager' + '.pickle')):
            try:
                f = open(os.path.join(path_model, 'manager' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                return tmp_dict
            except:
                raise ValueError('Cannot find model for %s', path_model)
        else:
            raise ValueError('Cannot find model for %s', path_model)

    def save(self):
        f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'db', 'path_model', 'static_data', 'thres_act', 'thres_split', 'use_db']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
