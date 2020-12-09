import pandas as pd
import numpy as np
import pickle, logging
import pika, uuid, time, json, os
import joblib
from joblib import Parallel, delayed
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.FuzzyClusteringManager.Clusterer_optimize_deep import clusterer
from Fuzzy_clustering.version3.FuzzyClusteringManager.TrainFuzzyManager import FuzzyManager
from Fuzzy_clustering.version3.FuzzyClusteringManager.Cluster_object import cluster_object
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

import time
# for timing
from contextlib import contextmanager
from timeit import default_timer


RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='FuzzyDatamanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def create_cluster(static_data, cluster_name, activations, X1, y1, X_cnn, X_lstm, split_test):

    cluster = cluster_object(static_data, cluster_name)
    if cluster.istrained==False:
        cluster.save_cluster_data(X1, y1, X_cnn, X_lstm, activations = activations, split_test = split_test)

    return cluster_name, cluster

class ModelTrainManager(object):

    def __init__(self, path_model):
        self.istrained = False
        self.clusters_created = False
        self.path_model = path_model
        try:
            self.load()
        except:
            pass

    def create_logger(self):
        self.logger = logging.getLogger('ModelTrainManager_' + self.model_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_model, 'log_' + self.model_name + '.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def init(self, static_data):
        self.data_variables = static_data['data_variables']
        self.static_data = static_data
        self.model_name = self.static_data['_id']
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']
        if not os.path.exists(os.path.join(self.static_data['path_data'], 'X_scaler.pickle')) \
                and not os.path.exists(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle')):
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

            sc = MinMaxScaler().fit(X.values)
            joblib.dump(sc, os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
            scale_y = MaxAbsScaler().fit(y.values)
            joblib.dump(scale_y, os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

            self.save()

    def load_data(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
        else:
            X_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
        else:
            X_lstm = np.array([])

        return X, y, X_cnn, X_lstm


    def split_test_data(self, X1, activations=None):
        if activations is None:
            indices = X1.index
            n_split = int(np.round(len(indices) * 0.7))
            split_test = indices[n_split + 1]
        else:
            split_test = None
            rate = 0.25
            while split_test is None and rate>0.1:
                split_indices = []
                for clust in activations.columns:
                    indices = activations[clust].index[activations[clust] >= self.thres_act].tolist()
                    if len(indices) > 0:
                        n_split = int(np.round(len(indices) * (1-rate)))
                        split_indices.append(indices[n_split + 1])
                split_test = pd.Series(split_indices).min()

                X_test = X1.loc[split_test:]
                if X_test.shape[0] > 0.4 * X1.shape[0]:
                    split_test = None
                    rate -=0.025
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
        self.create_logger()
        if not self.clusters_created:
            X, y, X_cnn, X_lstm = self.load_data()
            if y.isna().any().values[0]:
                X = X.drop(y.index[np.where(y.isna())[0]])
                if len(X_cnn.shape) > 1:
                    X_cnn = np.delete(X_cnn, np.where(y.isna())[0], axis=0)
                if len(X_lstm.shape) > 1:
                    X_lstm = np.delete(X_lstm, np.where(y.isna())[0], axis=0)
                y = y.drop(y.index[np.where(y.isna())[0]])
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux']>1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]

            sc = joblib.load(os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
            X1 = pd.DataFrame(sc.transform(X.values), columns=X.columns, index=X.index)

            scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

            y1 = pd.DataFrame(scale_y.transform(y.values), columns=y.columns, index=y.index)
            fuzzy_model = FuzzyManager(self.static_data)
            if fuzzy_model.istrained == False:
                response = fuzzy_model.train_fuzzy_clustering()
            else:
                response = 'Done'
            self.clusterer = clusterer(self.static_data)

            activations = self.clusterer.compute_activations(X1)

            if self.static_data['clustering']['is_Fuzzy']:
                self.split_test_data(X1, activations=activations)
            else:
                self.split_test_data(X1)

            self.clusters = {}
            if self.static_data['is_Global']:
                cluster = cluster_object(self.static_data, 'global')
                if cluster.istrained==False:
                    cluster.save_cluster_data(X1, y1, X_cnn, X_lstm, split_test=self.split_test)
                self.clusters['global'] = cluster
            if self.static_data['clustering']['is_Fuzzy']:
                results = Parallel(int(self.static_data['njobs']))\
                    (delayed(create_cluster)(self.static_data, cluster_name, activations, X1, y1, X_cnn, X_lstm, self.split_test)
                     for cluster_name in activations.columns)

                for res in results:
                    self.logger.info("Objective from linear models: rms %s and mae %s", res[1].lin_rms, res[1].lin_mae)
                    self.logger.info('Data info for cluster %s', res[1].cluster_name)
                    self.logger.info('Number of variables %s', str(res[1].D))
                    self.logger.info('Number of total samples %s', str(res[1].N_tot))
                    self.logger.info('Number of training samples %s', str(res[1].N_train))
                    self.logger.info('Number of validation samples %s', str(res[1].N_val))
                    self.logger.info('Number of testing samples %s', str(res[1].N_test))
                    self.logger.info('/n')
                    self.clusters[res[0]] = res[1]
            self.clusters_created = True
            self.save()
        return 'Done'

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

    def save(self):
        f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger','db', 'path_model', 'static_data','thres_act','thres_split','use_db']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer) or isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.str) or isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, bool):
            return bool(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            print(obj)
            raise TypeError('Object is not JSON serializable')

@server.consumer()
def FuzzyDatamanager(static_data):
    print(" [.] Receive project_group %s)" % static_data['projects_group'])
    model_manager = ModelTrainManager(static_data['path_model'])
    model_manager.init(static_data)
    model_response = model_manager.create_clusters_and_data()
    return model_response


if __name__=='__main__':
    server.run()
