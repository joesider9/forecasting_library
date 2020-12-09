import os
import pandas as pd
import numpy as np
import pickle
import logging, shutil, glob
import pymongo, joblib
from joblib import Parallel, delayed
from Fuzzy_clustering.ver_tf2.Clusterer_optimize_deep import cluster_optimize, clusterer
from sklearn.preprocessing import MinMaxScaler
from Fuzzy_clustering.ver_tf2.Cluster_train_regressors import cluster_train
from Fuzzy_clustering.ver_tf2.Global_train_regressor import global_train
from Fuzzy_clustering.ver_tf2.Cluster_train_regressor_TL import cluster_train_tl
from Fuzzy_clustering.ver_tf2.Global_train_regressor_TL import global_train_tl
from Fuzzy_clustering.ver_tf2.NWP_sampler import nwp_sampler
from Fuzzy_clustering.ver_tf2.Global_predict_regressor import global_predict
from Fuzzy_clustering.ver_tf2.Cluster_predict_regressors import cluster_predict
from Fuzzy_clustering.ver_tf2.Combine_train_model import Combine_train

import time
# for timing
from contextlib import contextmanager
from timeit import default_timer



@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


class ModelTrainManager(object):

    def __init__(self, path_model):
        self.istrained = False
        self.path_model = path_model
        try:
            self.load()
        except:
            pass

    def init(self, static_data, data_variables, use_db=False):
        self.data_variables = data_variables
        self.static_data = static_data
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']

        self.create_logger()
        self.use_db = use_db
        if use_db:
            self.db = self.open_db()

    def open_db(self):
        try:
            myclient = pymongo.MongoClient("mongodb://" + self.static_data['url'] + ":" + self.static_data['port'] + "/")

            project_db = myclient[self.static_data['_id']]
        except:
            self.logger.info('Cannot open Database')
            self.use_db=False
            project_db=None
            raise ConnectionError('Cannot open Database')
        self.logger.info('Open Database successfully')
        return project_db

    def create_logger(self):
        self.logger = logging.getLogger(self.static_data['_id'])
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_model, 'log_model.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def merge_old_data(self,X, y, X_cnn=np.array([]), X_lstm=np.array([])):
        data_path=self.static_data['path_data']
        if os.path.exists(os.path.join(data_path,'dataset_X.csv')):
            X1 = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
            y1 = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
            try:
                X=X.append(X1)
                y=y.append(y1)
                X=X.round(4)
                y=y.round(4)
                X['target'] = y
                X=X.drop_duplicates()
                y = X['target'].copy(deep=True)
                y = y.to_frame()
                y.columns=['target']
                X = X.drop(columns='target')
            except ImportError:
                raise AssertionError('Cannot merge the historical data with the new ones')
            X.to_csv(os.path.join(data_path, 'dataset_X.csv'))
            y.to_csv(os.path.join(data_path, 'dataset_y.csv'))
        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_3d = joblib.load(os.path.join(self.static_data['path_data'], 'dataset_cnn.pickle'))
            X_cnn = np.vstack([X_cnn, X_3d])
            joblib.dump(X_cnn, os.path.join(self.static_data['path_data'], 'dataset_cnn.pickle'))
        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_2d = joblib.load(os.path.join(self.static_data['path_data'], 'dataset_lstm.pickle'))
            X_lstm = np.vstack([X_lstm, X_2d])
            joblib.dump(X_lstm, os.path.join(self.static_data['path_data'], 'dataset_lstm.pickle'))

        self.logger.info('Data merged successfully')

        return X, y, X_cnn, X_lstm

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

        self.logger.info('Data loaded successfully')
        return X, y, X_cnn, X_lstm

    def backup(self,hor=None):
        #TODO write to backup checking the version of the model (if there are previous versions, write current model in different folder)
        if hor is None:
            for filename in glob.glob(os.path.join(self.path_model, '*.*')):
                shutil.copy(filename, self.static_data['path_backup'])
        else:
            for filename in glob.glob(os.path.join(self.path_model, '*.*')):
                shutil.copy(filename, os.path.join(self.static_data['path_backup'],'hor_'+str(hor)))

    def scale(self,X):
        self.sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
        self.save()
        return pd.DataFrame(self.sc.transform(X.values),columns=X.columns,index=X.index)

    def train_fuzzy_clustering(self, X, y1):
        N, D = X.shape
        n_split = int(np.round(N * 0.7))
        X_test = X.iloc[n_split + 1:]
        y_test = y1.iloc[n_split + 1:]

        X_train = X.iloc[:n_split]
        y_train = y1.iloc[:n_split]
        optimizer = cluster_optimize(self.static_data)
        if self.rated is None:
            rated = None
        else:
            rated = 20
        if self.static_data['type'] == 'fa':
            optimizer.run(X_train, y_train, X_test, y_test, rated, num_samples=300)
        else:
            optimizer.run(X_train, y_train, X_test, y_test, rated)
        self.save()

    def find_clusters_for_training(self, X_new, train):
        act_new = self.clusterer.compute_activations(X_new)

        if len(self.var_nonreg) > 0:
            X_new = X_new.drop(columns=self.var_nonreg).copy(deep=True)

        train_clust = []
        if not len(train) == 0:
            for clust in train:
                indices = act_new[clust].index[act_new[clust] >= self.thres_act].tolist()
                if len(indices) > 0:
                    inputs = X_new.loc[act_new[clust] >= self.thres_act]
                    cluster_dir = os.path.join(self.path_model, 'Regressor_layer/' + clust)
                    if not os.path.exists(cluster_dir):
                        os.makedirs(cluster_dir)
                    if not os.path.exists(os.path.join(cluster_dir, 'data')):
                        os.makedirs(os.path.join(cluster_dir, 'data'))
                    if not inputs.shape[0] == 0:
                        train_clust.append(clust)
        else:
            for clust in act_new.columns:
                indices = act_new[clust].index[act_new[clust] >= self.thres_act].tolist()
                if len(indices) > 0:
                    inputs = X_new.loc[act_new[clust] >= self.thres_act]
                    cluster_dir = os.path.join(self.path_model, 'Regressor_layer/' + clust)
                    if not os.path.exists(cluster_dir):
                        os.makedirs(cluster_dir)
                    if not os.path.exists(os.path.join(cluster_dir, 'data')):
                        os.makedirs(os.path.join(cluster_dir, 'data'))
                    if not inputs.shape[0] == 0:
                        train_clust.append(clust)

        return train_clust

    def split_test_data(self, activations, X1, y1, X_cnn, X_lstm):
        split_indices = []
        for clust in activations.columns:
            indices = activations[clust].index[activations[clust] >= self.thres_act].tolist()
            if len(indices) > 0:
                if len(indices) > 1000:
                    n_split = int(np.round(len(indices) * 0.75))
                    split_indices.append(indices[n_split + 1])
                else:
                    n_split = int(np.round(len(indices) * 0.85))
                    split_indices.append(indices[n_split + 1])
        split_test = pd.Series(split_indices).min()

        X_test = X1.loc[split_test:]
        if X_test.shape[0] > 0.35 * X1.shape[0]:
            split_test = None
        self.split_test = split_test
        return split_test

    def save_global_data(self, activations, X1, y1, X_cnn, X_lstm):
        # VARIABLES USED ONLY FOR CLUSTERING
        if len(self.var_nonreg) > 0:
            X1 = X1.drop(columns=self.var_nonreg).copy(deep=True)
        split_test = self.split_test

        self.logger.info('Save datasets for global model')

        cluster_dir=os.path.join(self.static_data['path_model'], 'Global_regressor')
        cluster_data_dir = os.path.join(cluster_dir, 'data')
        if not os.path.exists(cluster_data_dir):
            os.makedirs(cluster_data_dir)
        act = activations
        inputs = X1
        targets = y1

        inputs = inputs.drop(targets.index[pd.isnull(targets).values.ravel()])
        targets = targets.drop(targets.index[pd.isnull(targets).values.ravel()])

        targets = targets.drop(inputs.index[pd.isnull(inputs).any(1).values.ravel()])
        inputs = inputs.drop(inputs.index[pd.isnull(inputs).any(1).values.ravel()])
        if not split_test is None:
            test_indices = dict()
            test_indices['dates_train'] = inputs.index[inputs.index < split_test]
            test_ind = np.where(inputs.index < split_test)[0]
            test_ind.sort()
            test_indices['indices_train'] = test_ind

            test_indices['dates_test'] = inputs.index[inputs.index >= split_test]
            test_ind = np.where(inputs.index>=split_test)[0]
            test_ind.sort()
            test_indices['indices_test'] = test_ind
            joblib.dump(test_indices, os.path.join(cluster_data_dir, 'test_indices.pickle'))

        if not self.static_data['train_online']:
            inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
            targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
            act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
            self.logger.info('Data saved for global model')
            if len(X_cnn.shape) > 1:
                x_cnn = X_cnn
                joblib.dump(x_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))


            if len(X_lstm.shape) > 1:
                x_lstm = X_lstm
                joblib.dump(x_lstm, os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
        else:
            if not os.path.exists(os.path.join(cluster_data_dir, 'dataset_X.csv')):
                inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
                targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
                act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
                if len(X_cnn.shape) > 1:
                    x_cnn = X_cnn
                    joblib.dump(x_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                if len(X_lstm.shape) > 1:
                    x_lstm = X_lstm
                    joblib.dump(x_lstm, os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
                self.logger.info('Data saved for for global model')
            else:
                self.logger.info('load data from previous train loop  for global model')
                x_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'), index_col=0, header=[0], parse_dates=True, dayfirst=True)
                y_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'), index_col=0, header=[0],
                                 parse_dates=True, dayfirst=True)
                act_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'), index_col=0, header=[0],
                                   parse_dates=True, dayfirst=True)
                try:
                    self.logger.info('Merge data from previous train loop for global model')
                    inputs = x_old.append(inputs)
                    targets = y_old.append(targets)
                    act = act_old.append(act)
                    inputs = inputs.round(6)
                    targets = targets.round(6)
                    act = act.round(6)
                    inputs['target'] = targets
                    inputs['activation'] = act
                    inputs = inputs.drop_duplicates()
                    targets = inputs['target'].copy(deep=True)
                    act = inputs['activation'].copy(deep=True)
                    targets = targets.to_frame()
                    act = act.to_frame()
                    targets.columns = ['target']
                    act.columns = ['activation']
                    inputs = inputs.drop(columns=['target', 'activation'])
                    inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
                    targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
                    act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
                    if os.path.exists(os.path.join(cluster_data_dir, 'dataset_cnn.pickle')):
                        x_cnn = joblib.load(os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))
                        X_cnn = np.vstack([x_cnn, X_cnn])
                        joblib.dump(X_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                    if os.path.exists(os.path.join(cluster_data_dir, 'dataset_lstm.pickle')):
                        x_lstm = joblib.load(os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
                        X_lstm = np.vstack([x_lstm, X_lstm])
                        joblib.dump(X_lstm, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                    self.logger.info('Data merged and saved for global model')
                except ImportError:
                    print('Cannot merge the historical data with the new ones')
        self.logger.info('/n')

    def save_cluster_data(self, activations, X1, y1, X_cnn, X_lstm, train_clust_list):
        # VARIABLES USED ONLY FOR CLUSTERING
        if len(self.var_nonreg) > 0:
            X1 = X1.drop(columns=self.var_nonreg).copy(deep=True)
        split_test = self.split_test_data(activations, X1, y1, X_cnn, X_lstm)

        for clust in train_clust_list:
            self.logger.info('Save datasets for ' + clust)

            cluster_dir = os.path.join(self.path_model, 'Regressor_layer/' + clust)
            cluster_data_dir = os.path.join(cluster_dir, 'data')
            if (not os.path.exists(os.path.join(cluster_data_dir, 'dataset_X.csv')) and not self.static_data['train_online']) or \
                    (self.static_data['recreate_datasets'] and not self.static_data['train_online']):
                nind = np.where(activations[clust] >= self.thres_act)[0]
                nind.sort()

                act = activations.loc[activations[clust] >= self.thres_act, clust]
                inputs = X1.loc[activations[clust] >= self.thres_act]
                targets = y1.loc[activations[clust] >= self.thres_act]

                inputs = inputs.drop(targets.index[pd.isnull(targets).values.ravel()])
                targets = targets.drop(targets.index[pd.isnull(targets).values.ravel()])

                targets = targets.drop(inputs.index[pd.isnull(inputs).any(1).values.ravel()])
                inputs = inputs.drop(inputs.index[pd.isnull(inputs).any(1).values.ravel()])
                if not split_test is None:
                    test_indices = dict()
                    test_indices['dates_train'] = inputs.index[inputs.index < split_test]
                    test_ind = np.where(inputs.index < split_test)[0]
                    test_ind.sort()
                    test_indices['indices_train'] = test_ind

                    test_indices['dates_test'] = inputs.index[inputs.index >= split_test]
                    test_ind = np.where(inputs.index>=split_test)[0]
                    test_ind.sort()
                    test_indices['indices_test'] = test_ind
                    joblib.dump(test_indices, os.path.join(cluster_data_dir, 'test_indices.pickle'))

                if not self.static_data['train_online']:
                    inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
                    targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
                    act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
                    self.logger.info('Data saved for cluster %s', clust)
                    if len(X_cnn.shape) > 1:
                        x_cnn = X_cnn[nind]
                        joblib.dump(x_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))


                    if len(X_lstm.shape) > 1:
                        x_lstm = X_lstm[nind]
                        joblib.dump(x_lstm, os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
            elif self.static_data['train_online']:
                if not os.path.exists(os.path.join(cluster_data_dir, 'dataset_X.csv')):
                    inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
                    targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
                    act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
                    if len(X_cnn.shape) > 1:
                        x_cnn = X_cnn[nind]
                        joblib.dump(x_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                    if len(X_lstm.shape) > 1:
                        x_lstm = X_lstm[nind]
                        joblib.dump(x_lstm, os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
                    self.logger.info('Data saved for cluster %s', clust)
                else:
                    self.logger.info('load data from previous train loop for cluster %s', clust)
                    x_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'), index_col=0, header=[0], parse_dates=True, dayfirst=True)
                    y_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'), index_col=0, header=[0],
                                     parse_dates=True, dayfirst=True)
                    act_old = pd.read_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'), index_col=0, header=[0],
                                       parse_dates=True, dayfirst=True)
                    try:
                        self.logger.info('Merge data from previous train loop for cluster %s', clust)
                        inputs = x_old.append(inputs)
                        targets = y_old.append(targets)
                        act = act_old.append(act)
                        inputs = inputs.round(6)
                        targets = targets.round(6)
                        act = act.round(6)
                        inputs['target'] = targets
                        inputs['activation'] = act
                        inputs = inputs.drop_duplicates()
                        targets = inputs['target'].copy(deep=True)
                        act = inputs['activation'].copy(deep=True)
                        targets = targets.to_frame()
                        act = act.to_frame()
                        targets.columns = ['target']
                        act.columns = ['activation']
                        inputs = inputs.drop(columns=['target', 'activation'])
                        inputs.to_csv(os.path.join(cluster_data_dir, 'dataset_X.csv'))
                        targets.to_csv(os.path.join(cluster_data_dir, 'dataset_y.csv'))
                        act.to_csv(os.path.join(cluster_data_dir, 'dataset_act.csv'))
                        if os.path.exists(os.path.join(cluster_data_dir, 'dataset_cnn.pickle')):
                            x_cnn = joblib.load(os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))
                            X_cnn = np.vstack([x_cnn, X_cnn])
                            joblib.dump(X_cnn, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                        if os.path.exists(os.path.join(cluster_data_dir, 'dataset_lstm.pickle')):
                            x_lstm = joblib.load(os.path.join(cluster_data_dir, 'dataset_lstm.pickle'))
                            X_lstm = np.vstack([x_lstm, X_lstm])
                            joblib.dump(X_lstm, os.path.join(cluster_data_dir, 'dataset_cnn.pickle'))

                        self.logger.info('Data merged and saved for cluster %s', clust)
                    except ImportError:
                        print('Cannot merge the historical data with the new ones')
            self.logger.info('/n')

    def train(self, train=[]):

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
        X_new=X.copy(deep=True)

        if self.static_data['train_online']:
            X, y, X_cnn, X_lstm = self.merge_old_data(X, y, X_cnn=X_cnn, X_lstm=X_lstm)
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]
        X1 = self.scale(X)

        self.scale_y = MinMaxScaler(feature_range=(.1, 20)).fit(y.values)

        X_new = pd.DataFrame(self.sc.transform(X_new.values), columns=X_new.columns, index=X_new.index)

        y1 = pd.DataFrame(self.scale_y.transform(y.values), columns=y.columns, index=y.index)

        if not self.static_data['clustering']['is_clustering_trained'] and not os.path.exists(os.path.join(self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'])):
           self.train_fuzzy_clustering(X1, y1)

        self.clusterer=clusterer(self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'],self.static_data['type'])
        self.logger.info('Clusters created')
        #
        # train_clust_list = self.find_clusters_for_training(X_new, train)
        #
        # activations = self.clusterer.compute_activations(X1)
        #
        # self.save_cluster_data(activations, X1, y1, X_cnn, X_lstm, train_clust_list)
        # self.save_global_data(activations, X1, y1, X_cnn, X_lstm)
        #
        # # Obsolete
        # # if self.static_data['type'] in {'wind', 'pv'}:
        # #     create_nwp_sampler = nwp_sampler(self.static_data)
        # #     if create_nwp_sampler.istrained == False:
        # #         create_nwp_sampler.train(X1, X_cnn, gpu_id=self.static_data['CNN']['gpus'][0])
        #
        # self.regressors=dict()
        # glob_regressor = global_train(self.static_data, self.sc)
        # if glob_regressor.istrained==False:
        #     self.logger.info('Global regressor is training..')
        #     self.regressors['Global'] = glob_regressor.fit()
        #     self.logger.info('Global regressor trained..')
        #     self.save()
        # else:
        #     self.regressors['Global'] = glob_regressor.to_dict()
        # with elapsed_timer() as eval_elapsed:
        #     for clust in train_clust_list:
        #         t = time.process_time()
        #         print('Begin training of ' + clust)
        #         self.logger.info('Begin training of ' + clust)
        #
        #         clust_regressor = cluster_train(self.static_data, clust, self.sc)
        #         if clust_regressor.istrained==False:
        #             self.regressors[clust] = clust_regressor.fit()
        #         else:
        #             self.regressors[clust] = clust_regressor.to_dict()
        #
        #         self.save()
        #
        #         print('time %s' % str(eval_elapsed() / 60))
        #         self.logger.info('time %s', str((eval_elapsed() - t) / 60))
        #         print('finish training of ' + clust)
        #         self.logger.info('finish training of ' + clust)
        #
        #         t=eval_elapsed()
        # self.predict_regressors(X1, y1, X_cnn, X_lstm)
        #
        # combine_model_ = Combine_train(self.static_data)
        # self.combine_model = combine_model_.train()
        #
        # self.istrained = True
        # self.full_trained = True
        #
        # self.save()

    def train_tl_rules(self, static_data, clust, gpu, rule_model):

        clust_regressor = cluster_train_tl(static_data, clust, self.sc, gpu)
        regressor = clust_regressor.fit(rule_model=rule_model)
        return (clust, regressor)

    def train_TL(self, path_model_tl, train=[]):
        model_tl = self.load_to_transfer(path_model_tl)
        static_data_tl = self.static_data['tl_project']['static_data']
        self.sc = model_tl['sc']
        self.scale_y = model_tl['scale_y']
        X, y, X_cnn, X_lstm = self.load_data()
        if self.static_data['type'] == 'pv':
            index = np.where(X['flux'] > 1e-8)[0]
            X = X.iloc[index]
            y = y.iloc[index]
            X_cnn = X_cnn[index]
        X_new = X.copy(deep=True)

        if self.static_data['train_online']:
            X, y, X_cnn, X_lstm = self.merge_old_data(X, y, X_cnn=X_cnn, X_lstm=X_lstm)
            if self.static_data['type'] == 'pv':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]
        X1 = self.scale(X)
        # Obsolete
        # create_nwp_sampler = nwp_sampler(self.static_data)
        # if create_nwp_sampler.istrained == False:
        #     create_nwp_sampler.train(X1, X_cnn, gpu_id=self.static_data['CNN']['gpus'][0])

        self.scale_y = MinMaxScaler(feature_range=(.1, 20)).fit(y.values)

        X_new = pd.DataFrame(self.sc.transform(X_new.values), columns=X_new.columns, index=X_new.index)

        y1 = pd.DataFrame(self.scale_y.transform(y.values), columns=y.columns, index=y.index)

        fuzzy_file = os.path.join(static_data_tl['path_fuzzy_models'], static_data_tl['clustering']['cluster_file'])
        fmodel = joblib.load(fuzzy_file)
        joblib.dump(fmodel, os.path.join(self.static_data['path_fuzzy_models'], self.static_data['clustering']['cluster_file']))

        self.clusterer = clusterer(self.static_data['path_fuzzy_models'],
                                   self.static_data['clustering']['cluster_file'], self.static_data['type'])
        self.logger.info('Clusters created')

        train_clust_list = self.find_clusters_for_training(X_new, train)

        activations = self.clusterer.compute_activations(X1)

        self.save_cluster_data(activations, X1, y1, X_cnn, X_lstm, train_clust_list)
        self.save_global_data(activations, X1, y1, X_cnn, X_lstm)
        self.regressors = dict()

        gpus = np.tile(self.static_data['CNN']['gpus'], len(train_clust_list))
        glob_regressor = global_train_tl(self.static_data, self.sc, gpus[0])
        if glob_regressor.istrained == False:
            self.logger.info('Global regressor is training..')

            self.regressors['Global'] = glob_regressor.fit(rule_model=model_tl['regressors']['Global'])
            self.logger.info('Global regressor trained')
        else:
            self.regressors['Global'] = glob_regressor.to_dict()
        with elapsed_timer() as eval_elapsed:
            for k, clust in enumerate(train_clust_list):
                t = time.process_time()
                print('Begin training of ' +clust)
                self.logger.info('Begin training of ' + clust)


                clust_regressor = cluster_train_tl(self.static_data, clust, self.sc, gpus[k])
                if clust_regressor.istrained==False:
                    self.regressors[clust] = clust_regressor.fit(rule_model=model_tl['regressors'][clust])
                else:
                    self.regressors[clust] = clust_regressor.to_dict()

                print('time %s' % str(eval_elapsed() / 60))
                self.logger.info('time %s', str((eval_elapsed() - t) / 60))
                print('finish training of ' + clust)
                self.logger.info('finish training of ' + clust)
                self.save()
                t = eval_elapsed()
        self.predict_regressors(X1, y1, X_cnn, X_lstm)
        combine_model_ = Combine_train(self.static_data)
        self.combine_model = combine_model_.train()
        self.istrained = True
        self.full_trained = True
        self.save()

    def predict_regressors(self, X1, y1, X_cnn, X_lstm):
        data_path = self.static_data['path_data']
        if not self.split_test is None:
            X_test = X1.loc[X1.index >= self.split_test]
            y_test = y1.loc[X1.index >= self.split_test]
            test_ind = np.where(X1.index >= self.split_test)[0]
            test_ind.sort()
            if len(X_cnn.shape) > 1:
                X_cnn_test = X_cnn[test_ind]
            else:
                X_cnn_test = np.array([])
            if len(X_lstm.shape) > 1:
                X_lstm_test = X_lstm[test_ind]
            else:
                X_lstm_test = np.array([])

            pred_cluster = dict()
            act_test = self.clusterer.compute_activations(X_test)
            for clust in self.regressors.keys():
                if clust == 'Global':
                    if len(self.regressors['Global']['models']) > 0:
                        predict_module = global_predict(self.static_data)
                        pred_cluster['Global'] = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test)
                        pred_cluster['Global']['metrics'] = predict_module.evaluate(pred_cluster['Global'], y_test.values)
                        pred_cluster['Global']['dates'] = X_test.index
                        pred_cluster['Global']['index'] = np.arange(0, X_test.shape[0])
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
                    predict_module = cluster_predict(self.static_data, clust)
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
                            predictions[method] = pd.DataFrame(index=X_test.index, columns=[cl for cl in pred_cluster.keys()])
                        predictions[method].loc[pred_cluster[clust]['dates'], clust] = pred_cluster[clust][method].ravel()
                    elif method in {'metrics'}:
                        result_clust = pd.concat([result_clust, pred_cluster[clust][method]['mae'].rename(clust)], axis=1)
            result_clust.to_csv(os.path.join(data_path, 'result_of_clusters.csv'))
            joblib.dump(pred_cluster, os.path.join(data_path, 'predictions_by_cluster.pickle'))
            joblib.dump(predictions, os.path.join(data_path, 'predictions_by_method.pickle'))
            y_test.to_csv(os.path.join(data_path, 'target_test.csv'))
        else:
            self.static_data['combine_methods'] = ['average']

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
            if k not in ['logger','db', 'path_model', 'static_data','thres_act','thres_split','use_db']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

if __name__ == '__main__':
    from util_database import write_database
    from Fuzzy_clustering.ver_tf2.Projects_train_manager import ProjectsTrainManager

    static_data = write_database()
    project_manager = ProjectsTrainManager(static_data)
    project_manager.initialize()
    project_manager.create_datasets()
    project_manager.create_projects_relations()
    project = [pr for pr in project_manager.group_static_data if pr['_id'] == 'Lach'][0]
    static_data = project['static_data']

    model = ModelTrainManager(static_data['path_model'])
    model.init(project['static_data'], project_manager.data_variables)
    model.train()