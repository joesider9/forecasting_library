import os
import pandas as pd
import numpy as np
import pickle
import logging, shutil, glob
import pymongo, joblib
from Fuzzy_clustering.ver_tf2.Clusterer_optimize_deep import cluster_optimize, clusterer
from sklearn.preprocessing import MinMaxScaler
# from Fuzzy_clustering.ver_tf2.Regressor_layer import regressor_layer
# from Fuzzy_clustering.ver_tf2.Combine_module_ver2 import combine_model
# from Fuzzy_clustering.ver_tf2.CNN_stand_alone import cnn_3d
# from Fuzzy_clustering.ver_tf2.LSTM_stand_alone import lstm_3d
import time
# for timing
from contextlib import contextmanager
from timeit import default_timer
from sklearn.neural_network import MLPRegressor


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


class forecast_model(object):

    def __init__(self,static_data, use_db=False):
        self.static_data=static_data
        self.thres_split=static_data['clustering']['thres_split']
        self.thres_act=static_data['clustering']['thres_act']
        self.n_clusters=static_data['clustering']['n_clusters']
        self.rated=static_data['rated']
        self.var_imp=static_data['clustering']['var_imp']
        self.var_lin=static_data['clustering']['var_lin']
        self.var_nonreg=static_data['clustering']['var_nonreg']
        self.create_logger()
        self.use_db=use_db
        if use_db:
            self.db=self.open_db()

    def open_db(self):
        try:
            myclient = pymongo.MongoClient("mongodb://" + self.static_data['url'] + ":" + self.static_data['port'] + "/")

            project_db = myclient[self.static_data['_id']]
        except:
            self.logger.info('Cannot open Database')
            self.use_db=False
            project_db=None
            pass
        return project_db

    def create_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.static_data['path_project'], 'log_model.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def save_data(self,X,y):
        data_path=self.static_data['path_data']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(os.path.join(data_path,'dataset_X.csv')):
            X = X.round(4)
            y = y.round(4)
            X.to_csv(os.path.join(data_path,'dataset_X.csv'))
            y.to_csv(os.path.join(data_path,'dataset_y.csv'))
        else:
            X1 = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, parse_dates=True)
            y1 = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=[0], parse_dates=True)
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
                print('Cannot merge the historical data with the new ones')
            X.to_csv(os.path.join(data_path, 'dataset_X.csv'))
            y.to_csv(os.path.join(data_path, 'dataset_y.csv'))

        self.logger.info('Data saved')
        return X,y

    def load_data(self, path, filenameX,filenamey):
        X = pd.read_csv(os.path.join(path,filenameX), index_col=0, parse_dates=True)
        y = pd.read_csv(os.path.join(path,filenamey), index_col=0, header=None, names=['target'], parse_dates=True)

        self.logger.info('Data loaded')
        return X, y

    def backup(self,hor):
        for filename in glob.glob(os.path.join(self.static_data['path_project'], '*.*')):
            shutil.copy(filename, os.path.join(self.static_data['path_backup'],'hor_'+str(hor)))

    def scale(self,X):
        self.sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
        self.save()
        return pd.DataFrame(self.sc.transform(X.values),columns=X.columns,index=X.index)

    def train(self, X, y, update_data=True, lstm=False, X_lstm=None):
        self.lstm=lstm
        X_new=X.copy(deep=True)
        y_new = y.copy(deep=True)
        if update_data:
            X, y = self.save_data(X, y)
        X1 = self.scale(X)
        self.scale_y = MinMaxScaler(feature_range=(.1, 20)).fit(y.values)

        y1 = pd.DataFrame(self.scale_y.transform(y.values), columns=y.columns, index=y.index)

        if not self.static_data['clustering']['clustering_trained']: #and not os.path.exists(os.path.join(self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'])):
            N, D = X.shape
            n_split = int(np.round(N * 0.85))
            X_test = X.iloc[n_split + 1:]
            y_test = y1.iloc[n_split + 1:]

            X_train = X.iloc[:n_split]
            y_train = y1.iloc[:n_split]
            optimizer = cluster_optimize(self.static_data['type'], self.var_imp, self.var_lin, self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'],self.static_data['resampling'],self.static_data['RBF']['njobs'])
            optimizer.run(X_train, y_train, X_test, y_test, self.static_data['sklearn']['njobs'])

        self.clusterer=clusterer(self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'],self.static_data['type'])
        self.logger.info('Clusters created')

        act_new = self.clusterer.compute_activations(X_new)

        if len(self.var_nonreg)>0:
            X_new=X_new.drop(columns=self.var_nonreg).copy(deep=True)
        self.save()
        train_clust=[]
        for clust in act_new.columns:
            indices = act_new[clust].index[act_new[clust] >= self.thres_act].tolist()
            if len(indices)>0:
                inputs=X_new.loc[act_new[clust] >= self.thres_act]
                targets=y_new.loc[act_new[clust] >= self.thres_act]
                cluster_dir = os.path.join(self.static_data['path_project'], 'Regressor_layer/' + clust)
                if not os.path.exists(os.path.join(cluster_dir, 'data')):
                    os.makedirs((os.path.join(cluster_dir, 'data')))
                np.save(os.path.join(cluster_dir, 'data/X_train.npy'), inputs.values)
                np.save(os.path.join(cluster_dir, 'data/y_train.npy'), targets.values)
                if not inputs.shape[0]==0:
                    train_clust.append(clust)




        t=0
        self.clusterer.activations = self.clusterer.compute_activations(X)
        if len(self.var_nonreg)>0:
            X1=X1.drop(columns=self.var_nonreg).copy(deep=True)
        # train_clust=train_clust[7:]

        with elapsed_timer() as eval_elapsed:
            for clust in train_clust:
                # t = time.process_time()
                print('Begin training of ' + clust)
                indices = np.where(self.clusterer.activations[clust] >= self.thres_act)[0]
                act=self.clusterer.activations.loc[self.clusterer.activations[clust] >= self.thres_act, clust]
                inputs=X1.loc[self.clusterer.activations[clust] >= self.thres_act]
                targets=y1.loc[self.clusterer.activations[clust] >= self.thres_act]
                cluster_dir = os.path.join(self.static_data['path_project'], 'Regressor_layer/' + clust)
                if not os.path.exists(os.path.join(cluster_dir, 'data')):
                    os.makedirs((os.path.join(cluster_dir, 'data')))
                np.save(os.path.join(cluster_dir, 'data/X_train.npy'), inputs.values)
                np.save(os.path.join(cluster_dir, 'data/y_train.npy'), targets.values)
                targets_norm=y.loc[self.clusterer.activations[clust] >= self.thres_act]
                if len(targets.shape)==1:
                    ind=targets.index[pd.isnull(targets).nonzero()[0]]
                    if ind.shape[0]!=0:
                        inputs = inputs.drop(ind)
                        targets = targets.drop(ind)
                else:
                    ind=targets.index[pd.isnull(targets).any(1).nonzero()[0]]
                    if ind.shape[0] != 0:
                        inputs = inputs.drop(ind)
                        targets = targets.drop(ind)

                cluster_dir = os.path.join(self.static_data['path_project'], 'Regressor_layer/' + clust)
                clust_regressor = regressor_layer(self.static_data, cluster_dir)
                clust_regressor.train_cluster(inputs.values,targets.values,act.values)
                if hasattr(clust_regressor, 'trained'):
                    if (clust_regressor.trained != 'ok'):
                        if self.lstm:
                            if not X_lstm is None:
                                model = lstm_3d(cluster_dir, self.static_data['rated'])
                                nind = np.where(self.clusterer.activations[clust] >= self.thres_act)[0]
                                nind.sort()
                                model.train(X_lstm[nind], y.values[nind])
                else:
                    if self.lstm:
                        if not X_lstm is None:
                            model = lstm_3d(cluster_dir, self.static_data['rated'])
                            nind = np.where(self.clusterer.activations[clust] >= self.thres_act)[0]
                            nind.sort()
                            model.train(X_lstm[nind], y.values[nind])

                clust_regressor.trained = 'ok'
                clust_regressor.save(clust_regressor.cluster_dir)
                self.save()
                print('finish training of ' + clust)

                print('time %s' % str(eval_elapsed() / 60))
                self.logger.info('time %s', str((eval_elapsed() - t) / 60))

                print('finish training of ' + clust)
                t=eval_elapsed()

        predictions = self.predict_individual(X, y, X_lstm=X_lstm)

        combine_dir = os.path.join(self.static_data['path_project'], 'Combine_layer')
        if not os.path.exists(combine_dir):
            os.makedirs(combine_dir)
        # joblib.dump(predictions, os.path.join(combine_dir, 'pred_train_individual.pickle'))

        for comb_method in self.static_data['combine_methods']:
            comb_method_dir = os.path.join(combine_dir, comb_method)
            if not os.path.exists(comb_method_dir):
                os.makedirs(comb_method_dir)
            if comb_method not in {'cnn'}:
                if os.path.exists(os.path.join(comb_method_dir, 'combine_models.pickle')):
                    combine_models = joblib.load(os.path.join(comb_method_dir, 'combine_models.pickle'))
                else:
                    combine_models = dict()
                for rule in train_clust:
                    x_train = np.array([])
                    for method in predictions.keys():
                        if method not in {'activations'}:
                            p=self.scale_y.transform(predictions[method][rule].values.reshape(-1,1)).ravel()
                            if x_train.shape[0] == 0:
                                x_train = p
                            else:
                                x_train = np.vstack((x_train, p))
                    methods = [method for method in predictions.keys() if method not in {'activations'}]
                    x_train = x_train.T
                    try:
                        mask = np.any(~np.isnan(x_train), axis=1)
                        x_train = x_train[mask, :]
                        y_train1 = y1.values[mask]
                        comb = combine_model(comb_method, methods, 24, '', self.rated, comb_method_dir)
                        comb.train(x_train, y_train1)
                        combine_models[rule] = comb
                    except:
                        continue
                joblib.dump(combine_models, os.path.join(comb_method_dir, 'combine_models.pickle'))
            else:
                X_train3d = np.array([])
                for method in predictions.keys():
                    if method in {'activations'}:
                        continue
                    else:
                        x = predictions[method].copy()
                        if self.rated is None:
                            x[x.isnull()] = -999

                        else:
                            x[x.isnull()] = -self.rated
                            x = x / self.rated
                    if X_train3d.shape[0] == 0:
                        X_train3d = x.values
                    elif len(X_train3d.shape) == 2:
                        X_train3d = np.stack((X_train3d, x.values))
                    else:
                        X_train3d = np.vstack((X_train3d, x.values[np.newaxis, :, :]))

                X_train3d = X_train3d[:, :, :, np.newaxis].transpose(1, 2, 0, 3)
                combine_cnn = cnn_3d(comb_method_dir, self.rated)
                combine_cnn.train(X_train3d, y1.values)


    def initialize_predictions(self, activations, index):
        methods = []
        predictions = dict()
        predictions['activations'] = activations
        for model in self.static_data['project_methods'].keys():
            if self.static_data['project_methods'][model]['status'] == 'train' and \
                    self.static_data['project_methods'][model]['operation_mode'] == 'train':
                if model == 'ML_RBF_CNN':
                    methods.extend(['RBFNN', 'CNN'])
                    predictions['RBFNN'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                         columns=activations.columns, index=index))
                    predictions['CNN'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                       columns=activations.columns, index=index))
                elif model == 'ML_GA_RBF_OLS_CNN':
                    methods.extend(['GA_RBF_OLS', 'CNN_GA_OLS'])
                    predictions['GA_RBF_OLS'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                         columns=activations.columns, index=index))
                    predictions['CNN_GA_OLS'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                       columns=activations.columns, index=index))
                elif model == 'ML_RBF_OLS_CNN':
                    methods.extend(['RBF_OLS', 'CNN_OLS'])
                    predictions['RBF_OLS'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                              columns=activations.columns, index=index))
                    predictions['CNN_OLS'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                              columns=activations.columns, index=index))
                else:
                    methods.append(model)
                    predictions[model] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                                       columns=activations.columns, index=index))
        if self.lstm:
            methods.append('LSTM')
            predictions['LSTM'] = (pd.DataFrame(np.nan * np.zeros_like(activations.values),
                                               columns=activations.columns, index=index))
        return predictions, methods


    def predict_individual(self, X, y=None, X_lstm=None):
        self.load()
        self.clusterer=clusterer(self.static_data['path_fuzzy_models'],self.static_data['clustering']['cluster_file'],self.static_data['type'])
        self.clusterer.activations = self.clusterer.compute_activations(X)
        if isinstance(self.clusterer.activations, pd.Series):
            self.clusterer.activations = pd.DataFrame(self.clusterer.activations.to_dict(), index=[0])

        predictions, methods = self.initialize_predictions(self.clusterer.activations, X.index)

        X = pd.DataFrame(self.sc.transform(X.values), columns=X.columns, index=X.index)
        if len(self.var_nonreg)>0:
            X=X.drop(columns=self.var_nonreg).copy(deep=True)
        train_clust = []
        for clust in self.clusterer.activations.columns:
            indices = self.clusterer.activations[clust].index[self.clusterer.activations[clust] >= self.thres_act]
            inputs = X.loc[self.clusterer.activations[clust] >= self.thres_act]
            if not inputs.shape[0] == 0:
                train_clust.append(clust)

        for clust in train_clust:
            indices = self.clusterer.activations[clust].index[
                self.clusterer.activations[clust] > self.thres_act]

            inputs = X.loc[self.clusterer.activations[clust] > self.thres_act]
            # targets = y.loc[indices]

            if not inputs.shape[0] == 0:
                cluster_dir = os.path.join(self.static_data['path_project'], 'Regressor_layer/' + clust)
                if self.lstm:
                    if not X_lstm is None:
                        model = lstm_3d(cluster_dir, self.static_data['rated'])
                        nind = np.where(self.clusterer.activations[clust] >= self.thres_act)[0]
                        nind.sort()
                        pred = model.predict(X_lstm[nind])
                        predictions['LSTM'].loc[self.clusterer.activations[clust] > self.thres_act, clust]=pred
                # np.save(os.path.join(cluster_dir, 'data/X_test.npy'), inputs.values)
                # np.save(os.path.join(cluster_dir, 'data/y_test.npy'), targets.values)
                static_data = self.static_data
                clust_regressor = regressor_layer(static_data, cluster_dir)
                pred, regressors = clust_regressor.cluster_predict(inputs.values)
                for i in range(len(regressors)):
                    p=self.scale_y.inverse_transform(pred[:, i].reshape(-1,1))
                    p[p < 0] = 0
                    predictions[regressors[i]].loc[self.clusterer.activations[clust] > self.thres_act,clust] = p.ravel()

        return predictions


    def combine_cluster(self, pred_test):
        combine_dir = os.path.join(self.static_data['path_project'], 'Combine_layer')

        joblib.dump(pred_test, os.path.join(combine_dir, 'pred_test_individual.pickle'))
        predictions = np.array([])
        for comb_method in self.static_data['combine_methods']:
            comb_method_dir = os.path.join(combine_dir, comb_method)
            if not os.path.exists(comb_method_dir):
                continue
            if comb_method not in {'cnn'}:
                combine_models = joblib.load(os.path.join(comb_method_dir, 'combine_models.pickle'))
                predictions1 = np.array([])
                for rule in pred_test['activations'].columns:
                    pred = np.nan * np.zeros(pred_test['activations'].shape[0])
                    x_test = np.array([])
                    for method in pred_test.keys():
                        if method not in {'activations'}:
                            p=self.scale_y.transform(pred_test[method][rule].values.reshape(-1,1)).ravel()
                            if x_test.shape[0] == 0:
                                x_test = p
                            else:
                                x_test = np.vstack((x_test, p))
                    methods = [method for method in pred_test.keys() if method not in {'activations','RBFNN'}]
                    x_test = x_test.T
                    mask = np.any(~np.isnan(x_test), axis=1)
                    x_test = x_test[mask, :]
                    if x_test.shape[0]!=0:
                        pred[mask] = combine_models[rule].predict(x_test).ravel()
                    if predictions1.shape[0] == 0:
                        p=self.scale_y.inverse_transform(pred.reshape(-1,1)).ravel()
                        p[p < 0] = 0
                        predictions1 = p
                    else:
                        p=self.scale_y.inverse_transform(pred.reshape(-1,1)).ravel()
                        p[p < 0] = 0
                        predictions1 = np.vstack((predictions1, p))
                if predictions.shape[0] == 0:
                    predictions=self.combine_simple(predictions1.T)
                else:
                    predictions = np.vstack((predictions, self.combine_simple(predictions1.T)))
            else:
                combine_cnn = cnn_3d(comb_method_dir, self.rated)
                X_test3d = np.array([])
                for method in pred_test.keys():
                    if method in {'activations'}:
                        continue
                    else:
                        x = pred_test[method].copy()
                        if self.rated is None:
                            x[x.isnull()] = -999

                        else:
                            x[x.isnull()] = -self.rated
                            x = x / self.rated
                    if X_test3d.shape[0] == 0:
                        X_test3d = x.values
                    elif len(X_test3d.shape) == 2:
                        X_test3d = np.stack((X_test3d, x.values))
                    else:
                        X_test3d = np.vstack((X_test3d, x.values[np.newaxis, :, :]))
                X_test3d = X_test3d[:, :, :, np.newaxis].transpose(1, 2, 0, 3)
                pred = combine_cnn.predict(X_test3d)
                if predictions.shape[0] == 0:
                    p=self.scale_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                    p[p < 0] = 0
                    predictions = p
                else:
                    p = self.scale_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                    p[p < 0] = 0
                    predictions = np.vstack((predictions, p))
        return predictions.T


    def predict(self, X, y=None, X_lstm=None):
        self.load()
        pred_test = self.predict_individual(X,y=None,X_lstm=X_lstm)


        if len(self.static_data['combine_methods']) == 0:
            methods = [method for method in pred_test.keys() if method not in {'activations'}][0]
            prediction_tot = self.combine_simple(pred_test[methods])
            prediction_tot = prediction_tot.to_frame()
            predictions_combine = prediction_tot
        else:
            predictions = self.combine_cluster(pred_test)
            predictions_combine = pd.DataFrame(predictions, index=X.index,
                                               columns=self.static_data['combine_methods'])
            prediction_tot = self.combine_simple(predictions)
            prediction_tot = pd.DataFrame(prediction_tot, index=X.index, columns=['prediction_tot'])
        X = pd.DataFrame(self.sc.transform(X.values), columns=X.columns, index=X.index)
        if len(self.var_nonreg)>0:
            X=X.drop(columns=self.var_nonreg).copy(deep=True)
        if prediction_tot.isna().any().values[0] == True:
            indices = prediction_tot.index[prediction_tot.isna().to_numpy().ravel()]
            for ind in indices:
                act = pred_test['activations'].loc[ind]
                clust = act.idxmax()
                if isinstance(act, pd.Series):
                    print(act.to_frame().T)
                    act = act.to_frame().T

                inputs = X.loc[ind]
                cluster_dir = os.path.join(self.static_data['path_project'], 'Regressor_layer/' + clust)
                static_data = self.static_data
                clust_regressor = regressor_layer(static_data, cluster_dir)
                pred, regressors = clust_regressor.cluster_predict(inputs.values[np.newaxis, :])
                pred_nan, methods = self.initialize_predictions(act, act.index)

                for i in range(len(regressors)):
                    pred_nan[regressors[i]][clust].loc[ind] = pred[:, i]
                if len(self.static_data['combine_methods']) == 0:
                    methods = [method for method in pred_test.keys() if method not in {'activations'}][0]

                    pred_nan_fin = pred_nan[methods]
                    pred_nan_fin = self.combine_simple(pred_nan_fin)
                    if len(pred_nan_fin.shape) == 2:
                        pred_nan_fin = pred_nan_fin.ravel()
                    prediction_tot.loc[ind] = pred_nan_fin
                    predictions_combine = prediction_tot
                else:
                    pred_nan_fin = self.combine_cluster(pred_nan)
                    predictions_combine.loc[ind] = pred_nan_fin
                    pred_nan_fin = self.combine_simple(pred_nan_fin)
                    if len(pred_nan_fin.shape) == 2:
                        pred_nan_fin = pred_nan_fin.ravel()
                    prediction_tot.loc[ind] = pred_nan_fin
        print(prediction_tot)
        return prediction_tot, predictions_combine

    def combine_simple(self, predictions):
        if isinstance(predictions, pd.DataFrame):
            pred = predictions.mean(axis=1)
        else:
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            pred = np.nanmean(predictions, axis=1)
        # pred1 = predictions.multiply(activations)
        # pred1 = pred1.sum(axis=1).divide(activations.sum(axis=1))
        # pred1 = pred1[:, np.newaxis]
        return pred

    def load(self):
        model_path=self.static_data['path_project']
        if os.path.exists(os.path.join(self.static_data['path_project'], 'model' + '.pickle')):
            try:
                f = open(os.path.join(self.static_data['path_project'], 'model' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                self.save()
        else:
            self.save()


    def save(self):
        f = open(os.path.join(self.static_data['path_project'], 'model' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger','db','static_data','thres_act','thres_split','use_db']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()



