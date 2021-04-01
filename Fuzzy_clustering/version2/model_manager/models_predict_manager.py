import logging
import os
import pickle
import glob
import shutil

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed

from Fuzzy_clustering.version2.cluster_predict_manager.cluster_predict_manager import ClusterPredict
from Fuzzy_clustering.version2.combine_model_manager.combine_model_predict import CombineModelPredict
from Fuzzy_clustering.version2.fuzzy_clustering_manager.clusterer import Clusterer
from Fuzzy_clustering.version2.model_manager.data_manage_clusters import ClusterObject
from Fuzzy_clustering.version2.probabilistic_manager.proba_model_manager import proba_model_manager


class ModelPredictManager(object):

    def __init__(self, static_data):
        self.istrained = False
        self.path_model = static_data['path_model']
        try:
            self.load()
        except:
            pass
        if hasattr(self, 'is_trained'):
            self.istrained = self.is_trained
        self.static_data = static_data
        self.thres_act = static_data['clustering']['thres_act']
        self.rated = static_data['rated']
        file_x_scaler = os.path.join(self.static_data['path_data'], 'X_scaler.pickle')
        file_y_scaler = os.path.join(self.static_data['path_data'], 'Y_scaler.pickle')
        if os.path.exists(file_x_scaler) and os.path.exists(file_y_scaler):
            self.sc = joblib.load(file_x_scaler)
            self.scale_y = joblib.load(file_y_scaler)
        elif hasattr(self, 'sc') and hasattr(self, 'scale_y'):
            pass
        else:
            raise RuntimeError('Cannot find the data scalers')
        self.create_logger()

    def load_data_test(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X_test.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
        if os.path.exists(os.path.join(data_path, 'dataset_y_test.csv')):
            y = pd.read_csv(os.path.join(data_path, 'dataset_y_test.csv'), index_col=0, header=0, parse_dates=True,
                            dayfirst=True)
        else:
            y = None

        if os.path.exists(os.path.join(data_path, 'dataset_cnn_test.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn_test.pickle'))
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
        else:
            X_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm_test.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm_test.pickle'))
        else:
            X_lstm = np.array([])

        self.logger.info('Data loaded successfully')
        return X, X_cnn, X_lstm, y

    def load_data_train(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
        if os.path.exists(os.path.join(data_path, 'dataset_y.csv')):
            y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True,
                            dayfirst=True)
        else:
            y = None

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
        return X, X_cnn, X_lstm, y

    def create_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.static_data['path_project'], 'log_model_evaluation.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)
    def copytree(self, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
    def check_if_is_global(self):
        clust = 'global'
        path_old = os.path.join(self.static_data['path_model'], 'Global_regressor')
        path_new = os.path.join(self.static_data['path_model'], 'Regressor_layer/' + clust)
        if os.path.exists(path_old):
            if os.path.exists(path_new):
                shutil.rmtree(path_new)
            if not os.path.exists(path_new):
                os.makedirs(path_new)

            self.copytree(path_old, path_new)
            shutil.move(os.path.join(path_new, 'Global_models.pickle'), os.path.join(path_new, 'model_global.pickle'))
            shutil.rmtree(path_old)

        if os.path.exists(path_new):
            self.static_data['is_Global'] = True


    def predict_clusters(self, X_test, X_cnn_test, X_lstm_test, y_test=None, njobs=1):
        def predict_parallel(clust, cluster, static_data, X_test, act_test,
                             X_cnn_test, X_lstm_test,
                             y_test=None):

            predict_module = ClusterPredict(static_data, cluster)
            if clust == 'global':
                pred_cluster = predict_module.predict(X_test.values, X_cnn=X_cnn_test,
                                                      X_lstm=X_lstm_test)
                if y_test is not None:
                    pred_cluster['metrics'] = predict_module.evaluate(pred_cluster,
                                                                      y_test.values)
                pred_cluster['dates'] = X_test.index
                pred_cluster['index'] = np.arange(0, X_test.shape[0])
            else:
                dates = X_test.index[act_test >= self.thres_act]
                nind = np.where(act_test >= self.thres_act)[0]
                print(clust)
                print(X_test.shape)
                x = X_test.loc[dates]
                print(x.shape)
                if y_test is not None:
                    targ = y_test.loc[dates].values
                else:
                    targ = None
                if len(X_cnn_test.shape) > 1:
                    x_cnn = X_cnn_test[nind]
                else:
                    x_cnn = np.array([])
                if len(X_lstm_test.shape) > 1:
                    x_lstm = X_lstm_test[nind]
                else:
                    x_lstm = np.array([])
                pred_cluster = predict_module.predict(x.values, X_cnn=x_cnn, X_lstm=x_lstm)
                if targ is not None and targ.shape[0] > 0:
                    pred_cluster['metrics'] = predict_module.evaluate(pred_cluster, targ)
                pred_cluster['dates'] = dates
                pred_cluster['index'] = nind

            return clust, pred_cluster

        data_path = self.static_data['path_data']
        pred_cluster = dict()
        X_test = pd.DataFrame(self.sc.transform(X_test.values), columns=X_test.columns, index=X_test.index)
        if not y_test is None:
            y_test = pd.DataFrame(self.scale_y.transform(y_test.values), columns=y_test.columns, index=y_test.index)
        if self.static_data['clustering']['is_Fuzzy']:
            if not hasattr(self, 'clusterer'):
                self.clusterer = Clusterer(self.static_data)
            act_test = self.clusterer.compute_activations(X_test)
            act_test = self.check_if_all_nans(act_test)
        else:
            act_test = pd.DataFrame(index=X_test.index)
        # self.check_if_is_global()
        self.clusters = {}
        if self.static_data['is_Global']:
            cluster = ClusterObject(self.static_data, 'global')
            self.clusters['global'] = cluster
            if 'global' not in act_test.columns:
                act_test['global'] = 1
        if self.static_data['clustering']['is_Fuzzy']:
            for cluster_name in act_test.columns:
                cluster = ClusterObject(self.static_data, cluster_name)
                self.clusters[cluster_name] = cluster
        if njobs > 1:
            pred_clusters = Parallel(n_jobs=njobs)(
                delayed(predict_parallel)(clust, cluster, self.static_data, X_test, act_test[clust],
                                          X_cnn_test, X_lstm_test,
                                          y_test=y_test) for clust, cluster in self.clusters.items())
        else:
            pred_clusters=[]
            for clust, cluster in self.clusters.items():
                pred_clusters.append(predict_parallel(clust, cluster, self.static_data, X_test, act_test[clust],
                                          X_cnn_test, X_lstm_test,
                                          y_test=y_test))
        for pred_clust in pred_clusters:
            pred_cluster[pred_clust[0]] = pred_clust[1]
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

        if y_test is not None:
            result_clust.to_csv(os.path.join(data_path, 'result_of_clusters.csv'))
        return pred_cluster, predictions

    def predict_model(self, pred_cluster, predictions):
        combine_overall = CombineModelPredict(self.static_data)
        predictions_final = combine_overall.predict(predictions)

        for method, pred in predictions_final.items():
            pred = self.scale_y.inverse_transform(pred.reshape(-1, 1))
            pred[np.where(pred < 0)] = 0
            predictions_final[method] = pred

        return predictions_final

    def compute_metrics(self, pred, y):
        if self.rated is None:
            rated = y.ravel()
        else:
            rated = self.rated
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def evaluate(self, pred_all, y):
        result = pd.DataFrame(index=[method for method in pred_all.keys()], columns=['sse', 'rms', 'mae', 'mse'])
        for method, pred in pred_all.items():
            if isinstance(pred, pd.DataFrame):
                result.loc[method] = self.compute_metrics(pred.values, y)
            else:
                result.loc[method] = self.compute_metrics(pred, y)

        return result

    def predict_offline(self):
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data_train()

            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X_cnn = X_cnn[index]
                index = X.index[index]
                X = X.loc[index]
            else:
                index = indices

            pred_cluster, predictions_cluster = self.predict_clusters(X, X_cnn, X_lstm)
            predictions_final_temp = self.predict_model(pred_cluster, predictions_cluster)
            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            if self.static_data['is_probabilistic']:
                if not os.path.exists(os.path.join(self.static_data['path_data'], 'cvs_proba.pickle')):
                    proba_model = proba_model_manager(self.static_data)
                    if not proba_model.istrained:
                        from sklearn.model_selection import train_test_split
                        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
                        X_pred = np.array([])
                        for method, pred in predictions_final.items():
                            if X_pred.shape[0] == 0:
                                X_pred = scale_y.transform(predictions_final[method].values.reshape(-1, 1))
                            else:
                                X_pred = np.hstack((X_pred, scale_y.transform(predictions_final[method].values.reshape(-1, 1))))
                        X_pred[np.where(X_pred < 0)] = 0
                        ind_nan = np.where(np.all(~np.isnan(X_pred), axis=1))
                        X_pred = X_pred[ind_nan]
                        y = scale_y.transform(y.values[ind_nan].reshape(-1, 1))
                        cvs = []
                        for _ in range(3):
                            X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pred, y, test_size=0.15)
                            X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.15)
                            cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])

                        joblib.dump(cvs, os.path.join(self.static_data['path_data'], 'cvs_proba.pickle'))

            return predictions_final
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])

    def predict_proba(self, X):
        proba_model = proba_model_manager(self.static_data)
        if not proba_model.istrained:
            return proba_model.predict(X)
        else:
            raise ModuleNotFoundError('Probabilistic Model %s is not trained', self.static_data['_id'])

    def predict_online(self, X, X_cnn=np.array([]), X_lstm=np.array([]), njobs=1):
        if len(X_cnn.shape) > 1:
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
        if self.istrained:
            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = X.index[np.where(X['flux'] > 1e-8)[0]]
                X = X.loc[index]
                X_cnn = X_cnn[np.where(X['flux'] > 1e-8)[0]]
            else:
                index = indices

            pred_cluster, predictions = self.predict_clusters(X, X_cnn, X_lstm, njobs=njobs)
            predictions_final_temp = self.predict_model(pred_cluster, predictions)
            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            return predictions_final
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])

    def predict_test(self):
        data_path = self.static_data['path_data']
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data_test()
            y_test = y.copy()
            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]
                index = indices[index]
            else:
                index = indices

            pred_cluster, predictions = self.predict_clusters(X, X_cnn, X_lstm, y)
            predictions_final_temp = self.predict_model(pred_cluster, predictions)

            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            return predictions_final
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])
    def evaluate_short_term(self, best_method):
        data_path = self.static_data['path_data']
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data_test()
            X=X.fillna(method='bfill')
            y=y.fillna(method='bfill')
            X_cnn =np.nan_to_num(X_cnn,0)
            X_lstm =np.nan_to_num(X_lstm,0)

            y_test = y.copy()
            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]
                index = indices[index]
            else:
                index = indices

            pred_cluster, predictions = self.predict_clusters(X, X_cnn, X_lstm, y)
            predictions_final_temp = self.predict_model(pred_cluster, predictions)

            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

        return predictions_final[best_method], y_test

    def evaluate_all(self):
        data_path = self.static_data['path_data']
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data_test()
            y_test = y.copy()
            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                y = y.iloc[index]
                X_cnn = X_cnn[index]
                index = indices[index]
            else:
                index = indices

            pred_cluster, predictions = self.predict_clusters(X, X_cnn, X_lstm, y)
            predictions_final_temp = self.predict_model(pred_cluster, predictions)

            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            if y_test is not None:
                result_all = self.evaluate(predictions_final, y_test.values)
                result_all.to_csv(os.path.join(data_path, 'result_final.csv'))
                joblib.dump(predictions_final, os.path.join(data_path, 'predictions_final.pickle'))
                y_test.to_csv(os.path.join(data_path, 'target_test.csv'))
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])

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
