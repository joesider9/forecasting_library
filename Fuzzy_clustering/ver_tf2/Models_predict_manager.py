import os
import pandas as pd
import numpy as np
import pickle
import logging, shutil, glob
import pymongo, joblib
from Fuzzy_clustering.ver_tf2.Clusterer import clusterer
from Fuzzy_clustering.ver_tf2.Cluster_predict_regressors import cluster_predict
from Fuzzy_clustering.ver_tf2.Global_predict_regressor import global_predict
from Fuzzy_clustering.ver_tf2.Combine_predict_model import Combine_overall_predict
from Fuzzy_clustering.ver_tf2.util_database import write_database

class ModelPredictManager(object):

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
            myclient = pymongo.MongoClient(
                "mongodb://" + self.static_data['url'] + ":" + self.static_data['port'] + "/")

            project_db = myclient[self.static_data['_id']]
        except:
            self.logger.info('Cannot open Database')
            self.use_db = False
            project_db = None
            raise ConnectionError('Cannot open Database')
        self.logger.info('Open Database successfully')
        return project_db

    def load_data(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X_test.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        if os.path.exists(os.path.join(data_path, 'dataset_y_test.csv')):
            y = pd.read_csv(os.path.join(data_path, 'dataset_y_test.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        else:
            y=None

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


    def predict_regressors(self, X_test, X_cnn_test, X_lstm_test, y_test=None):
        data_path = self.static_data['path_data']
        pred_cluster = dict()
        X_test = pd.DataFrame(self.sc.transform(X_test.values), columns=X_test.columns, index=X_test.index)
        if not hasattr(self, 'clusterer'):
            self.clusterer = clusterer(self.static_data['path_fuzzy_models'],
                                       self.static_data['clustering']['cluster_file'], self.static_data['type'])
        act_test = self.clusterer.compute_activations(X_test)
        act_test = self.check_if_all_nans(act_test)
        for clust in self.regressors.keys():
            if clust == 'Global':
                if len(self.regressors['Global']['models']) > 0:
                    predict_module = global_predict(self.static_data)
                    pred_cluster['Global'] = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test)
                    if y_test is not None:
                        pred_cluster['Global']['metrics'] = predict_module.evaluate(pred_cluster['Global'], self.scale_y.transform(y_test.values))
                    pred_cluster['Global']['dates'] = X_test.index
                    pred_cluster['Global']['index'] = np.arange(0, X_test.shape[0])
            else:
                dates = X_test.index[act_test[clust] >= self.thres_act]
                nind = np.where(act_test[clust] >= self.thres_act)[0]
                nind.sort()

                x = X_test.loc[dates]
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
                predict_module = cluster_predict(self.static_data, clust)
                pred_cluster[clust] = predict_module.predict(x.values, X_cnn=x_cnn, X_lstm=x_lstm)
                if targ is not None and targ.shape[0]>0:
                    pred_cluster[clust]['metrics'] = predict_module.evaluate(pred_cluster[clust], self.scale_y.transform(targ))
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

        combine_overall = Combine_overall_predict(self.static_data)
        predictions_final = combine_overall.predict(pred_cluster, predictions)

        for method, pred in predictions_final.items():
            pred = self.scale_y.inverse_transform(pred.reshape(-1, 1))
            pred[np.where(pred<0)] = 0
            predictions_final[method] = pred

        if y_test is not None:
            result_clust.to_csv(os.path.join(data_path, 'result_of_clusters.csv'))

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

    def predict(self):
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data()

            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                X_cnn = X_cnn[index]
            else:
                index = indices

            predictions_final_temp = self.predict_regressors(X, X_cnn, X_lstm)
            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            return predictions_final
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])

    def predict_online(self, X, X_cnn= np.array([]), X_lstm= np.array([])):
        if len(X_cnn.shape)>1:
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
        if self.istrained:
            indices = X.index
            if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
                index = np.where(X['flux'] > 1e-8)[0]
                X = X.iloc[index]
                X_cnn = X_cnn[index]
            else:
                index = indices

            predictions_final_temp = self.predict_regressors(X, X_cnn, X_lstm)
            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=indices, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

            return predictions_final
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])

    def evaluate_all(self):
        data_path = self.static_data['path_data']
        if self.istrained:
            X, X_cnn, X_lstm, y = self.load_data()
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

            predictions_final_temp = self.predict_regressors(X, X_cnn, X_lstm, y)

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
            if indices.shape[0]>50:
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

if __name__ == '__main__':
    from util_database import write_database
    from Fuzzy_clustering.ver_tf2.Projects_train_manager import ProjectsTrainManager

    static_data = write_database()
    project_manager = ProjectsTrainManager(static_data)
    project_manager.initialize()
    project_manager.create_datasets(project_manager.data_eval, test=True)
    project = [pr for pr in project_manager.group_static_data if pr['_id'] == 'Lach'][0]
    static_data = project['static_data']

    model = ModelPredictManager(static_data['path_model'])
    model.init(project['static_data'], project_manager.data_variables)
    model.evaluate_all()