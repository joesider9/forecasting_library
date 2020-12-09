import numpy as np
import pandas as pd
import joblib, os, pickle
from Fuzzy_clustering.version3.ModelCombineManager.Clusterer import clusterer
from Fuzzy_clustering.version3.ModelCombineManager.ClusterPredictManager import ClusterPredict

class FullClusterPredictManager(object):

    def __init__(self, path_model, static_data):
        self.path_model = path_model
        self.static_data = static_data
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']
        try:
            self.load()
        except:
            pass

    def load_data(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
        else:
            X_cnn = np.array([])

        index = X.index
        if self.static_data['type'] == 'pv' and self.static_data['NWP_model'] == 'skiron':
            index = np.where(X['flux'] > 1e-8)[0]
            X = X.iloc[index]
            if X_cnn.shape[0]>0:
                X_cnn = X_cnn[index]
        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
        else:
            X_lstm = np.array([])

        return X, y, X_cnn, X_lstm, index

    def load_test_data(self):
        X, y, X_cnn, X_lstm, index = self.load_data()

        test_ind = np.where(X.index >= self.split_test)[0]
        index = index[index >= self.split_test]
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
        return X_test, y_test, X_cnn_test, X_lstm_test, index

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

    def predict_clusters(self, X_test = pd.DataFrame([]), y_test = pd.DataFrame([]), X_cnn_test = np.array([]), X_lstm_test = np.array([]), test = True):
        if X_test.shape[0]==0:
            offline = True
        else:
            offline = False
        if offline:
            if test:
                X_test, y_test, X_cnn_test, X_lstm_test, index = self.load_test_data()
            else:
                X_test, y_test, X_cnn_test, X_lstm_test, index = self.load_data()
        sc = joblib.load(os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
        pred_cluster = dict()
        X_test = pd.DataFrame(sc.transform(X_test.values), columns=X_test.columns, index=X_test.index)
        if y_test.shape[0]>0:
            y_test = pd.DataFrame(scale_y.transform(y_test.values), columns=y_test.columns, index=y_test.index)
        if not hasattr(self, 'clusterer'):
            self.clusterer = clusterer(self.static_data['path_fuzzy_models'])
        act_test = self.clusterer.compute_activations(X_test)
        act_test = self.check_if_all_nans(act_test)
        for clust in self.clusters.keys():
            predict_module = ClusterPredict(self.static_data, self.clusters[clust])
            if clust == 'global':
                if len(self.clusters[clust].methods) > 0:
                    pred_cluster[clust] = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test)
                    if y_test.shape[0] > 0:
                        pred_cluster[clust]['metrics'] = predict_module.evaluate(pred_cluster['global'], y_test.values)
                    pred_cluster[clust]['dates'] = X_test.index
                    pred_cluster[clust]['index'] = np.arange(0, X_test.shape[0])
            else:
                dates = X_test.index[act_test[clust] >= self.thres_act]
                nind = np.where(act_test[clust] >= self.thres_act)[0]
                nind.sort()

                x = X_test.loc[dates]
                if y_test.shape[0] > 0:
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
                if y_test.shape[0] > 0:
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
        if offline:
            if y_test.shape[0] > 0:
                return pred_cluster, predictions, y_test, index
            else:
                return pred_cluster, predictions, index
        else:
            if y_test.shape[0] > 0:
                return pred_cluster, predictions, y_test
            else:
                return pred_cluster, predictions

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
