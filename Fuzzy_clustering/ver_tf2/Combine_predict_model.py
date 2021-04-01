import os
import numpy as np
import joblib
from Fuzzy_clustering.ver_tf2.Sklearn_predict import sklearn_model_predict
from Fuzzy_clustering.ver_tf2.LSTM_predict_3d import LSTM_3d_predict

class Combine_overall_predict(object):
    def __init__(self, static_data):
        self.istrained = False
        self.model_dir = os.path.join(static_data['path_model'], 'Combine_module')
        try:
            self.load(self.model_dir)
        except:
            pass
        self.static_data = static_data
        self.model_type = static_data['type']
        self.combine_methods = static_data['combine_methods']
        self.methods = []
        for method in static_data['project_methods'].keys():
            if self.static_data['project_methods'][method]['Global'] == True and static_data['project_methods'][method][
                'status'] == 'train':
                if method == 'ML_RBF_ALL_CNN':
                    self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                elif method == 'ML_RBF_ALL':
                    self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
                else:
                    self.methods.append(method)
        self.methods += self.combine_methods

        self.rated = static_data['rated']
        self.n_jobs = 2 * static_data['njobs']

        self.data_dir = self.static_data['path_data']

    def bcp_predict(self, X, w):
        preds = []
        for inp in X:
            inp=inp.reshape(-1,1)
            mask=~np.isnan(inp)
            pred = np.matmul(w[mask.T]/np.sum(w[mask.T]), inp[mask])
            preds.append(pred)

        return np.array(preds)


    def lstm_predict(self, X, full=False):
        if full:
            cluster_dir = os.path.join(self.model_dir, 'LSTM_best')
        else:
            cluster_dir = os.path.join(self.model_dir, 'LSTM_combine')

        lstm_model = LSTM_3d_predict(self.static_data, self.rated, cluster_dir)
        if lstm_model.istrained==True:
            model = lstm_model.predict(X)
        else:
            raise ImportError('Cannot find LSTM for overall combine')

        return model

    def predict(self, pred_cluster, predictions, lstm=False):
        if self.istrained==True:
            pred_combine = dict()

            combine_method = 'average'
            for method in self.methods:
                if method in predictions.keys():
                    pred = predictions[method].mean(axis=1).values.astype('float').reshape(-1, 1)
                    pred[np.where(pred < 0)] = 0
                    pred_combine['average_' + method] = pred

            if hasattr(self, 'models'):
                combine_method = 'bcp'
                for method in self.combine_methods:
                    if 'bcp_'+method in self.models.keys():
                        pred = self.bcp_predict(predictions[method].values.astype('float'), self.models['bcp_'+method])
                        pred[np.where(pred < 0)] = 0
                        pred_combine['bcp_' + method] = pred

                for method in self.combine_methods:
                    X_pred = predictions[method].values.astype('float')
                    X_pred[np.where(np.isnan(X_pred))] = 0
                    X_pred /= 20
                    mlp_model = sklearn_model_predict(self.model_dir + '/' + method, self.rated, 'mlp', self.n_jobs)
                    if mlp_model.istrained == True:
                        pred = mlp_model.predict(X_pred)
                        pred[np.where(pred < 0)] = 0
                        pred_combine['mlp_' + method] = 20 * pred
            if lstm:
                X = np.array([])
                combine_method = 'lstm_full'
                N = predictions['average'].values.shape[0]
                for clust in pred_cluster.keys():
                    x = np.array([])
                    for method in pred_cluster[clust]:
                        if method in self.methods:
                            tmp = np.zeros([N, 1])
                            try:
                                tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method]
                            except:
                                tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method].reshape(-1, 1)
                            if x.shape[0] == 0:
                                x = tmp
                            else:
                                x = np.hstack((x, tmp))
                    if X.shape[0] == 0:
                        X = np.copy(x)
                    elif len(X.shape) == 2:
                        X = np.stack((X, x))
                    else:
                        X = np.vstack((X, x[np.newaxis, :, :]))
                X = np.transpose(X, [1, 0, 2]).astype('float')

                pred = self.lstm_predict(X, full=True)
                pred[np.where(pred < 0)] = 0
                pred_combine[combine_method] = 20 * pred

                X = np.array([])
                combine_method = 'lstm_combine'

                for clust in pred_cluster.keys():
                    x = np.array([])
                    for method in pred_cluster[clust]:
                        if method in self.combine_methods:
                            tmp = np.zeros([N, 1])
                            try:
                                tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method]
                            except:
                                tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method].reshape(-1, 1)
                            if x.shape[0] == 0:
                                x = tmp
                            else:
                                x = np.hstack((x, tmp))
                    if X.shape[0] == 0:
                        X = np.copy(x)
                    elif len(X.shape) == 2:
                        X = np.stack((X, x))
                    else:
                        X = np.vstack((X, x[np.newaxis, :, :]))
                X = np.transpose(X, [1, 0, 2]).astype('float')
                pred = self.lstm_predict(X)
                pred[np.where(pred < 0)] = 0
                pred_combine[combine_method] = 20 * pred
        else:
            raise ImportError('Combine overall model seems not trained')

        return pred_combine

    def load(self, pathname):
        cluster_dir = os.path.join(pathname)
        if os.path.exists(os.path.join(cluster_dir, 'combine_models.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'combine_models.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RLS model')
        else:
            raise ImportError('Cannot find RLS model')