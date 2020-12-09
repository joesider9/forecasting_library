import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Fuzzy_clustering.version3.project_manager.PredictModelManager.Model_3d_object import model3d_object
from Fuzzy_clustering.version3.project_manager.PredictModelManager.RBF_CNN_model import RBF_CNN_model
from Fuzzy_clustering.version3.project_manager.PredictModelManager.SKlearn_object import SKLearn_object
from Fuzzy_clustering.version3.project_manager.PredictModelManager.FS_object import FeatSelobject


class MultiEvaluator():
    def __init__(self, processes: int = 8):
        self.processes = processes

    def predict(self, i, x, model):
        return i, model.predict(x)

    def evaluate(self, X, model):
        partitions = 3000
        X_list=[]
        for i in range(0, X.shape[0], partitions):
            if (i+partitions+1)>X.shape[0]:
                X_list.append(X[i:])
            else:
                X_list.append(X[i:i+partitions])
        pred =Parallel(self.processes)(delayed(self.predict)(i, x, model) for i, x in enumerate(X_list))
        indices = np.array([p[0] for p in pred])
        predictions = np.array([])
        for ind in indices:
            if len(predictions.shape)==1:
                predictions = pred[ind][1]
            else:
                predictions = np.vstack((predictions, pred[ind][1]))

        return predictions

class ClusterPredict():
    def __init__(self, static_data, cluster):
        self.cluster = cluster
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.static_data = static_data
        self.model_type = static_data['type']
        self.methods = cluster.methods
        self.combine_methods = static_data['combine_methods']
        self.rated = static_data['rated']
        self.n_jobs = static_data['njobs']

    def pred_model(self, X, method, X_cnn=np.array([]), X_lstm=np.array([])):
        if (method == 'RBF_ALL'):
            model = RBF_CNN_model(self.static_data, self.cluster)
            pred = model.predict(X)
        elif (method == 'RBF_ALL_CNN'):
            model = RBF_CNN_model(self.static_data, self.cluster, cnn=True)
            pred = model.predict(X)
        elif method in {'CNN'}:
            model = model3d_object(self.static_data, self.cluster, method)
            pred = model.predict(X_cnn)
            pred[np.where(pred < 0)] = 0
        elif method in {'LSTM'}:
            model = model3d_object(self.static_data, self.cluster, method)
            pred = model.predict(X_lstm)
            pred[np.where(pred < 0)] = 0
        elif method in {'SVM', 'NUSVM', 'MLP', 'RF', 'XGB', 'elasticnet'}:
            model = SKLearn_object(self.static_data, self.cluster, method)
            pred = model.predict(X)
            pred[np.where(pred < 0)] = 0
        return pred

    def parallel_pred_model(self, X, method, X_cnn=np.array([]), X_lstm=np.array([])):
        parallel = MultiEvaluator(self.n_jobs)
        if (method == 'RBF_ALL'):
            model = RBF_CNN_model(self.static_data, self.cluster)
            pred = parallel.evaluate(X, model)
        elif (method == 'RBF_ALL_CNN'):
            model = RBF_CNN_model(self.static_data, self.cluster, cnn=True)
            pred = parallel.evaluate(X, model)
        elif method in {'CNN'}:
            model = model3d_object(self.static_data, self.cluster, method)
            pred = parallel.evaluate(X_cnn, model)
        elif method in {'LSTM'}:
            model = model3d_object(self.static_data, self.cluster, method)
            pred = parallel.evaluate(X_lstm, model)
        elif method in {'SVM', 'NUSVM', 'MLP', 'RF', 'XGB', 'elasticnet'}:
            model = SKLearn_object(self.static_data, self.cluster, method)
            pred = parallel.evaluate(X, model)
        return pred


    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = rated
        err = np.abs(pred.ravel() - y.ravel())/rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def evaluate(self, pred_all, y):
        result = pd.DataFrame(index=[method for method in pred_all.keys()], columns=['sse', 'rms', 'mae', 'mse'])
        for method, pred in pred_all.items():
            if not method in {'metrics'}:
                result.loc[method] = self.compute_metrics(pred, y, self.rated)
        return result

    def spark_predict(self, X, X_cnn=np.array([]), X_lstm=np.array([]), fs_reduced=False):
        if fs_reduced==False:
            fs_manager = FeatSelobject(self.cluster)
            if fs_manager.istrained==True:
                X = fs_manager.transform(X)

        predictions=dict()

        for method in self.methods:
            if X.shape[0]>0:
                pred = self.parallel_pred_model(X, method, X_cnn=X_cnn, X_lstm=X_lstm)

                for j in range(pred.shape[1]):
                    if np.any(np.isnan(pred[:, j])):
                        if np.sum(np.isnan(pred[:, j]))<=X.shape[0]/3:
                            pred[:, j][np.where(np.isnan(pred[:, j]))] = np.nanmean(pred[:, j])
                        else:
                            raise ValueError('There are nans in dataset of %s clust or model of method %s is not trained well', self.cluster_name, method)

                if method == 'RBF_ALL_CNN':
                    predictions['RBF_OLS'] = pred[:, 0].reshape(-1, 1)
                    predictions['GA_RBF_OLS'] = pred[:, 1].reshape(-1, 1)
                    predictions['RBFNN'] = pred[:, 2].reshape(-1, 1)
                    predictions['RBF-CNN'] = pred[:, 3].reshape(-1, 1)
                elif method == 'RBF_ALL':
                    predictions['RBF_OLS'] = pred[:, 0].reshape(-1, 1)
                    predictions['GA_RBF_OLS'] = pred[:, 1].reshape(-1, 1)
                    predictions['RBFNN'] = pred[:, 2].reshape(-1, 1)
                else:
                    if len(pred.shape)==1:
                        pred = pred.reshape(-1, 1)
                    predictions[method] = pred
            else:
                if method == 'RBF_ALL_CNN':
                    predictions['RBF_OLS'] = np.array([])
                    predictions['GA_RBF_OLS'] = np.array([])
                    predictions['RBFNN'] = np.array([])
                    predictions['RBF-CNN'] = np.array([])
                elif method == 'RBF_ALL':
                    predictions['RBF_OLS'] = np.array([])
                    predictions['GA_RBF_OLS'] = np.array([])
                    predictions['RBFNN'] = np.array([])
                else:
                    predictions[method] = np.array([])

        return predictions

    def predict(self, X, X_cnn=np.array([]), X_lstm=np.array([]), fs_reduced=False):


        if fs_reduced==False:
            fs_manager = FeatSelobject(self.cluster)
            if fs_manager.istrained==True:
                X = fs_manager.transform(X)

        predictions=dict()

        for method in self.methods:
            if X.shape[0]>0:
                pred = self.pred_model(X, method, X_cnn=X_cnn, X_lstm=X_lstm)

                for j in range(pred.shape[1]):
                    if np.any(np.isnan(pred[:, j])):
                        if np.sum(np.isnan(pred[:, j]))<=X.shape[0]/3:
                            pred[:, j][np.where(np.isnan(pred[:, j]))] = np.nanmean(pred[:, j])
                        else:
                            raise ValueError('There are nans in dataset of %s clust or model of method %s is not trained well', self.cluster_name, method)

                if method == 'RBF_ALL_CNN':
                    predictions['RBF_OLS'] = pred[:, 0].reshape(-1, 1)
                    predictions['GA_RBF_OLS'] = pred[:, 1].reshape(-1, 1)
                    predictions['RBFNN'] = pred[:, 2].reshape(-1, 1)
                    predictions['RBF-CNN'] = pred[:, 3].reshape(-1, 1)
                elif method == 'RBF_ALL':
                    predictions['RBF_OLS'] = pred[:, 0].reshape(-1, 1)
                    predictions['GA_RBF_OLS'] = pred[:, 1].reshape(-1, 1)
                    predictions['RBFNN'] = pred[:, 2].reshape(-1, 1)
                else:
                    if len(pred.shape)==1:
                        pred = pred.reshape(-1, 1)
                    predictions[method] = pred
            else:
                if method == 'RBF_ALL_CNN':
                    predictions['RBF_OLS'] = np.array([])
                    predictions['GA_RBF_OLS'] = np.array([])
                    predictions['RBFNN'] = np.array([])
                    predictions['RBF-CNN'] = np.array([])
                elif method == 'RBF_ALL':
                    predictions['RBF_OLS'] = np.array([])
                    predictions['GA_RBF_OLS'] = np.array([])
                    predictions['RBFNN'] = np.array([])
                else:
                    predictions[method] = np.array([])
        return predictions