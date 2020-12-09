import os

import joblib
import numpy as np

from Fuzzy_clustering.version2.deep_models.cnn_predict import CNN_predict
from Fuzzy_clustering.version2.deep_models.cnn_predict_3d import CNN_3d_predict
from Fuzzy_clustering.version2.deep_models.lstm_predict_3d import LSTM_3d_predict


class model3d():
    def __init__(self, static_data, cluster, method):
        self.cluster = cluster
        self.static_data = static_data
        self.method = str.lower(method)
        self.istrained = False
        if self.method == 'cnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'CNN_3d')
        elif self.method == 'lstm':
            self.model_dir = os.path.join(cluster.cluster_dir, 'LSTM_3d')
        elif self.method == 'rbfnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'RBFNN')
        elif self.method == 'rbf-cnn':
            self.model_dir = os.path.join(cluster.cluster_dir, 'RBF_CNN')
        try:
            self.load()

        except:
            pass

        self.static_data = static_data
        self.cluster_name = cluster.cluster_name
        self.rated = static_data['rated']
        self.probabilistic = False

    def predict(self, X, rbf_models=None):
        if self.istrained:
            if self.method == 'cnn':
                return self.predict_cnn(X)
            elif self.method == 'lstm':
                return self.predict_lstm(X)
            elif self.method == 'rbfnn':
                return self.predict_rbf(X)
            elif self.method == 'rbf-cnn':
                return self.predict_rbf_cnn(X, rbf_models)
        else:
            raise ImportError('Model %s is not trained for cluster %s of project', self.method,
                              self.cluster.cluster_name, self.static_data['_id'])

    def predict_rbf(self, X):
        if self.istrained:
            centroids = self.model['centroids']
            radius = self.model['Radius']
            w = self.model['W']
            s = X.shape
            d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [self.model['num_centr'], 1, 1]), [1, 0, 2]) - np.tile(
                np.expand_dims(centroids, axis=0), [s[0], 1, 1])
            d = np.sqrt(
                np.sum(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])), 2), axis=2))
            phi = np.exp((-1) * np.power(d, 2))
            pred = np.matmul(phi, w)
        else:
            raise ImportError('Model %s is not trained for cluster %s of project', self.method,
                              self.cluster.cluster_name, self.static_data['_id'])
        return pred

    def predict_lstm(self, X):
        model_predict = LSTM_3d_predict(self.model, self.scale_lstm, self.trial, self.probabilistic)
        if self.istrained:
            pred = model_predict.predict(X)
            pred[np.where(pred < 0)] = 0
            pred[np.where(pred > 1)] = 1
        else:
            raise ImportError('Model %s is not trained for cluster %s of project', self.method,
                              self.cluster.cluster_name, self.static_data['_id'])
        return pred

    def predict_cnn(self, X):
        model_predict = CNN_3d_predict(self.model, self.scale_cnn, self.trial, self.pool_size)
        if self.istrained:
            pred = model_predict.predict(X)
            pred[np.where(pred < 0)] = 0
            pred[np.where(pred > 1)] = 1
        else:
            raise ImportError('Model %s is not trained for cluster %s of project', self.method,
                              self.cluster.cluster_name, self.static_data['_id'])
        return pred

    def predict_rbf_cnn(self, X, rbf_models):
        model_predict = CNN_predict(self.model, self.scale_cnn, self.trial, self.pool_size, rbf_models)
        if self.istrained:
            pred = model_predict.predict(X)
        else:
            raise ImportError('Model %s is not trained for cluster %s of project', self.method,
                              self.cluster.cluster_name, self.static_data['_id'])
        return pred

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')
