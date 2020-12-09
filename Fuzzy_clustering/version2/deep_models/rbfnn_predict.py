import os
import pickle

import numpy as np


class rbf_model_predict(object):
    def __init__(self, static_data, rated, cluster_dir):
        self.static_data = static_data
        self.cluster = os.path.basename(cluster_dir)
        self.rated = rated
        self.cluster_dir = os.path.join(cluster_dir, 'RBFNN')
        self.model_dir = os.path.join(self.cluster_dir, 'model')
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def predict(self, X):
        p = []
        self.load(self.model_dir)
        if self.istrained:
            for i in range(len(self.models)):
                centroids = self.models[i]['centroids']
                radius = self.models[i]['Radius']
                w = self.models[i]['W']
                s = X.shape
                d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [self.num_centr, 1, 1]), [1, 0, 2]) - np.tile(
                    np.expand_dims(centroids, axis=0), [s[0], 1, 1])
                d = np.sqrt(
                    np.sum(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])), 2), axis=2))
                phi = np.exp((-1) * np.power(d, 2))
                p.append(np.matmul(phi, w))
            p = np.mean(np.array(p), axis=0)
        else:
            raise ModuleNotFoundError("Error on prediction of %s cluster. The model RBFNN seems not properly trained",
                                      self.cluster)
        return p

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'rbfnn' + '.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'rbfnn' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()

                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RBFNN model')
        else:
            raise ImportError('Cannot find RBFNN model')
