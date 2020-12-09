import os
import pickle

import joblib
import numpy as np


class sklearn_model_predict(object):

    def __init__(self, static_data, cluster_dir, rated, model_type, njobs):
        self.static_data = static_data
        self.njobs = njobs
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type = model_type

        try:
            self.load(self.model_dir)
        except:
            pass

    def compute_metrics(self, pred, y, rated):
        if rated is None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def predict(self, X):
        self.load(self.model_dir)
        if self.istrained:
            pred = self.model.predict(X).reshape(-1, 1)
        else:
            raise ModuleNotFoundError("Error on prediction of %s cluster. The model %s seems not properly trained",
                                      self.cluster, self.model_type)
        return pred

    def load(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        if os.path.exists(os.path.join(model_dir, 'model_all' + '.pickle')):
            try:
                f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open model_all model')
        else:
            raise ImportError('Cannot find model_all model')
