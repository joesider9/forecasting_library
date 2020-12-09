import os
import joblib

class SKLearn_object(object):
    def __init__(self, static_data, cluster, method):
        self.static_data = static_data
        self.cluster = cluster
        self.method = method
        self.istrained = False
        self.njobs = cluster.static_data['sklearn']['njobs']
        self.rated = cluster.static_data['rated']
        self.models = dict()
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.sk_models_dir = os.path.join(self.cluster_dir, 'SKLearn')
        self.model_dir = os.path.join(self.sk_models_dir, str.upper(method))
        try:
            self.load()
        except:
            pass

    def predict(self, X):
        if self.istrained==True and hasattr(self, 'models'):
            if self.method in self.models.keys():
                model_sklearn = self.load_model(self.model_dir)
                pred = model_sklearn.predict(X).reshape(-1, 1)
                pred[np.where(pred < 0)] = 0
                pred[np.where(pred > 1)] = 1
            else:
                raise ImportError('SKlearn Manager has not attribute %s for cluster %s of project', self.method, self.cluster.cluster_name, self.static_data['_id'])
        else:
            raise ImportError('SKlearn Manager has not attribute models for cluster %s of project', self.cluster.cluster_name, self.static_data['_id'])

        return pred

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'SKlearnManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'SKlearnManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open SKlearn model')
        else:
            raise ImportError('Cannot find SKlearn model')

    def load_model(self, model_dir):
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        return model