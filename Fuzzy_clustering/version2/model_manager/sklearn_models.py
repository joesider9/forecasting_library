import os

import joblib


class SKLearn_Model(object):
    def __init__(self, static_data, cluster, method, optimizer):
        self.static_data = static_data
        self.optimize_method = optimizer
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

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'SKlearnManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'SKlearnManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open SKlearn model')
        else:
            raise ImportError('Cannot find SKlearn model')
