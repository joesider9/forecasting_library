import os
import joblib

class SKLearn_object(object):
    def __init__(self, static_data, cluster, method, optimizer):
        self.static_data = static_data
        self.path_group =self.static_data['path_group']
        self.optimize_method = optimizer
        self.method = method
        self.istrained = False
        self.njobs = static_data['sklearn']['njobs']
        self.rated = static_data['rated']
        self.models = dict()
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.sk_models_dir = os.path.join(self.cluster_dir, 'SKLearn')
        self.model_dir = os.path.join(self.sk_models_dir, str.upper(method))
        if not os.path.exists(self.sk_models_dir):
            os.makedirs(self.sk_models_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
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