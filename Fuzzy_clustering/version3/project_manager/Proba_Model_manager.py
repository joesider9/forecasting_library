import os
import numpy as np
import joblib

class proba_model_manager():
    def __init__(self, static_data, params={}):
        if len(params)>0:
            self.params = params
            self.test = params['test']
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
        self.istrained = False
        self.method = 'mlp'
        self.model_dir = os.path.join(static_data['path_model'], 'Probabilistic')
        self.data_dir = self.static_data['path_data']

        if hasattr(self, 'test'):
            try:
                self.load(self.test_dir)
            except:
                pass
        else:
            try:
                self.load(self.model_dir)
            except:
                pass

        self.static_data = static_data
        self.cluster_name = static_data['_id']
        self.rated = static_data['rated']
        self.probabilistic = True
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)


    def load(self, path):
        if os.path.exists(os.path.join(path, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(path, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')





