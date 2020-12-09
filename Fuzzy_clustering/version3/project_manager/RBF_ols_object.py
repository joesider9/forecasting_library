import os
import joblib


class RBFOLS_manager_object(object):
    def __init__(self, static_data, cluster):
        self.static_data = static_data
        self.istrained = False
        self.njobs = cluster.static_data['sklearn']['njobs']
        self.models = dict()
        self.cluster_name = cluster.cluster_name
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.model_dir = os.path.join(self.cluster_dir, 'RBF_OLS')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load()
        except:
            pass



    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'RBFolsManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'RBFolsManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

