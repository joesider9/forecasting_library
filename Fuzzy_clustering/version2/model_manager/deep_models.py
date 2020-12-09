import os

import joblib


class model3d():
    def __init__(self, static_data, cluster, method):
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
        elif self.method == 'mlp_3d':
            self.model_dir = os.path.join(cluster.cluster_dir, 'MLP_3D')
        try:
            self.load()

        except:
            pass

        self.static_data = static_data
        self.cluster_name = cluster.cluster_name
        self.rated = static_data['rated']
        self.probabilistic = False

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')
