import joblib
import os

class model3d_object():
    def __init__(self, static_data, cluster, method, params):
        self.params = params
        self.test = params['test']
        self.method = str.lower(method)
        self.cluster = cluster
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
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.test_dir = self.model_dir
        try:
            self.load()
        except:
            pass
        if not self.istrained:
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
            try:
                self.load()
            except:
                if not os.path.exists(self.test_dir):
                    os.makedirs(self.test_dir)
                pass

        self.static_data = static_data
        self.cluster_name = cluster.cluster_name
        self.rated = static_data['rated']
        self.data_dir = cluster.data_dir
        self.probabilistic = False

    def load(self):
        if os.path.exists(os.path.join(self.test_dir, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.test_dir, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(self.test_dir, self.method + '.pickle'), compress=9)

