import os
import pickle

import joblib


class FeatSelManager(object):
    def __init__(self, cluster):
        self.istrained = False
        self.method = cluster.static_data['sklearn']['fs_method']
        self.njobs = cluster.static_data['njobs_feat_sel']
        self.inner_jobs = cluster.static_data['inner_jobs_feat_sel']
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.pca = None
        if self.method == 'boruta':
            self.model_dir = os.path.join(cluster.cluster_dir, 'FS/boruta')
        else:
            self.model_dir = os.path.join(cluster.cluster_dir, 'FS/PERM')
        try:
            self.load()
        except:
            pass

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def transform(self, X):
        if self.pca is None:
            return X[:, self.features]
        else:
            return self.pca.transform(X[:, self.features])

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'model_fs.pickle')):
            try:
                f = open(os.path.join(self.model_dir, 'model_fs.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
                self.pca = joblib.load(os.path.join(self.model_dir, 'pca.pickle'))
            except:
                raise ValueError('Cannot find model for %s', self.model_dir)
        else:
            raise ValueError('Cannot find model for %s', self.model_dir)
