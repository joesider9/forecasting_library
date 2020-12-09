import os
import pickle

import joblib


class FeatSelManager(object):
    def __init__(self, cluster):
        self.istrained = False
        self.static_data = cluster.static_data
        self.cluster_name = cluster.cluster_name
        self.method = cluster.static_data['sklearn']['fs_method']
        self.njobs = cluster.static_data['njobs_feat_sel']
        self.inner_jobs = cluster.static_data['inner_jobs_feat_sel']
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
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

    def load_data(self):
        data_path = self.data_dir
        if os.path.exists(os.path.join(data_path, 'cvs_full.pickle')):
            cvs = joblib.load(os.path.join(data_path, 'cvs_full.pickle'))
        else:
            raise ImportError('Cannot find data for cluster %s of %s', self.cluster_name, self.static_data['_id'])
        return cvs

    def save_data(self, cvs):
        data_path = self.data_dir
        for i in range(3):
            if self.pca is None:
                cvs[i][0] = cvs[i][0][:, self.features]
                cvs[i][2] = cvs[i][2][:, self.features]
                cvs[i][4] = cvs[i][4][:, self.features]
            else:
                cvs[i][0] = self.pca.transform(cvs[i][0][:, self.features])
                cvs[i][2] = self.pca.transform(cvs[i][2][:, self.features])
                cvs[i][4] = self.pca.transform(cvs[i][4][:, self.features])
        joblib.dump(cvs, os.path.join(data_path, 'cvs.pickle'))

    def fit(self):
        cvs = self.load_data()
        if self.method == 'boruta':
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_boruta import FS
            fs = FS(self.static_data, self.cluster_dir, self.njobs, path_group=self.static_data['path_group'])
        elif self.method == 'linearsearch':
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_linearsearch import FS
            fs = FS(self.static_data, self.cluster_dir, self.njobs, self.inner_jobs,
                    path_group=self.static_data['path_group'])
        else:
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_permutation import FS
            fs = FS(self.static_data, self.cluster_dir, self.njobs, self.inner_jobs,
                    path_group=self.static_data['path_group'])

        self.features, self.pca = fs.fit(cvs)
        self.save_data(cvs)
        self.istrained = True
        self.save()
        return 'Done'

    def fit_TL(self):
        cvs = self.load_data()
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)

        if self.method == 'boruta':
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_boruta import FS
            fs_trained = FS(static_data_tl, cluster_dir_tl, self.njobs, path_group=self.static_data['path_group'])
        elif self.method == 'linearsearch':
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_linearsearch import FS
            fs_trained = FS(static_data_tl, cluster_dir_tl, self.njobs, self.inner_jobs,
                            path_group=self.static_data['path_group'])
        else:
            from Fuzzy_clustering.version2.feature_selection_manager.feature_selection_permutation import FS
            fs_trained = FS(static_data_tl, cluster_dir_tl, self.njobs, self.inner_jobs,
                            path_group=self.static_data['path_group'])
        self.features = fs_trained.features
        self.pca = fs_trained.pca
        self.save_data(cvs)
        self.istrained = True
        self.save()
        return 'Done'

    def transform(self, X):
        if self.pca is None:
            return X[:, self.features]
        else:
            return self.pca.transform(X[:, self.features])

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'data_dir', 'cluster_dir', 'model_dir']:
                dict[k] = self.__dict__[k]
        return dict

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

    def save(self):
        joblib.dump(self.pca, os.path.join(self.model_dir, 'pca.pickle'))
        f = open(os.path.join(self.model_dir, 'model_fs.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'data_dir', 'cluster_dir', 'model_dir', 'pca']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
