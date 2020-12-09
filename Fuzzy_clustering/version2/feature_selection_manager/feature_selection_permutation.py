import copy
import joblib
import os

import numpy as np
from sklearn.decomposition import PCA

from Fuzzy_clustering.version2.sklearn_models.sklearn_models_optuna import sklearn_model


class FS(object):
    def __init__(self, static_data, model_path, njobs, inner_jobs, path_group=None):
        self.static_data = static_data
        self.path_group = path_group
        self.njobs = njobs
        self.inner_jobs = inner_jobs
        self.log_dir = os.path.join(model_path, 'FS/PERM')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def reduce_dim(self, cvs):
        ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
        gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

        njobs = int(ncpus - gpu_status)
        cpu_status = njobs
        joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))

        for i in range(3):
            cvs[i][0] = cvs[i][0][:, self.features]
            cvs[i][2] = cvs[i][2][:, self.features]
            cvs[i][4] = cvs[i][4][:, self.features]

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        X_train = np.vstack((X_train, X_val, X_test))
        y_train = np.vstack((y_train, y_val, y_test))

        reduction = np.linspace(48, self.N_tot, self.N_tot - 48) / np.logspace(0, 0.3, self.N_tot - 48)
        n_components = reduction[int(X_train.shape[1] - 48 - 1)]
        pca = PCA(n_components=n_components)
        pca.fit(X_train)

        return pca

    def fit(self, cvs):
        # logger = logging.getLogger('log_fs_permutation')
        # logger.setLevel(logging.INFO)
        # handler = logging.FileHandler(os.path.join(self.log_dir, 'log_fs_perm.log'), 'w')
        # handler.setLevel(logging.INFO)
        #
        # # create a logging format
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        #
        # # add the handlers to the logger
        # logger.addHandler(handler)

        print()
        print('Training the model (Fitting to the training data) ')
        # logger.info('Training the feature extraction ')

        method = 'rf'

        ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
        gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

        njobs = int(ncpus - gpu_status)
        cpu_status = njobs
        joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))

        regressor = sklearn_model(self.static_data, self.log_dir, 1, method, njobs, FS=True, path_group=self.path_group)
        regressor.train(cvs)

        self.N_tot = cvs[0][0].shape[1]

        features = np.arange(cvs[0][0].shape[1])
        np.random.shuffle(features)
        # features=features[np.argsort(regressor.model.feature_importances_)[::-1]]

        acc_test = regressor.acc_test

        # cv_result = regressor.cv_results.nlargest(10, 'acc')['params'].to_list()
        flag = True

        cvs_temp = copy.deepcopy(cvs)

        remove_features = []
        keep_features = []
        unchecked = np.copy(features)
        while flag:
            for f in unchecked:

                ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
                gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

                njobs = int(ncpus - gpu_status)
                cpu_status = njobs
                joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))

                features_temp = np.hstack(
                    (np.array(keep_features), np.delete(unchecked, np.where(unchecked == f)))).astype('int')
                reg_temp = sklearn_model(self.static_data, os.path.join(self.log_dir, 'temp'), 1, method, njobs,
                                         FS=True, path_group=self.path_group)
                for i in range(3):
                    cvs_temp[i][0] = copy.deepcopy(cvs[i][0][:, features_temp])
                    cvs_temp[i][2] = copy.deepcopy(cvs[i][2][:, features_temp])
                    cvs_temp[i][4] = copy.deepcopy(cvs[i][4][:, features_temp])
                reg_temp.train(cvs_temp)

                # cv_result = reg_temp.cv_results.nlargest(5, 'acc')['params'].to_list()
                if (reg_temp.acc_test - acc_test) < -0.005:
                    # logger.info('Remove feature %s accuracy: %s', str(f), str(reg_temp.acc_test))
                    print('Remove feature ', str(f), ' accuracy: ', str(reg_temp.acc_test))
                    remove_features.append(f)
                    unchecked = np.delete(unchecked, np.where(unchecked == f))
                    acc_test = reg_temp.acc_test
                    break
                else:
                    print('ADD feature ', str(f), ' accuracy:', str(reg_temp.acc_test))
                    # logger.info('ADD feature %s accuracy: %s', str(f), str(reg_temp.acc_test))
                    keep_features.append(f)
                    unchecked = np.delete(unchecked, np.where(unchecked == f))

            if unchecked.shape[0] == 0:
                flag = False
            else:
                np.random.shuffle(unchecked)

        features = np.array(keep_features)
        self.features = features

        if self.features.shape[0] > 48:
            pca = self.reduce_dim(cvs)
        else:
            pca = None
            # logger.info('Number of variables %s', str(self.features.shape[0]))
            # logger.info('Finish the feature extraction ')
        return features, pca
#
# def test_fs_permute(cvs, X_test1,  y_test1, cluster_dir):
#
#     logger = logging.getLogger('log_rbf_cnn_test.log')
#     logger.setLevel(logging.INFO)
#     handler = logging.FileHandler(os.path.join(cluster_dir, 'log_rbf_cnn_test.log'), 'a')
#     handler.setLevel(logging.INFO)
#
#     # create a logging format
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     # add the handlers to the logger
#     logger.addHandler(handler)
#
#     rated = None
#
#     static_data = write_database()
#
#     logger.info('Permutation Evaluation')
#     logger.info('/n')
#     method = 'svm'
#     model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
#     model_sklearn.train(cvs)
#     pred = model_sklearn.predict(X_test1)
#
#     metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
#     logger.info('before feature selection metrics')
#     logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
#
#     fs = FS(cluster_dir, static_data['sklearn']['njobs'])
#     features = fs.fit(cvs)
#     logger.info('Number of variables %s', str(features.shape[0]))
#
#     for i in range(3):
#         cvs[i][0] = cvs[i][0][:,features]
#         cvs[i][2] = cvs[i][2][:,features]
#         cvs[i][4] = cvs[i][4][:,features]
#
#     model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
#     model_sklearn.train(cvs)
#     pred = model_sklearn.predict(X_test1[:,features])
#
#     metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
#     logger.info('After feature selection metrics')
#     logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
