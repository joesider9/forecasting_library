import numpy as np
import pandas as pd
import logging, os
from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import sklearn_model
import copy
from util_database import write_database

class FS(object):
    def __init__(self, model_path, njobs):
        self.njobs=njobs
        self.log_dir=os.path.join(model_path, 'FS/PERM')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def fit(self, cvs):
        logger = logging.getLogger('log_fs_permutation')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, 'log_fs_perm.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print()
        print('Training the model (Fitting to the training data) ')
        logger.info('Training the feature extraction ')

        method = 'svm'

        regressor = sklearn_model(self.log_dir, 1, method, self.njobs)
        regressor.train(cvs)

        # Update classifier parameters
        estimator=regressor.model

        features = np.arange(cvs[0][0].shape[1])
        np.random.shuffle(features)
        # features=features[np.argsort(estimator.feature_importances_)]

        acc_test = regressor.acc_test

        cv_result = regressor.cv_results.nlargest(10, 'acc')['params'].to_list()
        flag = True

        cvs_temp = copy.deepcopy(cvs)

        remove_features = []
        keep_features = []
        unchecked =  np.copy(features)
        while flag:
            for f in unchecked:
                features_temp = np.hstack((np.array(keep_features),np.delete(unchecked, np.where(unchecked==f)))).astype('int')
                reg_temp = sklearn_model(os.path.join(self.log_dir, 'temp'), 1, method, self.njobs)
                for i in range(3):
                    cvs_temp[i][0] = copy.deepcopy(cvs[i][0][:, features_temp])
                    cvs_temp[i][2] = copy.deepcopy(cvs[i][2][:, features_temp])
                    cvs_temp[i][4] = copy.deepcopy(cvs[i][4][:, features_temp])
                reg_temp.train(cvs_temp)

                cv_result = reg_temp.cv_results.nlargest(5, 'acc')['params'].to_list()
                if reg_temp.acc_test < acc_test:
                    logger.info('Remove feature %s accuracy: %s', str(f), str(reg_temp.acc_test))
                    remove_features.append(f)
                    unchecked = np.delete(unchecked, np.where(unchecked == f))
                    acc_test = reg_temp.acc_test
                    break
                else:
                    logger.info('ADD feature %s accuracy: %s', str(f), str(reg_temp.acc_test))
                    keep_features.append(f)
                    unchecked = np.delete(unchecked, np.where(unchecked==f))

            if unchecked.shape[0]==0:
                flag = False
            else:
                np.random.shuffle(unchecked)

        features = np.array(keep_features)
        self.features = features

        logger.info('Number of variables %s', str(self.features.shape[0]))
        logger.info('Finish the feature extraction ')
        return features

def test_fs_permute(cvs, X_test1,  y_test1, cluster_dir):

    logger = logging.getLogger('log_rbf_cnn_test.log')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(cluster_dir, 'log_rbf_cnn_test.log'), 'a')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    rated = None

    static_data = write_database()

    logger.info('Permutation Evaluation')
    logger.info('/n')
    method = 'svm'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('before feature selection metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)

    fs = FS(cluster_dir, static_data['sklearn']['njobs'])
    features = fs.fit(cvs)
    logger.info('Number of variables %s', str(features.shape[0]))

    for i in range(3):
        cvs[i][0] = cvs[i][0][:,features]
        cvs[i][2] = cvs[i][2][:,features]
        cvs[i][4] = cvs[i][4][:,features]

    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1[:,features])

    metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('After feature selection metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
