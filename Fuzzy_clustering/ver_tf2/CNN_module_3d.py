import os
import numpy as np
import pandas as pd
import pickle
import glob
import shutil
import logging
import re, sys, joblib, bz2
import multiprocessing as mp
import tensorflow as tf
from joblib import Parallel, delayed
from Fuzzy_clustering.ver_tf2.CNN_tf_core_3d import CNN_3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from scipy.interpolate import interp2d
# from util_database import write_database
# from Fuzzy_clustering.ver_tf2.Forecast_model import forecast_model
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from Fuzzy_clustering.ver_tf2.CNN_predict_3d import CNN_3d_predict

def optimize_cnn(cnn, kernels, hsize, cnn_max_iterations, cnn_learning_rate, gpu, filters):
    flag = False
    for _ in range(3):
        try:
            acc_old_cnn, scale_cnn, model_cnn = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                    learning_rate=cnn_learning_rate, kernels=kernels, h_size=hsize, gpu_id=gpu,filters=filters)
            flag=True
        except:
            filters = int(filters/2)
            pass

    if not flag:
        acc_old_cnn=np.inf
        scale_cnn=None
        model_cnn=None


    return acc_old_cnn, kernels, hsize, scale_cnn, model_cnn, cnn.pool_size, cnn.trial, cnn_learning_rate


def predict(q, H, model):
    tf.config.set_soft_device_placement(True)
    pred = model.predict(H)

    q.put((pred[0]))


class cnn_3d_model():
    def __init__(self, static_data, rated, cluster_dir):
        self.static_data_all = static_data
        self.static_data = static_data['CNN']
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.cluster_cnn_dir = os.path.join(cluster_dir, 'CNN_3d')
        self.model_dir = os.path.join(self.cluster_cnn_dir, 'model')
        self.cluster_dir = cluster_dir
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        try:
            self.load(self.model_dir)
        except:
            pass

    def train_cnn(self, X, y):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print('CNN training...begin for %s ', self.cluster)
        logger.info('CNN training...begin for %s ', self.cluster)

        if len(y.shape)==1:
            y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        results =[]
        for trial in [0, 3]:
            if trial != 0:
                pool_size = [1, 2, 2]
            else:
                pool_size = [2, 1]

            cnn = CNN_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, pool_size, trial=trial)

            self.acc_cnn = np.inf
            gpus = np.tile(self.static_data['gpus'], 4)

            if trial==0:
                kernels=[
                    # [2, 2],
                    [2, 4],
                    [4, 2],
                    # [4, 4]
                ]
            else:
                kernels = [
                    [2, 4, 4],
                    # [2, 2, 2],
                    [3, 2, 2],
                    # [3, 4, 4]
                ]
            # res = optimize_cnn(cnn, kernels[0], self.static_data['h_size'],
            #                                                self.static_data['max_iterations'],
            #                                                self.static_data['learning_rate'],
            #                                                gpus[0],int(self.static_data['filters']))
            res = Parallel(n_jobs=len(self.static_data['gpus']))(
                delayed(optimize_cnn)(cnn, kernels[k], self.static_data['h_size'],
                                                           self.static_data['max_iterations'],
                                                           self.static_data['learning_rate'],
                                                           gpus[int(k)], int(self.static_data['filters'])) for k in range(2))
            results += res

        for r in results:
            logger.info("kernel: %s accuracy cnn: %s", r[1], r[0])

        acc_cnn = np.array([r[0] for r in results])
        self.acc_cnn, self.best_kernel, hsize, self.scale_cnn, model_cnn, self.pool_size, self.trial, lr= results[acc_cnn.argmin()]
        self.model = model_cnn
        train_res = pd.DataFrame.from_dict(model_cnn['error_func'], orient='index')

        train_res.to_csv(os.path.join(self.model_dir, 'train_result.csv'), header=None)

        cnn = CNN_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, self.pool_size,
                     trial=self.trial)

        self.acc_cnn = np.inf
        gpus = np.tile(self.static_data['gpus'], 4)
        h_size = [
            [1024, 256],
            [512, 128],

        ]

        results1 = Parallel(n_jobs=len(self.static_data['gpus']))(
            delayed(optimize_cnn)(cnn, self.best_kernel, h_size[k],
                                                       self.static_data['max_iterations'],
                                                       self.static_data['learning_rate'],
                                                       gpus[int(k)], int(self.static_data['filters'])) for k in range(2))
        for r in results1:
            logger.info("num neurons: 1st %s and 2nd %s with accuracy cnn: %s",
                        *r[2], r[0])
        results +=results1

        acc_cnn = np.array([r[0] for r in results])
        self.acc_cnn, self.best_kernel, self.best_h_size, self.scale_cnn, model_cnn, self.pool_size, self.trial, self.lr= results[acc_cnn.argmin()]
        self.model = model_cnn
        train_res = pd.DataFrame.from_dict(model_cnn['error_func'], orient='index')

        train_res.to_csv(os.path.join(self.model_dir, 'train_result_hsize.csv'), header=None)



        self.save(self.model_dir)

        # self.acc_cnn = np.inf
        # gpus = np.tile(self.static_data['gpus'], 4)
        # lrs = [0.5e-5, 1e-4]
        #
        #
        # results1 = Parallel(n_jobs=len(self.static_data['gpus']))(
        #     delayed(optimize_cnn)(cnn, self.best_kernel, self.best_h_size,
        #                                          self.static_data['max_iterations'],
        #                                          lrs[k],
        #                                          gpus[k], int(self.static_data['filters'])) for k in [0, 1])
        # for r in results1:
        #     logger.info("Learning rate: %s accuracy cnn: %s", r[7], r[0])
        #
        # results +=results1
        # acc_cnn = np.array([r[0] for r in results])
        # self.acc_cnn, self.best_kernel, self.best_h_size, self.scale_cnn, model_cnn, self.pool_size, self.trial, self.lr = results[acc_cnn.argmin()]
        # self.model = model_cnn
        # self.save(self.model_dir)

        logger.info("Best kernel: %s", self.best_kernel)
        logger.info("accuracy cnn: %s", self.acc_cnn)
        logger.info("num neurons: 1st %s and 2nd %s", *self.best_h_size)
        logger.info("with accuracy cnn: %s", self.acc_cnn)
        logger.info("Best learning rate: %s", self.lr)
        logger.info("Total accuracy cnn: %s", self.acc_cnn)
        logger.info('\n')


        self.istrained = True
        self.save(self.model_dir)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir', 'model']:
                dict[k] = self.__dict__[k]
        return dict

    def train_cnn_TL(self, X, y, model, gpu):

        if len(y.shape)==1:
            y = y.reshape(-1, 1)

        print('CNN training...begin for %s ', self.cluster)

        X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        cnn = CNN_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, model['pool_size'], model['trial'])

        self.acc_cnn = np.inf
        gpus = np.tile(gpu, 2)


        if not 'lr' in model.keys():
            model['lr'] = 5e-5
        results = Parallel(n_jobs=len(self.static_data['gpus']))(
            delayed(optimize_cnn)(cnn, model['best_kernel'], model['best_h_size'],
                                                 self.static_data['max_iterations'],
                                                 model['lr'],
                                                 gpus[k]) for k in [0])

        self.acc_cnn, self.best_kernel, self.best_h_size, self.scale_cnn, model_cnn, self.pool_size, self.trial, self.lr  = results[0]
        self.model = model_cnn

        self.save(self.model_dir)

        self.istrained = True
        self.save(self.model_dir)
        return self.to_dict()

    def predict(self, X):
        cnn = CNN_3d_predict(self.static_data_all, self.rated, self.cluster_dir)
        return  cnn.predict(X)

    def move_files(self, path1, path2):
        for filename in glob.glob(os.path.join(path1, '*.*')):
            shutil.copy(filename, path2)

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'cnn' + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(cluster_dir, 'cnn' + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self, pathname):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'model_dir', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(pathname, 'cnn' + '.pickle'), compress=9)

if __name__ == '__main__':
    if sys.platform == 'linux':
        sys_folder = '/media/smartrue/HHD1/George/models/'
    else:
        sys_folder = 'D:/models/'

    path_project = sys_folder + '/Crossbow/Bulgaria_ver2/pv/Lach/model_ver0'
    clust = 'rule.4'
    cluster_dir = path_project + '/Regressor_layer/' + clust
    model_dir = cluster_dir + '/CNN_3d/model'
    data_dir = cluster_dir + '/data'

    logger = logging.getLogger('log_train_' + clust + '.log')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(path_project, 'log_train_' + clust + '.log'), 'w')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)


    static_data = joblib.load(os.path.join(path_project, 'static_data.pickle'))
    X = joblib.load(os.path.join(data_dir, 'dataset_cnn.pickle'))
    y = pd.read_csv(os.path.join(data_dir, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
    y = y.values


    static_data['CNN']['max_iterations'] =1001
    cnn_model_3d = cnn_3d_model(static_data, static_data['rated'], cluster_dir)
    if cnn_model_3d.istrained == True:
        cnn_model_3d.istrained = False

    N_tot = X.shape[0]

    n_split = int(np.round(X.shape[0] * 0.85))

    X_test = X[n_split + 1:]
    y_test = y[n_split + 1:]

    X = X[:n_split]
    y = y[:n_split]

    model = cnn_model_3d.train_cnn(X, y)



    pred = cnn_model_3d.predict(X_test)
    metrics_cnn = cnn_model_3d.compute_metrics(pred, y_test, static_data['rated'])

    logger.info('cnn 3d')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_cnn)
