import os
import numpy as np
import pandas as pd
import pickle
import glob
import shutil
import logging, joblib
import re
import multiprocessing as mp
from joblib import Parallel, delayed
import tensorflow as tf
from Fuzzy_clustering.ver_tf2.RBFNN_module import rbf_model
from Fuzzy_clustering.ver_tf2.CNN_tf_core import CNN
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp2d

from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from Fuzzy_clustering.ver_tf2.RBF_ols import rbf_ols_module
from Fuzzy_clustering.ver_tf2.CNN_predict import CNN_predict

def optimize_cnn(cnn, kernels, hsize, cnn_max_iterations, cnn_learning_rate, gpu, filters):
    flag = False
    for _ in range(3):
        try:
            acc_old_cnn, scale_cnn, model_cnn = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                    learning_rate=cnn_learning_rate, kernels=kernels, h_size=hsize, gpu_id=gpu, filters=filters)
            flag = True
        except:
            filters = int(filters / 2)
        pass


    if not flag:
        acc_old_cnn = np.inf
        scale_cnn = None
        model_cnn = None

    return acc_old_cnn, kernels[1], hsize, cnn_learning_rate, scale_cnn, model_cnn


def predict(q, H, model):
    tf.config.set_soft_device_placement(True)
    pred = model.predict(H)

    q.put((pred[0]))


class cnn_model():
    def __init__(self, static_data, rated, cluster_dir, rbf_dir):

        self.static_data_all = static_data
        self.static_data = static_data['CNN']
        self.static_data_rbf = static_data['RBF']
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.istrained = False
        self.rbf_dir = rbf_dir
        self.cluster_dir = cluster_dir
        if isinstance(rbf_dir, list):
            self.rbf_method = 'RBF_ALL'
            self.cluster_cnn_dir = os.path.join(cluster_dir, 'RBF_ALL/CNN')
            self.model_dir = os.path.join(self.cluster_cnn_dir, 'model')
            self.rbf = rbf_model(self.static_data_rbf, self.rated, cluster_dir)
            self.rbf.models=[]
            for dir in rbf_dir:
                rbf_method = os.path.basename(dir)
                cluster_rbf_dir = os.path.join(dir, 'model')

                if rbf_method == 'RBFNN':
                    rbf = rbf_model(self.static_data_rbf, self.rated, cluster_dir)

                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[0])
                elif rbf_method == 'RBF_OLS':
                    rbf = rbf_ols_module(cluster_dir, rated, self.static_data_rbf['njobs'], GA=False)
                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[-1])
                elif rbf_method == 'GA_RBF_OLS':
                    rbf = rbf_ols_module(cluster_dir, rated, self.static_data_rbf['njobs'], GA=True)
                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[0])
                else:
                    raise ValueError('Cannot recognize RBF method')

        else:
            self.rbf_method = os.path.basename(rbf_dir)
            cluster_rbf_dir = os.path.join(rbf_dir, 'model')
            self.cluster_cnn_dir = os.path.join(rbf_dir, 'CNN')
            self.model_dir = os.path.join(self.cluster_cnn_dir, 'model')
            if self.rbf_method == 'RBFNN':
                self.rbf = rbf_model(self.static_data_rbf, self.rated, cluster_dir)
            elif self.rbf_method == 'RBF_OLS':
                self.rbf = rbf_ols_module(cluster_dir, rated, self.static_data_rbf['njobs'], GA=False)
            elif self.rbf_method == 'GA_RBF_OLS':
                self.rbf = rbf_ols_module(cluster_dir, rated, self.static_data_rbf['njobs'], GA=True)
            else:
                raise ValueError('Cannot recognize RBF method')
            try:
                self.rbf.load(cluster_rbf_dir)
            except:
                raise ImportError('Cannot load RBFNN models')


        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        try:
            self.load(self.model_dir)
        except:
            pass

    def train_cnn(self, cvs):
        logger = logging.getLogger('log_train_' + self.cluster + '.log')
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
        logger.info('CNN training...begin for method %s ', self.rbf_method)

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        cnn = CNN(self.static_data, self.rated, self.rbf.models, X_train, y_train, X_val, y_val, X_test, y_test)

        self.acc_cnn = np.inf
        gpus = np.tile(self.static_data['gpus'], 4)

        # k=2
        # optimize_cnn(cnn, self.temp_dir + str(k), [2, k],
        #                       self.static_data['max_iterations'], self.static_data['learning_rate'],
        #                       gpus[int(k / 4)])
        # pool = mp.Pool(processes=len(self.static_data['gpus']))
        # result = [pool.apply_async(optimize_cnn, args=(cnn, self.temp_dir + str(k), [2, k], self.static_data['h_size'],
        #                                                self.static_data['max_iterations'],
        #                                                self.static_data['learning_rate'],
        #                                                gpus[int(k / 4)])) for k in [2, 12]]
        # results = [p.get() for p in result]
        # pool.close()
        # pool.terminate()
        # pool.join()
        results = Parallel(n_jobs=len(self.static_data['gpus']))(
            delayed(optimize_cnn)(cnn, [2, k], self.static_data['h_size'],
                                                       self.static_data['max_iterations'],
                                                       self.static_data['learning_rate'],
                                                       gpus[i], int(self.static_data['filters'])) for i, k in enumerate([4, 12]))
        for r in results:
            logger.info("kernel: %s accuracy cnn: %s", r[1], r[0])

        acc_cnn = np.array([r[0] for r in results])
        self.acc_cnn, self.best_kernel, hsize, lr, self.scale_cnn, model_cnn = results[acc_cnn.argmin()]
        self.model = model_cnn
        train_res = pd.DataFrame.from_dict(model_cnn['error_func'], orient='index')

        train_res.to_csv(os.path.join(self.model_dir, 'train_result.csv'), header=None)


        self.save(self.model_dir)

        try:
            self.acc_cnn = np.inf
            gpus = np.tile(self.static_data['gpus'], 4)
            h_size=[
                [1024, 256],
                [512, 128],

            ]

            results1 = Parallel(n_jobs=len(self.static_data['gpus']))(
                delayed(optimize_cnn)(cnn, [2, self.best_kernel], h_size[k],
                                                           self.static_data['max_iterations'],
                                                           self.static_data['learning_rate'],
                                                           gpus[int(k)], int(self.static_data['filters'])) for k in range(2))
            for r in results1:
                logger.info("num neurons: 1st %s and 2nd %s with accuracy cnn: %s", *r[2], r[0])

            results += results1
            acc_cnn = np.array([r[0] for r in results])
            self.acc_cnn, self.best_kernel, self.best_h_size, self.lr, self.scale_cnn, model_cnn = results[acc_cnn.argmin()]
            self.model = model_cnn
            train_res = pd.DataFrame.from_dict(model_cnn['error_func'], orient='index')

            train_res.to_csv(os.path.join(self.model_dir, 'train_result_hsize.csv'), header=None)

            logger.info("Best kernel: %s", self.best_kernel)
            logger.info("accuracy cnn: %s", self.acc_cnn)
            logger.info("num neurons: 1st %s and 2nd %s", *self.best_h_size)
            logger.info("with accuracy cnn: %s", self.acc_cnn)
            logger.info("Best learning rate: %s", self.lr)
            logger.info("Total accuracy cnn: %s", self.acc_cnn)
            logger.info('\n')
            self.istrained = True
            self.save(self.model_dir)

        except:
            pass

            # self.acc_cnn = np.inf
            # gpus = np.tile(self.static_data['gpus'], 4)
            # lrs = [1e-6, 1e-4]

            # k=2
            # optimize_cnn(cnn, self.temp_dir + str(k), [2, k],
            #                       self.static_data['max_iterations'], self.static_data['learning_rate'],
            #                       gpus[int(k / 4)])
            # pool = mp.Pool(processes=len(self.static_data['gpus']))
            # result = [pool.apply_async(optimize_cnn, args=(cnn, self.temp_dir + str(k), [2, self.best_kernel], self.best_h_size,
            #                                                self.static_data['max_iterations'],
            #                                                lrs[k],
            #                                                gpus[k])) for k in [0, 1]]
            # results = [p.get() for p in result]
            # pool.close()
            # pool.terminate()
            # pool.join()
            # results1 = Parallel(n_jobs=len(self.static_data['gpus']))(
            #     delayed(optimize_cnn)(cnn, [2, self.best_kernel], self.best_h_size,
            #                                                self.static_data['max_iterations'],
            #                                                lrs[k],
            #                                                gpus[k], int(self.static_data['filters'])) for k in [0, 1])
            # for r in results1:
            #     lr = r[3]
            #     logger.info("Learning rate: %s accuracy cnn: %s", lr, r[0])
            #
            # results += results1
            # acc_cnn = np.array([r[0] for r in results])
            # self.acc_cnn, self.best_kernel, self.best_h_size, self.lr, self.scale_cnn, model_cnn = results[acc_cnn.argmin()]
            # self.model = model_cnn



        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger','static_data_all','model_dir', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir','rbf_dir', 'model']:
                dict[k] = self.__dict__[k]
        return dict

    def train_cnn_TL(self, cvs, model, gpu):

        print('CNN training...begin for %s ', self.cluster)

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        cnn = CNN(self.static_data, self.rated, self.rbf.models, X_train, y_train, X_val, y_val, X_test, y_test)

        self.acc_cnn = np.inf
        gpus = np.tile(gpu, 1)


        results = Parallel(n_jobs=len(self.static_data['gpus']))(
            delayed(optimize_cnn)(cnn, [2, model['best_kernel']], model['best_h_size'],
                                                       self.static_data['max_iterations'],
                                                       model['lr'],
                                                       gpus[k], int(self.static_data['filters'])) for k in [0])

        acc_cnn = np.array([r[0] for r in results])
        self.acc_cnn, best_kernel, best_h_size, lr, self.scale_cnn, model_cnn = results[acc_cnn.argmin()]
        self.model = model_cnn
        self.lr = model['lr']
        self.best_h_size = model['best_h_size']
        self.best_kernel = model['best_kernel']

        self.istrained = True
        self.save(self.model_dir)

        return self.to_dict()

    def rbf_map(self,X, num_centr, centroids, radius):
        hmap_list = []
        s = X.shape
        d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [num_centr, 1, 1]), [1, 0, 2]) - np.tile(
            np.expand_dims(centroids, axis=0), [s[0], 1, 1])
        d = np.sqrt(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])), 2))
        phi = np.exp((-1) * np.power(d, 2))

        return np.transpose(phi,[1, 0, 2])

    def rescale(self, arr, nrows, ncol):
        W, H = arr.shape
        new_W, new_H = (nrows, ncol)
        xrange = lambda x: np.linspace(0, 1, x)

        f = interp2d(xrange(H), xrange(W), arr, kind="linear")
        new_arr = f(xrange(new_H), xrange(new_W))

        return new_arr
    def predict(self,X):
        cnn = CNN_predict(self.static_data_all, self.rated, self.cluster_dir, self.rbf_dir)
        return cnn.predict(X)

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
            if k not in ['logger','static_data_all','model_dir', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir','rbf_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(pathname, 'cnn' + '.pickle'), compress=9)
