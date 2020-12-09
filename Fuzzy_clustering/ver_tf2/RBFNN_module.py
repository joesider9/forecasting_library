import os
import gc
import tensorflow as tf
import numpy as np
import pickle
import glob
import shutil
import multiprocessing as mp
import pandas as pd
from Fuzzy_clustering.ver_tf2.RBFNN_tf_core import RBFNN
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
import logging
from joblib import Parallel, delayed
# from util_database import write_database
# from Fuzzy_clustering.ver_tf2.Forecast_model import forecast_model
# from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous

def optimize_rbf(rbf, X_train, y_train, X_val, y_val, X_test, y_test, num_centr, lr,  gpu):
    acc_old = np.inf
    acc_old, centroids, radius, w, model = rbf.train(X_train, y_train, X_val, y_val, X_test, y_test, num_centr, lr, gpu_id=gpu)

    return num_centr, lr, acc_old, model


class rbf_model(object):
    def __init__(self, static_data,  rated, cluster_dir):
        self.static_data=static_data
        self.cluster = os.path.basename(cluster_dir)
        self.rated=rated
        self.cluster_dir=os.path.join(cluster_dir, 'RBFNN')
        self.model_dir = os.path.join(self.cluster_dir, 'model')
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def train_core(self, X_train, y_train, X_val, y_val, X_test, y_test, ncs, lrs):
        self.gpu = True
        nproc = self.static_data['njobs']
        gpus = np.tile(self.static_data['gpus'], ncs.shape[0]*lrs.shape[0])

        RBFnn = RBFNN(self.model_dir, rated=self.rated, max_iterations=self.static_data['max_iterations'])
        # n=0
        # optimize_rbf(RBFnn, cvs[n][0], cvs[n][1], cvs[n][2], cvs[n][3], X_test, y_test, nc[n], gpus[n])

        # pool = mp.Pool(processes=nproc)
        #
        # result = []
        # k=0
        # for n in range(ncs.shape[0]):
        #     for lr in range(lrs.shape[0]):
                # optimize_rbf(RBFnn, X_train, y_train, X_val, y_val, X_test, y_test, ncs[n], lrs[lr], gpus[k])
                # result.append(pool.apply_async(optimize_rbf,
                #                    args=(RBFnn, X_train, y_train, X_val, y_val, X_test, y_test, ncs[n], lrs[lr], gpus[k])))
        k = np.arange(ncs.shape[0]*lrs.shape[0])
        # optimize_rbf(RBFnn, X_train, y_train, X_val, y_val, X_test, y_test, ncs[0], lrs[0], gpus[0])
        results = Parallel(n_jobs=nproc)(
            delayed(optimize_rbf)(RBFnn, X_train, y_train, X_val, y_val, X_test, y_test, ncs[n], lrs[lr], gpus[i+j]) for i, n in enumerate(range(ncs.shape[0])) for j, lr in enumerate(range(lrs.shape[0])))
                # k+=1


        # results = [p.get() for p in result]
        # pool.close()
        # pool.terminate()
        # pool.join()

        r = pd.DataFrame(results, columns=['num_centr', 'lr', 'acc', 'model'])

        self.num_centr = r.loc[r['acc'].idxmin()]['num_centr']
        self.lr = r.loc[r['acc'].idxmin()]['lr']
        self.rbf_performance = r['acc'].min()

        self.save(self.model_dir)
        gc.collect()

        models = [r2[3] for r2 in results]
        return models

    def rbf_train(self, cvs):

        logger = logging.getLogger('RBFNN ADAM_train_' + self.cluster)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print('RBFNN ADAM training...begin')
        logger.info('RBFNN ADAM training...begin for %s', self.cluster)


        nc = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 52]

        # nc = [12]


        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        ncs = np.array(nc)
        lrs=np.array([self.static_data['learning_rate']])

        models = self.train_core(X_train, y_train, X_val, y_val, X_test, y_test, ncs, lrs)

        same = 1
        for model in models:
            logger.info('Best number of centers training ')
            logger.info('Model with num centers %s ', str(model['num_centr']))
            logger.info('val_mae  %s, val_mse  %s, val_sse  %s, val_rms %s ', *model['metrics'])
            logger.info('Model trained with max iterations %s ', str(model['best_iteration']))
            train_res = pd.DataFrame.from_dict(model['error_func'], orient='index')
            if not os.path.exists(
                    os.path.join(self.model_dir, 'train_centers_result_' + str(model['num_centr']) + '.csv')):
                train_res.to_csv(
                    os.path.join(self.model_dir, 'train_centers_result_' + str(model['num_centr']) + '.csv'),
                    header=None)
            else:
                train_res.to_csv(os.path.join(self.model_dir,
                                              'train_centers_result_' + str(model['num_centr']) + '_' + str(
                                                  same) + '.csv'), header=None)
                same += 1

        logger.info('temporary performance %s ', str(self.rbf_performance))
        logger.info('temporary RBF number %s ', str(self.num_centr))
        logger.info('\n')

        logger.info('\n')


        if self.num_centr >= 5 and self.static_data['Fine_tuning']:
            logger.info('Begin fine tuning....')
            print('Begin fine tuning....')
            ncs = np.hstack(
                [np.arange(self.num_centr - 2, self.num_centr - 1), np.arange(self.num_centr + 1, self.num_centr + 3)])
            models = self.train_core(X_train, y_train, X_val, y_val, X_test, y_test, ncs, lrs)
            same = 1
            for model in models:
                logger.info('fine tunninig training ')
                logger.info('Model with num centers %s ', str(model['num_centr']))
                logger.info('val_mae  %s, val_mse  %s, val_sse  %s, val_rms %s ', *model['metrics'])
                logger.info('Model trained with max iterations %s ', str(model['best_iteration']))
                train_res = pd.DataFrame.from_dict(model['error_func'], orient='index')
                if not os.path.exists(
                        os.path.join(self.model_dir, 'train_fine_tune_result_' + str(model['num_centr']) + '.csv')):
                    train_res.to_csv(
                        os.path.join(self.model_dir, 'train_fine_tune_result_' + str(model['num_centr']) + '.csv'),
                        header=None)
                else:
                    train_res.to_csv(os.path.join(self.model_dir,
                                                  'train_fine_tune_result_' + str(model['num_centr']) + '_' + str(
                                                      same) + '.csv'), header=None)
                    same += 1

            logger.info('After fine tuning performance %s ', str(self.rbf_performance))
            logger.info('After fine tuning  RBF number %s ', str(self.num_centr))
            logger.info('\n')

        ncs = np.array([self.num_centr])
        lrs=np.array([1e-3, 5e-4, 1e-4, 5e-5])

        models = self.train_core(X_train, y_train, X_val, y_val, X_test, y_test, ncs, lrs)

        same=1
        for model in models:
            logger.info('Best Learning rate training ')
            logger.info('Model with num centers %s ', str(model['num_centr']))
            logger.info('val_mae  %s, val_mse  %s, val_sse  %s, val_rms %s ', *model['metrics'])
            logger.info('Model trained with max iterations %s ', str(model['best_iteration']))
            train_res = pd.DataFrame.from_dict(model['error_func'], orient='index')
            if not os.path.exists(os.path.join(self.model_dir,'train_lr_result_' + str(model['num_centr']) + '.csv')):
                train_res.to_csv(os.path.join(self.model_dir,'train_lr_result_' + str(model['num_centr']) + '.csv'), header=None)
            else:
                train_res.to_csv(os.path.join(self.model_dir, 'train_lr_result_' + str(model['num_centr']) + '_'+ str(same) + '.csv'), header=None)
                same+=1

        logger.info('Tuning lr performance %s ', str(self.rbf_performance))

        logger.info('Tuning lr  is %s ', str(self.lr))
        logger.info('\n')


        ncs = np.array([self.num_centr])

        ncs = np.repeat(ncs, 3)
        gpus = np.tile(self.static_data['gpus'], ncs.shape[0])


        RBFnn = RBFNN(self.model_dir, rated=self.rated, max_iterations=self.static_data['max_iterations'])

        nproc = self.static_data['njobs']
        # pool = mp.Pool(processes=nproc)
        #
        # result = [pool.apply_async(optimize_rbf, args=(
        #     RBFnn, cvs[n][0], cvs[n][1].reshape(-1, 1), cvs[n][2], cvs[n][3].reshape(-1, 1), X_test, y_test, ncs[n], self.lr, gpus[n])) for n in
        #           range(ncs.shape[0])]
        #
        # results = [p.get() for p in result]
        # pool.close()
        # pool.terminate()
        # pool.join()
        #

        results = Parallel(n_jobs=nproc)(
            delayed(optimize_rbf)(RBFnn, cvs[n][0], cvs[n][1].reshape(-1, 1), cvs[n][2], cvs[n][3].reshape(-1, 1), X_test, y_test, ncs[n], self.lr, gpus[n]) for n in range(ncs.shape[0]))

        r = pd.DataFrame(results, columns=['num_centr','lr', 'acc', 'model'])
        r2 = r.groupby(['num_centr'])['model'].apply(lambda x: np.squeeze([x]))
        r1 = r.groupby(['num_centr']).mean()

        self.acc_old = r1['acc'].values[0]
        r2 = r2[self.num_centr]

        self.models = [r2[i] for i in range(3)]

        self.rbf_performance = self.acc_old
        self.istrained = True
        self.save(self.model_dir)
        gc.collect()


        same=1
        for model in self.models:
            logger.info('Final training ')
            logger.info('Model with num centers %s ', str(model['num_centr']))
            logger.info('val_mae  %s, val_mse  %s, val_sse  %s, val_rms %s ', *model['metrics'])
            logger.info('Model trained with max iterations %s ', str(model['best_iteration']))
            train_res = pd.DataFrame.from_dict(model['error_func'], orient='index')
            if not os.path.exists(os.path.join(self.model_dir,'train_fin_result_' + str(model['num_centr']) + '.csv')):
                train_res.to_csv(os.path.join(self.model_dir,'train_fin_result_' + str(model['num_centr']) + '.csv'), header=None)
            else:
                train_res.to_csv(os.path.join(self.model_dir, 'train_fin_result_' + str(model['num_centr']) + '_'+ str(same) + '.csv'), header=None)
                same+=1


        logger.info('final performance %s ', str(self.rbf_performance))
        logger.info('final RBF number %s ', str(self.num_centr))
        logger.info('RBFNN training...end for %s', self.cluster)
        logger.info('\n')

        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'logger', 'cluster_dir','model_dir', 'model']:
                dict[k] = self.__dict__[k]
        return dict

    def predict(self,X):
        p=[]
        self.load(self.model_dir)
        for i in range(len(self.models)):
            centroids=self.models[i]['centroids']
            radius=self.models[i]['Radius']
            w=self.models[i]['W']
            s = X.shape
            d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [self.num_centr, 1, 1]), [1, 0, 2]) - np.tile(
                np.expand_dims(centroids, axis=0), [s[0], 1, 1])
            d = np.sqrt(np.sum(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])),2), axis=2))
            phi = np.exp((-1) * np.power(d,2))
            p.append(np.matmul(phi, w))
        p=np.mean(np.array(p),axis=0)
        return p

    def rbf_train_TL(self, cvs, model, gpu):


        print('RBFNN ADAM training...begin')


        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]

        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        ncs = np.array([model['num_centr']])

        ncs = np.repeat(ncs, 3)
        gpus = np.tile(gpu, ncs.shape[0])

        RBFnn = RBFNN(self.model_dir, rated=self.rated, max_iterations=self.static_data['max_iterations'])

        nproc = self.static_data['njobs']

        results = Parallel(n_jobs=nproc)(
            delayed(optimize_rbf)(RBFnn, cvs[n][0], cvs[n][1].reshape(-1, 1), cvs[n][2], cvs[n][3].reshape(-1, 1),
                                  X_test, y_test, ncs[n], model['lr'], gpus[n]) for n in range(ncs.shape[0]))

        r = pd.DataFrame(results, columns=['num_centr', 'lr', 'acc', 'model'])
        r2 = r.groupby(['num_centr'])['model'].apply(lambda x: np.squeeze([x]))
        r1 = r.groupby(['num_centr']).mean()

        self.num_centr = model['num_centr']
        self.lr = model['lr']
        self.acc_old = r1['acc'].values[0]
        r2 = r2[self.num_centr]

        self.models = [r2[i] for i in range(3)]

        self.rbf_performance = self.acc_old
        self.istrained = True
        self.save(self.model_dir)
        gc.collect()

        return self.to_dict()

    def move_files(self, path1, path2):
        for filename in glob.glob(os.path.join(path1, '*.*')):
            shutil.copy(filename, path2)

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse=np.sum(np.square(pred.ravel()-y.ravel()))
        rms=np.sqrt(np.mean(np.square(err)))
        mae=np.mean(err)
        mse = sse/y.shape[0]

        return [sse, rms, mae, mse]

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'rbfnn' + '.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'rbfnn' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()

                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RBFNN model')
        else:
            raise ImportError('Cannot find RBFNN model')


    def save(self, pathname):
        f = open(os.path.join(pathname, 'rbfnn' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'logger', 'cluster_dir','model_dir']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

# if __name__=='__main__':
#     cluster_dir='D:/APE_net_ver2/Regressor_layer/rule.2'
#     data_dir='D:/APE_net_ver2/Regressor_layer/rule.2/data'
#
#     rated = None
#
#     static_data = write_database()
#     X=np.load(os.path.join(data_dir, 'X_train.npy'))
#     y=np.load(os.path.join(data_dir, 'y_train.npy'))
#     forecast = forecast_model(static_data, use_db=False)
#     forecast.load()
#     X=X[:,0:-1]
#     X = forecast.sc.transform(X)
#     y= forecast.scale_y.transform(y)
#     # scy = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1,1))
#     #
#     # y = scy.transform(y.reshape(-1,1)).ravel()
#     N, D = X.shape
#     n_split = int(np.round(N * 0.85))
#     X_test1 = X[n_split + 1:, :]
#     y_test1 = y[n_split + 1:]
#     X = X[:n_split, :]
#     y = y[:n_split]
#
#     # X_train, X_val, y_train,  y_val=split_continuous(X, y, test_size=0.15, random_state=42)
#     X_train, X_test, y_train, y_test=split_continuous(X, y, test_size=0.15, random_state=42)
#
#     model_rbf = rbf_model(static_data['RBF'], static_data['type'], static_data['rated'], cluster_dir)
#     model_rbf.rbf_train(X, y, X_test, y_test)
#     pred = model_rbf.predict(X_test1)
#     metrics_single = model_rbf.compute_metrics(pred, y_test1, 0)
#
#     print('Single width')
#     print('sse, rms, mae, mse')
#     print(metrics_single)