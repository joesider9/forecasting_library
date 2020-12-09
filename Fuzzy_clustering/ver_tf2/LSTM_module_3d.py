import os
import numpy as np
import pandas as pd
import pickle
import glob
import shutil, joblib
import logging
import re
import multiprocessing as mp
import tensorflow as tf
from joblib import Parallel, delayed
from Fuzzy_clustering.ver_tf2.LSTM_tf_core_3d import LSTM_3d
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp2d
# from util_database import write_database
# from Fuzzy_clustering.ver_tf2.Forecast_model import forecast_model
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from Fuzzy_clustering.ver_tf2.LSTM_predict_3d import LSTM_3d_predict

def optimize_lstm(lstm, units, hold_prob, lstm_max_iterations, lstm_learning_rate, gpu):
    # try:
    acc_old_lstm, scale_lstm, model_lstm = lstm.train(max_iterations=lstm_max_iterations,
                                                learning_rate=lstm_learning_rate, units=units, hold_prob=hold_prob, gpu_id=gpu)
    # except:
    #     acc_old_lstm=np.inf
    #     scale_lstm=None
    #     model_lstm=None
    #     pass

    return acc_old_lstm, scale_lstm, model_lstm, units, hold_prob, lstm_learning_rate, lstm.trial


def predict(q, H, model):
    tf.config.set_soft_device_placement(True)
    pred = model.predict(H)

    q.put((pred[0]))


class lstm_3d_model():
    def __init__(self, static_data, rated, cluster_dir, probabilistc=False):
        self.static_data_all = static_data
        self.static_data = static_data['CNN']
        self.probabilistic = probabilistc
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.cluster_lstm_dir = os.path.join(cluster_dir, 'LSTM_3d')
        self.model_dir = os.path.join(self.cluster_lstm_dir, 'model')
        self.cluster_dir = cluster_dir
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def train_lstm(self, X, y):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print('LSTM training...begin for %s ', self.cluster)
        logger.info('LSTM training...begin for %s ', self.cluster)

        if len(y.shape)==1:
            y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

        min_units = np.maximum(X_train.shape[1], 64)
        results =[]
        for trial in [2, 3]:
            if trial==0:
                units = [[min_units],
                         [2*min_units]]
            elif trial==1:
                units = [
                    [2*min_units, 1024]
                        ]
            elif trial==2:
                units = [
                    [2*min_units, 1024, min_units]
                        ]
            else:
                units = [
                    [2*min_units, 1024, min_units, 1024]
                         ]
            lstm = LSTM_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, trial=trial, probabilistc=self.probabilistic)

            self.acc_lstm = np.inf
            gpus = np.tile(self.static_data['gpus'], 4)

            res = Parallel(n_jobs=2*len(self.static_data['gpus']))(
                delayed(optimize_lstm)(lstm,  size, 1, self.static_data['max_iterations'],
                                                           self.static_data['learning_rate'],
                                                           gpus[i]) for i, size in enumerate(units))
            results += res

        for r in results:
            logger.info("Trial %s Units: %s accuracy lstm: %s", r[6], r[1], r[0])

        acc_lstm = np.array([r[0] for r in results])
        self.acc_lstm, self.scale_lstm, model_lstm, self.best_units, hold_prob, lr, self.trial = results[acc_lstm.argmin()]

        self.model = model_lstm

        logger.info("Best trial: %s", self.trial)
        logger.info("Best units: %s", self.best_units)
        logger.info("accuracy lstm: %s", self.acc_lstm)

        lstm = LSTM_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, trial=self.trial, probabilistc=self.probabilistic)

        self.acc_lstm = np.inf
        gpus = np.tile(self.static_data['gpus'], 4)
        # prob_holds = [0.5, 0.75]
        #
        # res = Parallel(n_jobs=len(self.static_data['gpus']))(
        #     delayed(optimize_lstm)(lstm, self.best_units, prob,
        #                                                self.static_data['max_iterations'],
        #                                                self.static_data['learning_rate'],
        #                                                gpus[i]) for i, prob in enumerate(prob_holds))
        # for r in res:
        #     logger.info("prob_hold: %s with accuracy lstm: %s",
        #                 r[4], r[0])
        # results += res
        # acc_lstm = np.array([r[0] for r in results])
        # self.acc_lstm, self.scale_lstm, model_lstm, self.best_units, self.best_prob_holds, lr, self.trial  = results[acc_lstm.argmin()]
        # self.model = model_lstm
        #
        # logger.info("Best prob_hold %s", self.best_prob_holds)
        # logger.info("with accuracy lstm: %s", self.acc_lstm)
        self.best_prob_holds=1
        self.acc_lstm = np.inf
        gpus = np.tile(self.static_data['gpus'], 4)
        lrs = [1e-6, 1e-5]

        res = Parallel(n_jobs=2*len(self.static_data['gpus']))(
            delayed(optimize_lstm)(lstm, self.best_units, self.best_prob_holds,
                                                 self.static_data['max_iterations'],
                                                 lr,
                                                 gpus[i]) for i, lr in enumerate(lrs))
        for r in res:
            logger.info("Learning rate: %s accuracy lstm: %s", r[5], r[0])

        results += res
        acc_lstm = np.array([r[0] for r in results])
        self.acc_lstm, self.scale_lstm, model_lstm, self.best_units, self.best_prob_holds, self.lr, self.trial   = results[acc_lstm.argmin()]
        self.model = model_lstm
        logger.info("Best learning rate: %s", self.lr)
        logger.info("Total accuracy lstm: %s", self.acc_lstm)
        logger.info('\n')
        self.save(self.model_dir)

        self.istrained = True
        self.save(self.model_dir)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'temp_dir', 'cluster_lstm_dir', 'cluster_dir', 'model']:
                dict[k] = self.__dict__[k]
        return dict

    def train_lstm_TL(self, X, y, model, gpu):


        print('LSTM training...begin for %s ', self.cluster)

        X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        lstm = LSTM_3d(self.static_data, self.rated, X_train, y_train, X_val, y_val, X_test, y_test, trial=model['trial'])

        self.acc_lstm = np.inf
        gpus = np.tile(gpu, 2)

        results = Parallel(n_jobs=len(self.static_data['gpus']))(
            delayed(optimize_lstm)(lstm, model['best_units'], model['best_prob_holds'],
                                                 self.static_data['max_iterations'],
                                                 model['lr'],
                                                 gpus[k]) for k in [0])

        self.acc_lstm, self.scale_lstm, model_lstm, self.best_units, self.best_prob_holds, self.lr, self.trial = results[0]
        self.model = model_lstm

        self.save(self.model_dir)

        self.istrained = True
        self.save(self.model_dir)
        return self.to_dict()

    def predict(self,X):
        lstm = LSTM_3d_predict(self.static_data_all, self.rated,self.cluster_dir)
        return  lstm.predict(X)

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
        if os.path.exists(os.path.join(cluster_dir, 'lstm' + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(cluster_dir, 'lstm' + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open LSTM model')
        else:
            raise ImportError('Cannot find LSTM model')

    def save(self, pathname):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'static_data_all', 'model_dir', 'temp_dir', 'cluster_lstm_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(pathname, 'lstm' + '.pickle'), compress=9)
#

if __name__ == '__main__':
    from util_database import write_database
    from Fuzzy_clustering.ver_tf2.Projects_train_manager import ProjectsTrainManager
    from Fuzzy_clustering.ver_tf2.Models_train_manager import ModelTrainManager

    static_data = write_database()
    project_manager = ProjectsTrainManager(static_data)
    project_manager.initialize()
    project_manager.create_datasets()
    project_manager.create_projects_relations()
    project = [pr for pr in project_manager.group_static_data if pr['_id'] == 'Lach'][0]
    static_data = project['static_data']

    model = ModelTrainManager(static_data['path_model'])
    model.init(project['static_data'], project_manager.data_variables)

    model_dir = os.path.join(static_data['path_model'], 'Combine_module')
    cluster_dir = os.path.join(model_dir, 'LSTM_full')
    data_dir = static_data['path_data']

    X = joblib.load(os.path.join(data_dir, 'predictions_for_check.pickle'))
    y = pd.read_csv(os.path.join(data_dir, 'target_test.csv'), index_col=0, header=[0],
                                     parse_dates=True, dayfirst=True)
    static_data['CNN']['max_iterations'] = 1001
    combine_model = lstm_3d_model(static_data, static_data['rated'], cluster_dir)
    if combine_model.istrained == True:
        combine_model.istrained = False

    combine_model.train_lstm(X, y.values)