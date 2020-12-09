import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging, pickle
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR, LinearSVR, NuSVR

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import multiprocessing as mp


def fit_and_score(model, cvs, params, rated):
    model.set_params(**params)
    rms_val=[]
    rms_test=[]
    for cv in cvs:
        model.fit(cv[0], cv[1].ravel())
        if rated is None:
            ypred = model.predict(cv[2]).ravel()
            rms_val.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[3].ravel()) / cv[3].ravel()))))
            ypred = model.predict(cv[4]).ravel()
            rms_test.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[5].ravel()) / cv[5].ravel()))))
        else:
            ypred = model.predict(cv[2]).ravel()
            rms_val.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[3].ravel())))))
            ypred = model.predict(cv[4]).ravel()
            rms_test.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[5].ravel())))))
    return 0.4*np.mean(rms_val)+0.6*np.mean(rms_test), params

class sklearn_model(object):

    def __init__(self,static_data, cluster_dir,rated,model_type,njobs, path_group = None):
        self.static_data = static_data
        self.path_group =path_group
        self.njobs=njobs
        self.rated=rated
        self.cluster = os.path.basename(cluster_dir)
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        self.istrained = False
        self.optimizer = 'grid_search'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type=model_type
        logger = logging.getLogger('gridsearch_train_' + '_' + self.model_type + self.cluster)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(cluster_dir, 'log_grid_train_' + self.cluster + '.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger=logger
        try:
            self.load(self.model_dir)
        except:
            pass

    def fit_model(self, cvs, model, params):
        print()
        print('Training the model {}'.format(self.model_type))

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape)==1 and len(cvs[0][5].shape)==1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        pms = []
        for p in ParameterGrid(params):
            pms.append((p))

        ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
        gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

        njobs = int(ncpus - gpu_status)
        cpu_status = njobs
        joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))

        pool = mp.Pool(processes=self.njobs)
        result = [pool.apply_async(fit_and_score, args=(model, cvs, d, self.rated)) for d in pms]
        results = [p.get() for p in result]
        pool.close()
        pool.terminate()
        pool.join()
        # results = Parallel(n_jobs=self.njobs)(
        #     delayed(fit_and_score)(model, cvs, d) for d in pms)
        r = pd.DataFrame(results, columns=[ 'rms', 'params'])
        params = r['params'].loc[r['rms'].idxmin()]
        rms = r['rms'].min()
        model.set_params(**params)
        model.fit(X,y.ravel())
        return model, params, rms

    def compute_metrics(self, pred, y, rated):
        if rated is None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def fit_model1(self, model, params, cvs):
        model.set_params(**params)
        rms_val = []
        rms_test = []
        for cv in cvs:
            model.fit(cv[0], cv[1].ravel())
            ypred = model.predict(cv[2]).ravel()
            if self.rated is None:
                acc = np.mean(np.abs(ypred - cv[3].ravel()) / cv[3].ravel())
            else:
                acc = np.mean(np.abs(ypred - cv[3].ravel()))
            rms_val.append(acc)
            ypred = model.predict(cv[4]).ravel()
            if self.rated is None:
                acc = np.mean(np.abs(ypred - cv[5].ravel()) / cv[5].ravel())
            else:
                acc = np.mean(np.abs(ypred - cv[5].ravel()))
            rms_test.append(acc)

        return 0.4 * np.mean(rms_val) + 0.6 * np.mean(rms_test), np.mean(rms_test)

    def train(self, cvs, init_params=[]):
        print('training...')

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()

        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        if 'xgb' in str.lower(self.model_type):
            param_grid = {

                # 'learning_rate': [0.001, 0.01, 0.1],
                # 'max_depth': [0, 3, 18, 24, 75, 100, 200, 250],
                'reg_alpha': [0.001, 1],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [2, 3, 6, 12, 18, 24, 32, 44, 56, 75, 86, 100],
                'min_child_weight': [1, 3, 5, 8],
                'gamma': [0.001, 0.01, 0.1],
                'subsample': [0.4, 0.75, 1.0],
                'colsample_bytree': [0.4, 0.75, 1.0],

            }

            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        elif 'rf' in str.lower(self.model_type):
            param_grid = {

                # 'learning_rate': [0.001, 0.01, 0.1],
                # 'max_depth': [0, 3, 18, 24, 75, 100, 200, 250],


                'max_depth': [None, 2, 3, 6, 12, 18, 24, 32, 44, 56, 75, 86, 100],
                'min_samples_split': [2, 5, 10, 15, 20, 25],
                'min_samples_leaf': [1, 2, 5, 10, 15, 20, 25],
                'max_features': ["auto","sqrt", 0.8, 0.6, 0.4]

            }

            model = RandomForestRegressor(random_state=42, n_estimators=500)

        elif str.lower(self.model_type)=='nusvm':
            C_range = np.logspace(-4, 4, 18)
            gamma_range = np.logspace(-12, 3, 52)
            nu = np.linspace(0.01, 0.90, 10)

            param_grid = dict(gamma=gamma_range, C=C_range, nu=nu)

            model = NuSVR(max_iter=150000)
        elif str.lower(self.model_type)== 'svm':
            C_range = np.logspace(-4, 4, 18)
            gamma_range = np.logspace(-12, 3, 52)

            param_grid = dict(gamma=gamma_range, C=C_range)

            model = SVR(max_iter=150000)

        elif 'lsvm' in str.lower(self.model_type):
            C_range = np.logspace(-4, 4, 13)

            param_grid = dict(C=C_range)

            model = LinearSVR(max_iter=150000)
        elif 'mlp' in str.lower(self.model_type):
            param_grid = {'hidden_layer_sizes': [8, 12, 24, 48, 64, 96, 128, 180, 256, 360, 450],
                          'alpha': [0.00005, 0.0005, 0.005, 0.05]}

            model = MLPRegressor(max_iter=1000, early_stopping=True)


        self.model, self.best_params, self.accuracy = self.fit_model(cvs, model, param_grid)

        self.accuracy, self.acc_test = self.fit_model1(self.model, self.best_params, cvs)
        self.model.set_params(**self.best_params)
        self.model.fit(X, y.ravel())
        self.logger.info('Best params')
        self.logger.info(self.best_params)
        self.logger.info('Final mae %s', str(self.acc_test))
        self.logger.info('Final rms %s', str(self.accuracy))
        self.logger.info('finish train for model %s', self.model_type)
        self.istrained = True
        self.save(self.model_dir)

        return self.to_dict()

    def train_TL(self, cvs, params):
        self.best_params = params
        print('training...')

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()

        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        if 'xgb' in str.lower(self.model_type):
            param_grid = {

                # 'learning_rate': [0.001, 0.01, 0.1],
                # 'max_depth': [0, 3, 18, 24, 75, 100, 200, 250],
                'reg_alpha': [0.001, 1],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [2, 3, 6, 12, 18, 24, 32, 44, 56, 75, 86, 100],
                'min_child_weight': [1, 3, 5, 8],
                'gamma': [0.001, 0.01, 0.1],
                'subsample': [0.4, 0.75, 1.0],
                'colsample_bytree': [0.4, 0.75, 1.0],

            }

            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        elif 'rf' in str.lower(self.model_type):
            param_grid = {

                # 'learning_rate': [0.001, 0.01, 0.1],
                # 'max_depth': [0, 3, 18, 24, 75, 100, 200, 250],


                'max_depth': [None, 2, 3, 6, 12, 18, 24, 32, 44, 56, 75, 86, 100],
                'min_samples_split': [2, 5, 10, 15, 20, 25],
                'min_samples_leaf': [1, 2, 5, 10, 15, 20, 25],
                'max_features': ["auto","sqrt", 0.8, 0.6, 0.4]

            }

            model = RandomForestRegressor(random_state=42, n_estimators=500)

        elif str.lower(self.model_type)=='nusvm':
            C_range = np.logspace(-4, 4, 18)
            gamma_range = np.logspace(-12, 3, 52)
            nu = np.linspace(0.01, 0.90, 10)

            param_grid = dict(gamma=gamma_range, C=C_range, nu=nu)

            model = NuSVR(max_iter=150000)
        elif str.lower(self.model_type)== 'svm':
            C_range = np.logspace(-4, 4, 18)
            gamma_range = np.logspace(-12, 3, 52)

            param_grid = dict(gamma=gamma_range, C=C_range)

            model = SVR(max_iter=150000)

        elif 'lsvm' in str.lower(self.model_type):
            C_range = np.logspace(-4, 4, 13)

            param_grid = dict(C=C_range)

            model = LinearSVR(max_iter=150000)
        elif 'mlp' in str.lower(self.model_type):
            param_grid = {'hidden_layer_sizes': [8, 12, 24, 48, 64, 96, 128, 180, 256, 360, 450],
                          'alpha': [0.00005, 0.0005, 0.005, 0.05]}

            model = MLPRegressor(max_iter=1000, early_stopping=True)

        self.model = model
        self.accuracy, self.acc_test = self.fit_model1(self.model, self.best_params, cvs)
        self.model.set_params(**self.best_params)
        self.model.fit(X, y.ravel())
        self.logger.info('Best params')
        self.logger.info(self.best_params)
        self.logger.info('Final mae %s', str(self.acc_test))
        self.logger.info('Final rms %s', str(self.accuracy))
        self.logger.info('finish train for model %s', self.model_type)
        self.istrained = True
        self.save(self.model_dir)

        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'model']:
                dict[k] = self.__dict__[k]
        return dict

    def predict(self,X):
        self.load(self.model_dir)
        return self.model.predict(X)

    def load(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        if os.path.exists(os.path.join(model_dir, 'model_all' + '.pickle')):
            try:
                f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open model_all model')
        else:
            raise ImportError('Cannot find model_all model')

    def save(self, model_dir):
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()