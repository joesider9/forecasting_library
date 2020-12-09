import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging, pickle
import joblib
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from optuna.samplers import TPESampler
import optuna

class sklearn_model(object):

    def __init__(self,static_data, cluster_dir,rated,model_type,njobs, FS=False, path_group=None):
        self.static_data = static_data
        self.path_group=path_group
        self.njobs=njobs
        self.FS = FS
        self.rated=rated
        self.cluster = os.path.basename(cluster_dir)
        self.istrained = False
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type=model_type
        self.optimizer = 'optuna'
        self.cluster = os.path.basename(cluster_dir)
        logger = logging.getLogger('optuna_train_' + '_' + self.model_type + self.cluster)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(cluster_dir, 'log_opt_train_' + self.cluster + '.log'), 'w')
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

    def fit_model(self, model, params):
        model.set_params(**params)
        rms_val = []
        rms_test = []
        for cv in self.cvs:
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
        return 0.4*np.mean(rms_val)+0.6*np.mean(rms_test)

    def objective(self, trial):
        if 'xgb' in str.lower(self.model_type):
            params = {'learning_rate': trial.suggest_uniform('learning_rate', 0.0001, 0.5),
                      'max_depth': trial.suggest_int('max_depth', 1, 150),
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                      'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
                      'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 1.0),
                      'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                      'gamma': trial.suggest_uniform('gamma', 0.01, 10),
                      'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1.0)}
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        elif 'rf' in str.lower(self.model_type):
            params = {
                     'max_depth': trial.suggest_int('max_depth', 1, 150),
                     'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None, 0.8, 0.6, 0.4]),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 250),
                     'min_samples_split': trial.suggest_int('min_samples_split', 2, 250),
                     }
            model = RandomForestRegressor(n_estimators=500, random_state=42)
        elif str.lower(self.model_type)=='svm':
            params = {'C':trial.suggest_uniform('C', 1e-4, 1e3),
                      'kernel': trial.suggest_categorical('kernel',['linear', 'poly', 'rbf', 'sigmoid']),
                      'gamma':trial.suggest_uniform('gamma',  1e-2, 10)}
            model = SVR(max_iter=1000000)
        elif str.lower(self.model_type)=='nusvm':
            params = {'nu': trial.suggest_uniform('nu', 0.01, 0.99),
                      'C': trial.suggest_uniform('C', 1e-4, 1e5),
                      'gamma':trial.suggest_uniform('gamma',  1e-2, 10)}
            model = NuSVR(max_iter=1000000)
        elif 'mlp' in str.lower(self.model_type):
            n_layers = trial.suggest_int('n_layers', 1, 2)
            layers=[]
            for i in range(n_layers):
                layers.append(trial.suggest_int('n_units_l{}'.format(i), 3, 800))
            params = {'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
                          }

            model = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000, early_stopping=True)

        return self.fit_model(model, params)

    def objective_fs(self, trial):

        params = {
            'max_depth': trial.suggest_categorical('max_depth', [1, 2, 3, 5, 10, 16, 24, 36, 52, 76, 96, 128, 150]),

        }
        model = RandomForestRegressor(n_estimators=100, n_jobs=self.inner_jobs, random_state=42, max_features=2/3)

        return self.fit_model(model, params)

    def apply_params(self, params, X, y):

        if 'xgb' in str.lower(self.model_type):
            model = xgb.XGBRegressor(random_state=42)
        elif 'rf' in str.lower(self.model_type):
            model = RandomForestRegressor(random_state=42)
        elif str.lower(self.model_type)=='svm':
            model = SVR(max_iter=1000000)
        elif str.lower(self.model_type)=='nusvm':
            model = NuSVR(max_iter=1000000)
        elif 'mlp' in str.lower(self.model_type):
            n_layers = params['n_layers']
            layers = []
            for i in range(n_layers):
                layers.append(params['n_units_l{}'.format(i)])
            params = {
                      'alpha': params['alpha'],
                      }

            model = MLPRegressor(hidden_layer_sizes=layers, max_iter=4000, early_stopping=True)
        model.set_params(**params)
        model.fit(X, y.ravel())
        return model

    def fit_model1(self, params, cvs):
        if 'xgb' in str.lower(self.model_type):
            model = xgb.XGBRegressor(random_state=42)
        elif 'rf' in str.lower(self.model_type):
            model = RandomForestRegressor(random_state=42)
        elif str.lower(self.model_type) == 'svm':
            model = SVR(max_iter=150000)
        elif str.lower(self.model_type) == 'nusvm':
            model = NuSVR(max_iter=150000)
        else:
            n_layers = params['n_layers']
            layers = []
            for i in range(n_layers):
                layers.append(params['n_units_l{}'.format(i)])
            params = {
                'alpha': params['alpha'],
            }

            model = MLPRegressor(hidden_layer_sizes=layers, max_iter=4000, early_stopping=True)
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

    def train(self, cvs, init_params=[], n_trials=500, inner_jobs = 1):
        self.inner_jobs = inner_jobs
        print('training...')
        print('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('Begin train for model %s', self.model_type)
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.cvs = cvs

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape)==1 and len(cvs[0][5].shape)==1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()

        if not self.path_group is None:
            ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
            gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

            njobs = int(ncpus - gpu_status)
            cpu_status = njobs
            joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))
        else:
            njobs = self.njobs

        study = optuna.create_study(sampler=TPESampler())
        if self.FS:
            study.optimize(self.objective_fs, n_trials=int(np.maximum(njobs, 10)), n_jobs=njobs)
        else:
            study.optimize(self.objective, n_trials=n_trials, n_jobs=njobs)

        self.best_params = study.best_params
        self.accuracy, self.acc_test = self.fit_model1(study.best_params, cvs)

        self.model = self.apply_params(study.best_params, X, y)
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
        print('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('Begin train for model %s', self.model_type)
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.cvs = cvs

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape)==1 and len(cvs[0][5].shape)==1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()

        self.accuracy, self.acc_test = self.fit_model1(self.best_params, cvs)

        self.model = self.apply_params(self.best_params, X, y)
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

    def predict(self, X):
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
                raise ImportError('Cannot open model_optuna model')
        else:
            raise ImportError('Cannot find model_optuna model')

    def save(self, model_dir):
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
