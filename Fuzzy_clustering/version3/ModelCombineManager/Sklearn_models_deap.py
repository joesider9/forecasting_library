import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging, pickle
import joblib
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from Fuzzy_clustering.version3.ModelCombineManager.GA_param_search import EvolutionaryAlgorithmSearchCV


class sklearn_model(object):

    def __init__(self, static_data, cluster_dir, rated, model_type, njobs, is_combine=False, path_group=None):
        self.static_data = static_data
        self.path_group = path_group
        self.rated=rated
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type = model_type
        self.is_combine = is_combine
        self.optimizer = 'deap'
        self.istrained = False
        self.cluster = os.path.basename(cluster_dir)
        logger = logging.getLogger('deap_train_' + '_' + self.model_type + self.cluster)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_deap_train_' + self.cluster + '.log'), 'w')
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
        self.njobs=njobs

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

    def train(self, cvs, init_params=[], FS=False, inner_jobs=1):
        print('training with deap...')

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape)==1 and len(cvs[0][5].shape)==1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        self.D, self.N = X.shape


        if 'elasticnet' in str.lower(self.model_type):
            X_train = cvs[0][0]
            y_train = cvs[0][1].reshape(-1, 1)
            X_val = cvs[0][2]
            y_val = cvs[0][3].reshape(-1, 1)
            X_test = cvs[0][4]
            y_test = cvs[0][5].reshape(-1, 1)
            X_train = np.vstack((X_train, X_val, X_test))
            y_train = np.vstack((y_train, y_val, y_test))
            model = ElasticNetCV(cv=5, max_iter=4000)
            model.fit(X_train, y_train.ravel())

            self.best_params = model.get_params()
            ypred = model.predict(X_test).ravel()
            if self.rated is None:
                self.accuracy = np.mean(np.abs(ypred - y_test.ravel()) / y_test.ravel())
            else:
                self.accuracy = np.mean(np.abs(ypred - y_test.ravel()))
            self.acc_test = self.accuracy
            self.model = model

            self.logger.info('Best params')
            self.logger.info(self.best_params)
            self.logger.info('Final mae %s', str(self.acc_test))
            self.logger.info('Final rms %s', str(self.accuracy))
            self.logger.info('finish train for model %s', self.model_type)
            self.istrained = True
            self.save(self.model_dir)

            return self.to_dict()

        else:
            if 'xgb' in str.lower(self.model_type):
                params = {'learning_rate': np.logspace(-5, -1, num=6, base=10),
                          'max_depth': np.unique(np.linspace(1, 150, num=50).astype('int')),
                          'colsample_bytree': np.linspace(0.4, 1.0, num=60),
                          'colsample_bynode': np.linspace(0.4, 1.0, num=60),
                          'subsample': np.linspace(0.2, 1.0, num=6),
                          'gamma': np.linspace(0.001, 2, num=20),
                          'reg_alpha': np.linspace(0, 1.0, num=12)}
                model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
                ngen = self.static_data['sklearn']['gen']
                npop = self.static_data['sklearn']['pop']
            elif 'rf' in str.lower(self.model_type):
                if FS:
                    params = {
                        'max_depth': [1, 2, 3, 5, 10, 16, 24, 36, 52, 76, 96, 128, 150],
                    }
                    model = RandomForestRegressor(n_estimators=100, n_jobs=inner_jobs,random_state=42, max_features=2/3)
                    ngen = 2
                    npop = 4
                else:
                    params = {
                              'max_depth': np.unique(np.linspace(1, 130, num=50).astype('int')),
                              'max_features': ['auto', 'sqrt', 'log2', None, 0.8, 0.6, 0.4],
                              'min_samples_leaf': np.unique(np.linspace(1, cvs[0][0].shape[0]-10, num=50).astype('int')),
                              'min_samples_split': np.unique(np.linspace(2, cvs[0][0].shape[0]-10, num=50).astype('int')),
                              }
                    model =  RandomForestRegressor(n_estimators=500, random_state=42)
                    ngen = self.static_data['sklearn']['gen']
                    npop = self.static_data['sklearn']['pop']
            elif str.lower(self.model_type)=='svm':
                params = {'C': np.logspace(-2, 3, num=100, base=10),
                          'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                          'gamma': list(np.linspace(0.001, 2, num=100)) + ['scale', 'auto']}
                model = SVR(max_iter=1000000)
                ngen = self.static_data['sklearn']['gen']
                npop = self.static_data['sklearn']['pop']
            elif str.lower(self.model_type)=='nusvm':
                params = {'nu': np.linspace(0.01, 0.99, num=10),
                          'C': np.logspace(-1, 5, num=100, base=10),
                          'gamma': np.linspace(0.01, 10, num=100)}
                model = NuSVR(max_iter=1000000)
                ngen = self.static_data['sklearn']['gen']
                npop = self.static_data['sklearn']['pop']
            elif 'mlp' in str.lower(self.model_type):
                if not self.is_combine:
                    params = {'hidden_layer_sizes': np.linspace(4, 800, num=50).astype('int'),
                              'alpha': np.linspace(1e-5, 1e-1, num=4),
                              }
                else:
                    params = {'hidden_layer_sizes': np.linspace(4, 250, num=50).astype('int'),
                              'activation': ['identity', 'tanh', 'relu'],
                              'alpha': np.linspace(1e-5, 1e-1, num=4),
                              }

                model = MLPRegressor(max_iter=1000, early_stopping=True)
                ngen = 5
                npop = self.static_data['sklearn']['pop']

        if not self.path_group is None:
            ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
            gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

            njobs = int(ncpus - gpu_status)
            cpu_status = njobs
            joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))
        else:
            njobs = self.njobs


        cv = EvolutionaryAlgorithmSearchCV(estimator=model,
                                           params=params,
                                           scoring='neg_root_mean_squared_error',
                                           cv=3,
                                           rated=self.rated,
                                           verbose=1,
                                           population_size=npop,
                                           gene_mutation_prob=0.8,
                                           gene_crossover_prob=0.8,
                                           tournament_size=3,
                                           generations_number=ngen,
                                           refit=False,
                                           init_params=init_params,
                                           n_jobs=njobs,
                                           path_group=self.path_group)

        cv.fit(cvs)

        self.best_params = cv.best_params_

        self.accuracy, self.acc_test = self.fit_model1(model, self.best_params, cvs)

        self.model = model
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
        print('training with deap...')

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape)==1 and len(cvs[0][5].shape)==1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        self.D, self.N = X.shape
        if 'xgb' in str.lower(self.model_type):
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

        elif 'rf' in str.lower(self.model_type):
            model =  RandomForestRegressor(n_estimators=500, random_state=42)

        elif str.lower(self.model_type)=='svm':
            model = SVR(max_iter=1000000)

        elif str.lower(self.model_type)=='nusvm':
            model = NuSVR(max_iter=1000000)

        elif 'mlp' in str.lower(self.model_type):
            model = MLPRegressor(max_iter=1000, early_stopping=True)
        else:
            raise ValueError(' Cannot recognize sklearn method')

        self.accuracy, self.acc_test = self.fit_model1(model, self.best_params, cvs)

        self.model = model
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
                raise ImportError('Cannot open model_deap model')
        else:
            raise ImportError('Cannot find model_deap model')

    def save(self, model_dir):
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
