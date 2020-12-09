import logging
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.svm import SVR

from Fuzzy_clustering.ver_tf2.GA_param_search import EvolutionaryAlgorithmSearchCV


class sklearn_model_tl(object):

    def __init__(self, cluster_dir, rated, model_type, njobs):
        self.njobs = 2 * njobs
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.istrained = False
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type = model_type
        self.cluster = os.path.basename(cluster_dir)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(cluster_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger
        try:
            self.load(self.model_dir)
        except:
            pass

    def compute_metrics(self, pred, y, rated):
        if rated is None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel())
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def train_deap(self, cvs, params, init_params=[]):
        self.best_params = params
        print('training with deap...')

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        self.D, self.N = X.shape
        if 'xgb' in str.lower(self.model_type):
            params = {'learning_rate': np.logspace(-5, -1, num=6, base=10),
                      'max_depth': np.unique(np.linspace(1, 50, num=50).astype('int')),
                      'colsample_bytree': np.linspace(0.4, 1.0, num=6),
                      'colsample_bynode': np.linspace(0.4, 1.0, num=6),
                      'subsample': np.linspace(0.2, 1.0, num=6),
                      'gamma': np.linspace(0.001, 2, num=20),
                      'reg_alpha': np.linspace(0, 1.0, num=12)}
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

        elif 'rf' in str.lower(self.model_type):
            params = {
                'max_depth': np.unique(np.linspace(1, 50, num=50).astype('int')),
                'max_features': ['auto', 'sqrt', 'log2', None, 0.8, 0.6, 0.4],
                'min_samples_leaf': np.linspace(.01, 0.5, num=5),
                'min_samples_split': np.linspace(.01, 0.5, num=5),
            }
            model = RandomForestRegressor(n_estimators=500, random_state=42)

        elif str.lower(self.model_type) == 'svm':
            params = {'C': np.logspace(-5, 5, num=100, base=10),
                      'gamma': np.linspace(0.0001, 2, num=100)}
            model = SVR(max_iter=150000)

        elif str.lower(self.model_type) == 'nusvm':
            params = {'nu': np.linspace(0.01, 0.99, num=10),
                      'C': np.logspace(-5, 5, num=100, base=10),
                      'gamma': np.linspace(0.0001, 2, num=100)}
            model = NuSVR(max_iter=500000)

        elif 'mlp' in str.lower(self.model_type):

            params = {'hidden_layer_sizes': np.linspace(4, 800, num=50).astype('int'),
                      'alpha': np.linspace(1e-5, 1e-1, num=4),
                      }

            model = MLPRegressor(max_iter=1000, early_stopping=True)

        if len(init_params) == 0:
            cv = EvolutionaryAlgorithmSearchCV(estimator=model,
                                               params=params,
                                               scoring='neg_root_mean_squared_error',
                                               cv=3,
                                               verbose=1,
                                               population_size=20,
                                               gene_mutation_prob=0.8,
                                               gene_crossover_prob=0.8,
                                               tournament_size=3,
                                               generations_number=4,
                                               refit=False,
                                               init_params=init_params,
                                               n_jobs=self.njobs)
        else:
            cv = EvolutionaryAlgorithmSearchCV(estimator=model,
                                               params=params,
                                               scoring='neg_root_mean_squared_error',
                                               cv=3,
                                               verbose=1,
                                               population_size=20,
                                               gene_mutation_prob=0.8,
                                               gene_crossover_prob=0.8,
                                               tournament_size=3,
                                               generations_number=4,
                                               refit=False,
                                               init_params=init_params,
                                               n_jobs=self.njobs)
        cv.fit(cvs)

        self.model = cv.best_estimator_
        self.accuracy = cv.best_score_
        self.best_params = cv.best_params_
        params = dict()
        for i, ind in cv.all_history_[-1].genealogy_history.items():
            if len(ind.fitness.getValues()) > 0:
                params[i] = dict()
                params[i]['acc'] = ind.fitness.getValues()[0]
                params[i]['params'] = np.array(ind)

        self.cv_results = pd.DataFrame.from_dict(params, orient='index')
        self.cv_results = self.cv_results.drop_duplicates('acc')
        logs = []
        for log in cv.all_logbooks_:
            for l in log:
                logs.append(l)
        logs = pd.DataFrame(logs)
        logs.to_csv(os.path.join(self.model_dir, 'trials.csv'))
        self.save(self.model_dir)
        pred = self.model.predict(cvs[0][4])
        if self.rated is None:
            self.acc_test = np.mean(np.abs(pred - cvs[0][5].ravel()) / cvs[0][5].ravel())
        else:
            self.acc_test = np.mean(np.abs(pred - cvs[0][5]) / self.rated)

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

    def create_model(self):
        if 'xgb' in str.lower(self.model_type):
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=self.njobs)
        elif 'rf' in str.lower(self.model_type):
            model = RandomForestRegressor(n_estimators=500, random_state=42)
        elif str.lower(self.model_type) == 'svm':

            model = SVR(max_iter=150000)
        elif str.lower(self.model_type) == 'nusvm':

            model = NuSVR(max_iter=500000)
        elif 'mlp' in str.lower(self.model_type):
            model = MLPRegressor(max_iter=1000, early_stopping=True)
        return model

    def fit_model1(self, model, params, cvs):
        model.set_params(**params)
        rms_val = []
        rms_test = []
        for cv in cvs:
            model.fit(cv[0], cv[1].ravel())
            ypred = model.predict(cv[2]).ravel()
            rms_val.append(np.sqrt(np.mean(np.square(ypred - cv[3].ravel()))))
            ypred = model.predict(cv[4]).ravel()
            rms_test.append(np.sqrt(np.mean(np.square(ypred - cv[5].ravel()))))

        return model, 0.4 * np.mean(rms_val) + 0.6 * np.mean(rms_test) + 2 * np.abs(
            np.mean(rms_val) - np.mean(rms_test))

    def train(self, cvs, params):
        self.best_params = params
        print('training...')
        print('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('Begin train for model %s', self.model_type)
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.cvs = cvs

        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()

        model = self.create_model()
        self.model, self.accuracy = self.fit_model1(model, self.best_params, cvs)
        pred = self.model.predict(cvs[0][4])
        if self.rated is None:
            self.acc_test = np.mean(np.abs(pred.ravel() - cvs[0][5].ravel()))
        else:
            self.acc_test = np.mean(np.abs(pred.ravel() - cvs[0][5].ravel()))

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
            if k not in ['logger']:
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
