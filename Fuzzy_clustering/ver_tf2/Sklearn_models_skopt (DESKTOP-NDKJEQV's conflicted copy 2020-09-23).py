import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging, pickle
import joblib
from sklearn.model_selection import train_test_split
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from Fuzzy_clustering.ver_tf2.skopt import forest_minimize
from Fuzzy_clustering.ver_tf2.skopt.space import Real, Integer, Categorical
from Fuzzy_clustering.ver_tf2.skopt.utils import use_named_args

class sklearn_model(object):

    def __init__(self,cluster_dir,rated,model_type,njobs, init_params=None):
        self.init_params=init_params
        self.njobs=2*njobs
        self.rated=rated
        self.cluster = os.path.basename(cluster_dir)
        self.model_dir = os.path.join(cluster_dir, str.upper(model_type))
        self.istrained = False
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type=model_type
        self.optimizer = 'optuna'

        logger = logging.getLogger('skopt_train_' + '_' + self.model_type + self.cluster)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(cluster_dir, 'log_skopt_train_' + self.cluster + '.log'), 'w')
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


    def create_model(self):
        if 'xgb' in str.lower(self.model_type):

            params = [
                Real(0.4, 1, name="colsample_bylevel"),
                Real(0.4, 1, name="colsample_bytree"),
                Real(0.01, 1, name="gamma"),
                Real(0.00001, 1, name="learning_rate"),
                Integer(1, 100, name="max_depth"),
                Real(1, 10, name="min_child_weight"),
                Real(0.001, 2, name="reg_alpha"),
                Real(0.4, 1, name="subsample"),
            ]
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=self.njobs)
        elif 'rf' in str.lower(self.model_type):
            params = [
                Integer(1, 80, name="max_depth"),
                Categorical(['auto', 'sqrt', 'log2', None, 0.8, 0.6, 0.4], name="max_features"),
                Integer(1, 250, name="min_samples_leaf"),
                Integer(2, 250,  name="min_samples_split"),
            ]
            model = RandomForestRegressor(n_estimators=500, random_state=42)
        elif str.lower(self.model_type)=='svm':

            params = [
                Real(1e-3, 10, name="gamma"),
                Real(1e-2, 1e5, name="C"),
            ]
            model = SVR(max_iter=1000000)
        elif str.lower(self.model_type)=='nusvm':

            params = [
                Real(1e-3, 10, name="gamma"),
                Real(1e-2, 1e5, name="C"),
                Real(0.01, 0.99, name="nu"),
            ]
            model = NuSVR(max_iter=1000000)
        elif 'mlp' in str.lower(self.model_type):
            params = [
                Integer(5, 800, name="hidden_layer_sizes"),
                Real(1e-5, 1e-1, name="alpha")]

            model = MLPRegressor(max_iter=4000, early_stopping=True)
        return params, model

    def apply_params(self, X, y, model, **params):

        model.set_params(**params)
        model.fit(X, y.ravel())
        return model

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
                acc = np.mean(np.abs(ypred - cv[3].ravel()) / self.rated)
            rms_val.append(acc)
            ypred = model.predict(cv[4]).ravel()
            if self.rated is None:
                acc = np.mean(np.abs(ypred - cv[5].ravel()) / cv[5].ravel())
            else:
                acc = np.mean(np.abs(ypred - cv[5].ravel()) / self.rated)
            rms_test.append(acc)

        return 0.4 * np.mean(rms_val) + 0.6 * np.mean(rms_test), np.mean(rms_test)

    def train(self, cvs, init_params=[]):
        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        self.D, self.N = X.shape
        print('training...')
        print('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('%s training...begin for %s ', self.model_type, self.cluster)
        self.logger.info('Begin train for model %s', self.model_type)


        params, model = self.create_model()

        @use_named_args(params)
        def fit_model(**params):
            model.set_params(**params)
            rms_val = []
            rms_test = []
            for cv in cvs:
                model.fit(cv[0], cv[1].ravel())
                ypred = model.predict(cv[2]).ravel()
                rms_val.append(np.sqrt(np.mean(np.square(ypred - cv[3].ravel()))))
                ypred = model.predict(cv[4]).ravel()
                rms_test.append(np.sqrt(np.mean(np.square(ypred - cv[5].ravel()))))

            return 0.4*np.mean(rms_val)+0.6*np.mean(rms_test)

        gp_result = forest_minimize(func=fit_model,
                                dimensions=params,
                                n_calls=30,
                                    n_random_starts=41,
                                x0=self.init_params,
                                n_jobs=self.njobs)
        best_params=dict()
        for param, value in zip(params, gp_result.x):
            best_params[param.name]=value
        self.best_params = best_params
        self.cv_results = [gp_result.x_iters[v] for v in gp_result.func_vals.argsort()[:20]]
        self.model =model
        self.accuracy, self.acc_test = self.fit_model1(self.model, self.best_params, cvs)


        self.model.set_params(**best_params)
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
                raise ImportError('Cannot open model_skopt model')
        else:
            raise ImportError('Cannot find model_skopt model')

    def save(self, model_dir):
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        f = open(os.path.join(model_dir, 'model_all' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()


def test_skopt(cvs, X_test1,  y_test1, cluster_dir):

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

    logger.info('Scikit Optimize Evaluation')
    logger.info('/n')
    logger.info('/n')
    logger.info('SVM train')
    method = 'svm'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    if model_sklearn.istrained==True:
        model_sklearn.istrained=False
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    logger.info('Best params')
    logger.info(model_sklearn.best_params)
    logger.info('Final mae %s', str(model_sklearn.acc_test))
    logger.info('Final total %s', str(model_sklearn.accuracy))

    metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('SVM metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)

    logger.info('finish train for model %s', model_sklearn.model_type)

    logger.info('/n')

    logger.info('nu-SVM train')
    method = 'nusvm'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    if model_sklearn.istrained==True:
        model_sklearn.istrained=False
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    logger.info('Best params')
    logger.info(model_sklearn.best_params)
    logger.info('Final mae %s', str(model_sklearn.acc_test))
    logger.info('Final total %s', str(model_sklearn.accuracy))

    metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('nu-SVM metricsv')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
    logger.info('finish train for model %s', model_sklearn.model_type)
    logger.info('/n')

    logger.info('XGB train')
    method = 'xgb'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    if model_sklearn.istrained == True:
        model_sklearn.istrained = False
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    logger.info('Best params')
    logger.info(model_sklearn.best_params)
    logger.info('Final mae %s', str(model_sklearn.acc_test))
    logger.info('Final total %s', str(model_sklearn.accuracy))

    metrics_xgb = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('Xboost metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_xgb)
    logger.info('finish train for model %s', model_sklearn.model_type)
    logger.info('/n')

    logger.info('RF train')
    method = 'RF'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    if model_sklearn.istrained == True:
        model_sklearn.istrained = False
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    logger.info('Best params')
    logger.info(model_sklearn.best_params)
    logger.info('Final mae %s', str(model_sklearn.acc_test))
    logger.info('Final total %s', str(model_sklearn.accuracy))

    metrics_mlp = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('RF metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_mlp)
    logger.info('/n')
    logger.info('finish train for model %s', model_sklearn.model_type)
    logger.info('MLP train')
    method = 'mlp'
    model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
    if model_sklearn.istrained == True:
        model_sklearn.istrained = False
    model_sklearn.train(cvs)
    pred = model_sklearn.predict(X_test1)

    logger.info('Best params')
    logger.info(model_sklearn.best_params)
    logger.info('Final mae %s', str(model_sklearn.acc_test))
    logger.info('Final total %s', str(model_sklearn.accuracy))

    metrics_mlp = model_sklearn.compute_metrics(pred, y_test1, rated)
    logger.info('MLP metrics')
    logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_mlp)
    logger.info('finish train for model %s', model_sklearn.model_type)
    logger.info('/n')