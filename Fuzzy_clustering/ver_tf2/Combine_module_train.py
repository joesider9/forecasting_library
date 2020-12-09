import os
import pickle, logging
import numpy as np
import joblib, tqdm
import pandas as pd
from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import sklearn_model
from Fuzzy_clustering.ver_tf2.Cluster_predict_regressors import cluster_predict
from Fuzzy_clustering.ver_tf2.Global_predict_regressor import global_predict
from Fuzzy_clustering.ver_tf2.Data_Sampler import DataSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

class combine_model(object):
    def __init__(self, static_data, cluster_dir, x_scaler=None, is_global=False):
        self.istrained = False
        self.combine_methods = static_data['combine_methods']
        self.cluster_name = os.path.basename(cluster_dir)
        self.cluster_dir = cluster_dir
        self.model_dir = os.path.join(self.cluster_dir, 'Combine')
        try:
            self.load(self.model_dir)
        except:
            pass

        self.x_scaler = x_scaler
        self.static_data = static_data

        self.model_type=static_data['type']
        self.is_global = is_global
        self.methods = []
        if static_data['resampling']:
            if is_global:
                self.resampling = False
                for method in static_data['project_methods'].keys():
                    if self.static_data['project_methods'][method]['Global'] == True:
                        if method == 'ML_RBF_ALL_CNN':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                        elif method == 'ML_RBF_ALL':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN' ])
                        else:
                            self.methods.append(method)
            else:
                self.resampling = True
                for method in static_data['project_methods'].keys():
                    if static_data['project_methods'][method]['status'] == 'train':
                        if method == 'ML_RBF_ALL_CNN':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                        elif method == 'ML_RBF_ALL':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN' ])
                        else:
                            self.methods.append(method)
        else:
            if is_global:
                self.resampling = False
                for method in static_data['project_methods'].keys():
                    if self.static_data['project_methods'][method]['Global'] == True:
                        if method == 'ML_RBF_ALL_CNN':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                        elif method == 'ML_RBF_ALL':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN' ])
                        else:
                            self.methods.append(method)
            else:
                self.resampling = False
                for method in static_data['project_methods'].keys():
                    if static_data['project_methods'][method]['status'] == 'train':
                        if method == 'ML_RBF_ALL_CNN':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                        elif method == 'ML_RBF_ALL':
                            self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN' ])
                        else:
                            self.methods.append(method)

        self.rated=static_data['rated']


        self.n_jobs = static_data['njobs']

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.data_dir = os.path.join(self.cluster_dir, 'data')
        logger = logging.getLogger('combine_train_procedure')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_combine.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def simple_stack(self, x, y):
        if x.shape[0]==0:
            x = y
        else:
            x = np.vstack((x, y))
        return x

    def rls_fit(self, X, y):
        P = 1e-4 * np.eye(self.weight_size)
        C = np.array([])
        err = np.array([])
        preds = np.array([])
        w = np.ones([1, self.weight_size]) / self.weight_size
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp=inp.reshape(-1,1)
            pred = np.matmul(w, inp)
            e = targ - pred
            if err.shape[0]==0:
                sigma = 1
            else:
                sigma = np.square(np.std(err))

            c = np.square(e)*np.matmul( np.matmul(np.transpose(inp), P), inp)/(sigma*(1+np.matmul( np.matmul(np.transpose(inp), P), inp)))
            C = self.simple_stack(C, c)
            R=np.random.chisquare(inp.shape[0],C.shape[0])
            censored=R>C
            f=np.mean(censored)
            l=0.85+(0.999-0.85)*f
            P=(1/l)*(P-(np.matmul(np.matmul(P,np.matmul(inp,np.transpose(inp))),P))/(l+np.matmul( np.matmul(np.transpose(inp), P), inp)))
            w+=np.transpose(np.matmul(P,inp)*e)
            w[np.where(w < 0)] = 0
            w /= np.sum(w)
            err = self.simple_stack(err, e)
            if count>7000:
                break
            count+=1
        return w

    def bcp_fit(self, X, y):
        sigma=np.nanstd(y-X,axis=0).reshape(-1,1)
        err = []
        preds = []
        w = np.ones([1, self.weight_size]) / self.weight_size
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp=inp.reshape(-1,1)
            mask=~np.isnan(inp)
            pred = np.matmul(w[mask.T], inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p=np.exp(-1*np.square((targ-inp[mask].T)/(np.sqrt(2*np.pi)*sigma[mask])))
            w[mask.T]=((w[mask.T]*p)/np.sum(w[mask.T]*p))
            w[np.where(w < 0)] = 0
            w /= np.sum(w)

            count+=1
        return w

    def resampling_for_combine(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):
        if (not self.x_scaler is None):
            if self.is_global:
                predict_module = global_predict(self.static_data)
            else:
                predict_module = cluster_predict(self.static_data, self.cluster_name)

            self.logger.info('Make predictions of testing set not used in training')
            self.logger.info('/n')

            pred_test_comb = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test, fs_reduced=False)

            result_test_comb = predict_module.evaluate(pred_test_comb, y_test.values)

            result_test_comb = result_test_comb.sort_values(by=['mae'])

            self.logger.info('Make predictions of sampling set with nwp_sampler')
            self.logger.info('/n')

            sampler_dl = DataSampler(self.static_data, self.cluster_name, self.x_scaler, method='ADASYN')
            sampler_dl.istrained = False
            # if not os.path.exists(os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle')):
            # if not sampler_dl.istrained:
            if len(X_cnn_test.shape) > 1:
                X_sampl, y_sampl, X_cnn_sampl, X_lstm_sampl = sampler_dl.imblearn_sampling(X=X_test, y=y_test,
                                                                                           act=act_test,
                                                                                           X_cnn=X_cnn_test)
            elif len(X_lstm_test.shape) > 1:
                X_sampl, y_sampl, X_cnn_sampl, X_lstm_sampl = sampler_dl.imblearn_sampling(X=X_test, y=y_test, act=act_test,
                                                                           X_lstm=X_lstm_test)
            else:
                raise NotImplementedError('X_lstm sampling not implemented yet')

            if len(X_cnn_test.shape) > 1:
                pred_nwp_dl_resample = predict_module.predict(X_sampl.values, X_cnn=X_cnn_sampl, fs_reduced=False)
            elif len(X_lstm_test.shape) > 1:
                pred_nwp_dl_resample = predict_module.predict(X_sampl.values, X_lstm=X_lstm_sampl, fs_reduced=False)
            else:
                raise NotImplementedError('X_lstm sampling not implemented yet')

            joblib.dump(pred_nwp_dl_resample, os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle'))
            joblib.dump(y_sampl, os.path.join(self.data_dir, 'y_resample_dl.pickle'))
                # joblib.dump(y_cnn_sampl, os.path.join(self.data_dir, 'y_cnn_resample_dl.pickle'))
            # else:
            #     pred_nwp_dl_resample = joblib.load(os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle'))
            #     y_sampl = joblib.load(os.path.join(self.data_dir, 'y_resample_dl.pickle'))
            #     # y_cnn_sampl = joblib.load(os.path.join(self.data_dir, 'y_cnn_resample_dl.pickle'))


            result_nwp_dl_resample = predict_module.evaluate(pred_nwp_dl_resample, y_sampl)

            result = pd.concat({'on_test': result_test_comb['mae'],
                                # 'with_nwp_dl_resample_org': result_nwp_dl_resample_org['mae'],
                                'with_nwp_dl_resample': result_nwp_dl_resample['mae']}, axis=1)
            result.to_csv(os.path.join(self.data_dir, 'result_sampling.csv'))

            return pred_nwp_dl_resample, y_sampl.reshape(-1,1), result_nwp_dl_resample.astype(float)
        else:
            raise ValueError('Scaler or data indices are not set')





    def without_resampling(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):
        if (not self.x_scaler is None):
            X_test = X_test.values
            y_test = y_test.values
            if self.is_global:
                predict_module = global_predict(self.static_data)
            else:
                predict_module = cluster_predict(self.static_data, self.cluster_name)

            self.logger.info('Make predictions of testing set not used in training')
            self.logger.info('/n')

            pred_test_comb = predict_module.predict(X_test, X_cnn=X_cnn_test, X_lstm=X_lstm_test, fs_reduced=False)

            result_test_comb = predict_module.evaluate(pred_test_comb, y_test)


            if len(y_test.shape) == 1:
                y_test = y_test[:, np.newaxis]
            return pred_test_comb, y_test, result_test_comb.astype(float)
        else:
            raise ValueError('Scaler or data indices are not set')

    def train(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):
        if X_test.shape[0]>0 and len(self.methods)>1:
            if self.model_type in {'pv', 'wind'}:
                if self.resampling == True:
                    pred_resample, y_resample, results = self.resampling_for_combine(X_test, y_test, act_test, X_cnn_test, X_lstm_test)
                else:
                    pred_resample, y_resample, results = self.without_resampling(X_test, y_test,act_test,
                                                                                       X_cnn_test, X_lstm_test)
            elif self.model_type in {'load'}:
                if self.resampling == True:
                    pred_resample, y_resample, results = self.resampling_for_combine(X_test, y_test, act_test,
                                                                                       X_cnn_test, X_lstm_test)
                else:
                    pred_resample, y_resample, results = self.without_resampling(X_test, y_test, act_test,
                                                                                 X_cnn_test, X_lstm_test)
            elif self.model_type in {'fa'}:
                if self.resampling == True:
                    pred_resample, y_resample, results = self.resampling_for_combine(X_test, y_test, act_test,
                                                                                       X_cnn_test, X_lstm_test)
                else:
                    pred_resample, y_resample, results = self.without_resampling(X_test, y_test, act_test,
                                                                                 X_cnn_test, X_lstm_test)

            self.best_methods = results.nsmallest(4,'mae').index.tolist()
            results = results.loc[self.best_methods]
            results['diff'] = results['mae'] - results['mae'].iloc[0]
            best_of_best = results.iloc[np.where(results['diff'] <= 0.01)].index.tolist()
            if len(best_of_best)==1:
                best_of_best.append(self.best_methods[1])
            self.best_methods = best_of_best
            X_pred = np.array([])
            for method in sorted(self.best_methods):
                if X_pred.shape[0]==0:
                    X_pred=pred_resample[method]
                else:
                    X_pred = np.hstack((X_pred,pred_resample[method]))
            X_pred /= 20
            X_pred[np.where(X_pred<0)] = 0
            y_resample /=20
            X_pred, y_resample = shuffle(X_pred, y_resample)
            self.weight_size = len(self.best_methods)
            self.model = dict()
            for combine_method in self.combine_methods:
                if combine_method=='rls':
                    self.logger.info('RLS training')
                    self.logger.info('/n')
                    self.model[combine_method] = dict()
                    w = self.rls_fit(X_pred, y_resample)

                    self.model[combine_method]['w']=w

                elif combine_method=='bcp':
                    self.logger.info('BCP training')
                    self.logger.info('/n')
                    self.model[combine_method] = dict()
                    w= self.bcp_fit(X_pred, y_resample)
                    self.model[combine_method]['w'] = w

                elif combine_method == 'mlp':
                    self.logger.info('MLP training')
                    self.logger.info('/n')
                    cvs = []
                    for _ in range(3):
                        X_train, X_test1, y_train, y_test1 = train_test_split(X_pred, y_resample, test_size=0.15)
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
                        cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                    mlp_model= sklearn_model(self.model_dir, self.rated, 'mlp', self.n_jobs, is_combine=True)
                    self.model[combine_method] = mlp_model.train(cvs)

                elif combine_method == 'bayesian_ridge':
                    self.logger.info('bayesian_ridge training')
                    self.logger.info('/n')
                    self.model[combine_method] = BayesianRidge()
                    self.model[combine_method].fit(X_pred, y_resample)

                elif combine_method == 'elastic_net':
                    self.logger.info('elastic_net training')
                    self.logger.info('/n')
                    self.model[combine_method] = ElasticNetCV(cv=5)
                    self.model[combine_method].fit(X_pred, y_resample)
                elif combine_method == 'ridge':
                    self.logger.info('ridge training')
                    self.logger.info('/n')
                    self.model[combine_method] = RidgeCV(cv=5)
                    self.model[combine_method].fit(X_pred, y_resample)
            self.logger.info('End of combine models training')
        else:
            self.combine_methods= ['average']
        self.istrained = True
        self.save(self.model_dir)

    def predict(self,X):
        X_pred = np.array([])
        if not hasattr(self, 'best_methods'):
            self.best_methods = X.keys()
        for method in sorted(self.best_methods):
            if X_pred.shape[0]==0:
                X_pred = X[method]
            else:
                X_pred = np.hstack((X_pred, X[method]))
        X_pred /= 20
        if not hasattr(self,'model'):
            raise ValueError('The combine models does not exist')
        pred_combine = dict()
        for combine_method in self.combine_methods:
            if combine_method == 'rls':
                pred = np.matmul(self.model[combine_method]['w'], X_pred)
            elif combine_method == 'bcp':
                pred = np.matmul(self.model[combine_method]['w'], X_pred)

            elif combine_method == 'mlp':
                self.model[combine_method] = sklearn_model(self.model_dir, self.rated, 'mlp', self.n_jobs)
                pred = self.model[combine_method].predict(X_pred)

            elif combine_method == 'bayesian_ridge':
                self.model[combine_method] = BayesianRidge(fit_intercept=False)
                pred = self.model[combine_method].predict(X_pred)

            elif combine_method == 'elastic_net':
                self.model[combine_method] = ElasticNetCV(cv=5, fit_intercept=False)
                pred = self.model[combine_method].predict(X_pred)
            elif combine_method == 'ridge':
                self.model[combine_method] = RidgeCV(cv=5, fit_intercept=False)
                pred = self.model[combine_method].predict(X_pred)

            elif combine_method == 'isotonic':
                self.model[combine_method] = IsotonicRegression(y_min=0, y_max=1)
                pred = self.model[combine_method].predict(X_pred)
            else:
                pred = np.mean(X_pred, axis=1).reshape(-1, 1)

            pred_combine[combine_method] = 20 * pred

        return pred_combine


    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'combine_models.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'combine_models.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RLS model')
        else:
            raise ImportError('Cannot find RLS model')


    def save(self, pathname):
        cluster_dir = pathname
        f = open(os.path.join(cluster_dir, 'combine_models.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
        f.close()


    def resampling_for_combine_obsolete(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):
        if (not self.x_scaler is None):
            if self.is_global:
                predict_module = global_predict(self.static_data)
            else:
                predict_module = cluster_predict(self.static_data, self.cluster_name)

            self.logger.info('Make predictions of testing set not used in training')
            self.logger.info('/n')

            pred_test_comb = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test, fs_reduced=False)

            result_test_comb = predict_module.evaluate(pred_test_comb, y_test.values)

            result_test_comb = result_test_comb.sort_values(by=['mae'])

            self.logger.info('Make predictions of sampling set with nwp_sampler')
            self.logger.info('/n')

            sampler_dl = DataSampler(self.static_data, self.cluster_name, self.x_scaler, method='ADASYN')
            sampler_dl.istrained = False
            if not os.path.exists(os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle')):
                if not sampler_dl.istrained:
                    if len(X_cnn_test.shape) > 1 and len(X_lstm_test.shape) > 1:
                        raise NotImplementedError('X_lstm sampling not implemented yet')
                    elif len(X_cnn_test.shape) > 1:
                        X_sampl, y_sampl, X_cnn_sampl = sampler_dl.nwp_dl_sampling(X=X_test, y=y_test, act=act_test,
                                                                                   X_cnn=X_cnn_test)
                        # X_sampl, y_sampl, X_cnn_sampl, y_cnn_sampl = sampler_dl.nwp_dl_sampling()

                    elif len(X_lstm_test.shape) > 1:
                        raise NotImplementedError('X_lstm sampling not implemented yet')
                    else:
                        X_sampl, y_sampl, X_cnn_sampl = sampler_dl.nwp_dl_sampling(X=X_test, y=y_test, act=act_test)
                        # X_sampl, y_sampl, X_cnn_sampl, y_cnn_sampl = sampler_dl.nwp_dl_sampling()

                if len(X_cnn_test.shape) > 1 and len(X_lstm_test.shape) > 1:
                    raise NotImplementedError('X_lstm sampling not implemented yet')
                elif len(X_cnn_test.shape) > 1:
                    pred_nwp_dl_resample = predict_module.spark_predict(X_sampl.values, X_cnn=X_cnn_sampl, fs_reduced=False)
                elif len(X_lstm_test.shape) > 1:
                    raise NotImplementedError('X_lstm sampling not implemented yet')
                else:
                    pred_nwp_dl_resample = predict_module.spark_predict(X_sampl.values, fs_reduced=False)

                joblib.dump(pred_nwp_dl_resample, os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle'))
                joblib.dump(y_sampl, os.path.join(self.data_dir, 'y_resample_dl.pickle'))
                # joblib.dump(y_cnn_sampl, os.path.join(self.data_dir, 'y_cnn_resample_dl.pickle'))
            else:
                pred_nwp_dl_resample = joblib.load(os.path.join(self.data_dir, 'prediction_nwp_dl_resample.pickle'))
                y_sampl = joblib.load(os.path.join(self.data_dir, 'y_resample_dl.pickle'))
                # y_cnn_sampl = joblib.load(os.path.join(self.data_dir, 'y_cnn_resample_dl.pickle'))


            result_nwp_dl_resample = predict_module.evaluate(pred_nwp_dl_resample, y_sampl)

            result = pd.concat({'on_test': result_test_comb['mae'],
                                # 'with_nwp_dl_resample_org': result_nwp_dl_resample_org['mae'],
                                'with_nwp_dl_resample': result_nwp_dl_resample['mae']}, axis=1)
            result.to_csv(os.path.join(self.data_dir, 'result_sampling.csv'))

            return pred_nwp_dl_resample, y_sampl, pred_test_comb
        else:
            raise ValueError('Scaler or data indices are not set')