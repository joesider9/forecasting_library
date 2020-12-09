import os
import pickle, logging
import numpy as np
import joblib, tqdm
import pandas as pd
from Fuzzy_clustering.version3.ClusterCombineManager.Sklearn_models_deap import sklearn_model
from Fuzzy_clustering.version3.ClusterCombineManager.Data_Sampler import DataSampler
from Fuzzy_clustering.version3.ClusterCombineManager.ClusterPredictManager import ClusterPredict
from rabbitmq_rpc.server import RPCServer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.utils import shuffle
from Fuzzy_clustering.version3.ClusterCombineManager.Cluster_object import cluster_object
import pika, uuid, time, json

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='ClusterCombinemanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class ClusterCombiner():
    def __init__(self, static_data, cluster):
        self.istrained= False
        self.cluster = cluster
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.model_dir = os.path.join(self.cluster_dir, 'Combine')
        self.static_data = static_data
        self.model_type = static_data['type']
        self.methods = []
        for method in cluster.methods:
            if method == 'RBF_ALL_CNN':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
            elif  method == 'RBF_ALL':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
            else:
                self.methods.append(method)
        self.combine_methods = static_data['combine_methods']
        self.rated = static_data['rated']
        self.n_jobs = static_data['sklearn']['njobs']
        self.resampling = static_data['resampling']
        try:
            self.load(self.model_dir)
        except:
            pass
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.data_dir = os.path.join(self.cluster_dir, 'data')

        logger = logging.getLogger('combine_'+ self.cluster_name + '_' + static_data['_id'])
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_combine.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def resampling_for_combine(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):

        predict_module = ClusterPredict(self.static_data, self.cluster)


        self.logger.info('Make predictions of testing set not used in training')
        self.logger.info('/n')

        pred_test_comb = predict_module.predict(X_test.values, X_cnn=X_cnn_test, X_lstm=X_lstm_test, fs_reduced=False)

        result_test_comb = predict_module.evaluate(pred_test_comb, y_test.values)

        result_test_comb = result_test_comb.sort_values(by=['mae'])

        self.logger.info('Make predictions of sampling set with nwp_sampler')
        self.logger.info('/n')

        sampler_dl = DataSampler(self.static_data, self.cluster_name, method='ADASYN')
        sampler_dl.istrained = False

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

        result_nwp_dl_resample = predict_module.evaluate(pred_nwp_dl_resample, y_sampl)

        result = pd.concat({'on_test': result_test_comb['mae'],
                            # 'with_nwp_dl_resample_org': result_nwp_dl_resample_org['mae'],
                            'with_nwp_dl_resample': result_nwp_dl_resample['mae']}, axis=1)
        result.to_csv(os.path.join(self.data_dir, 'result_sampling.csv'))

        return pred_nwp_dl_resample, y_sampl.reshape(-1,1), result_nwp_dl_resample.astype(float)

    def without_resampling(self, X_test, y_test, act_test, X_cnn_test, X_lstm_test):

        X_test = X_test.values
        y_test = y_test.values
        predict_module = ClusterPredict(self.static_data, self.cluster)

        self.logger.info('Make predictions of testing set not used in training')
        self.logger.info('/n')

        pred_test_comb = predict_module.predict(X_test, X_cnn=X_cnn_test, X_lstm=X_lstm_test, fs_reduced=False)

        result_test_comb = predict_module.evaluate(pred_test_comb, y_test)


        if len(y_test.shape) == 1:
            y_test = y_test[:, np.newaxis]
        return pred_test_comb, y_test, result_test_comb.astype(float)

    def load_data(self):
        data_path = self.data_dir
        if os.path.exists(os.path.join(data_path, 'dataset_X_test.csv')):
            X_test = pd.read_csv(os.path.join(data_path, 'dataset_X_test.csv'), index_col=0, header=0, parse_dates=True,
                        dayfirst=True)
            y_test = pd.read_csv(os.path.join(data_path, 'dataset_y_test.csv'), index_col=0, header=0, parse_dates=True,
                            dayfirst=True)
            act_test = pd.read_csv(os.path.join(data_path, 'dataset_act_test.csv'), index_col=0, header=0, parse_dates=True,
                              dayfirst=True)

            if os.path.exists(os.path.join(data_path, 'dataset_cnn_test.pickle')):
                X_cnn_test = joblib.load(os.path.join(data_path, 'dataset_cnn_test.pickle'))
            else:
                X_cnn_test = np.array([])

            if os.path.exists(os.path.join(data_path, 'dataset_lstm_test.pickle')):
                X_lstm_test = joblib.load(os.path.join(data_path, 'dataset_lstm_test.pickle'))
            else:
                X_lstm_test = np.array([])
        else:
            X_test = pd.DataFrame([])
            y_test = pd.DataFrame([])
            act_test = pd.DataFrame([])
            X_cnn_test = np.array([])
            X_lstm_test = np.array([])
        return X_test, y_test, act_test, X_cnn_test, X_lstm_test

    def train(self):
        X_test, y_test, act_test, X_cnn_test, X_lstm_test = self.load_data()
        if X_test.shape[0]>0 and len(self.methods)>1 and self.istrained==False:
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
            best_of_best = results.iloc[np.where(results['diff'] <= 0.02)].index.tolist()
            if len(best_of_best)==1:
                best_of_best.extend([best_of_best[0], best_of_best[0], self.best_methods[1]])
            elif len(best_of_best)==2:
                best_of_best.extend([best_of_best[0], best_of_best[0]])
            elif len(best_of_best)==3:
                best_of_best.append(best_of_best[0])

            self.best_methods = best_of_best
            X_pred = np.array([])
            for method in sorted(self.best_methods):
                if X_pred.shape[0]==0:
                    X_pred=pred_resample[method]
                else:
                    X_pred = np.hstack((X_pred,pred_resample[method]))
            X_pred[np.where(X_pred<0)] = 0
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
                        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pred, y_resample, test_size=0.15)
                        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.15)
                        cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                    mlp_model= sklearn_model(self.static_data, self.model_dir, self.rated, 'mlp', self.n_jobs, is_combine=True, path_group=self.static_data['path_group'])
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

        return 'Done'


    def simple_stack(self, x, y):
        if x.shape[0] == 0:
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

        for _ in range(3):
            count = 0
            for inp, targ in tqdm.tqdm(zip(X, y)):
                inp = inp.reshape(-1, 1)
                pred = np.matmul(w, inp)
                e = targ - pred
                if err.shape[0] == 0:
                    sigma = 1
                else:
                    sigma = np.square(np.std(err))

                c = np.square(e) * np.matmul(np.matmul(np.transpose(inp), P), inp) / (
                            sigma * (1 + np.matmul(np.matmul(np.transpose(inp), P), inp)))
                C = self.simple_stack(C, c)
                R = np.random.chisquare(inp.shape[0], C.shape[0])
                censored = R > C
                f = np.mean(censored)
                l = 0.75 + (0.999 - 0.75) * f
                P = (1 / l) * (P - (np.matmul(np.matmul(P, np.matmul(inp, np.transpose(inp))), P)) / (
                            l + np.matmul(np.matmul(np.transpose(inp), P), inp)))
                w += np.transpose(np.matmul(P, inp) * e)
                w[np.where(w < 0)] = 0
                w /= np.sum(w)
                err = self.simple_stack(err, e)
                if count > 7000:
                    break
                count += 1
        return w

    def bcp_fit(self, X, y):
        sigma = np.nanstd(y - X, axis=0).reshape(-1, 1)
        err = []
        preds = []
        w = np.ones([1, self.weight_size]) / self.weight_size
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp = inp.reshape(-1, 1)
            mask = ~np.isnan(inp)
            pred = np.matmul(w[mask.T], inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p = np.exp(-1 * np.square((targ - inp[mask].T) / (np.sqrt(2 * np.pi) * sigma[mask])))
            w[mask.T] = ((w[mask.T] * p) / np.sum(w[mask.T] * p))
            w[np.where(w < 0)] = 0
            w /= np.sum(w)

            count += 1
        return w
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer) or isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.str) or isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, bool):
            return bool(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            print(obj)
            raise TypeError('Object is not JSON serializable')


@server.consumer()
def combine_manager(static_data):
    print(" [.] Receive cluster %s)" % static_data['cluster_name'])
    cluster = cluster_object(static_data, static_data['cluster_name'])
    combine_model = ClusterCombiner(static_data, cluster)
    if combine_model.istrained == False:
        cb_response = {'result': combine_model.train(), 'cluster_name': cluster.cluster_name,
                       'project': static_data['_id']}
    else:
        cb_response = {'result': 'Done', 'cluster_name': cluster.cluster_name, 'project': static_data['_id']}
    return cb_response

if __name__=='__main__':
    server.run()
