import pika, uuid, time, json, os
import joblib
import numpy as np
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.SKlearnModels.Cluster_object import cluster_object

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='SKlearnmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)


class SKLearn_Manager(object):
    def __init__(self, static_data, cluster, method, optimizer):
        self.static_data = static_data
        self.path_group = self.static_data['path_group']
        self.optimize_method = optimizer
        self.method = method
        self.istrained = False
        self.njobs = static_data['sklearn']['njobs']
        self.rated = static_data['rated']
        self.models = dict()
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.sk_models_dir = os.path.join(self.cluster_dir, 'SKLearn')
        self.model_dir = os.path.join(self.sk_models_dir, str.upper(method))
        if not os.path.exists(self.sk_models_dir):
            os.makedirs(self.sk_models_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load()
        except:
            pass

    def load_data(self):
        data_path = self.data_dir
        cvs = joblib.load(os.path.join(data_path, 'cvs.pickle'))
        return cvs

    def fit(self):
        if self.optimize_method == 'deap':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_deap import sklearn_model
        elif self.optimize_method == 'optuna':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_optuna import sklearn_model
        else:
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_grid import sklearn_model

        if self.istrained == False:
            cvs = self.load_data()
            model_sklearn = sklearn_model(self.static_data, self.sk_models_dir, self.rated, self.method, self.njobs,
                                          path_group=self.path_group)
            if model_sklearn.istrained == False:
                print('Train ', self.method, ' ', self.cluster_name)
                self.models[self.method] = model_sklearn.train(cvs)
            else:
                self.models[self.method] = model_sklearn.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def fit_TL(self):
        if self.optimize_method == 'deap':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_deap import sklearn_model
        elif self.optimize_method == 'optuna':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_optuna import sklearn_model
        else:
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_grid import sklearn_model
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_sklearn_TL = sklearn_model(static_data_tl, cluster_dir_tl, static_data_tl['rated'], self.method,
                                         self.njobs)
        if self.istrained == False:
            cvs = self.load_data()
            model_sklearn = sklearn_model(self.static_data, self.sk_models_dir, self.rated, self.method, self.njobs)
            if model_sklearn.istrained == False:
                self.models[self.method] = model_sklearn.train_TL(cvs, model_sklearn_TL.best_params)
            else:
                self.models[self.method] = model_sklearn.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'SKlearnManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'SKlearnManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open SKlearn model')
        else:
            raise ImportError('Cannot find SKlearn model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'cluster_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.model_dir, 'SKlearnManager.pickle'))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
<<<<<<< HEAD:forecast_library/Fuzzy_clustering/version3/SKlearnModels/SKlearnManager.py
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
def SKlearnmanager(static_data):
    print(" [.] Receive cluster %s)" % static_data['cluster_name'])
    cluster = cluster_object(static_data, static_data['cluster_name'])
    method_sk = static_data['method']
    optimizer = static_data['optimize_method']
    sk_model = SKLearn_Manager(static_data, cluster, method_sk, optimizer)
    if sk_model.istrained == False:
        sk_response = {'result': sk_model.fit(), 'cluster_name': cluster.cluster_name, 'project': static_data['_id']}
    else:
        sk_response = {'result': 'Done', 'cluster_name': cluster.cluster_name, 'project': static_data['_id']}
    return sk_response

if __name__=='__main__':
    server.run()
