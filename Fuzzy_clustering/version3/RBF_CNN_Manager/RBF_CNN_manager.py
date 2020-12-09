import joblib
from Fuzzy_clustering.version3.RBF_CNN_Manager.CNN_tf_core import CNN
from Fuzzy_clustering.version3.RBF_CNN_Manager.RBF_CNN_model import RBF_CNN_model
import pika, uuid, time, json, os
import numpy as np
from rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.RBF_CNN_Manager.Cluster_object import cluster_object

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='RBF_CNN_manager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class rbf_cnn_manager():
    def __init__(self, static_data, cluster, method, params):
        self.params = params
        self.test = params['test']
        self.method = str.lower(method)
        self.cluster = cluster
        self.istrained = False
        self.model_dir = os.path.join(cluster.cluster_dir, 'RBF_CNN')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.test_dir = self.model_dir
        try:
            self.load()
        except:
            pass
        if not self.istrained:
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
            try:
                self.load()
            except:
                if not os.path.exists(self.test_dir):
                    os.makedirs(self.test_dir)
                pass

        self.static_data = static_data
        self.cluster_name = cluster.cluster_name
        self.rated = static_data['rated']
        self.data_dir = cluster.data_dir
        self.probabilistic = False



    def fit(self):
        if self.istrained == False:
            return self.optimize_rbf_cnn()
        else:
            return self.acc

    def fit_TL(self):
        if self.istrained == False:
            return self.optimize_rbf_cnn_TL()
        else:
            return self.acc

    def load_data(self):
        if os.path.exists(os.path.join(self.data_dir, 'dataset_X.csv')):
            cvs = joblib.load(os.path.join(self.data_dir, 'cvs.pickle'))
        else:
            cvs = np.array([])
        return cvs

    def load_rbf_models(self):
        model_rbfs = RBF_CNN_model(self.static_data, self.cluster, cnn=False)
        rbf_models = [model_rbfs.model_rbf_ols.models, model_rbfs.model_rbf_ga.models, model_rbfs.model_rbfnn.model]
        return rbf_models

    def optimize_rbf_cnn(self):
        self.trial = self.params['trial']
        self.pool_size = self.params['pool_size']
        self.kernels = self.params['kernels']
        self.lr = self.params['lr']
        self.hsize = self.params['h_size']
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = self.static_data['CNN']['filters']
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.static_data_cnn = self.static_data['CNN']
        self.static_data_rbf = self.static_data['RBF']

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        self.rbf_models = self.load_rbf_models()

        cnn = CNN(self.static_data, self.rated, self.rbf_models, X_train, y_train, X_val, y_val, X_test, y_test, self.pool_size, self.trial)

        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                              learning_rate=self.lr, kernels=self.kernels,
                                                              h_size=self.hsize, filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
            pass

        if not flag:
            self.acc = np.inf
            self.scale_cnn = None
            self.model = None
        self.istrained=True
        self.save()
        return self.acc

    def load(self):
        if os.path.exists(os.path.join(self.test_dir, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.test_dir, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'temp_dir', 'cluster_cnn_dir', 'cluster_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(self.test_dir, self.method + '.pickle'), compress=9)



    def optimize_rbf_cnn_TL(self):
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_TL_dir = os.path.join(cluster_dir_tl, 'RBF_CNN')
        model_TL = joblib.load(os.path.join(model_TL_dir, self.method + '.pickle'))

        self.trial = model_TL['trial']
        self.pool_size = model_TL['pool_size']
        self.kernels = model_TL['kernels']
        self.lr = model_TL['lr']
        self.hsize = model_TL['h_size']
        cnn_max_iterations = self.static_data['CNN']['max_iterations']
        self.filters = model_TL['filters']
        cvs = self.load_data()
        self.N = cvs[0][0].shape[1]
        self.D = cvs[0][0].shape[0] + cvs[0][2].shape[0] + cvs[0][4].shape[0]
        self.static_data_cnn = self.static_data['CNN']
        self.static_data_rbf = self.static_data['RBF']

        X_train = cvs[0][0]
        y_train = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)

        self.rbf_models = self.load_rbf_models()

        cnn = CNN(self.static_data, self.rated, self.rbf_models, X_train, y_train, X_val, y_val, X_test, y_test, self.pool_size, self.trial)

        flag = False
        for _ in range(3):
            try:
                self.acc, self.scale_cnn, self.model = cnn.train_cnn(max_iterations=cnn_max_iterations,
                                                              learning_rate=self.lr, kernels=self.kernels,
                                                              h_size=self.hsize, filters=self.filters)
                flag = True
                break
            except:
                self.filters = int(self.filters / 2)
            pass

        if not flag:
            self.acc = np.inf
            self.scale_cnn = None
            self.model = None
        self.istrained=True
        self.save()
        return self.acc


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
def deep_manager(static_data):
    print(" [.] Receive cluster %s)" % static_data['cluster_name'])
    cluster = cluster_object(static_data, static_data['cluster_name'])
    model_method = static_data['method']
    params = static_data['params']
    model_3d = rbf_cnn_manager(static_data, cluster, model_method, params=params)
    if model_3d.istrained == False:
        response = {'result': model_3d.fit(), 'cluster_name': cluster.cluster_name, 'project': static_data['_id'],
                    'test': params['test'], 'method': model_method}
    else:
        response = {'result': model_3d.acc, 'cluster_name': cluster.cluster_name, 'project': static_data['_id'],
                    'test': params['test'], 'method': model_method}
    return response

if __name__=='__main__':
    server.run()
