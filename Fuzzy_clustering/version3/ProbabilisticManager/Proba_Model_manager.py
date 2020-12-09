import numpy as np
import joblib
from rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.ProbabilisticManager.MLP_tf_core import MLP
import pika, uuid, time, json, os


RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='Probamanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class proba_model_manager():
    def __init__(self, static_data, params={}):
        if len(params)>0:
            self.params = params
            self.test = params['test']
            self.test_dir = os.path.join(self.model_dir, 'test_' + str(self.test))
        self.istrained = False
        self.method = 'mlp'
        self.model_dir = os.path.join(static_data['path_model'], 'Probabilistic')
        self.data_dir = self.static_data['path_data']

        if hasattr(self, 'test'):
            try:
                self.load(self.test_dir)
            except:
                pass
        else:
            try:
                self.load(self.model_dir)
            except:
                pass

        self.static_data = static_data
        self.cluster_name = static_data['_id']
        self.rated = static_data['rated']
        self.probabilistic = True
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def fit(self):
        if self.istrained == False:

            return self.optimize_mlp()

        else:
            return self.acc

    def load_data(self):
        if os.path.exists(os.path.join(self.data_dir, 'cvs_proba.pickle')):
            cvs = joblib.load(os.path.join(self.data_dir, 'cvs_proba.pickle'))
        else:
            raise ImportError('Predictions for probabilistic not found ')

        return cvs


    def optimize_mlp(self):
        self.trial = self.params['trial']
        self.units = self.params['units']
        self.act_func = self.params['act_func']
        self.lr = self.params['lr']
        lstm_max_iterations = self.static_data['MLP']['max_iterations']
        self.hold_prob = self.static_data['MLP']['hold_prob']
        cvs = self.load_data()
        mlp = MLP(self.static_data, self.rated, cvs[0], cvs[1], cvs[2], cvs[3], cvs[4], cvs[5], trial=self.trial,
                       probabilistc=self.probabilistic)
        # try:
        self.acc, self.model = mlp.train(max_iterations=lstm_max_iterations,
                                                          learning_rate=self.lr, units=self.units,
                                                          hold_prob=self.hold_prob, act_func=self.act_func)
        # except:
        #     acc_old_lstm=np.inf
        #     scale_lstm=None
        #     model_lstm=None
        #     pass
        self.istrained=True
        self.save()
        return self.acc

    def load(self, path):
        if os.path.exists(os.path.join(path, self.method + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(path, self.method + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data_all', 'static_data', 'temp_dir', 'model_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(self.test_dir, self.method + '.pickle'), compress=9)



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
def proba_manager(static_data):
    model_method = static_data['method']
    params = static_data['params']
    model_proba = proba_model_manager(static_data, params=params)
    if model_proba.istrained == False:
        response = {'result': model_proba.fit(), 'project': static_data['_id'],
                    'test': params['test'], 'method': model_method}
    else:
        response = {'result': model_proba.acc, 'project': static_data['_id'],
                    'test': params['test'], 'method': model_method}
    return response

if __name__=='__main__':
    server.run()
