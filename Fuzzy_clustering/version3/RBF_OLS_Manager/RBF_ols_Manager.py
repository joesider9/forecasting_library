import pika, uuid, time, json, os
from Fuzzy_clustering.version3.RBF_OLS_Manager.RBF_ols import rbf_ols_module
import joblib
import numpy as np
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.RBF_OLS_Manager.Cluster_object import cluster_object

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='RBFOLSmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class RBFOLS_Manager(object):
    def __init__(self, static_data, cluster):
        self.static_data = static_data
        self.istrained = False
        self.njobs = cluster.static_data['sklearn']['njobs']
        self.models = dict()
        self.cluster_name = cluster.cluster_name
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.model_dir = os.path.join(self.cluster_dir, 'RBF_OLS')
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
        if self.istrained == False:
            cvs = self.load_data()
            model_rbf_ols = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'], self.static_data['sklearn']['njobs'], GA=False
                                           , path_group=self.static_data['path_group'])
            model_rbf_ga = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'], self.static_data['sklearn']['njobs'], GA=True
                                          , path_group=self.static_data['path_group'])
            if model_rbf_ols.istrained == False:
                max_samples = 1500
                for _ in range(4):
                    try:
                        print('Train RBFOLS ', self.cluster_name)
                        self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf(cvs, max_samples=max_samples)
                        break
                    except:
                        print('Reduce training set')
                        max_samples -= 500
                        continue
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False:
                max_samples = 1500
                for _ in range(4):
                    try:
                        print('Train GA-RBF ', self.cluster_name)
                        self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf(cvs, max_samples=max_samples)
                        break
                    except:
                        print('Reduce training set')
                        max_samples -= 500
                        continue
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            self.istrained=True
            self.save()
        return 'Done'

    def fit_TL(self):
        if self.istrained == False:
            static_data_tl = self.static_data['tl_project']['static_data']
            cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)

            model_rbf_ols_TL = rbf_ols_module(static_data_tl, cluster_dir_tl, static_data_tl['rated'],
                                           self.static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga_TL = rbf_ols_module(static_data_tl, cluster_dir_tl, static_data_tl['rated'],
                                          self.static_data['sklearn']['njobs'], GA=True)
            cvs = self.load_data()
            model_rbf_ols = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'], self.static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'], self.static_data['sklearn']['njobs'], GA=True)
            if model_rbf_ols.istrained == False:
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf_TL(cvs, model_rbf_ols_TL.models)
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False:
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf_TL(cvs, model_rbf_ga_TL.models)
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            self.istrained=True
            self.save()
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'RBFolsManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'RBFolsManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger',  'static_data', 'model_dir', 'cluster_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.model_dir, 'RBFolsManager.pickle'))


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
def RBFOLSmanager(static_data):
    print(" [.] Receive cluster %s)" % static_data['cluster_name'])
    cluster = cluster_object(static_data, static_data['cluster_name'])
    rbf_ols_model = RBFOLS_Manager(static_data, cluster)
    if rbf_ols_model.istrained == False:
        rbf_ols_response = {'result': rbf_ols_model.fit(), 'cluster_name': cluster.cluster_name,
                            'project': static_data['_id']}
    else:
        rbf_ols_response = {'result': 'Done', 'cluster_name': cluster.cluster_name, 'project': static_data['_id']}
    return rbf_ols_response

if __name__=='__main__':
    server.run()
