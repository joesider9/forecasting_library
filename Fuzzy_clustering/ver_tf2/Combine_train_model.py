import os
import numpy as np
import pandas as pd
import joblib, logging, tqdm
from Fuzzy_clustering.ver_tf2.LSTM_module_3d import lstm_3d_model
from sklearn.model_selection import train_test_split
from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import sklearn_model

class Combine_train(object):
    def __init__(self, static_data):
        self.istrained = False
        self.model_dir = os.path.join(static_data['path_model'], 'Combine_module')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass
        self.static_data = static_data
        self.model_type = static_data['type']
        self.combine_methods = static_data['combine_methods']
        self.methods = []
        for method in static_data['project_methods'].keys():
            if self.static_data['project_methods'][method]['Global'] == True and static_data['project_methods'][method]['status'] == 'train':
                if method == 'ML_RBF_ALL_CNN':
                    self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
                elif method == 'ML_RBF_ALL':
                    self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
                else:
                    self.methods.append(method)
        self.methods += self.combine_methods
        self.weight_size_full = len(self.methods)
        self.weight_size = len(self.combine_methods)
        self.rated = static_data['rated']
        self.n_jobs = 2 * static_data['njobs']

        self.data_dir = self.static_data['path_data']

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_combine_model.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger


    def bcp_fit(self, X, y):
        sigma=np.nanstd((y-X).astype(float),axis=0).reshape(-1,1)
        err = []
        preds = []
        w = np.ones([1, X.shape[1]]) / X.shape[1]
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp=inp.reshape(-1,1)
            mask=~np.isnan(inp)
            pred = np.matmul(w[mask.T]/np.sum(w[mask.T]), inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p=np.exp(-1*np.square((targ-inp[mask].T)/(np.sqrt(2*np.pi)*sigma[mask])))
            w[mask.T]=((w[mask.T]*p)/np.sum(w[mask.T]*p))

            count+=1
        return w

    def lstm_fit(self, X, y, full=False):
        if full:
            cluster_dir = os.path.join(self.model_dir, 'LSTM_best')
        else:
            cluster_dir = os.path.join(self.model_dir, 'LSTM_combine')

        lstm_model = lstm_3d_model(self.static_data, self.rated, cluster_dir)
        if lstm_model.istrained==False:
            model = lstm_model.train_lstm(X, y)
        else:
            model = lstm_model.to_dict()

        return model

    def train(self,lstm=False):
        if len(self.combine_methods)>1:
            if os.path.exists(os.path.join(self.data_dir, 'predictions_by_method.pickle')):
                pred_cluster = joblib.load(os.path.join(self.data_dir, 'predictions_by_cluster.pickle'))
                predictions = joblib.load(os.path.join(self.data_dir, 'predictions_by_method.pickle'))
                y = pd.read_csv(os.path.join(self.data_dir, 'target_test.csv'), index_col=0, header=[0],
                                     parse_dates=True, dayfirst=True)

                self.models = dict()
                if lstm:
                    X = np.array([])
                    combine_method = 'lstm_full'

                    for clust in pred_cluster.keys():
                        x = np.array([])
                        for method in pred_cluster[clust]:
                            if method in self.methods:
                                tmp = np.zeros_like(y.values.reshape(-1, 1))
                                try:
                                    tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method]
                                except:
                                    tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method].reshape(-1, 1)
                                if x.shape[0] == 0:
                                    x = tmp
                                else:
                                    x = np.hstack((x, tmp))
                        if X.shape[0] == 0:
                            X = np.copy(x)
                        elif len(X.shape) == 2:
                            X = np.stack((X, x))
                        else:
                            X = np.vstack((X, x[np.newaxis, :, :]))
                    X = np.transpose(X, [1, 0, 2]).astype('float')
                    y_pred = y.values / 20
                    self.models[combine_method] = self.lstm_fit(X, y_pred, full=True)

                    X = np.array([])
                    combine_method = 'lstm_combine'

                    for clust in pred_cluster.keys():
                        x = np.array([])
                        for method in pred_cluster[clust]:
                            if method in self.combine_methods:
                                tmp = np.zeros_like(y.values.reshape(-1, 1))
                                try:
                                    tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method]
                                except:
                                    tmp[pred_cluster[clust]['index']] = pred_cluster[clust][method].reshape(-1, 1)
                                if x.shape[0] == 0:
                                    x = tmp
                                else:
                                    x = np.hstack((x, tmp))
                        if X.shape[0] == 0:
                            X = np.copy(x)
                        elif len(X.shape) == 2:
                            X = np.stack((X, x))
                        else:
                            X = np.vstack((X, x[np.newaxis, :, :]))
                    X = np.transpose(X, [1, 0, 2]).astype('float')
                    y_pred = y.values / 20
                    self.models[combine_method] = self.lstm_fit(X, y_pred)

                for method in self.combine_methods:
                    pred = predictions[method].values.astype('float')
                    pred[np.where(np.isnan(pred))] = 0
                    pred /= 20
                    y_pred = y.values/20
                    cvs = []
                    for _ in range(3):
                        X_train, X_test1, y_train, y_test1 = train_test_split(pred, y_pred, test_size=0.15)
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
                        cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                    mlp_model = sklearn_model(self.model_dir + '/'+method, self.rated, 'mlp', self.n_jobs)
                    if mlp_model.istrained == False:
                        self.models['mlp_'+method] = mlp_model.train(cvs)
                    else:
                        self.models['mlp_' + method] = mlp_model.to_dict()
                combine_method = 'bcp'
                for method in self.combine_methods:
                    self.models['bcp_'+method] = self.bcp_fit(predictions[method].values.astype('float'), y.values)


            else:
                raise ValueError('Prediction of regressors missing')
        else:
            self.combine_methods = ['average']
        self.istrained = True
        self.save(self.model_dir)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'temp_dir', 'cluster_lstm_dir', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        return dict

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

if __name__ == '__main__':
    from util_database import write_database
    from Fuzzy_clustering.ver_tf2.Projects_train_manager import ProjectsTrainManager
    from Fuzzy_clustering.ver_tf2.Models_train_manager import ModelTrainManager

    static_data = write_database()
    project_manager = ProjectsTrainManager(static_data)
    project_manager.initialize()
    project_manager.create_datasets()
    project_manager.create_projects_relations()
    project = [pr for pr in project_manager.group_static_data if pr['_id'] == 'Lach'][0]
    static_data = project['static_data']

    model = ModelTrainManager(static_data['path_model'])
    model.init(project['static_data'], project_manager.data_variables)

    combine_model = Combine_train(model.static_data)

    combine_model.train()