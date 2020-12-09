import os
import numpy as np
import joblib
from Fuzzy_clustering.version3.project_manager.PredictModelManager.Sklearn_combine_predict import sklearn_model_predict

class CombineModelPredict(object):
    def __init__(self, static_data):
        self.static_data = static_data
        self.istrained = False
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model_type = self.static_data['type']
        self.combine_methods = self.static_data['combine_methods']
        methods = [method for method in self.static_data['project_methods'].keys() if
                        self.static_data['project_methods'][method] == True]



        try:
            self.load(self.model_dir)
        except:
            pass
        self.methods = []
        for method in methods:
            if method == 'RBF_ALL_CNN':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN', 'RBF-CNN'])
            elif method == 'RBF_ALL':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS', 'RBFNN'])
            else:
                self.methods.append(method)
        self.methods += self.combine_methods
        self.weight_size_full = len(self.methods)
        self.weight_size = len(self.combine_methods)
        self.rated = self.static_data['rated']
        self.n_jobs = self.static_data['sklearn']['njobs']
        self.data_dir = self.static_data['path_data']

    def bcp_predict(self, X, w):
        preds = []
        for inp in X:
            inp=inp.reshape(-1,1)
            mask=~np.isnan(inp)
            pred = np.matmul(w[mask.T]/np.sum(w[mask.T]), inp[mask])
            preds.append(pred)

        return np.array(preds)

    def predict(self, predictions):
        if self.istrained==True:
            pred_combine = dict()
            self.combine_methods = [method for method in self.combine_methods if method in predictions.keys()]
            combine_method = 'average'
            for method in self.methods:
                pred = predictions[method].mean(axis=1).values.astype('float').reshape(-1, 1)
                pred[np.where(pred < 0)] = 0
                pred_combine['average_' + method] = pred

            combine_method = 'bcp'
            for method in self.combine_methods:
                if 'bcp_'+method in self.models.keys():
                    pred = self.bcp_predict(predictions[method].values.astype('float'), self.models['bcp_'+method])
                    pred[np.where(pred < 0)] = 0
                    pred_combine['bcp_' + method] = pred

            for method in self.combine_methods:
                X_pred = predictions[method].values.astype('float')
                X_pred[np.where(np.isnan(X_pred))] = 0
                mlp_model = sklearn_model_predict(self.model_dir + '/' + method, self.rated, 'mlp', self.n_jobs)
                if mlp_model.istrained == True:
                    pred = mlp_model.predict(X_pred)
                    pred[np.where(pred < 0)] = 0
                    pred_combine['mlp_' + method] = pred
        else:
            raise ImportError('Combine overall model seems not trained')

        return pred_combine

    def load(self, pathname):
        cluster_dir = os.path.join(pathname)
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