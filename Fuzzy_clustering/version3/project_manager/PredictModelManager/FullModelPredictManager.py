import numpy as np
import pandas as pd
import joblib, os, pickle
from Fuzzy_clustering.version3.project_manager.PredictModelManager.CombineModelPredict import CombineModelPredict

class FullModelPredictManager(object):

    def __init__(self, path_model, static_data):
        self.path_model = path_model
        self.static_data = static_data
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']
        try:
            self.load()
        except:
            pass

    def predict_model(self, pred_cluster, predictions, scale = True):
        combine_overall = CombineModelPredict(self.static_data)
        predictions_final = combine_overall.predict(predictions)
        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

        for method, pred in predictions_final.items():
            if scale:
                pred = scale_y.inverse_transform(pred.reshape(-1, 1))
            else:
                pred = pred.reshape(-1, 1)
            pred[np.where(pred<0)] = 0
            predictions_final[method] = pred

        return predictions_final

    def load(self):
        if os.path.exists(os.path.join(self.path_model, 'manager' + '.pickle')):
            try:
                f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                if 'path_model' in tmp_dict.keys():
                    del tmp_dict['path_model']
                self.__dict__.update(tmp_dict)
            except:
                raise ValueError('Cannot find model for %s', self.path_model)
        else:
            raise ValueError('Cannot find model for %s', self.path_model)