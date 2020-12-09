import joblib, os
import pandas as pd
import numpy as np
from Fuzzy_clustering.version3.project_manager.PredictModelManager.FullClusterPredictManager import FullClusterPredictManager
from Fuzzy_clustering.version3.project_manager.PredictModelManager.FullModelPredictManager import FullModelPredictManager
from Fuzzy_clustering.version3.project_manager.Proba_Model_manager import proba_model_manager


class ProbaDataManager(object):

    def __init__(self, static_data):
        self.path_model = static_data['path_model']
        self.static_data = static_data

    def prepare_data(self):
        clusters_predict_manager = FullClusterPredictManager(self.path_model, self.static_data)
        pred_cluster, predictions_cluster, y_all, y, index, index_all = clusters_predict_manager.predict_clusters(test = False)
        model_predict_manager = FullModelPredictManager(self.path_model, self.static_data)
        predictions_final_temp = model_predict_manager.predict_model(pred_cluster, predictions_cluster, scale=False)
        predictions_final = dict()
        for method, pred in predictions_final_temp.items():
            pred_temp = pd.DataFrame(0, index=index_all, columns=[method])
            pred_temp.loc[index, method] = pred
            predictions_final[method] = pred_temp
        proba_model = proba_model_manager(self.static_data)
        if not proba_model.istrained:
            from sklearn.model_selection import train_test_split
            scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
            X_pred = np.array([])
            for method, pred in predictions_final.items():
                if X_pred.shape[0] == 0:
                    X_pred = scale_y.transform(predictions_final[method].reshape(-1, 1))
                else:
                    X_pred = np.hstack((X_pred, scale_y.transform(predictions_final[method].reshape(-1, 1))))
            X_pred[np.where(X_pred < 0)] = 0

            cvs = []
            for _ in range(3):
                X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pred, y_all, test_size=0.15)
                X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.15)
                cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])

            joblib.dump(X_pred, os.path.join(self.static_data['path_data'], 'cvs_proba.pickle'))
