import joblib
import numpy as np
import os
import pandas as pd

from Fuzzy_clustering.version3.project_manager.PredictModelManager.FullClusterPredictManager import \
    FullClusterPredictManager
from Fuzzy_clustering.version3.project_manager.PredictModelManager.FullModelPredictManager import \
    FullModelPredictManager


class ProjectsEvalManager():
    def __init__(self, project):
        self.project = project
        self.static_data = self.project.static_data
        self.rated = self.static_data['rated']

    def compute_metrics(self, pred, y):
        if self.rated is None:
            rated = y.ravel()
        else:
            rated = self.rated
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def evaluate(self, pred_all, y):
        result = pd.DataFrame(index=[method for method in pred_all.keys()], columns=['sse', 'rms', 'mae', 'mse'])
        for method, pred in pred_all.items():
            if isinstance(pred, pd.DataFrame):
                result.loc[method] = self.compute_metrics(pred.values, y)
            else:
                result.loc[method] = self.compute_metrics(pred, y)

        return result

    def evaluate_all(self):
        data_path = self.static_data['path_data']
        if self.project.is_trained:
            clusters_predict_manager = FullClusterPredictManager(self.project.path_model, self.project.static_data)
            pred_cluster, predictions_cluster, y_test_all, y_test, index, index_all = clusters_predict_manager.predict_clusters(
                test=True)
            model_predict_manager = FullModelPredictManager(self.project.path_model, self.project.static_data)
            predictions_final_temp = model_predict_manager.predict_model(pred_cluster, predictions_cluster, scale=True)
            predictions_final = dict()
            for method, pred in predictions_final_temp.items():
                pred_temp = pd.DataFrame(0, index=index_all, columns=[method])
                pred_temp.loc[index, method] = pred
                predictions_final[method] = pred_temp

                if y_test is not None:
                    result_all = self.evaluate(predictions_final, y_test_all.values)
                    result_all.to_csv(os.path.join(data_path, 'result_final.csv'))
                    joblib.dump(predictions_final, os.path.join(data_path, 'predictions_final.pickle'))
                    y_test.to_csv(os.path.join(data_path, 'target_test.csv'))
        else:
            raise ModuleNotFoundError('Model %s is not trained', self.static_data['_id'])
