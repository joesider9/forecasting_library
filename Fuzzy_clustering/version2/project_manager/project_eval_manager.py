import joblib
import os
import sys
import numpy as np
import pandas as pd

from Fuzzy_clustering.version2.model_manager.models_predict_manager import ModelPredictManager
from Fuzzy_clustering.version2.project_manager.projects_data_manager import ProjectsDataManager


class ProjectsEvalManager:
    def __init__(self, static_data):
        self.static_data = static_data
        self.nwp_model = static_data['NWP_model']
        self.nwp_resolution = static_data['NWP_resolution']
        self.project_owner = static_data['project_owner']
        self.projects_group = static_data['projects_group']
        self.area_group = static_data['area_group']
        self.version_group = static_data['version_group']
        self.version_model = static_data['version_model']
        self.data_variables = static_data['data_variables']
        self.methods = [method for method in static_data['project_methods'].keys() if
                        static_data['project_methods'][method] == True]
        self.path_group = self.static_data['path_group']
        self.group_static_data = joblib.load(os.path.join(self.path_group, 'static_data_projects.pickle'))

        self.model_type = self.static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_nwp_group = self.static_data['path_nwp_group']

    def evaluate(self):
        projects = self.collect_projects()
        for project in projects:
            project.evaluate_all()

    def eval_short_term(self, horizon=4, best_method = 'average'):
        projects = self.collect_projects()
        project_data_manager = ProjectsDataManager(self.static_data, is_test = None)
        if hasattr(project_data_manager, 'data_eval'):
            nwp_response = project_data_manager.nwp_extractor()
            data_eval = project_data_manager.data_eval
            if self.static_data['ts_resolution'] == '15min':
                window = np.arange(0, 60 * horizon + 0.2, 15)
            else:
                window = np.arange(0, 60 * horizon + 0.2, 60)

            for hor in window:
                for project in projects:
                    if hor == 0:
                        predictions = pd.DataFrame(data_eval.values, index=data_eval.index, columns=[hor])
                        observations = pd.DataFrame(data_eval.values, index=data_eval.index, columns=[hor])
                        joblib.dump(predictions,
                                    os.path.join(project.static_data['path_data'], 'predictions_short_term.pickle'))
                        joblib.dump(observations,
                                    os.path.join(project.static_data['path_data'], 'observations_short_term.pickle'))
                    else:
                        pred, y = project.evaluate_short_term(best_method)
                        pred.index = pred.index - pd.DateOffset(minutes=hor)
                        y.index = y.index - pd.DateOffset(minutes=hor)
                        predictions[hor] = np.nan
                        predictions[hor].loc[pred.index] = pred.values.ravel()
                        observations[hor] = np.nan
                        observations[hor].loc[y.index] = y.values.ravel()
                        joblib.dump(predictions,
                                    os.path.join(project.static_data['path_data'], 'predictions_short_term.pickle'))
                        joblib.dump(observations,
                                    os.path.join(project.static_data['path_data'], 'observations_short_term.pickle'))
                        result = pd.DataFrame(index=[project.static_data['_id']],
                                              columns=window)
                        if project.static_data['rated'] is None:
                            rated = observations
                        else:
                            rated = project.static_data['rated']
                        err = np.abs(predictions - observations) / rated
                        mae = np.mean(err, axis=0)
                        print(mae)
                _ = project_data_manager.create_short_term_datasets(data_eval)
            for project in projects:
                predictions = joblib.load(os.path.join(project.static_data['path_data'], 'predictions_short_term.pickle'))
                observations = joblib.load(os.path.join(project.static_data['path_data'], 'observations_short_term.pickle'))
                result = pd.DataFrame(index=[project.static_data['_id']],
                                      columns=window)
                if project.static_data['rated'] is None:
                    rated = observations
                else:
                    rated = project.static_data['rated']
                err = np.abs(predictions - observations) / rated
                mae = np.mean(err, axis=0)
                result.loc[project.static_data['_id']] = mae

                result.to_csv(os.path.join(project.static_data['path_data'], 'result_short_term.csv'))


    def collect_projects(self):
        projects = []
        for project in self.group_static_data:
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelPredictManager(project['static_data'])
                if project_model.istrained == True:
                    projects.append(project_model)
                else:
                    raise ValueError('Project is not trained ', project['_id'])

        return projects
