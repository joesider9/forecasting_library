import joblib
import os
import sys

from Fuzzy_clustering.version2.model_manager.models_predict_manager import ModelPredictManager


class ProjectsPredictManager():
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
        self.model_type = self.static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_group = self.static_data['path_group']
        self.path_nwp_group = self.static_data['path_nwp_group']
        self.methods = [method for method in static_data['project_methods'].keys() if
                        static_data['project_methods'][method] == True]
        self.group_static_data = joblib.load(os.path.join(self.path_group, 'static_data_projects.pickle'))

    def predict_offline(self):
        projects = self.collect_projects()
        predictions = dict()
        for project in projects:
            predictions[project.static_data['_id']] = project.predict_offline()
        return predictions

    def collect_projects(self):
        projects = []
        for project in self.group_static_data:
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelPredictManager(project['static_data'])
                if project_model.istrained == True:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle')):
                        projects.append(project_model)
                    else:
                        raise ValueError('Cannot find project ', project['_id'], ' datasets')

        return projects
