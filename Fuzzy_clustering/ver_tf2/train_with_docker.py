import os
import joblib
from Fuzzy_clustering.ver_tf2.Models_train_manager import ModelTrainManager


class ModelTrain():

    def __init__(self, project_owner, projects_group, project_name, version, model_type, njobs, gpus):
        sys_folder = 'D:/models/'
        self.model_type = model_type
        version1 = 1
        path_group = sys_folder + project_owner + '/' + projects_group + '_ver' + str(version1)+ '/' + model_type
        path_project = path_group + '/' + project_name
        self.path_model = path_project + '/model_ver' + str(version)
        path_data = self.path_model + '/DATA'
        path_fuzzy_models = self.path_model + '/fuzzy_models'
        path_nwp_group = sys_folder + project_owner + '/' + projects_group + '_ver' + str(
            version) + '/nwp'
        temp = {
                'path_project': path_project,
                'path_model': self.path_model,
                'path_data': path_data,
                'pathnwp': path_nwp_group,
                'path_fuzzy_models': path_fuzzy_models,
                'njobs':njobs,
                }
        static_data = joblib.load(os.path.join(self.path_model, 'static_data.pickle'))
        static_data['RBF']['njobs'] = njobs
        static_data['RBF']['gpus'] = gpus
        static_data['CNN']['njobs'] = njobs
        static_data['CNN']['gpus'] = gpus

        static_data.update(temp)
        self.project = {'_id' : project_name, 'static_data' : static_data}

    def fit(self):
        project_model = ModelTrainManager(self.project['static_data']['path_model'])
        # if project_model.istrained == False:
        data_variables = self.project['static_data']['data_variables']
        project_model.init(self.project['static_data'], data_variables)
        if self.model_type in {'wind', 'pv'}:
            if os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_X.csv')) \
                    and os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_y.csv')) \
                    and os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_cnn.pickle')):
                if self.project['static_data']['transfer_learning'] == False:
                    project_model.train()
                # else:
                #     project_model.train_TL(project['static_data']['tl_project']['static_data']['path_model'])
            else:
                raise ValueError('Cannot find project ', self.project['_id'], ' datasets')

        elif self.model_type in {'load'}:
            if os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_X.csv')) \
                    and os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_y.csv')) \
                    and os.path.exists(
                os.path.join(self.project['static_data']['path_data'], 'dataset_lstm.pickle')):
                project_model.train()
        elif self.model_type in {'fa'}:
            if os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_X.csv')) \
                    and os.path.exists(os.path.join(self.project['static_data']['path_data'], 'dataset_y.csv')) \
                    and os.path.exists(
                os.path.join(self.project['static_data']['path_data'], 'dataset_lstm.pickle')):
                if self.project['static_data']['transfer_learning'] == False:
                    project_model.train()
        else:
            raise ValueError('Cannot recognize model type')