import os
from collections import defaultdict

from Fuzzy_clustering.version2.common_utils.logging import create_logger
from Fuzzy_clustering.version2.dataset_manager.create_dataset_for_load_lv import dataset_creator_LV
from Fuzzy_clustering.version2.dataset_manager.create_dataset_for_load import dataset_creator_load
from Fuzzy_clustering.version2.dataset_manager.create_dataset_for_load_scada import dataset_creator_scada
from Fuzzy_clustering.version2.dataset_manager.create_datasets_dense import DatasetCreatorDense
from Fuzzy_clustering.version2.dataset_manager.create_datasets_point import DatasetCreatorPoint
from Fuzzy_clustering.version2.dataset_manager.create_datasets_for_fa import dataset_creator_ecmwf
from Fuzzy_clustering.version2.dataset_manager.create_datasets_for_fa import dataset_creator_xmachina
from Fuzzy_clustering.version2.dataset_manager.create_datasets_pca import DatasetCreatorPCA


class DatasetCreator:
    def __init__(self, static_data, group_static_data, test):
        self.test = test
        self.group_static_data = group_static_data
        self.static_data = static_data
        self.file_data = static_data['data_file_name']
        self.project_owner = static_data['project_owner']
        self.projects_group = static_data['projects_group']
        self.area_group = static_data['area_group']
        self.version_group = static_data['version_group']
        self.version_model = static_data['version_model']
        self.weather_in_data = static_data['weather_in_data']
        self.nwp_model = static_data['NWP_model']
        self.nwp_resolution = static_data['NWP_resolution']
        self.data_variables = static_data['data_variables']

        self.model_type = static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_group = self.static_data['path_group']
        self.path_nwp_group = self.static_data['path_nwp_group']

        self.logger = create_logger(logger_name=f'DataManager_{self.model_type}', abs_path=self.path_group,
                                    logger_path='log_data_manager.log')

        # if self.test is True creates the files with suffix _test that are used in evaluation procedure
        # else if self.test is False creates the files that are used in training procedure
        if self.test is not None:
            self.dataset_x = 'dataset_X_test.csv' if self.test else 'dataset_X.csv'
            self.dataset_y = 'dataset_y_test.csv' if self.test else 'dataset_y.csv'
            self.dataset_lstm = 'dataset_lstm_test.pickle' if self.test else 'dataset_lstm.pickle'
            self.dataset_cnn = 'dataset_cnn_test.pickle' if self.test else 'dataset_cnn.pickle'
        else:
            self.dataset_x = 'dataset_X_test.csv'
            self.dataset_y = 'dataset_y_test.csv'
            self.dataset_lstm = 'dataset_lstm_test.pickle'
            self.dataset_cnn = 'dataset_cnn_test.pickle'

    def create_datasets(self, data):
        """

        This function calls the data creators (PCA or dense) of every problem (wind, pv, load etc). Removes old files if
        'recreate_datasets' parameter in static_data is True in order to create new files

        :param data:
        pandas dataframe each column might be a different project for the cases wind and pv or for the cases of
        load and Gas the first column is the project and the following columns are explanatory variables

        :return:
        'Done' if the files are created successfully

        """
        if (self.static_data['recreate_datasets']) or (self.test is None):
            for project in self.group_static_data:
                path_prefix = project['static_data']['path_data']
                dataset_x_path = os.path.join(path_prefix, self.dataset_x)
                if os.path.exists(dataset_x_path):
                    os.remove(dataset_x_path)

                dataset_y_path = os.path.join(path_prefix, self.dataset_y)
                if os.path.exists(dataset_y_path):
                    os.remove(dataset_y_path)

                dataset_cnn_path = os.path.join(path_prefix, self.dataset_cnn)
                if os.path.exists(dataset_cnn_path):
                    os.remove(dataset_cnn_path)

                dataset_lstm_path = os.path.join(path_prefix, self.dataset_lstm)
                if os.path.exists(dataset_lstm_path):
                    os.remove(dataset_lstm_path)

        project_info = defaultdict(list)
        for project in self.group_static_data:
            path_prefix = project['static_data']['path_data']

            # FIXME: With that implementation you can't have a dataset with PCA and Dense at the same time.
            if not (os.path.exists(os.path.join(path_prefix, self.dataset_x)) or
                    os.path.exists(os.path.join(path_prefix, self.dataset_y))):
                # TODO: Cleanse or filter wrong/missing values
                project_info['projects'].append(project)
                project_info['path_prefixes'].append(path_prefix)
                project_info['project_cols'].append(project['_id'])
        print(project_info)
        if len(project_info['projects']) > 0:
            if self.model_type in {'pv', 'wind'}:
                self.load_pv_wind(data, project_info)
            elif self.model_type == 'load':
                self.load_energy(data, project_info)
            elif self.model_type == 'fa':
                self.load_gas(data, project_info)
            else:
                raise ValueError(f"Cannot recognize model type {self.model_type}")
        return 'Done'

    def load_pv_wind(self, data, project_info):

        if self.static_data['compress_data'] == 'PCA':
            print()

            for project, path_prefix in zip(project_info['projects'], project_info['path_prefixes']):
                if self.static_data['recreate_datasets']:  # FIXME: Why do we delete it only under PCA?
                    os.remove(os.path.join(path_prefix, 'nwps_3d.pickle'))
                if project['_id'] != self.projects_group + '_' + self.model_type:
                    # Datasets are processed and created within the functions
                    dataset = DatasetCreatorPCA(project,
                                                data=data[project['_id']].dropna(),
                                                n_jobs=self.static_data['njobs'],
                                                test=self.test)
                    dataset.make_dataset_res()
                    self.logger.info('Dataset using PCA for testing constructed for %s', project['_id'])
                else:  # All projects under a country are executed, PCA can't process that kind of data.
                    dataset = DatasetCreatorDense(self.projects_group,
                                                  project,
                                                  data[project['_id']].dropna(),
                                                  self.path_nwp_group,
                                                  self.nwp_model,
                                                  self.nwp_resolution,
                                                  self.data_variables,
                                                  njobs=self.static_data['njobs'],
                                                  test=self.test)
                    if self.test is not None:
                        dataset.make_dataset_res()
                    else:
                        dataset.make_dataset_res_short_term()

        elif self.static_data['compress_data'] == 'dense':
            self.logger.info(f'Start creating dataset using dense compression for country {self.projects_group}')
            for project in project_info['projects']:
                self.logger.info(f"Start creating dataset using dense compression for projects {project['_id']}")
            if len(project_info['project_cols']) > 0:
                dataset = DatasetCreatorDense(self.projects_group,
                                              project_info['projects'],
                                              data[project_info['project_cols']].dropna(axis=1),
                                              self.path_nwp_group,
                                              self.nwp_model,
                                              self.nwp_resolution,
                                              self.data_variables,
                                              njobs=self.static_data['njobs'],
                                              test=self.test)

                if self.test is not None:
                    dataset.make_dataset_res()
                else:
                    dataset.make_dataset_res_short_term()
                self.logger.info(f'Dataset using dense compression for country {self.projects_group} created')
        elif self.static_data['compress_data'] == 'point':
            self.logger.info(f'Start creating dataset using dense compression for country {self.projects_group}')
            for project in project_info['projects']:
                self.logger.info(f"Start creating dataset using dense compression for projects {project['_id']}")
            if len(project_info['project_cols']) > 0:
                dataset = DatasetCreatorPoint(self.projects_group,
                                              project_info['projects'],
                                              data[project_info['project_cols']].dropna(),
                                              self.path_nwp_group,
                                              self.nwp_model,
                                              self.nwp_resolution,
                                              self.data_variables,
                                              njobs=self.static_data['njobs'],
                                              test=self.test)

                if self.test is not None:
                    dataset.make_dataset_res()
                else:
                    dataset.make_dataset_res_short_term()
                self.logger.info(f'Dataset using dense compression for country {self.projects_group} created')

        else:
            raise ValueError(
                f"Cannot recognize dimensionality reduction method {self.static_data['compress_data']}")

    def load_energy(self, data, project_info):

        if len(project_info['project_cols']) > 0:
            self.logger.info('Start creating dataset for load')
        if self.nwp_model == 'ecmwf':
            if project_info['project_cols'][0] == 'SCADA':
                dataset = dataset_creator_scada(self.projects_group, project_info['projects'], data,
                                                self.path_nwp_group,
                                                self.nwp_model, self.nwp_resolution, self.data_variables,
                                                njobs=self.static_data['njobs'], test=self.test)
                dataset.make_dataset_scada()
            elif project_info['project_cols'][0] == 'lv_load':
                dataset = dataset_creator_LV(self.projects_group, project_info['projects'], data, self.path_nwp_group,
                                             self.nwp_model, self.nwp_resolution, self.data_variables,
                                             njobs=self.static_data['njobs'], test=self.test)
                dataset.make_dataset_lv()
        elif self.nwp_model in {'gfs', 'skiron'}:
            dataset = dataset_creator_load(self.projects_group, project_info['projects'], data,
                                            self.path_nwp_group,
                                            self.nwp_model, self.nwp_resolution, self.data_variables,
                                            njobs=self.static_data['njobs'], test=self.test)
            if self.test is not None:
                dataset.make_dataset_load()
            else:
                dataset.make_dataset_load_short_term()

    def load_gas(self, data, project_info):

        if len(project_info['project_cols']) > 0:
            self.logger.info('Start creating dataset for fa')

        if self.nwp_model == 'ecmwf':
            dataset = dataset_creator_ecmwf(self.projects_group, project_info['projects'], data,
                                            self.path_nwp_group,
                                            self.nwp_model, self.nwp_resolution,
                                            self.data_variables,
                                            njobs=self.static_data['njobs'], test=self.test)
            dataset.make_dataset_ecmwf()
        elif self.nwp_model == 'xmachina':
            dataset = dataset_creator_xmachina(self.projects_group, project_info['projects'], data,
                                               self.path_nwp_group,
                                               self.nwp_model, self.nwp_resolution,
                                               self.data_variables,
                                               njobs=self.static_data['njobs'], test=self.test)
            if self.version_model == 0:
                dataset.make_dataset_xmachina_curr()
            else:
                dataset.make_dataset_xmachina_dayahead()
