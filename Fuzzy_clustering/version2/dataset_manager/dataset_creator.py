import logging
import os
import sys

from Fuzzy_clustering.version2.dataset_manager.create_dataset_for_load_lv import dataset_creator_LV
from Fuzzy_clustering.version2.dataset_manager.create_dataset_for_load_scada import dataset_creator_scada
from Fuzzy_clustering.version2.dataset_manager.create_datasets_dense import dataset_creator_dense
from Fuzzy_clustering.version2.dataset_manager.create_datasets_for_fa import dataset_creator_ecmwf
from Fuzzy_clustering.version2.dataset_manager.create_datasets_for_fa import dataset_creator_xmachina
from Fuzzy_clustering.version2.dataset_manager.create_datasets_pca import DatasetCreatorPCA


class DatasetCreator:
    '''
    Interface of data creators. This file calls the data creators (PCA or dense) of every problem (wind, pv, load etc)

    Args:
         static_data: dict of all info that come from config.py
         group_static_data: list of the projects

    '''
    def __init__(self, static_data, group_static_data):
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

        self.model_type = self.static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_group = self.static_data['path_group']
        self.path_nwp_group = self.static_data['path_nwp_group']
        self.create_logger()

    def create_logger(self):
        self.logger = logging.getLogger('DataManager_' + self.model_type)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_group, 'log_data_manager.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def create_datasets(self, data, test):
        '''
        This function calls the data creators (PCA or dense) of every problem (wind, pv, load etc). Removes old files if
        'recreate_datasets' parameter in static_data is True in order to create new files

        :param data:
        pandas dataframe each column might be a different project for the cases wind and pv or for the cases of
        load and Gas the first column is the project and the following columns are explanatory variables

        :param test: Boulean
        if True creates the files with suffix _test that are used in evaluation procedure
        if False creates the files that are used in training procedure

        :return:
        'Done' if the files are created successfully
        '''
        if self.static_data['recreate_datasets']:
            self.logger.info('Previous dataset will be removed')
            if test:
                for project in self.group_static_data:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn_test.pickle')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_cnn_test.pickle'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle'))
            else:
                for project in self.group_static_data:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_X.csv'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle'))
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle')):
                        os.remove(os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle'))

        if self.model_type in ('pv', 'wind'):
            if self.static_data['compress_data'] == 'PCA':
                for project in self.group_static_data:  # Create nwp/power_output for each farm (project)
                    if self.static_data['recreate_datasets']:
                        os.remove(os.path.join(project['static_data']['path_data'], 'nwps_3d.pickle'))
                    if project['_id'] != self.projects_group + '_' + self.model_type:
                        if data[project['_id']].dropna().count() > 24:
                            if test:
                                if not os.path.exists(
                                        os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                                        or not os.path.exists(
                                    os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                                    self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                    dataset = DatasetCreatorPCA(project, data=data[project['_id']].dropna(),
                                                                n_jobs=self.static_data['njobs'], test=test)
                                    dataset.make_dataset_res()
                                    self.logger.info('Dataset using PCA for testing constructed for %s', project['_id'])
                            else:
                                if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                        or not os.path.exists(
                                    os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                                    self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                    dataset = DatasetCreatorPCA(project, data=data[project['_id']].dropna(),
                                                                n_jobs=self.static_data['njobs'])
                                    dataset.make_dataset_res()
                                    self.logger.info('Dataset for training using PCA constructed for %s', project['_id'])

                        elif project['_id'] == self.projects_group + '_' + self.model_type:
                            if test:
                                if not os.path.exists(
                                        os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                                        or not os.path.exists(
                                    os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                                    self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                    dataset = dataset_creator_dense(self.projects_group, project,
                                                                    data[project['_id']].dropna(),
                                                                    self.path_nwp_group, self.nwp_model,
                                                                    self.nwp_resolution, self.model_type,
                                                                    njobs=self.static_data['njobs'],
                                                                    test=test)
                                    dataset.make_dataset_res()
                                    self.logger.info('Dataset uisng PCA constructed for %s', project['_id'])
                            else:
                                if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                        or not os.path.exists(
                                    os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                                    self.logger.info('Start creating dataset using PCA for %s', project['_id'])
                                    dataset = dataset_creator_dense(self.projects_group, project,
                                                                    data[project['_id']].dropna(),
                                                                    self.path_nwp_group,
                                                                    self.nwp_model, self.nwp_resolution,
                                                                    self.data_variables,
                                                                    njobs=self.static_data['njobs'], test=test)
                                    dataset.make_dataset_res()
                                    self.logger.info('Dataset uisng PCA constructed for %s', project['_id'])

            elif self.static_data['compress_data'] == 'dense':
                project_col = []
                projects = []
                for project in self.group_static_data:
                    if test:
                        if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                                or not os.path.exists(
                            os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                            if data[project['_id']].dropna().count() > 24:
                                projects.append(project)
                                project_col.append(project['_id'])
                    else:
                        if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                or not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                            if data[project['_id']].dropna().count() > 24:
                                projects.append(project)
                                project_col.append(project['_id'])
                self.logger.info('Start creating dataset using dense compression for country %s', self.projects_group)
                self.logger.info('Start creating dataset using dense compression for projects %s', project_col)
                if len(project_col) > 0:
                    dataset = dataset_creator_dense(self.projects_group, projects, data[project_col].dropna(),
                                                    self.path_nwp_group,
                                                    self.nwp_model, self.nwp_resolution, self.data_variables,
                                                    njobs=self.static_data['njobs'], test=test)

                    dataset.make_dataset_res()
                    self.logger.info('Dataset using dense compression for country %s created', self.projects_group)

        elif self.model_type == 'load':
            project_col = []
            projects = []
            for project in self.group_static_data:
                if test:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                            or not os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                        projects.append(project)
                        project_col.append(project['_id'])
                else:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            or not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                        projects.append(project)
                        project_col.append(project['_id'])

            if len(project_col) > 0:
                self.logger.info('Start created dataset for load')
                if self.nwp_model == 'ecmwf':
                    if project_col[0] == 'SCADA':
                        dataset = dataset_creator_scada(self.projects_group, projects, data, self.path_nwp_group,
                                                        self.nwp_model, self.nwp_resolution, self.data_variables,
                                                        njobs=self.static_data['njobs'], test=test)
                        dataset.make_dataset_scada()
                    elif project_col[0] == 'lv_load':
                        dataset = dataset_creator_LV(self.projects_group, projects, data, self.path_nwp_group,
                                                     self.nwp_model, self.nwp_resolution, self.data_variables,
                                                     njobs=self.static_data['njobs'], test=test)
                        dataset.make_dataset_lv()

        elif self.model_type == 'fa':
            project_col = []
            projects = []
            for project in self.group_static_data:
                if test:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                            or not os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                        projects.append(project)
                        project_col.append(project['_id'])
                else:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            or not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                        projects.append(project)
                        project_col.append(project['_id'])

            if len(project_col) > 0:
                self.logger.info('Start created dataset for fa')
                if self.nwp_model == 'ecmwf':
                    dataset = dataset_creator_ecmwf(self.projects_group, projects, data, self.path_nwp_group,
                                                    self.nwp_model, self.nwp_resolution, self.data_variables,
                                                    njobs=self.static_data['njobs'], test=test)
                    dataset.make_dataset_ecmwf()
                elif self.nwp_model == 'xmachina':
                    if self.version_model == 0:
                        dataset = dataset_creator_xmachina(self.projects_group, projects, data, self.path_nwp_group,
                                                           self.nwp_model, self.nwp_resolution, self.data_variables,
                                                           njobs=self.static_data['njobs'], test=test)
                        dataset.make_dataset_xmachina_curr()
                    else:
                        dataset = dataset_creator_xmachina(self.projects_group, projects, data, self.path_nwp_group,
                                                           self.nwp_model, self.nwp_resolution, self.data_variables,
                                                           njobs=self.static_data['njobs'], test=test)
                        dataset.make_dataset_xmachina_dayahead()

        else:
            raise ValueError('Cannot recognize model_type or dimentional reduction method')
        return 'Done'
