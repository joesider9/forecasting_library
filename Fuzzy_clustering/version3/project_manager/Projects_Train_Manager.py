import sys
import os
import pandas as pd
import numpy as np
import logging, shutil, glob, json
import copy, joblib

from Fuzzy_clustering.version3.project_manager.ModelManager.Models_train_manager import ModelTrainManager
from Fuzzy_clustering.version3.project_manager.ModelManager.Models_predict_manager import ModelPredictManager
from Fuzzy_clustering.version3.project_manager.correlation_link_projects import ProjectLinker
from Fuzzy_clustering.version3.project_manager.rabbitmq_client import rabbit_client_data, rabbit_client_nwp
import time
# for timing
from contextlib import contextmanager
from timeit import default_timer



@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


class ProjectsTrainManager(object):

    def __init__(self,static_data):
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
        data_file_name = os.path.basename(self.file_data)
        if 'load' in data_file_name:
            self.model_type = 'load'
        elif 'pv' in data_file_name:
            self.model_type = 'pv'
        elif 'wind' in data_file_name:
            self.model_type = 'wind'
        elif 'fa' in data_file_name:
            self.model_type = 'fa'
        else:
            raise IOError('Wrong data file name. Use one of load_ts.csv, wind_ts.csv, pv_ts.csv')

        if self.static_data['Docker']:
            if sys.platform == 'linux':
                self.sys_folder = '/models/'
                if self.nwp_model == 'skiron' and self.nwp_resolution == 0.05:
                    self.path_nwp = '/nwp_grib/SKIRON'
                elif self.nwp_model == 'skiron' and self.nwp_resolution == 0.1:
                    self.path_nwp = '/nwp_grib/SKIRON_low'
                elif self.nwp_model == 'ecmwf':
                    self.path_nwp = '/nwp_grib/ECMWF'
                else:
                    self.path_nwp = None
            else:
                if self.nwp_model == 'ecmwf':
                    self.sys_folder = '/models/'
                    self.path_nwp = '/nwp_grib/ECMWF'
                else:
                    self.sys_folder = '/models/'
                    self.path_nwp = None
        else:
            if sys.platform == 'linux':
                self.sys_folder = '/media/smartrue/HHD1/George/models/'
                if self.nwp_model == 'skiron' and self.nwp_resolution==0.05:
                    self.path_nwp = '/media/smartrue/HHD2/SKIRON'
                elif self.nwp_model == 'skiron' and self.nwp_resolution==0.1:
                    self.path_nwp = '/media/smartrue/HHD2/SKIRON_low'
                elif self.nwp_model == 'ecmwf':
                    self.path_nwp = '/media/smartrue/HHD2/ECMWF'
                else:
                    self.path_nwp = None
            else:
                if self.nwp_model == 'ecmwf':
                    self.sys_folder = 'D:/models/'
                    self.path_nwp = 'D:/Dropbox/ECMWF'
                else:
                    self.sys_folder = 'D:/models/'
                    self.path_nwp = None

        self.path_group = self.sys_folder + self.project_owner + '/' + self.projects_group + '_ver' + str(self.version_group)+ '/' + self.model_type
        if not os.path.exists(self.path_group):
            os.makedirs(self.path_group)
        self.path_nwp_group = self.sys_folder + self.project_owner + '/' + self.projects_group + '_ver' + str(self.version_group) + '/nwp'
        if not os.path.exists(self.path_nwp_group):
            os.makedirs(self.path_nwp_group)
        self.create_logger()


    def create_logger(self):
        self.logger = logging.getLogger('ProjectsTrainManager_' + self.model_type)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_group, 'log_' + self.projects_group + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)


    def check_project_names(self):
        flag = True
        if self.model_type in {'wind', 'pv'}:
            for name in self.projects:
                if name not in self.coord.index.to_list() and name != self.projects_group + '_' + self.model_type and name != 'APE_net':
                    flag = False
                    self.logger.info('There is inconsistency to files data and coord for the project %s', name)
            if flag==False:
                raise ValueError('Inconcistency in project names between data and coord')

        if self.use_rated:
            for name in self.projects:
                if name not in self.rated.index.to_list() and name != self.projects_group + '_' + self.model_type:
                    flag = False
                    self.logger.info('There is inconsistency to files data and rated for the project %s', name)
            if flag==False:
                raise ValueError('Inconcistency in project names between data and rated')

        return flag

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_data, header=0, index_col=0, parse_dates=True, dayfirst=True)
        except:
            self.logger.info('Cannot import timeseries from the file %s', self.file_data)
            raise IOError('Cannot import timeseries from the file %s', self.file_data)
        self.logger.info('Timeseries imported successfully from the file %s', self.file_data)

        if 'total' in self.data.columns:
            self.data = self.data.rename(columns={'total':self.projects_group + '_' + self.model_type})
        if self.static_data['Evaluation_start'] != None:
            if self.model_type == 'fa':
                try:
                    eval_date = pd.to_datetime(self.static_data['Evaluation_start'], format='%d%m%Y %H:%M')
                    self.data_eval = self.data.iloc[np.where(self.data.index>eval_date-pd.DateOffset(days=372))]
                    self.data = self.data.iloc[np.where(self.data.index<=eval_date)]
                except:
                    raise ValueError('Wrong date format, use %d%m%Y %H:%M. Or the date does not exist in the dataset')
            else:
                try:
                    eval_date = pd.to_datetime(self.static_data['Evaluation_start'], format='%d%m%Y %H:%M')
                    self.data_eval = self.data.iloc[np.where(self.data.index > eval_date)]
                    self.data = self.data.iloc[np.where(self.data.index <= eval_date)]
                except:
                    raise ValueError(
                        'Wrong date format, use %d%m%Y %H:%M. Or the date does not exist in the dataset')
        self.projects = []
        if self.model_type == 'load':
            self.projects.append(self.data.columns[0])
        elif self.model_type == 'fa':
            self.projects.append('fa')
        else:
            for name in self.data.columns:
                if name=='total':
                    name = self.projects_group + '_' + self.model_type
                self.projects.append(name)

        if self.weather_in_data == False:
            try:
                self.coord = pd.read_csv(self.file_coord, header=None, index_col=0)

            except:
                self.logger.info('Cannot import coordinates from the file %s', self.file_coord)
                raise IOError('Cannot import coordinates from the file %s', self.file_coord)
            self.logger.info('Coordinates imported successfully from the file %s', self.file_coord)
        else:
            self.logger.info('Coordinates in the data')

        if self.use_rated:
            try:
                self.rated = pd.read_csv(self.file_rated, header=None, index_col=0)
            except:
                self.logger.info('Cannot import Rated Power from the file %s', self.file_rated)
                raise IOError('Cannot import Rated Power from the file %s', self.file_rated)
            self.logger.info('Rated Power imported successfully from the file %s', self.file_rated)

        self.logger.info('Data loaded successfully')

    def create_area(self, coord, resolution):
        if self.nwp_resolution == 0.05:
            levels = 4
            round_coord = 1
        else:
            levels = 2
            round_coord = 0

        if coord!=None:
            if isinstance(coord, list):
                if len(coord)==2:
                    lat = coord[0]
                    long = coord[1]
                    lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20, resolution)
                    lat1 = lat_range[np.abs(lat_range - lat).argmin()]-self.nwp_resolution/10
                    lat2 = lat_range[np.abs(lat_range - lat).argmin()]+self.nwp_resolution/10


                    long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20, resolution)
                    long1 = long_range[np.abs(long_range - long).argmin()] - self.nwp_resolution / 10
                    long2 = long_range[np.abs(long_range - long).argmin()] + self.nwp_resolution / 10

                    area=[[lat1 - self.nwp_resolution*levels, long1 - self.nwp_resolution*levels],
                                 [lat2 + self.nwp_resolution*levels, long2 + self.nwp_resolution*levels]]
                elif len(coord)==4:
                    area = list(np.array(coord).reshape(2,2))
                else:
                    raise ValueError('Wrong coordinates. Should be point (lat, long) or area [lat1, long1, lat2, long2]')
            elif isinstance(coord, dict):
                area = dict()
                for key, value in coord.items():
                    if len(value)==2:
                        lat = value[0]
                        long = value[1]
                        lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20,
                                              resolution)
                        lat1 = lat_range[np.abs(lat_range - lat).argmin()] - self.nwp_resolution / 10
                        lat2 = lat_range[np.abs(lat_range - lat).argmin()] + self.nwp_resolution / 10

                        long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20,
                                               resolution)
                        long1 = long_range[np.abs(long_range - long).argmin()] - self.nwp_resolution / 10
                        long2 = long_range[np.abs(long_range - long).argmin()] + self.nwp_resolution / 10

                        area[key] = [[lat1 - self.nwp_resolution*levels, long1 - self.nwp_resolution*levels],
                                     [lat2 + self.nwp_resolution*levels, long2 + self.nwp_resolution*levels]]
                    else:
                        area[key] = np.array(value).reshape(2,2)
            else:
                raise ValueError('Wrong coordinates. Should be dict or list')
        else:
            area = dict()
        self.logger.info('Areas created succesfully')

        return area

    def initialize(self):
        data_file_name = os.path.basename(self.file_data)
        if os.path.exists(os.path.join(os.path.dirname(self.file_data), 'coord_auto_' + self.model_type + '.csv')):
            self.file_coord = os.path.join(os.path.dirname(self.file_data), 'coord_auto_' + self.model_type + '.csv')
        else:
            self.file_coord = os.path.join(os.path.dirname(self.file_data), 'coord_' + self.model_type + '.csv')
        if not os.path.exists(self.file_coord) and self.weather_in_data==False:
            raise IOError('File with coordinates does not exist')

        self.file_rated = os.path.join(os.path.dirname(self.file_data), 'rated_' + self.model_type + '.csv')
        if not os.path.exists(self.file_rated):
            if self.model_type in {'wind', 'pv'} and self.projects_group not in {'APE_net'}:
                raise ValueError('Provide rated_power for each project. The type of projects is %s', self.model_type)
            self.use_rated = False
        else:
            self.use_rated = True

        self.load_data()

        self.group_static_data = []
        if self.check_project_names():
            for project_name in self.projects:
                path_project = self.path_group + '/' + project_name
                if not os.path.exists(path_project):
                    os.makedirs(path_project)
                path_backup = self.path_group + '/backup_models/' + project_name
                if not os.path.exists(path_backup):
                    os.makedirs(path_backup)

                path_model = path_project + '/model_ver' + str(self.version_model)
                if not os.path.exists(path_model):
                    os.makedirs(path_model)

                path_data = path_model + '/DATA'
                if not os.path.exists(path_data):
                    os.makedirs(path_data)
                path_fuzzy_models = path_model + '/fuzzy_models'
                if not os.path.exists(path_fuzzy_models):
                    os.makedirs(path_fuzzy_models)
                if self.use_rated:
                    if  project_name == self.projects_group + '_' + self.model_type and project_name not in self.rated.index.to_list():
                        rated = self.rated.sum().to_list()[0]
                    else:
                        rated = self.rated.loc[project_name].to_list()[0]
                else:
                    rated = None
                if hasattr(self, 'coord'):
                    if project_name=='APE_net' or self.model_type=='load' or project_name == self.projects_group + '_' + self.model_type:
                        coord = dict()
                        for name, latlong in self.coord.iterrows():
                            coord[name] = latlong.values.tolist()
                    else:
                        coord = self.coord.loc[project_name].to_list()
                else:
                    coord = None
                area = self.create_area(coord, self.nwp_resolution)

                temp = {'_id': project_name,
                               'owner': self.project_owner,
                               'project_group': self.projects_group,
                               'type': self.model_type,
                               'location': coord,
                               'areas': area,
                               'rated': rated,
                               'path_project': path_project,
                               'path_model': path_model,
                               'version_group': self.version_group,
                               'version_model': self.version_model,
                               'path_backup': path_backup,
                               'path_data': path_data,
                               'pathnwp': self.path_nwp_group,
                               'path_fuzzy_models': path_fuzzy_models,
                               'run_on_platform': False,
                        }
                static_data=dict()
                for key, value in self.static_data.items():
                    static_data[key] = value
                for key, value in temp.items():
                    static_data[key] = value

                self.group_static_data.append({'_id' : project_name, 'static_data' : static_data})
                joblib.dump(static_data, os.path.join(path_model, 'static_data.pickle'))
                with open(os.path.join(path_model, 'static_data.txt'), 'w') as file:
                    for k, v in static_data.items():
                        if not isinstance(v, dict):
                            file.write(str(k) + ' >>> ' + str(v) + '\n\n')
                        else:
                            file.write(str(k) + ' >>> ' + '\n')
                            for kk, vv in v.items():
                                file.write('\t' + str(kk) + ' >>> ' + str(vv) + '\n')

            joblib.dump(self.group_static_data, os.path.join(self.path_group, 'static_data_projects.pickle'))
            self.logger.info('Static data of all projects created')


    def nwp_extractor(self, data):
        static_data = copy.deepcopy(self.static_data)
        data.index = data.index.astype(str)
        static_data['data'] = data.to_dict()
        client = rabbit_client_nwp()
        return client.call(static_data)

    def create_datasets(self, data, test=False):
        static_data = copy.deepcopy(self.static_data)
        data.index = data.index.astype(str)
        static_data['data'] = data.to_dict()
        static_data['test'] = test
        static_data['group_static_data'] = self.group_static_data
        client = rabbit_client_data()
        return client.call(static_data)

    def create_projects_relations(self):
        if os.path.exists(os.path.join(self.path_group, 'static_data_projects.pickle')):
            self.group_static_data = joblib.load(os.path.join(self.path_group, 'static_data_projects.pickle'))
        else:
            self.initialize()
        if self.static_data['enable_transfer_learning'] and len(self.group_static_data)>1:
            self.logger.info('Create projects relations')
            transfer_learning_linker = ProjectLinker(self.group_static_data)
            self.training_project_groups = transfer_learning_linker.find_relations()

            projects = dict()
            for project in self.group_static_data:
                if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                    projects[project['_id']] = project

            count_projects = 0
            for project_name, project in projects.items():
                if project_name not in self.training_project_groups.keys():
                    for main_project, group in self.training_project_groups.items():
                        if project_name in group:
                            project['static_data']['transfer_learning'] = True
                            project['static_data']['tl_project'] = projects[main_project]
                            count_projects+=1
                else:
                    count_projects += 1

            if len(projects) != count_projects:
                raise RuntimeError('Some projects does not include in a transfer learning project group')

            for project in self.group_static_data:
                if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                    project['static_data']['transfer_learning'] = projects[project['_id']]['static_data']['transfer_learning']
                    project['static_data']['tl_project'] = projects[project['_id']]['static_data']['tl_project']

    def fit(self):

        for project in self.group_static_data:
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelTrainManager(project['static_data']['path_model'])
                # if project_model.istrained == False:
                project_model.init(project['static_data'], self.data_variables)
                if self.model_type  in {'wind', 'pv'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))\
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle')):
                        if project['static_data']['transfer_learning'] == False:
                            self.logger.info('Start train project %s', project['_id'])
                            project_model.train()
                        # else:
                        #     project_model.train_TL(project['static_data']['tl_project']['static_data']['path_model'])
                    else:
                        raise ValueError('Cannot find project ', project['_id'], ' datasets')

                elif self.model_type in {'load'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle')):
                        self.logger.info('Start train project %s', project['_id'])
                        project_model.train()
                elif self.model_type in {'fa'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle')):
                        if project['static_data']['transfer_learning'] == False:
                            self.logger.info('Start train project %s', project['_id'])
                            project_model.train()
                else:
                    raise ValueError('Cannot recognize model type')
        if self.static_data['enable_transfer_learning']:
            for project in self.group_static_data:
                if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                    project_model = ModelTrainManager(project['static_data']['path_model'])
                    if project_model.istrained == False:
                        project_model.init(project['static_data'], self.data_variables)
                        if self.model_type  in {'wind', 'pv'}:
                            if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                    and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))\
                                    and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle')):
                                if project['static_data']['transfer_learning'] == True:
                                    self.logger.info('Start train project %s', project['_id'])
                                    project_model.train_TL(project['static_data']['tl_project']['static_data']['path_model'])
                            else:
                                raise ValueError('Cannot find project ', project['_id'], ' datasets')

                        else:
                            raise ValueError('Cannot recognize model type')

    def evaluate(self):
        self.initialize()
        self.create_datasets(self.data_eval, test=True)

        for project in self.group_static_data:
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelPredictManager(project['static_data']['path_model'])
                # if project_model.istrained == False:
                project_model.init(project['static_data'], self.data_variables)
                if self.model_type  in {'wind', 'pv'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))\
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn_test.pickle')):

                        self.logger.info('Evaluate project %s', project['_id'])
                        project_model.evaluate_all()
                        
                    else:
                        raise ValueError('Cannot find project ', project['_id'], ' datasets')

                elif self.model_type in {'load'}:
                    raise NotImplementedError('load model manager not implemented yet')

                elif self.model_type in {'fa'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')) \
                            and os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle')):
                        self.logger.info('Evaluate project %s', project['_id'])
                        project_model.evaluate_all()

                else:
                    raise ValueError('Cannot recognize model type')
