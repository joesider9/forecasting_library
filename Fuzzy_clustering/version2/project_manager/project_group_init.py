import os

import joblib
import numpy as np
import pandas as pd

from Fuzzy_clustering.version2.common_utils.logging import create_logger


def compute_area_grid(lat, long, resolution, round_coord, levels):
    lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20,
                          resolution)
    lat1 = lat_range[np.abs(lat_range - lat).argmin()] - resolution / 10
    lat2 = lat_range[np.abs(lat_range - lat).argmin()] + resolution / 10

    long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20,
                           resolution)
    long1 = long_range[np.abs(long_range - long).argmin()] - resolution / 10
    long2 = long_range[np.abs(long_range - long).argmin()] + resolution / 10

    return [[lat1 - resolution * levels, long1 - resolution * levels],
            [lat2 + resolution * levels, long2 + resolution * levels]]


class ProjectGroupInit:
    """
    Class responsible for managing and loading the
    power output or load.

    """

    def __init__(self, static_data):
        """

        Parameters
        ----------
        static_data: python dict
            contains all the information required to load the power measurement for specific project(s).
        """
        self.static_data = static_data  # dict containing information about project paths, model structure and training
        # params, input file, see in util_database_timos.py and config_timos.py
        self.file_data = static_data['data_file_name']  # input .csv file PROBLEM_TYPE + '_ts.csv' i.e. wind_ts.csv
        self.project_owner = static_data[
            'project_owner']  # Name of project owner or research program i.e. my_projects or CROSSBOW
        self.projects_group = static_data['projects_group']  # Name of the country
        self.area_group = static_data['area_group']  # coordinates of the country
        self.version_group = static_data['version_group']
        self.version_model = static_data['version_model']
        self.weather_in_data = static_data[
            'weather_in_data']  # True if input file contains more columns than the power output column
        self.nwp_model = static_data['NWP_model']
        self.nwp_resolution = static_data['NWP_resolution']
        self.data_variables = static_data['data_variables']  # Variable names used

        self.projects = []  # list containing all the parks, we're interested in. Each park is considered as a project.
        self.use_rated = True

        self.model_type = self.static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_group = self.static_data['path_group']
        self.path_nwp_group = self.static_data['path_nwp_group']
        self.group_static_data = []
        self.logger = create_logger(logger_name=f'ProjectInitManager_{self.model_type}', abs_path=self.path_group,
                                    logger_path=f'log_{self.projects_group}.log', write_type='a')

    def initialize(self):
        if os.path.exists(os.path.join(os.path.dirname(self.file_data), 'coord_auto_' + self.model_type + '.csv')):
            self.file_coord = os.path.join(os.path.dirname(self.file_data), 'coord_auto_' + self.model_type + '.csv')
        else:
            self.file_coord = os.path.join(os.path.dirname(self.file_data), 'coord_' + self.model_type + '.csv')

        if not os.path.exists(self.file_coord) and not self.weather_in_data:
            raise IOError('File with coordinates does not exist')

        self.file_rated = os.path.join(os.path.dirname(self.file_data), 'rated_' + self.model_type + '.csv')
        if not os.path.exists(self.file_rated):
            if self.model_type in {'wind', 'pv'} and self.projects_group not in {'IPTO'}:
                raise ValueError('Provide rated_power for each project. The type of projects is %s', self.model_type)
            self.use_rated = False
        else:
            self.use_rated = True

        self.load_power_of_parks()  # Loads power output, coordinates and rated power.

        if len(self.projects) == 0:
            raise ImportError('No project loaded. check the input file in configuration')

        if self.check_project_names():
            for project_name in self.projects:
                path_project = self.path_group + '/' + project_name
                if not os.path.exists(path_project):
                    os.makedirs(path_project)
                path_model = path_project + '/model_ver' + str(self.version_model)
                if not os.path.exists(path_model):
                    os.makedirs(path_model)
                path_backup = self.path_group + '/backup_models/' + project_name + '/model_ver' + str(
                    self.version_model)
                if not os.path.exists(path_backup):
                    os.makedirs(path_backup)
                path_data = path_model + '/DATA'
                if not os.path.exists(path_data):
                    os.makedirs(path_data)
                path_fuzzy_models = path_model + '/fuzzy_models'
                if not os.path.exists(path_fuzzy_models):
                    os.makedirs(path_fuzzy_models)
                if self.use_rated:
                    if project_name == self.projects_group + '_' + self.model_type and project_name not in self.rated.index.to_list():
                        rated = self.rated.sum().to_list()[0]
                    else:
                        rated = self.rated.loc[project_name].to_list()[0]
                else:
                    rated = None
                if hasattr(self, 'coord'):
                    if project_name == 'APE_net' or self.model_type == 'load' or project_name == self.projects_group + '_' + self.model_type:
                        coord = dict()
                        for name, lat_long in self.coord.iterrows():
                            coord[name] = lat_long.values.tolist()
                    else:
                        coord = self.coord.loc[project_name].to_list()  # [lat, long]
                else:
                    coord = None
                area = self.create_area(coord)

                temp = {'_id': project_name,
                        'owner': self.project_owner,
                        'project_group': self.projects_group,
                        'type': self.model_type,
                        'location': coord,
                        'areas': area,
                        'rated': rated,
                        'path_project': path_project,
                        'path_model': path_model,
                        'path_group': self.path_group,
                        'version_group': self.version_group,
                        'version_model': self.version_model,
                        'path_backup': path_backup,
                        'path_data': path_data,
                        'pathnwp': self.path_nwp_group,
                        'path_fuzzy_models': path_fuzzy_models,
                        'run_on_platform': False,
                        }
                static_data = dict()
                for key, value in self.static_data.items():
                    static_data[key] = value
                for key, value in temp.items():
                    static_data[key] = value

                self.group_static_data.append({'_id': project_name, 'static_data': static_data})
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

    def check_project_names(self):
        flag = True
        if self.model_type in {'wind', 'pv'}:
            for name in self.projects:
                if name not in self.coord.index.to_list() and name != self.projects_group + '_' + self.model_type and name != 'APE_net':
                    flag = False
                    self.logger.info('There is inconsistency to files data and coord for the project %s', name)
            if not flag:
                raise ValueError('Inconsistency in project names between data and coord')

        if self.use_rated:
            for name in self.projects:
                if name not in self.rated.index.to_list() and name != self.projects_group + '_' + self.model_type:
                    flag = False
                    self.logger.info('There is inconsistency to files data and rated for the project %s', name)
            if not flag:
                raise ValueError('Inconsistency in project names between data and rated')

        return flag

    def load_power_of_parks(self):
        try:
            # Data containing power output or load. Each column refers to a different wind, pv park.
            self.data = pd.read_csv(self.file_data, header=0, index_col=0, parse_dates=True, dayfirst=True)
        except Exception:
            self.logger.info(f'Cannot import timeseries from the file {self.file_data}')
            raise IOError(f'Cannot import timeseries from the file {self.file_data}')
        self.logger.info('Timeseries imported successfully from the file %s', self.file_data)

        if 'total' in self.data.columns:  # In some cases, the total output of all parks is included.
            self.data = self.data.rename(
                columns={'total': self.projects_group + '_' + self.model_type})  # e.g group = 'Greece'

        if self.static_data['Evaluation_start']:

            valid_combination = True
            time_offset = pd.DateOffset(hours=0)

            if self.model_type == 'fa':
                time_offset = pd.DateOffset(days=372)
            elif self.model_type == 'load':
                if self.data.columns[0] == 'lv_load':
                    time_offset = pd.DateOffset(days=9001)
                else:
                    if self.static_data['horizon'] == 'short-term':
                        time_offset = pd.DateOffset(hours = 350)
                    else:
                        if self.static_data['ts_resolution'] == 'hourly':
                            time_offset = pd.DateOffset(hours = 9001)
                        elif self.static_data['ts_resolution'] == '15min':
                            time_offset = pd.DateOffset(minutes = 60 * 9001)

            if valid_combination:
                try:
                    eval_date = pd.to_datetime(self.static_data['Evaluation_start'], format='%d%m%Y %H:%M')
                    self.data_eval = self.data.iloc[np.where(self.data.index > eval_date - time_offset)]
                    self.data = self.data.iloc[np.where(self.data.index <= eval_date)]
                except Exception:
                    raise ValueError('Wrong date format, use %d%m%Y %H:%M. Or the date does not exist in the dataset')

        if self.model_type == 'load':
            self.projects.append(self.data.columns[0])
        elif self.model_type == 'fa':
            if self.version_model == 0:
                self.projects.append('fa_curr_morning')
            elif self.version_model == 1:
                self.projects.append('fa_ahead_morning')
            else:
                raise ValueError(
                    'Version model should be 0 for current day and 1 for day ahead otherwise choose another group version')
        else:
            for name in self.data.columns:
                var = f'{self.projects_group}_{self.model_type}' if name == 'total' else name
                self.projects.append(var)
        if not self.weather_in_data:
            try:
                # For each of the park, load its coordinates. (lat,long) single tuple
                self.coord = pd.read_csv(self.file_coord, header=None, index_col=0)
            except Exception:
                self.logger.info('Cannot import coordinates from the file %s', self.file_coord)
                raise IOError('Cannot import coordinates from the file %s', self.file_coord)
            self.logger.info('Coordinates imported successfully from the file %s', self.file_coord)
        else:
            self.logger.info('Coordinates in the data')

        if self.use_rated:
            try:
                # For each park, read its rated (maximum) power.
                self.rated = pd.read_csv(self.file_rated, header=None, index_col=0)
            except Exception:
                self.logger.info('Cannot import Rated Power from the file %s', self.file_rated)
                raise IOError('Cannot import Rated Power from the file %s', self.file_rated)
            self.logger.info('Rated Power imported successfully from the file %s', self.file_rated)

        self.logger.info('Data loaded successfully')

    def create_area(self, coord):
        levels = 4 if self.nwp_resolution == 0.05 else 2
        round_coord = 1 if self.nwp_resolution == 0.05 else 0

        if coord is None:
            area = dict()
        elif isinstance(coord, list):
            if len(coord) == 2:
                lat, long = coord[0], coord[1]
                area = compute_area_grid(lat, long, self.nwp_resolution, round_coord, levels)
            elif len(coord) == 4:
                area = list(np.array(coord).reshape(2, 2))
            else:
                raise ValueError(
                    'Wrong coordinates. Should be point (lat, long) or area [lat1, long1, lat2, long2]')
        elif isinstance(coord, dict):
            area = dict()
            for key, value in coord.items():
                if len(value) == 2:
                    lat, long = value[0], value[1]
                    area[key] = compute_area_grid(lat, long, self.nwp_resolution, round_coord, levels)
                else:
                    area[key] = np.array(value).reshape(2, 2)
        else:
            raise ValueError('Wrong coordinates. Should be dict or list')

        self.logger.info('Areas created successfully')

        return area
