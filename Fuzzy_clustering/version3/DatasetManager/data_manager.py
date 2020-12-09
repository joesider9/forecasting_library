import pika, uuid, time, json, os, shutil, sys, logging
import numpy as np
import pandas as pd
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.DatasetManager.create_datasets_PCA import dataset_creator_PCA
from Fuzzy_clustering.version3.DatasetManager.create_datasets_dense import dataset_creator_dense
from Fuzzy_clustering.version3.DatasetManager.create_datasets_for_fa import dataset_creator_ecmwf, dataset_creator_xmachina
from Fuzzy_clustering.version3.DatasetManager.create_dataset_for_load_SCADA import dataset_creator_scada
from Fuzzy_clustering.version3.DatasetManager.create_dataset_for_load_LV import dataset_creator_LV

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='data_manager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)


class Dataset_creator():
    def __init__(self, static_data ,group_static_data):
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
                if self.nwp_model == 'skiron' and self.nwp_resolution == 0.05:
                    self.path_nwp = '/media/smartrue/HHD2/SKIRON'
                elif self.nwp_model == 'skiron' and self.nwp_resolution == 0.1:
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

        self.path_group = self.sys_folder + self.project_owner + '/' + self.projects_group + '_ver' + str(
            self.version_group) + '/' + self.model_type
        if not os.path.exists(self.path_group):
            os.makedirs(self.path_group)
        self.path_nwp_group = self.sys_folder + self.project_owner + '/' + self.projects_group + '_ver' + str(
            self.version_group) + '/nwp'
        if not os.path.exists(self.path_nwp_group):
            os.makedirs(self.path_nwp_group)
        self.create_logger()

    def create_logger(self):
        self.logger = logging.getLogger('DataManager_' + self.model_type)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_group, 'log_data_manager.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def create_datasets(self, data, test):

        if self.static_data['recreate_datasets']:
            self.logger.info('Previous dataset is gona removed')
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
        if self.static_data['compress_data'] == 'PCA' and self.model_type in {'pv', 'wind'}:
            for project in self.group_static_data:
                if self.static_data['recreate_datasets']:
                    os.remove(os.path.join(project['static_data']['path_data'], 'nwps_3d.pickle'))
                if project['_id'] != self.projects_group + '_' + self.model_type:
                    if data[project['_id']].dropna().count() > 24:
                        if test:
                            if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                                    or not os.path.exists(
                                os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                                self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                dataset = dataset_creator_PCA(project, data=data[project['_id']].dropna(),
                                                              njobs=self.static_data['njobs'], test=test)
                                dataset.make_dataset_res()
                                self.logger.info('Dataset using PCA for testing constructed for %s', project['_id'])
                        else:
                            if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                    or not os.path.exists(
                                os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                                self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                dataset = dataset_creator_PCA(project, data=data[project['_id']].dropna(),
                                                              njobs=self.static_data['njobs'])
                                dataset.make_dataset_res()
                                self.logger.info('Dataset for training using PCA constructed for %s', project['_id'])

                    elif project['_id'] == self.projects_group + '_' + self.model_type:
                        if test:
                            if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                                    or not os.path.exists(
                                os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                                self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                dataset = dataset_creator_dense(self.projects_group, project, data[project['_id']].dropna(),
                                                                self.path_nwp_group, self.nwp_model,
                                                                self.nwp_resolution, self.model_type,
                                                                njobs=self.static_data['njobs'], test=test)
                                dataset.make_dataset_res()
                                self.logger.info('Dataset uisng PCA constructed for %s', project['_id'])
                        else:
                            if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                                    or not os.path.exists(
                                os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                                self.logger.info('Start created dataset uisng PCA for %s', project['_id'])
                                dataset = dataset_creator_dense(self.projects_group, project, data[project['_id']].dropna(),
                                                                self.path_nwp_group,
                                                                self.nwp_model, self.nwp_resolution, self.data_variables,
                                                                njobs=self.static_data['njobs'], test=test)
                                dataset.make_dataset_res()
                                self.logger.info('Dataset uisng PCA constructed for %s', project['_id'])

        elif self.static_data['compress_data'] == 'dense' and self.model_type in {'pv', 'wind'}:
            project_col = []
            projects = []
            for project in self.group_static_data:
                if test:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv')) \
                            or not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv')):
                        if data[project['_id']].dropna().count() > 24:
                            projects.append(project)
                            project_col.append(project['_id'])
                else:
                    if not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            or not os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                        if data[project['_id']].dropna().count() > 24:
                            projects.append(project)
                            project_col.append(project['_id'])
            self.logger.info('Start creating dataset using dense for country %s', self.projects_group)
            self.logger.info('Start creating dataset using dense for projects %s', project_col)
            if len(project_col) > 0:
                dataset = dataset_creator_dense(self.projects_group, projects, data[project_col].dropna(),
                                                self.path_nwp_group,
                                                self.nwp_model, self.nwp_resolution, self.data_variables,
                                                njobs=self.static_data['njobs'], test=test)

                dataset.make_dataset_res()
                self.logger.info('Dataset using dense for country %s created', self.projects_group)

        elif self.model_type in {'load'}:
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

        elif self.model_type in {'fa'}:
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
                    if self.version_model==0:
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer) or isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.str) or isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, bool):
            return bool(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            print(obj)
            raise TypeError('Object is not JSON serializable')

@server.consumer()
def data_manager(static_data):
    group_static_data = static_data['group_static_data']
    test = static_data['test']
    data = static_data['data']
    data = pd.DataFrame.from_dict(data)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

    print(" [.] Receive project_group %s)" % static_data['projects_group'])
    data_manager = Dataset_creator(static_data, group_static_data)
    response = data_manager.create_datasets(data, test=test)

    return response

if __name__=='__main__':
    server.run()
