import pika, uuid, time, json, os, shutil, sys, logging
import numpy as np
import pandas as pd
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer
from Fuzzy_clustering.version3.NwpManager.skiron_extractor import skiron_Extractor
from Fuzzy_clustering.version3.NwpManager.ecmwf_extractor import ecmwf_Extractor



RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='nwp_manager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class nwp_extractor():
    def __init__(self, static_data):
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
        self.logger = logging.getLogger('NWP_Manager_' + self.model_type)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_group, 'log_nwp.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def extract(self, data):
        if not self.path_nwp is None:
            if self.static_data['recreate_nwp_files']:
                shutil.rmtree(self.path_nwp_group)
                os.makedirs(self.path_nwp_group)
            count = 0
            if self.nwp_model == 'skiron':
                for t in data.index:
                    if not os.path.exists(
                            os.path.join(self.path_nwp_group, 'skiron_' + t.strftime('%d%m%y') + '.pickle')):
                        count += 1
            else:
                for t in data.index:
                    if not os.path.exists(
                            os.path.join(self.path_nwp_group, 'ecmwf_' + t.strftime('%d%m%y') + '.pickle')):
                        count += 1
            if count > 20:
                self.logger.info('Start extract nwps')
                if self.nwp_model == 'skiron' and sys.platform == 'linux':
                    nwp_extractor = skiron_Extractor(self.projects_group, self.path_nwp, self.nwp_resolution,
                                                     self.path_nwp_group, data.index, self.area_group,
                                                     njobs=2 * self.static_data['njobs'])
                    nwp_extractor.extract_nwps()
                elif self.nwp_model == 'ecmwf':
                    nwp_extractor = ecmwf_Extractor(self.projects_group, self.path_nwp, self.nwp_resolution,
                                                    self.path_nwp_group, data.index, self.area_group,
                                                    njobs=2 * self.static_data['njobs'])
                    nwp_extractor.extract_nwps()
                else:
                    raise IOError('Windows does not support pygrib. You should extract nwps in a linux system')
                self.logger.info('Finish extract nwps')
            else:
                self.logger.info('Found nwp files')

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
def nwp_manager(static_data):
    data = static_data['data']
    data = pd.DataFrame.from_dict(data)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

    print(" [.] Receive project_group for nwp extract %s)" % static_data['projects_group'])
    nwp_manager = nwp_extractor(static_data)
    response = nwp_manager.extract(data)
    return response

if __name__=='__main__':
    server.run()
