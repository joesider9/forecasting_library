import os
import shutil
import sys

from Fuzzy_clustering.version2.common_utils.logging import create_logger
from Fuzzy_clustering.version2.nwp_manager.ecmwf_extractor import EcmwfExtractor
from Fuzzy_clustering.version2.nwp_manager.skiron_extractor import SkironExtractor
from Fuzzy_clustering.version2.nwp_manager.gfs_extractor import GfsExtractor


class NwpExtractor:
    def __init__(self, static_data, is_test=False):
        self.is_test = is_test
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

        self.logger = create_logger(logger_name=f'NwpManager_{self.model_type}',
                                    abs_path=self.path_group,
                                    logger_path='log_nwp.log')

    def extract(self, data):
        if self.path_nwp is not None:
            if self.static_data['recreate_nwp_files']:
                shutil.rmtree(self.path_nwp_group)
                os.makedirs(self.path_nwp_group)

            count = 0
            for t in data.index:
                nwp_file = f"{self.nwp_model}_{t.strftime('%d%m%y')}.pickle"
                if not os.path.exists(os.path.join(self.path_nwp_group, nwp_file)):
                    count += 1
            if count > 20:
                self.logger.info('Start extracting nwps')
                if self.nwp_model == 'skiron' and sys.platform == 'linux':
                    nwp_extractor = SkironExtractor(self.projects_group, self.path_nwp, self.nwp_resolution,
                                                    self.path_nwp_group, data.index, self.area_group,
                                                    n_jobs=self.static_data['njobs'])
                    nwp_extractor.extract_nwps()
                elif self.nwp_model == 'ecmwf':
                    nwp_extractor = EcmwfExtractor(self.projects_group, self.path_nwp, self.nwp_resolution,
                                                   self.path_nwp_group, data.index, self.area_group,
                                                   n_jobs=self.static_data['njobs'])
                    nwp_extractor.extract_nwps()
                elif self.nwp_model == 'gfs':
                    nwp_extractor = GfsExtractor(self.projects_group, self.path_nwp, self.nwp_resolution,
                                                   self.path_nwp_group, data.index, self.area_group,
                                                   n_jobs=self.static_data['njobs'])
                    nwp_extractor.extract_nwps()
                elif self.nwp_model == 'openweather':
                    pass
                else:
                    raise IOError('Windows does not support pygrib. You should extract nwps in a linux system')
                self.logger.info('Finish extract nwps')
            else:
                self.logger.info(f"Didn't find enough nwp files, found {count}")

        return 'Done'
