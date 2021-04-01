import getpass
import os
import sys

import tensorflow as tf

from Fuzzy_clustering.version2.template.constants import *
from Fuzzy_clustering.version2.template.default_project import ALL_FIELDS
from Fuzzy_clustering.version2.template.default_project import default_project_values

Docker = False

if Docker:
    sys_folder = '/models/'
    nwp_folder = '/nwp_grib/'
    data_file_name = './data/wind_ts.csv'
    project_path = None
else:
    data_file_name = f'D:/Dropbox/current_codes/PycharmProjects/'

    if sys.platform == LINUX:
        user_name = getpass.getuser()
        if user_name == 'timos':
            project_path = '/home/timos/Desktop/LoadForecasting/Projects'
            sys_folder = f'{project_path}/models/'
            nwp_folder = f'{project_path}/nwps/'
            data_file_name = f'{project_path}/data/'
        else:
            sys_folder = '/media/smartrue/HHD1/George/models/'
            nwp_folder = '/media/smartrue/HHD2/'
            project_path = None
    else:
        sys_folder = 'D:/models/'
        nwp_folder = 'D:/Dropbox/'
        project_path = None

projects = {

    'Petalas': {
        DOCKER: Docker,
        PROJECT_OWNER: 'my_projects',
        PROJECTS_GROUP: 'Greece',
        DATA_FILE_NAME: f'{data_file_name}Petalas/data/wind_ts.csv',
        NWP_FOLDER: nwp_folder,
        SYS_FOLDER: sys_folder,
        AREA_GROUP: [[33.9426055, 18.929342], [42.047856, 30.0400288]],
        NWP_MODEL: ECMWF,
        NWP_RESOLUTION: 0.1,
        VERSION_GROUP: 1,
        N_JOBS: 4,
        VERSION_MODEL: 1,
        ΝUM_GEN: 1,
        LSTM_MAX_ITERATIONS: 10,
        N_JOBS_LSTM: 1,
        N_CLUSTERS: 10,
        UNITS: 64,
        FEATURE_SELECTION: False,
        DENSE_LAYER_DIM: 64,
        IS_GLOBAL: True,
        IS_FUZZY: True,
        EVALUATION: '10062019 01:00',
        PROJECT_METHODS: {RBF_ALL_CNN: False, RBF_ALL: False, CNN: False, LSTM: False, MLP_3D: False,
                          SVM: True, NU_SVM: False, MLP: False, RF: True, XGB: False, ELASTIC_NET: False},

    },

    'AzoresDayAhead': {
        DOCKER: Docker,
        PROJECT_OWNER: 'EDA',
        PROJECTS_GROUP: 'Azores',
        DATA_FILE_NAME: f'{data_file_name}Azores/data/St_Miguel/load/DayAhead/load_ts.csv',
        NWP_FOLDER: nwp_folder,
        SYS_FOLDER: sys_folder,
        N_JOBS: 4,
        N_GPUS: 1,
        AREA_GROUP: [[36.5, -33], [40.5, -23.5]],
        NWP_MODEL: GFS,
        NWP_RESOLUTION: 0.25,
        IS_GLOBAL: True,
        IS_FUZZY: True,
        FEATURE_SELECTION: False,
        DIMENSIONALITY_REDUCTION: False,
        EVALUATION: '01072020 00:00',
        PROJECT_METHODS: {RBF_ALL_CNN: False, RBF_ALL: False, CNN: False, LSTM: True, MLP_3D: False,
                          SVM: True, NU_SVM: True, MLP: False, RF: True, XGB: True, ELASTIC_NET: False},
        LSTM_MAX_ITERATIONS: 10,
        N_JOBS_LSTM: 1,
        ΝUM_GEN: 1,
        N_CLUSTERS: 10,
        UNITS: 64,
        DENSE_LAYER_DIM: 64,
    },

    'AzoresOneHourAhead': {
        DOCKER: Docker,
        PROJECT_OWNER: 'EDA',
        PROJECTS_GROUP: 'Azores',
        DATA_FILE_NAME: f'{data_file_name}Azores/data/St_Miguel/load/OneHour/load_ts.csv',
        NWP_FOLDER: nwp_folder,
        SYS_FOLDER: sys_folder,
        AREA_GROUP: [[36.5, -33], [40.5, -23.5]],
        NWP_MODEL: GFS,
        NWP_RESOLUTION: 0.25,
        IS_GLOBAL: True,
        FEATURE_SELECTION: False,
        DIMENSIONALITY_REDUCTION: True,
        IS_FUZZY: True,
        ΝUM_GEN: 1,
        N_JOBS_LSTM: 1,
        UNITS: 16,

    }
}

# Overwrite any previous default value and make sure
# all required fields are present in a project

for project in projects:
    projects[project] = {**default_project_values, **projects[project]}
    assert all(
        item in ALL_FIELDS for item in projects[project].keys()), 'All projects must have all the settings defined'

tf.autograph.set_verbosity(0)
# Silence info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
