import os
import sys

Docker = False
if Docker:
    sys_folder = '/models/'
    nwp_folder = '/nwp_grib/'
    data_file_name = 'data/St_Miguel/wind/OneHour/DayAhead/wind_ts.csv'
else:
    if sys.platform == 'linux':
        sys_folder = '/media/smartrue/HHD1/George/models/'
        nwp_folder = '/media/smartrue/HHD2/'
        data_file_name = '/home/smartrue/PycharmProjects/Azores/data/St_Miguel/wind/DayAhead/wind_ts.csv'
    else:
        sys_folder = 'D:/models/'
        nwp_folder = 'D:/Dropbox/'
        data_file_name = 'D:/Dropbox/current_codes/PycharmProjects/Azores/data/St_Miguel/wind/DayAhead/wind_ts.csv'

project_owner = 'EDA'
projects_group = 'Azores'
area_group = [[36.5, -33], [40.5, -23.5]]
version_group = 0
version_model = 0
horizon = 'day_ahead'
ts_resolution = 'hourly'

if sys.platform != 'linux':
    njobs = 8  # ALL CPUS
    njobs_fuzzy = 8  # HAlf CPUs
    njobs_feat_sel = 8  # 3/4 CPUS
    inner_jobs_feat_sel= 1
    njobs_rbfols = 8
    njobs_rbfnn = 4
    njobs_sklearn = 8
    njobs_lstm = 2
    njobs_cnn_3d = 2
    njobs_cnn = 1
    njobs_mlp = 2
    intra_op = 2

    ngpus = 1
else:
    njobs = 18  # ALL CPUS
    njobs_fuzzy = 18  # HAlf CPUs
    njobs_feat_sel = 18  # 3/4 CPUS
    inner_jobs_feat_sel= 1
    njobs_rbfols = 18
    njobs_rbfnn = 3
    njobs_sklearn = 18
    njobs_lstm = 2
    njobs_cnn_3d = 2
    njobs_cnn = 1
    njobs_mlp = 3
    intra_op = 2

    ngpus = 2


Evaluation = '01072019 00:00' # date or None

RECREATE_NWP_FILES = False
RECREATE_DATASETS = False
weather_in_data = False

NWP_model = 'gfs'  # Skiron, ECMWF, gfs
NWP_resolution = 0.25  # 0.05 or 0.04, 0.25

compress_data = 'dense'  # PCA or dense

ENABLE_TRANSFER_LEARNING = False
transfer_learning = False
tl_project = ''
feature_selection_method = 'linearsearch' #boruta or permutation
dimentional_reduction = False

combine_methods = ['rls', 'bcp', 'mlp', 'bayesian_ridge', 'elastic_net', 'ridge',
                   'average']  # 'rls', 'bcp', 'mlp', 'bayesian_ridge', 'elastic_net', 'ridge', 'average'
exclude_method_combine = []

is_Global = False

is_Fuzzy = True
cluster_file='best_fuzzy.pkl'
is_clustering_trained=False
is_probabilistic=False

project_methods = {}

project_methods['RBF_ALL_CNN'] = True
project_methods['RBF_ALL'] = False
project_methods['CNN'] = True
project_methods['LSTM'] = False
project_methods['MLP_3D'] = False
project_methods['SVM'] = False
project_methods['NUSVM'] = False
project_methods['MLP'] = True
project_methods['RF'] = True
project_methods['XGB'] = False
project_methods['elasticnet'] = False

Sklearn_optimizer = 'deap'

file_name = os.path.basename(data_file_name)

if 'load' in file_name:
    model_type = 'load'
elif 'pv' in file_name:
    model_type = 'pv'
elif 'wind' in file_name:
    model_type = 'wind'
elif 'fa' in file_name:
    model_type = 'fa'
else:
    raise IOError('Wrong data file name. Use one of load_ts.csv, wind_ts.csv, pv_ts.csv')

# Project features
# Base fuzzy variable to balance the imbalanced datasets
# Implementation for ONLY ONE base variable
if model_type == 'pv':
    var_imp = [{'hour': [{'flux': {'mfs': 1}},
                         {'flux': {'mfs': 2}},
                         {'flux': {'mfs': 2}},
                         {'flux': {'mfs': 3}},
                         {'flux': {'mfs': 4}},
                         {'flux': {'mfs': 3}},
                         {'flux': {'mfs': 2}},
                         {'flux': {'mfs': 2}},
                         {'flux': {'mfs': 1}},
                         ]}]
    var_lin = ['flux', 'cloud']
    var_nonreg = []

elif model_type == 'wind':
    var_imp = [{'direction': [{'wind': {'mfs': 3}},
                              {'wind': {'mfs': 3}},
                              {'wind': {'mfs': 3}},
                              {'wind': {'mfs': 3}},
                              ]}]
    var_lin = ['wind', 'direction', 'p_wind']
    var_nonreg = []
elif model_type == 'load':
    var_imp = [
        {'sp_index': [
            {'hour': {'mfs': 8}, 'month': {'mfs': 5}},
            {'hour': {'mfs': 3}, 'month': {'mfs': 2}},
            {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
        ]
        },
        # {'sp_index': [
        #     {'hour': {'mfs': 5}, 'month': {'mfs': 8}},
        #     {'hour': {'mfs': 2}, 'month': {'mfs': 3}},
        #     {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
        # ]
        # },
        # {'sp_index': [
        #     {'hour': {'mfs': 6}, 'Temp': {'mfs': 7}},
        #     {'hour': {'mfs': 2}, 'Temp': {'mfs': 3}},
        #     {'hour': {'mfs': 2}, 'Temp': {'mfs': 2}},
        # ]
        # }
    ]
    var_lin = ['month', 'sp_index', 'dayweek', 'Temp'] + ['load_' + str(k) for k in range(0, 12)]
    var_nonreg = []
elif model_type == 'fa':
    var_imp = [
        {'sp_index': [
            {'month': {'mfs': 4}},
            {'month': {'mfs': 1}},
        ]
        },
        {'sp_index': [
            {'temp_max': {'mfs': 6}},
            {'temp_max': {'mfs': 1}},
        ]
        },
        {'sp_index': [
            {'month': {'mfs': 2}, 'temp_max': {'mfs': 2}},
            {'month': {'mfs': 1}, 'temp_max': {'mfs': 1}},
        ]
        }
    ]
    var_lin = ['month', 'sp_index', 'dayweek', 'temp_max', 'hdd_h2'] + ['Ath24_' + str(i) for i in range(10)]
    var_nonreg = []
else:
    var_imp = {}
    var_lin = []
    var_nonreg = []

if model_type == 'pv':
    variables = ['Cloud', 'Flux', 'Temperature', 'WS']
elif model_type == 'wind':
    variables = ['WS', 'WD', 'Temperature']
elif model_type == 'load':
    variables = ['Cloud', 'Flux', 'WS', 'WD', 'Temperature']
elif model_type == 'fa':
    variables = ['Cloud', 'Flux', 'WS', 'WD', 'Temperature']
else:
    variables = []

# Clustering
n_clusters = 200

thres_act = 0.01
thres_split = 0.8



clustering_train_online=False

add_rules_indvidual=False
import_external_rules = [] # horizon

resampling=True
resampling_thres=200

#Mongo
url='localhost'
port='27017'

resampling_on_var = [['cloud', 0], ['flux', 1], ['Temp', 4], ['wind', 5],
                     ]  # NAME OF THE VARIABLES FOR RESAMPLING AND INDEX OF THE VARIABLES ON 3D DATA NOT NECESSARY FOR WIND AND PV

check_fuzzy_models = False

# Feature selection

fs_status = 'notok'
gen_fz = 20
pop_fz = 50

gen_sk = 20
pop_sk = 50
# RBFNN
gen_rbf = 100
pop_rbf = 50
population = 50
max_iterations = 60000
learning_rate = 0.5e-4
mean_var = 0.005
std_var = 0.005

fine_tuning = False

# CNN
filters = 24
units = 24
hold_prob = 1
pool_size = [2, 1]
h_size = [2048, 512]
cnn_max_iterations = 30000
cnn_learning_rate = 1e-4

mlp_max_iterations = 30000
mlp_learning_rate = 1e-4
