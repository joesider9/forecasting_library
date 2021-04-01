import getpass
import os
import sys
RECREATE_NWP_FILES = "recreate_nwp_files"
RECREATE_DATASETS = "recreate_datasets"
WEATHER_IN_DATA = "weather_in_data"
ENABLE_TRANSFER_LEARNING = "enable_transfer_learning"

PROJECT_OWNER = 'project_owner'
PROJECTS_GROUP = 'projects_group'
AREA_GROUP = 'area_group'

DATA_FILE_NAME = 'data_file_name'
NWP_FOLDER = 'nwp_folder'
SYS_FOLDER = "sys_folder"
NWP = 'nwp'

DOCKER = 'docker'
PROJECT_SUFFIX = 'project_suffix'
VERSION_GROUP = 'version_group'
VERSION_MODEL = 'version_model'
N_JOBS = 'n_jobs'
N_JOBS_FUZZY = 'n_jobs_fuzzy'
N_JOBS_FEAT_SEL = "n_jobs_feat_sel"
INNER_JOBS_FEAT_SEL = 'inner_jobs_feat_sel'
N_JOBS_RBF_OLS = 'n_jobs_rbf_ols'
N_JOBS_RBF_NN = "n_jobs_rbf_nn"
N_JOBS_SKLEARN = "n_jobs_sklearn"
N_JOBS_LSTM = "n_jobs_lstm"
N_JOBS_CNN_3D = "n_jobs_cnn_3d"
N_JOBS_CNN = "n_jobs_cnn"
N_JOBS_MLP = "n_jobs_mlp"
INTRA_OP = 'intra_op'
N_GPUS = 'n_gpus'
EVALUATION = "evaluation"  # date or None
NWP_MODEL = "nwp_model"  # skiron, ecmwf or gfs
NWP_RESOLUTION = "nwp_resolution"  # 0.05 or 0.04
TRANSFER_LEARNING = "transfer_learning"
TL_PROJECT = 'tl_project'
FEATURE_SELECTION_METHOD = 'feature_selection_method'
DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
COMPRESS_DATA = 'compress_data'  # PCA or dense
COMBINE_METHODS = "combine_methods"
EXCLUDE_METHOD_COMBINE = "exclude_method_combine"
IS_GLOBAL = "is_global"  # True
IS_FUZZY = "is_fuzzy"  # False
CLUSTER_FILE = 'cluster_file'
IS_CLUSTERING_TRAINED = "is_clustering_trained"
IS_PROBABILISTIC = "is_probabilistic"
PROJECT_METHODS = "project_methods"
SKLEARN_OPTIMIZER = 'sklearn_optimizer'
SKLEARN = 'sklearn'
GLOBAL = 'global'
HYPER_PARAMETER_TUNING = 'hyperparameter_tuning'
# forecast types
MODEL_TYPE = 'model_type'
LOAD = 'load'
GAS = 'gas'
WIND = 'wind'
PV = 'pv'
TARGET = 'target'
SKLEARN_MANAGER_PICKLE = 'sklearn_manager.pickle'
FEATURE_SELECTION = 'feature_selection'
PERM = 'perm'
COMBINE = 'combine'
REGRESSOR_LAYER = 'regressor_layer'
# variables
CLOUD = 'cloud'
FLUX = 'flux'
TEMPERATURE = 'temperature'
HUMID = 'humid'
WS = 'ws'  # wind speed
WD = 'wd'  # wind direction

U_WIND = 'u_wind'
V_WIND = 'v_wind'

TEMP_MAX = 'temp_max'
TEMP_MIN = 'temp_min'
TEMP_MEAN = 'temp_mean'
TEMP_MONTH = 'temp_month'
TEMP_SP_DAYS = 'temp_sp_days'

TEMPERATURE_SORT = 'temp'
CLOUD_SORT = 'cl'
FLUX_SORT = 'fl'

SP_INDEX = 'sp_index'
HOUR = 'hour'
MONTH = 'month'
DAY_WEEK = 'day_week'
HDD_H2 = 'hdd_h2'
P_WIND = 'p_wind'
DIRECTION = 'direction'

# fuzzy parameters

VAR_IMP = "var_imp"
VAR_LIN = "var_lin"
VAR_NON_REG = "var_non_reg"
MFS = 'mfs'

DATA_VARIABLES = 'data_variables'
# nwp providers
GFS = 'gfs'
ECMWF = 'ecmwf'
SKIRON = 'skiron'
SKIRON_LOW = 'skiron_low'
# compress type
DENSE = 'dense'
PCA = 'pca'
PATH_NWP = 'path_nwp'
PATH_GROUP = 'path_group'
PATH_NWP_GROUP = 'path_nwp_group'

# feature selection algoriithm
BORUTA = 'boruta'
LINEAR_SEARCH = 'linear_search'

# fit methods
RLS = 'rls'
BCP = 'bcp'
MLP = 'mlp'
BAYESIAN_RIDGE = 'bayesian_ridge'
ELASTIC_NET = 'elastic_net'
RIDGE = 'ridge'
AVERAGE = 'average'

# REGRESSOR TYPES

RBF_ALL_CNN = 'rbf_all_cnn'
RBF_ALL = 'rbf_all'
CNN = 'cnn'
LSTM = 'lstm'
MLP_3D = 'mlp_3d'
SVM = 'svm'
NU_SVM = 'nu_svm'
RF = 'rf'
XGB = 'xgb'
CLUSTERING = 'clustering'
RBF = 'rbf'

DEAP = 'deap'
NOTOK = 'notok'

N_CLUSTERS = "n_clusters"
THRESH_ACT = "thresh_act"
THRESH_SPLIT = "thresh_split"
TRAIN_ONLINE = "train_online"
ADD_RULES_INDIVIDUAL = "add_rules_individual"
IMPORT_EXTERNAL_RULES = "import_external_rules"
RESAMPLING = "resampling"
RESAMPLING_THRESH = "resampling_thresh"
BEST_FUZZY_PICKLE = 'best_fuzzy_pickle'

# Mongo
URL = 'localhost'
PORT = '27017'

RESAMPLING_ON_VAR = 'resampling_on_var'

CHECK_FUZZY_MODELS = "check_fuzzy_models"

# Feature selection
POP = 'pop'
GEN = 'gen'
FS_STATUS = 'fs_status'
GEN_FZ = "gen_fz"
POP_FZ = 'pop_fz'
GEN_SK = 'gen_sk'
POP_SK = 'pop_sk'
ΝUM_GEN = 'num_gen'

# RBFNN
GEN_RBF = 'gen_rbf'
POP_RBF = 'pop_rbf'
POPULATION = 'population'
MAX_ITERATIONS = 'max_iterations'
LEARNING_RATE = 'learning_rate'
MEAN_VAR = 'mean_var'
STD_VAR = 'std_var'
FINE_TUNING = 'fine_tuning'

# CNN
FILTERS = 'filters'
UNITS = 'units'
HOLD_PROB = 'hold_prob'
POOL_SIZE = 'pool_size'
H_SIZE = 'h_size'
CNN_MAX_ITERATIONS = 'cnn_max_iterations'
CNN_LEARNING_RATE = 'cnn_learning_rate'

# LSTM
LSTM_MAX_ITERATIONS = 'lstm_max_iterations'
LSTM_LEARNING_RATE = 'lstm_learning_rate'
DENSE_LAYER_DIM = 'dense_layer_dim'
MLP_MAX_ITERATIONS = 'MLP_MAX_ITERATIONS'
MLP_LEARNING_RATE = 'MLP_LEARNING_RATE'
DAYS_OFFSET = 'days_offset'
# Projects shared information


# project specific information
ID = '_id'
OWNER = 'owner'
PROJECT_GROUP = 'project_group'
LOCATION = 'location'
AREA = 'area'
RATED_POWER = 'rated_power'
PATH_PROJECT = 'path_project'
PATH_MODEL = 'path_model'
PATH_BACKUP = 'path_backup'
PATH_DATA = 'path_data'
PATH_FUZZY_MODELS = 'path_fuzzy_models'
RUN_ON_PLATFORM = 'run_on_platform'
STATIC_DATA = 'static_data'
STATIC_DATA_PICKLE = 'static_data.pickle'
STATIC_DATA_TXT = 'static_data.txt'
STATIC_DATA_PROJECTS_PICKLE = 'static_data_projects.pickle'
APE_NET = 'APE_net'

LINUX = 'linux'
DONE = 'done'

LAT = 'lat'
LONG = 'long'
TRAIN_SPLIT = 'train_split'

# Project paths

MODEL_VERSION = 'model_ver'
BACKUP_MODEL = 'backup_models'
FUZZY_MODELS = 'fuzzy_models'
DATA = 'data'

# Projects Name

TOTAL = 'total'
ST_MIGUEL = 'St_Miguel'

CNN_3D = 'cnn_3d'
LSTM_3D = 'lstm_3d'
RBF_NN = 'rbf_nn'
RBF_CNN = 'rbf_cnn'
