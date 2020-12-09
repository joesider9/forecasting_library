#Folder locations
# Folder locations
import sys, os
if sys.platform == 'linux':
    njobs = 8
    gpus = ['/device:GPU:0','/device:GPU:1']
    sys_folder = '/media/smartrue/HHD1/George/models/'
    cnn_path_temp = sys_folder
    data_file_name = '/home/smartrue/PycharmProjects/FA_Forecast_ver2/data/fa_ts.csv'
else:
    gpus = ['/device:GPU:0']
    njobs = 3
    sys_folder = 'D:/models/'
    cnn_path_temp = sys_folder
    data_file_name = 'D:/Dropbox/current_codes/PycharmProjects/FA_Forecast_ver2/data/fa_ts.csv'

project_owner = 'my_projects'
projects_group = 'Greece'
area_group = [[33.9426055, 18.929342], [42.047856, 30.0400288]]
version = 1
Evaluation = '01032020 01:00'

AUTO_COORDS_FIND = False
RECREATE_NWP_FILES = False
RECREATE_DATASETS = False
is_clustering_trained=False
ENABLE_TRANSFER_LEARNING = False
cluster_file='best_fuzzy.pkl'
weather_in_data = False

file_name = os.path.basename(data_file_name)

NWP_model = 'ecmwf' #Skiron, ECMWF
NWP_resolution = 0.1 # 0.05 or 0.04

compress_data = 'dense' #PCA or dense

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

#Mongo
url='localhost'
port='27017'

#Project features
if model_type == 'pv':
    var_imp={'hour':{'mfs':6, 'range':[-0.04, 1.04]},
             'flux':{'mfs':4, 'range':[-0.04, 1.04]},
             # 'cloud':{'mfs':2, 'range':[-0.04, 1.04]},
             # 'month':{'mfs':4, 'range':[-0.04, 1.04]}
             }
    var_lin=['flux', 'cloud']
    var_nonreg=[]

elif model_type == 'wind':
    var_imp = {'wind':{'mfs':4, 'range':[-0.04, 1.04]},
               'direction':{'mfs':3, 'range':[-0.04, 1.04]},
               # 'p_wind':{'mfs':2, 'range':[-0.04, 1.04]}
             }
    var_lin = ['wind', 'direction', 'p_wind']
    var_nonreg=[]
elif model_type == 'load':
    var_imp = {'hour':{'mfs':6, 'range':[-0.04, 1.04]},
               'temp_max':{'mfs':4, 'range':[-0.04, 1.04]},
               'sp_index':{'mfs':2, 'range':[-0.04, 1.04]},
               'month':{'mfs':4, 'range':[-0.04, 1.04]},
               'load':{'mfs':3, 'range':[-0.04, 1.04]}
             }
    var_lin = ['month', 'sp_index', 'dayweek', 'Temp'] + ['other.' + str(k) for k in range(0, 8)]
    var_nonreg = []
elif model_type == 'fa':
    var_imp = {'Temp':{'mfs':4, 'range':[-0.04, 1.040]},
               'sp_index':{'mfs':2, 'range':[-0.04, 1.04]},
               'month':{'mfs':4, 'range':[-0.04, 1.04]},
               }
    var_lin = ['month', 'sp_index', 'dayweek', 'Temp'] + ['Ath24_'+str(i) for i in range(10)]
    var_nonreg = []
else:
    var_imp = {}
    var_lin = []
    var_nonreg = []
project_methods={}

project_methods['ML_RBF_ALL_CNN']={
        'Global':True,
        'status': 'train', # not_trained, train,trained
        }
project_methods['ML_RBF_ALL']={
    'Global': False,
    'status': 'not_trained', # not_trained, train,trained
        }
project_methods['ML_CNN_3d']={
    'Global': False,
    'status': 'not_trained', # not_trained, train,trained
        }
project_methods['ML_LSTM_3d']={
    'Global': True,
    'status': 'train', # not_trained, train,trained
        }
project_methods['ML_SVM']={
    'Global': True,
    'status': 'train', # not_trained, train,trained
    'sklearn_method' :'deap' #deap, optuna, skopt, grid_search
                                    }
project_methods['ML_NUSVM']={
    'Global': False,
    'status': 'not_trained', # not_trained, train,trained
    'sklearn_method' :'deap' #deap, optuna, skopt, grid_search
                                    }
project_methods['ML_MLP']={
    'Global': True,
    'status': 'train', # not_trained, train,trained
    'sklearn_method' :'deap' #deap, optuna, skopt, grid_search
                                    }
project_methods['ML_RF']={
    'Global': True,
    'status': 'train', # not_trained, train,trained
    'sklearn_method' :'deap' #deap, optuna, skopt, grid_search
        }
project_methods['ML_XGB']={
    'Global': True,
    'status': 'train',  # not_trained, train,trained
    'sklearn_method': 'deap'  # deap, optuna, skopt, grid_search
                           }

feature_selection_method = 'boruta' #boruta or permutation

combine_methods=['rls', 'bcp', 'mlp', 'bayesian_ridge', 'elastic_net', 'ridge', 'average'] #'rls', 'bcp', 'mlp', 'bayesian_ridge', 'elastic_net', 'ridge', 'average'
exclude_method_combine = []


is_Fuzzy = True

transfer_learning = False
tl_project = ''


if model_type == 'pv':
    variables = ['Cloud', 'Flux', 'Temperature', 'WS']
elif model_type == 'wind':
    variables = ['WS', 'WD', 'Temperature']
elif model_type == 'load':
    raise NotImplementedError()
elif model_type == 'fa':
    variables = ['Cloud', 'Flux', 'WS', 'WD', 'Temperature']
else:
    variables = []

#Clustering
n_clusters=200

thres_act=0.01
thres_split=0.8




clustering_train_online=False

add_rules_indvidual=False
import_external_rules = [] # horizon

resampling=True
resampling_thres=200
resampling_on_var = [['cloud',0], ['flux',[1,2,3]], ['Temp',4], ['wind',5]] #NAME OF THE VARIABLES FOR RESAMPLING AND INDEX OF THE VARIABLES ON 3D DATA

check_fuzzy_models = False

# Feature selection

fs_status='notok'
#RBFNN
max_iterations=100000
learning_rate=0.5e-4
mean_var=0.005
std_var=0.005

fine_tuning = False


#CNN
filters=24
pool_size=[2,1]
h_size=[2048,512]
cnn_max_iterations= 30000
cnn_learning_rate=1e-5
#
# data=['APE_ekk','APE_temp']
