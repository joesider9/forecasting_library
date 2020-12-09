from Fuzzy_clustering.ver_tf2.SKlearn_models import test_grdsearch
from Fuzzy_clustering.ver_tf2.Sklearn_models_skopt import test_skopt
from Fuzzy_clustering.ver_tf2.Sklearn_models_optuna import test_optuna
from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import test_deap
from Fuzzy_clustering.ver_tf2.Feature_selection_boruta import test_boruta
from Fuzzy_clustering.ver_tf2.Feature_selection_permutation import test_fs_permute
from util_database import write_database
from Fuzzy_clustering.ver_tf2.Forecast_model import forecast_model
from sklearn.model_selection import train_test_split
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import pandas as pd
import os, sys

if __name__ == '__main__':
    if sys.platform == 'linux':
        sys_folder = '/media/smartrue/HHD1/George/models/'
    else:
        sys_folder = 'D:/models/'
    project_name = 'APE_net_ver2'
    project_country = 'APE_net_ver2'
    project_owner = '4cast_models'
    path_project = sys_folder + project_owner + '/' + project_country + '/' + project_name
    cluster_dir = path_project +'/Regressor_layer/rule.14'
    data_dir = path_project + '/Regressor_layer/rule.14/data'

    static_data = write_database()
    X = pd.read_csv(os.path.join(data_dir, 'dataset_X.csv'), index_col=0,
                    parse_dates=True, dayfirst=True)
    y = pd.read_csv(os.path.join(data_dir, 'dataset_y.csv'), index_col=0, parse_dates=True, dayfirst=True)
    forecast = forecast_model(static_data, use_db=False)
    forecast.load()

    forecast.sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
    forecast.scale_y = MinMaxScaler(feature_range=(.1, 20)).fit(y.values)
    X = forecast.sc.transform(X)
    y = forecast.scale_y.transform(y)
    # scy = MinMaxScaler(feature_range=(0, 1)).fit(y.reshape(-1,1))
    #
    # y = scy.transform(y.reshape(-1,1)).ravel()
    N, D = X.shape
    n_split = int(np.round(N * 0.85))
    X_test1 = X[n_split + 1:, :]
    y_test1 = y[n_split + 1:]
    X = X[:n_split, :]
    y = y[:n_split]
    X_train, X_test, y_train, y_test = split_continuous(X, y, test_size=0.15, random_state=42)
    cvs = []
    for _ in range(3):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        cvs.append([X_train, y_train, X_val, y_val, X_test, y_test])
    cvs_cache = copy.deepcopy(cvs)
    test_deap(cvs, X_test1,  y_test1, cluster_dir)
    test_optuna(cvs, X_test1,  y_test1, cluster_dir)
    test_skopt(cvs, X_test1,  y_test1, cluster_dir)
    test_grdsearch(cvs, X_test1,  y_test1, cluster_dir)
    test_fs_permute(cvs, X_test1,  y_test1, cluster_dir)
    cvs = copy.deepcopy(cvs_cache)
    test_boruta(cvs, X_test1,  y_test1, cluster_dir)
