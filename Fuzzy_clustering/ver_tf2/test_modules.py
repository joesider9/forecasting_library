import joblib, os
import pandas as pd
import numpy as np
from Fuzzy_clustering.ver_tf2.Models_train_manager import ModelTrainManager

model_path = 'D:/models/my_projects/APE_net_ver1/pv/APE_net/model_ver0'
rule = 'rule.8'

cluster_dir = os.path.join(model_path, 'Regressor_layer/' + rule)
data_path = os.path.join(cluster_dir, 'data')

static_data = joblib.load(os.path.join(model_path, 'static_data.pickle'))
model = ModelTrainManager(path_model=model_path)
model.load()


def split_test_data(X, y, act, X_cnn=np.array([]), X_lstm=np.array([]), test_indices=None):
    N_tot, D = X.shape
    if not test_indices is None:
        X_test = X.loc[test_indices['dates_test']]
        y_test = y.loc[test_indices['dates_test']]
        act_test = act.loc[test_indices['dates_test']]

        X = X.loc[test_indices['dates_train']]
        y = y.loc[test_indices['dates_train']]
        act = act.loc[test_indices['dates_train']]

        if len(X_cnn.shape) > 1:
            X_cnn_test = X_cnn[test_indices['indices_test']]
            X_cnn = X_cnn[test_indices['indices_train']]
        else:
            X_cnn_test = np.array([])

        if len(X_lstm.shape) > 1:
            X_lstm_test = X_lstm[test_indices['indices_test']]
            X_lstm = X_lstm[test_indices['indices_train']]
        else:
            X_lstm_test = np.array([])
    else:
        X_test = pd.DataFrame([])
        y_test = pd.DataFrame([])
        act_test = pd.DataFrame([])
        X_cnn_test = np.array([])
        X_lstm_test = np.array([])

    N_test = X_test.shape[0]
    return X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test

def load_data():
    X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
    y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
    act = pd.read_csv(os.path.join(data_path, 'dataset_act.csv'), index_col=0, header=0, parse_dates=True,
                      dayfirst=True)

    if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
        X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
        if X_cnn.shape[1] == 6:
            X_cnn = X_cnn.transpose([0, 2, 3, 1])
    else:
        X_cnn = np.array([])

    if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
        X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
    else:
        X_lstm = np.array([])
    if os.path.exists(os.path.join(data_path, 'test_indices.pickle')):
        test_indices = joblib.load(os.path.join(data_path, 'test_indices.pickle'))
    else:
        test_indices = None

    return X, y, act, X_cnn, X_lstm, test_indices


def test_combine_module():
    from Fuzzy_clustering.ver_tf2.Combine_module_train import combine_model

    X, y, act, X_cnn, X_lstm, test_indices = load_data()
    X, y, act, X_cnn, X_lstm, X_test, y_test, act_test, X_cnn_test, X_lstm_test = split_test_data(X, y,
                                                                                                       act,
                                                                                                       X_cnn=X_cnn,
                                                                                                       X_lstm=X_lstm,
                                                                                                       test_indices=test_indices)
    comb_model = combine_model(static_data, cluster_dir, model.sc)
    comb_model.istrained = False
    comb_model.train(X_test, y_test, act_test, X_cnn_test, X_lstm_test)

def test_cluster_module():
    from Fuzzy_clustering.ver_tf2.Cluster_train_regressors import cluster_train

    cluster_model = cluster_train(static_data, rule, model.sc)
    cluster_model.istrained=False
    cluster_model.fit()

if __name__ == '__main__':
    # test_combine_module()
    test_cluster_module()