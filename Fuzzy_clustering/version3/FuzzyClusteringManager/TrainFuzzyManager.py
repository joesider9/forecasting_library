import os, joblib, pickle
import pandas as pd
import numpy as np
from Fuzzy_clustering.version3.FuzzyClusteringManager.Clusterer import clusterer
from Fuzzy_clustering.version3.FuzzyClusteringManager.Clusterer_optimize_deep import cluster_optimize

class FuzzyManager():
    def __init__(self,static_data):
        self.istrained = False
        self.static_data = static_data
        self.rated = static_data['rated']
        self.fuzzy_path = self.static_data['path_fuzzy_models']
        self.fuzzy_file = self.static_data['clustering']['cluster_file']
        self.model_type = self.static_data['type']
        try:
            self.load()
        except:
            pass

    def load_data(self):
        data_path = self.static_data['path_data']
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        return X, y

    def train_fuzzy_clustering(self):
        X, y = self.load_data()
        if y.isna().any().values[0]:
            X = X.drop(y.index[np.where(y.isna())[0]])
            y = y.drop(y.index[np.where(y.isna())[0]])

        N, D = X.shape
        sc = joblib.load(os.path.join(self.static_data['path_data'], 'X_scaler.pickle'))
        X1 = pd.DataFrame(sc.transform(X.values), columns=X.columns, index=X.index)

        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))

        y1 = pd.DataFrame(scale_y.transform(y.values), columns=y.columns, index=y.index)

        if self.static_data['type'] == 'fa':
            n_split = int(np.round(N * 0.7))
            X_test = X1.iloc[n_split + 1:]
            y_test = y1.iloc[n_split + 1:]

            X_train = X1.iloc[:n_split]
            y_train = y1.iloc[:n_split]
            optimizer = cluster_optimize(self.static_data)
            if optimizer.istrained==False:
                if self.rated is None:
                    rated = None
                else:
                    rated = self.rated
                optimizer.run(X_train, y_train, X_test, y_test, rated, num_samples=200, n_ratio=0.8, ngen=300)
        else:
            n_split = int(np.round(N * 0.7))
            X_test = X1.iloc[n_split + 1:]
            y_test = y1.iloc[n_split + 1:]

            X_train = X1.iloc[:n_split]
            y_train = y1.iloc[:n_split]
            optimizer = cluster_optimize(self.static_data)
            if optimizer.istrained == False:
                if self.rated is None:
                    rated = None
                else:
                    rated = self.rated
                optimizer.run(X_train, y_train, X_test, y_test, rated)
        self.clusterer = clusterer(self.static_data)
        self.istrained = True
        self.save()
        return 'Done'

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data']:
                dict[k] = self.__dict__[k]
        return dict

    def load(self):
        if os.path.exists(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict={}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'fuzzy_file', 'fuzzy_path', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open rule fuzzy model')
        else:
            raise ImportError('Cannot find rule fuzzy model')


    def save(self):
        f = open(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'fuzzy_file', 'fuzzy_path', 'n_jobs']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()