import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d

def rescale(arr, nrows, ncol):
    W, H = arr.shape
    new_W, new_H = (nrows, ncol)
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_H), xrange(new_W))

    return new_arr


class DataSampler(object):
    def __init__(self, static_data, clust, method='ADASYN'):
        self.static_data=static_data
        self.method = method
        self.model_type=static_data['type']
        self.data_variables = self.static_data['data_variables']
        self.n_jobs=static_data['njobs']
        self.cluster_dir=os.path.join(static_data['path_model'], 'Regressor_layer/' +clust)
        self.data_dir = os.path.join(self.cluster_dir, 'data')
        self.istrained = False

        try:
            self.load(self.data_dir)
        except:
            pass

    def stack_3d(self, X, sample):

        if X.shape[0] == 0:
            X = sample
        elif len(sample.shape) != len(X.shape):
            X = np.vstack((X, sample[np.newaxis]))
        else:
            X = np.vstack((X.transpose([3, 0, 1, 2]), sample.transpose([3, 0, 1, 2]))).transpose([1, 2 , 3, 0])
        return X

    def imblearn_fit_var3d(self, X, X_3d, y, variables, random_state=42):

        from Fuzzy_clustering.ver_tf2.Adasyn_var3d import ADASYN

        flag = False
        Std = 0.01
        while (flag == False and Std <= 1):
            try:
                std = np.maximum(Std * np.std(y), 0.2)
                yy = np.digitize(y, np.arange(np.min(y), np.max(y), std), right=True)
                bins = np.arange(np.min(y), np.max(y), std)
                bins = bins[(np.bincount(yy.ravel()) >= 2)[:-1]]
                yy = np.digitize(y, bins, right=True)
                # if Std==0.01 and np.max(yy)!=0:
                #     strategy = {cl:int(100*X.shape[0]/np.max(yy)) for cl in np.unique(yy)}
                # else:
                strategy = "auto"
                if np.unique(yy).shape[0] == 1:
                    X2 = X
                    X_3d2 = X_3d
                    yy2 = y
                    return X2, yy2
                if np.any(np.bincount(yy.ravel()) < 2):
                    for cl in np.where(np.bincount(yy.ravel()) < 2)[0]:
                        X = X[np.where(yy != cl)[0]]
                        X_3d = X_3d[np.where(yy != cl)[0]]
                        y = y[np.where(yy != cl)[0]]
                        yy = yy[np.where(yy != cl)[0]]
                sm = ADASYN(sampling_strategy=strategy, random_state=random_state,variables=variables,
                                n_neighbors=np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)


                X2, X_3d2, yy2 = sm.fit_resample(X, X_3d, yy.ravel(), y.ravel())

                flag = True
            except:
                Std *= 10
        if flag == True:
            return X2, X_3d2, yy2
        else:
            return X, X_3d, y

    def imblearn_fit_var2d(self, X, X_3d, y, variables, variables_3d, random_state=42):

        from Fuzzy_clustering.ver_tf2.Adasyn_var2d import ADASYN

        flag = False
        Std = 0.01
        while (flag == False and Std <= 1):
            try:
                std = np.maximum(Std * np.std(y), 0.2)
                yy = np.digitize(y, np.arange(np.min(y), np.max(y), std), right=True)
                bins = np.arange(np.min(y), np.max(y), std)
                bins = bins[(np.bincount(yy.ravel()) >= 2)[:-1]]
                yy = np.digitize(y, bins, right=True)
                # if Std==0.01 and np.max(yy)!=0:
                #     strategy = {cl:int(100*X.shape[0]/np.max(yy)) for cl in np.unique(yy)}
                # else:
                strategy = "auto"
                if np.unique(yy).shape[0] == 1:
                    X2 = X
                    X_3d2 = X_3d
                    yy2 = y
                    return X2, X_3d2, yy2
                if np.any(np.bincount(yy.ravel()) < 2):
                    for cl in np.where(np.bincount(yy.ravel()) < 2)[0]:
                        X = X[np.where(yy != cl)[0]]
                        X_3d = X_3d[np.where(yy != cl)[0]]
                        y = y[np.where(yy != cl)[0]]
                        yy = yy[np.where(yy != cl)[0]]
                sm = ADASYN(sampling_strategy=strategy, random_state=random_state,variables=variables, variables_3d=variables_3d,
                                n_neighbors=np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)


                X2, X_3d2, yy2 = sm.fit_resample(X, X_3d, yy.ravel(), y.ravel())

                flag = True
            except:
                Std *= 10
        if flag == True:
            return X2, X_3d2, yy2
        else:
            return X, X_3d, y

    def imblearn_sampling(self, X=np.array([]), y=np.array([]), act=np.array([]), X_cnn=np.array([]),
                          X_lstm=np.array([])):

        X_org = X.copy(deep=True)
        y_org = y.copy(deep=True)
        y = y.values
        act = act.values

        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        if len(act.shape) == 1:
            act = act[:, np.newaxis]

        if self.model_type in ['pv', 'wind']:
            var_resample = []
            var_resample_3d = []
            for variable in self.static_data['resampling_on_var']:
                var_resample += [i for i, var in enumerate(X.columns) if
                                 str.lower(variable[0]) in str.lower(var)]

            X2, X_cnn, y2 = self.imblearn_fit_var3d(
                X.values, X_cnn, y, var_resample)
        elif self.model_type in {'fa', 'load'}:
            var_resample = []
            var_resample_3d = []
            for variable in self.static_data['resampling_on_var']:
                var_resample += [i for i, var in enumerate(X.columns) if
                                str.lower(variable[0]) in str.lower(var)]
                if not variable[1] in var_resample_3d:
                    var_resample_3d.append(variable[1])

            X2, X_lstm, y2 = self.imblearn_fit_var2d(
                X.values, X_lstm, y, var_resample, var_resample_3d)

        else:
            raise ValueError('wrong type model for resampling')
        X2 = pd.DataFrame(X2, columns=X_org.columns)
        return X2, y2, X_cnn, X_lstm

    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'sampler.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'sampler.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                dict = {}
                for k in tmp_dict.keys():
                    if k in ['logger', 'static_data']:
                        dict[k] = tmp_dict[k]
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open sampler')
        else:
            raise ImportError('Cannot find sampler' )

    def save(self, pathname):
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        f = open(os.path.join(pathname, 'sampler.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

