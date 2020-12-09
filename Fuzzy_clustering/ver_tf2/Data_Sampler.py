import os
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous
from Fuzzy_clustering.ver_tf2.NWP_sampler import nwp_sampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time, logging, warnings, joblib


def rescale(arr, nrows, ncol):
    W, H = arr.shape
    new_W, new_H = (nrows, ncol)
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_H), xrange(new_W))

    return new_arr


class DataSampler(object):
    def __init__(self, static_data, clust, scale_x, method='ADASYN'):
        self.static_data=static_data
        self.method = method
        self.scale_x = scale_x
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

    def load_data(self):
        data_path = self.data_dir
        X = pd.read_csv(os.path.join(data_path, 'dataset_X.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        y = pd.read_csv(os.path.join(data_path, 'dataset_y.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)
        act = pd.read_csv(os.path.join(data_path, 'dataset_act.csv'), index_col=0, header=0, parse_dates=True, dayfirst=True)

        if os.path.exists(os.path.join(data_path, 'dataset_cnn.pickle')):
            X_cnn = joblib.load(os.path.join(data_path, 'dataset_cnn.pickle'))
            if X_cnn.shape[1]==6:
                X_cnn = X_cnn.transpose([0, 2, 3, 1])
        else:
            X_cnn = np.array([])

        if os.path.exists(os.path.join(data_path, 'dataset_lstm.pickle')):
            X_lstm = joblib.load(os.path.join(data_path, 'dataset_lstm.pickle'))
        else:
            X_lstm = np.array([])

        return X, y, act, X_cnn, X_lstm

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


                X2, X_3d, yy2 = sm.fit_resample(X, X_3d, yy.ravel(), y.ravel())

                flag = True
            except:
                Std *= 10
        if flag == True:
            return X2, X_3d, yy2
        else:
            raise RuntimeError('Cannot make resampling ')

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
                    yy2 = y
                    return X2, yy2
                if np.any(np.bincount(yy.ravel()) < 2):
                    for cl in np.where(np.bincount(yy.ravel()) < 2)[0]:
                        X = X[np.where(yy != cl)[0]]
                        X_3d = X_3d[np.where(yy != cl)[0]]
                        y = y[np.where(yy != cl)[0]]
                        yy = yy[np.where(yy != cl)[0]]
                sm = ADASYN(sampling_strategy=strategy, random_state=random_state,variables=variables, variables_3d=variables_3d,
                                n_neighbors=np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)


                X2, X_3d, yy2 = sm.fit_resample(X, X_3d, yy.ravel(), y.ravel())

                flag = True
            except:
                Std *= 10
        if flag == True:
            return X2, X_3d, yy2
        else:
            raise RuntimeError('Cannot make resampling ')

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

    def create_sample(self, data, dates):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            dataset_X = pd.concat([dataset_X, pd.DataFrame(dates, columns=['hour','month'])])
        for var in sorted(self.data_variables):
            if ((var == 'WS') and (self.static_data['type']=='wind')) or ((var == 'Flux') and (self.static_data['type']=='pv')):
                X0 = np.squeeze(data[var + '_prev'])
                X0_level0 = X0[:, 2, 2].reshape(-1, 1)

                X1 = np.squeeze(data[var])

                X1_level1 = X1[:, 2, 2].reshape(-1, 1)

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = np.percentile(X1_level3d, [5, 50, 95])

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = np.mean(X1_level3u)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = np.percentile(X1_level4d, [5, 50, 95])

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = np.percentile(X1_level4u, [5, 50, 95])

                X2 = np.squeeze(data[var + '_next'])

                X2_level0 = X2[:, 2, 2].reshape(-1, 1)

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col =  ['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(1)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X0_level0, X2_level0, X1_level1, X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                X[np.where(X < 0)] = 0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X,  columns=col)], axis=1)

            elif var in {'WD', 'Cloud'}:
                X1 = np.squeeze(data[var])

                X1_level1 = X1[:, 2, 2].reshape(-1, 1)

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = np.percentile(X1_level3d, [5, 50, 95])

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = np.mean(X1_level3u)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = np.percentile(X1_level4d, [5, 50, 95])

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = np.percentile(X1_level4u, [5, 50, 95])

                var_name = 'cloud' if var=='Cloud' else 'direction'
                var_sort = 'cl' if var=='Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(1)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1, X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                if var=='Cloud':
                    X[np.where(X<0)] = 0
                    X[np.where(X>100)] = 100
                else:
                    X[np.where(X<0)] = 0
                    X[np.where(X>360)] = 360

                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type']=='pv')):
                X2 = np.squeeze(data[var])
                X2_level0 = X2[2, 2]
                var_name = 'Temp' if var == 'Temperature' else 'wind'

                col = [var_name]

                X = X2_level0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, columns=col)], axis=1)
            else:
                continue
        return dataset_X

    def create_sample_pca(self, data, dates):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':

            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(dates, index=dates, columns=['hour', 'month'])])
        for var in sorted(self.data_variables):
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                X0 = np.squeeze(data[var + '_prev'])
                X0_level0 = X0[:, 2, 2].reshape(-1, 1)

                X1 = np.squeeze(data[var])

                X1_level1 = X1[:, 2, 2].reshape(-1, 1)

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = self.PCA_transform(X1_level3d, 3, level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = self.PCA_transform(X1_level3u, 2, level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = self.PCA_transform(X1_level4d, 3, level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = self.PCA_transform(X1_level4u, 3, level)

                X2 = np.squeeze(data[var + '_next'])

                X2_level0 = X2[:, 2, 2].reshape(-1, 1)

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col =['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X0_level0, X2_level0, X1_level1, X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                X[np.where(X < 0)] = 0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, columns=col)], axis=1)

            elif var in {'WD', 'Cloud'}:
                X1 = np.squeeze(data[var])

                X1_level1 = X1[: ,2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = self.PCA_transform(X1_level3d, 3, level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = self.PCA_transform(X1_level3u, 2, level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = self.PCA_transform(X1_level4d, 3, level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = self.PCA_transform(X1_level4u, 3, level)

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1.reshape(-1,1), X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                if var == 'Cloud':
                    X[np.where(X < 0)] = 0
                    X[np.where(X > 100)] = 100
                else:
                    X[np.where(X < 0)] = 0
                    X[np.where(X > 360)] = 360

                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                X2 = np.squeeze(data[var])
                X2_level0 = X2[: ,2, 2]
                var_name = 'Temp' if var == 'Temperature' else 'wind'

                col = [var_name]

                X = X2_level0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, columns=col)], axis=1)
            else:
                continue
        return dataset_X

    def PCA_transform(self, data, components, level):

        fname = os.path.join(self.static_data['path_data'], 'kpca_' + level + '.pickle')
        models = joblib.load(fname)
        data_scaled = models['scaler'].transform(data)
        data_compress = models['kpca'].transform(data_scaled)

        return data_compress

    def nwp_dl_sampling_obsolete(self, X=np.array([]), y=np.array([]), act=np.array([]), X_cnn=np.array([]), X_lstm=np.array([])):
        if len(X.shape) <= 1:
            X, y, act, X_cnn, X_lstm = self.load_data()
        if len(X_lstm.shape) > 1:
            raise NotImplementedError('TODO the same for X_lstm as X_cnn')
        X_org = pd.DataFrame(self.scale_x.inverse_transform(X.values), index=X.index, columns=X.columns)
        y = y.values
        act = act.values
        try:
            self.load(self.data_dir)
        except:
            pass


        self.columns_dl = []
        sampler = nwp_sampler(self.static_data)

        self.x2_dl = np.array([])
        self.y2_dl = np.array([])
        hour_val = np.array([])
        if self.model_type == 'pv':
            for h in np.unique(X_org['hour']):
                ind = np.where(X_org['hour'] == h)[0]
                x2, y2 = self.imblearn_fit(
                    pd.concat([X[sampler.columns].iloc[ind], X_org['month'].iloc[ind]], axis=1).values, y[ind])
                cl_ind = np.where(np.array(sampler.columns)=='cloud')
                x2[np.where(x2[:, cl_ind] > 1), cl_ind] = 1
                if len(self.x2_dl.shape)>1:
                    self.x2_dl = np.vstack((self.x2_dl, x2))
                    self.y2_dl = np.vstack((self.y2_dl, y2.reshape(-1, 1)))
                    hour_val =np.vstack((hour_val, h*np.ones([x2.shape[0],1])))
                else:
                    self.x2_dl = x2
                    self.y2_dl = y2.reshape(-1, 1)
                    hour_val = h*np.ones([x2.shape[0],1])



            # x2, self.y2 = self.imblearn_fit(pd.concat([X[sampler.columns], X_org[['hour', 'month']]], axis=1).values, y)
            mths = np.unique(X_org['month'].values)
            rg = np.min(np.diff(np.unique(X_org['month'].values))) / 2
            for m in mths:
                self.x2_dl[np.isclose(self.x2_dl[:, -1], m, atol=rg), -1] = m
            self.x_dates_dl = np.hstack((hour_val,self.x2_dl[:,-1].reshape(-1,1))).astype('int')
            self.x2_dl = self.x2_dl[:,:-1]
        else:
            self.x2_dl, self.y2_dl = self.imblearn_fit(X[sampler.columns].values, y)
            self.x_dates_dl = []

        i = 0
        self.nwp_dl = dict()
        N = self.x2_dl.shape[0]
        for var in sorted(self.data_variables):
            if ((var == 'WS') and (self.model_type == 'wind')) or ((var == 'Flux') and (self.model_type == 'pv')):

                self.columns_dl.append('flux' if var == 'Flux' else 'wind')
                model1_name = 'flux' if var == 'Flux' else 'wind'
                if sampler.istrained:
                    var_name = 'flux' if var == 'Flux' else 'wind'
                    self.columns_dl.extend(['p_' + var_name] + [var_name] + ['n_' + var_name])
                    model1_name = 'p_' + var_name
                    max_val = 1000 if var == 'Flux' else 30
                    self.nwp_dl[var + '_prev'] = max_val * sampler.run_models(self.x2_dl, model1_name)
                    self.nwp_dl[var + '_prev'][np.where(self.nwp_dl[var + '_prev'] < 0)] = 0

                    model1_name = var_name
                    self.nwp_dl[var] = max_val * sampler.run_models(self.x2_dl, model1_name)
                    self.nwp_dl[var][np.where(self.nwp_dl[var] < 0)] = 0

                    model1_name = 'n_' + var_name
                    self.nwp_dl[var + '_next'] = max_val * sampler.run_models(self.x2_dl, model1_name)
                    self.nwp_dl[var + '_next'][np.where(self.nwp_dl[var + '_next'] < 0)] = 0

                i += 3

            elif var in {'WD', 'Cloud'}:

                self.columns_dl.append('cloud' if var == 'Cloud' else 'direction')
                model2_name = 'cloud' if var == 'Cloud' else 'direction'
                max_val = 100 if var == 'Cloud' else 360
                if sampler.istrained:
                    self.nwp_dl[var] = max_val * sampler.run_models(self.x2_dl, model2_name)
                    self.nwp_dl[var][np.where(self.nwp_dl[var] < 0)] = 0
                    self.nwp_dl[var][np.where(self.nwp_dl[var] > max_val)] = max_val
                i += 1
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.model_type == 'pv')):
                self.columns_dl.append('Temp' if var == 'Temperature' else 'wind')
                model2_name = 'Temp' if var == 'Temperature' else 'wind'
                max_val = 320 if var == 'Temperature' else 30

                if sampler.istrained:
                    self.nwp_dl[var] = max_val * sampler.run_models(self.x2_dl, model2_name)
                    self.nwp_dl[var][np.where(self.nwp_dl[var] < 0)] = 0
                i += 1
            else:
                i += 1

        if len(X_cnn.shape) == 4:
            self.X_cnn_dl = np.array([])
            for var in sorted(self.data_variables):
                if ((var == 'WS') and (self.model_type == 'wind')) or ((var == 'Flux') and (self.model_type == 'pv')):
                    self.X_cnn_dl = self.stack_3d(self.X_cnn_dl, self.nwp_dl[var + '_prev'])
                    self.X_cnn_dl = self.stack_3d(self.X_cnn_dl, self.nwp_dl[var])
                    self.X_cnn_dl = self.stack_3d(self.X_cnn_dl, self.nwp_dl[var + '_next'])
                elif var in {'WD', 'Cloud'}:

                    self.X_cnn_dl = self.stack_3d(self.X_cnn_dl, self.nwp_dl[var])

                elif (var in {'Temperature'}) or ((var == 'WS') and (self.model_type == 'pv')):
                    self.X_cnn_dl = self.stack_3d(self.X_cnn_dl, self.nwp_dl[var])

        if self.static_data['compress_data'] == 'dense':
            self.X_dl = self.create_sample(self.nwp_dl, self.x_dates_dl)
        else:
            self.X_dl = self.create_sample_pca(self.nwp_dl, self.x_dates_dl)
        self.y2_dl = self.y2_dl[np.where(~pd.isnull(self.X_dl).any(axis=1))[0].ravel()]

        self.X_dl = self.X_dl.drop(np.where(pd.isnull(self.X_dl).any(axis=1))[0].ravel())
        self.X_dl = pd.DataFrame(self.scale_x.transform(self.X_dl.values), index=self.X_dl.index, columns=self.X_dl.columns)
        # self.X = pd.concat([self.X, X.reset_index()], axis=0, join='inner', ignore_index=False, copy=True).reset_index(drop=True)
        # self.y2 = np.vstack((self.y2.reshape(-1, 1), y))
        self.y2_dl = self.y2_dl.reshape(-1, 1)
        # if len(X_cnn.shape) == 4:
        #     self.X_cnn = np.vstack((self.X_cnn, X_cnn))


        return self.X_dl, self.y2_dl, self.X_cnn_dl

    def nwp_sampling_obsolete(self, X=np.array([]), y=np.array([]), act=np.array([]), X_cnn=np.array([]), X_lstm=np.array([])):
        if len(X.shape) <= 1:
            X, y, act, X_cnn, X_lstm = self.load_data()
        if len(X_lstm.shape) > 1:
            raise NotImplementedError('TODO the same for X_lstm as X_cnn')
        X_org = pd.DataFrame(self.scale_x.inverse_transform(X.values), index=X.index, columns=X.columns)
        y = y.values
        act = act.values
        try:
            self.load(self.data_dir)
        except:
            pass


        self.columns = []

        hour_val = np.array([])
        if self.model_type == 'pv':
            cnn_shape = X_cnn.shape
            x_cnn = np.copy(X_cnn)
            x_cnn = x_cnn.reshape(-1, np.prod(cnn_shape[1:]))
            scale_cnn = MinMaxScaler()
            x_cnn = scale_cnn.fit_transform(x_cnn)
            x_cnn = np.hstack((x_cnn, X_org[['hour', 'month']].values))
            x_cnn2, y_cnn2 = self.imblearn_fit(x_cnn, y)
            mths = np.unique(X_org['month'].values)
            rg = np.min(np.diff(np.unique(X_org['month'].values))) / 2
            for m in mths:
                x_cnn2[np.isclose(x_cnn2[:, -1], m, atol=rg), -1] = m
            hrs = np.unique(X_org['hour'].values)
            rg = np.min(np.diff(np.unique(X_org['hour'].values))) / 2
            for h in hrs:
                x_cnn2[np.isclose(x_cnn2[:, -2], h, atol=rg), -2] = h
            self.x_dates = x_cnn2[:,-2:]
            x_cnn2 = x_cnn2[:,:-2]
            x_cnn2 = scale_cnn.inverse_transform(x_cnn2)
            self.x_cnn2 = x_cnn2.reshape(-1, *cnn_shape[1:])
            self.y_cnn2 = y_cnn2
        else:
            cnn_shape = X_cnn.shape
            x_cnn = np.copy(X_cnn)
            x_cnn = x_cnn.reshape(-1, np.prod(cnn_shape[1:]))
            scale_cnn = MinMaxScaler()
            x_cnn = scale_cnn.fit_transform(x_cnn)
            x_cnn = np.hstack((x_cnn, X_org['hour', 'month'].values))
            self.x_cnn2, y_cnn2 = self.imblearn_fit(x_cnn, y)
            self.x_dates = []
            self.y_cnn2 = y_cnn2
        self.X_cnn = self.x_cnn2
        self.y2 = self.y_cnn2.reshape(-1, 1)
        i = 0
        self.nwp = dict()
        for var in sorted(self.data_variables):
            if ((var == 'WS') and (self.model_type == 'wind')) or ((var == 'Flux') and (self.model_type == 'pv')):

                self.columns.append('flux' if var == 'Flux' else 'wind')
                var_name = 'flux' if var == 'Flux' else 'wind'
                self.columns.extend(['p_' + var_name] + [var_name] + ['n_' + var_name])

                self.nwp[var + '_prev'] = self.x_cnn2[:, :, :, i]
                self.nwp[var + '_prev'][np.where(self.nwp[var + '_prev'] < 0)] = 0

                self.nwp[var] = self.x_cnn2[:, :, :, i+1]
                self.nwp[var][np.where(self.nwp[var] < 0)] = 0

                self.nwp[var + '_next'] = self.x_cnn2[:, :, :, i+2]
                self.nwp[var + '_next'][np.where(self.nwp[var + '_next'] < 0)] = 0

                i += 3

            elif var in {'WD', 'Cloud'}:

                self.columns.append('cloud' if var == 'Cloud' else 'direction')
                max_val = 100 if var == 'Cloud' else 360
                self.nwp[var] = self.x_cnn2[:, :, :, i]
                self.nwp[var][np.where(self.nwp[var] < 0)] = 0
                self.nwp[var][np.where(self.nwp[var] > max_val)] = max_val
                i += 1
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.model_type == 'pv')):
                self.columns.append('Temp' if var == 'Temperature' else 'wind')

                self.nwp[var] = self.x_cnn2[:, :, :, i]
                self.nwp[var][np.where(self.nwp[var] < 0)] = 0
                i += 1
            else:
                i += 1

        if not hasattr(self, 'X'):
            if self.static_data['compress_data'] == 'dense':
                self.X = self.create_sample(self.nwp, self.x_dates)
            else:
                self.X = self.create_sample_pca(self.nwp, self.x_dates)
            self.y2 = self.y2[np.where(~pd.isnull(self.X).any(axis=1))[0].ravel()]
            if len(X_cnn.shape) == 4:
                self.X_cnn = self.X_cnn[np.where(~pd.isnull(self.X).any(axis=1))[0].ravel()]
            self.X = self.X.drop(np.where(pd.isnull(self.X).any(axis=1))[0].ravel())
            self.X = pd.DataFrame(self.scale_x.transform(self.X.values), index=self.X.index, columns=self.X.columns)
            # self.X = pd.concat([self.X, X.reset_index()], axis=0, join='inner', ignore_index=False, copy=True).reset_index(drop=True)
            # self.y2 = np.vstack((self.y2.reshape(-1, 1), y))
            self.y2 = self.y2.reshape(-1, 1)
            # if len(X_cnn.shape) == 4:
            #     self.X_cnn = np.vstack((self.X_cnn, X_cnn))
        else:
            pass
        return self.X, self.y2, self.X_cnn

    def imblearn_fit_obsolete(self, X, y, random_state=42):
        if self.model_type == {'pv', 'wind'}:
            from Fuzzy_clustering.ver_tf2.imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, SMOTE, ADASYN
        else:
            from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, SMOTE, ADASYN

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
                if np.unique(yy).shape[0]==1:
                    X2 = X
                    yy2 = y
                    return X2, yy2
                if np.any(np.bincount(yy.ravel())<2):
                    for cl in np.where(np.bincount(yy.ravel())<2)[0]:
                        X = X[np.where(yy!=cl)[0]]
                        y = y[np.where(yy!=cl)[0]]
                        yy = yy[np.where(yy!=cl)[0]]
                if self.method == 'ADASYN':
                    sm = ADASYN(sampling_strategy=strategy, random_state=random_state, n_neighbors=np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)
                elif self.method == 'SVMSMOTE':
                    sm = SVMSMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=np.min(np.bincount(yy.ravel()) - 1),
                                  m_neighbors=2 * np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)
                else:
                    sm = BorderlineSMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=np.min(np.bincount(yy.ravel()) - 1),
                                         m_neighbors=2 * np.min(np.bincount(yy.ravel()) - 1),
                                         n_jobs=self.n_jobs)

                try:
                    X2, yy2 = sm.fit_resample(X, yy.ravel())
                except:
                    sm = SMOTE(sampling_strategy=strategy, random_state=random_state,
                                k_neighbors=np.min(np.bincount(yy.ravel()) - 1), n_jobs=self.n_jobs)
                    X2, yy2 = sm.fit_resample(X, yy.ravel())

                X2 = X2[X.shape[0]+1:]
                X2[np.where(X2<0)] = 0
                yy2 = yy2[X.shape[0]+1:]
                yy2 = bins[yy2 - 1]
                flag = True
            except:
                Std *= 10


        if flag == True:
            return X2, yy2
        else:
            raise RuntimeError('Cannot make resampling ')