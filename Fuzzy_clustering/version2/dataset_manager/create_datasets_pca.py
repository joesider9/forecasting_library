import logging
import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from pytz import timezone
from scipy.interpolate import interp2d
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from Fuzzy_clustering.version2.common_utils.logging import create_logger


def my_scorer(estimator, X):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)


def rescale(arr, nrows, ncol):
    W, H = arr.shape
    new_W, new_H = (nrows, ncol)
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_H), xrange(new_W))

    return new_arr


def rescale_mean(arr):
    arr_new = np.zeros([int(np.ceil(arr.shape[0] / 2)), int(np.ceil(arr.shape[1] / 2))])
    for i in range(0, arr.shape[0], 2):
        for j in range(0, arr.shape[1], 2):
            arr_new[int((i + 1) / 2), int((j + 1) / 2)] = np.mean(arr[i:i + 2, j:j + 2])
    return arr_new


def stack_2d(X, sample, compress):
    if compress:
        sample = rescale_mean(sample)

    if len(sample.shape) == 3:
        if X.shape[0] == 0:
            X = sample
        elif len(X.shape) == 3:
            X = np.stack((X, sample))
        else:
            X = np.vstack((X, sample[np.newaxis, :, :, :]))
    elif len(sample.shape) == 2:
        if X.shape[0] == 0:
            X = sample
        elif len(X.shape) == 2:
            X = np.stack((X, sample))
        else:
            X = np.vstack((X, sample[np.newaxis, :, :]))
    elif len(sample.shape) == 4:
        if X.shape[0] == 0:
            X = sample
        elif len(X.shape) == 4:
            X = np.stack((X, sample))
        else:
            X = np.vstack((X, sample[np.newaxis, :, :, :, :]))
    return X


def stack_3d(X, sample):
    if X.shape[0] == 0:
        X = sample
    elif len(sample.shape) != len(X.shape):
        X = np.vstack((X, sample[np.newaxis]))
    else:
        X = np.vstack((X, sample))
    return X


def check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
    flag = True
    for var in variables:
        if nwp[var].shape[0] == 0 and nwp_next[var].shape[0] == 0 and nwp_prev[var].shape[0] == 0:
            flag = False
            break
    return flag


def stack_daily_nwps(t, p_dates, path_nwp_project, nwp_model, areas, variables, compress, model_type):
    x = np.array([])
    x_3d = np.array([])
    data_var = dict()
    for var in variables:
        if ((var == 'WS') and (model_type == 'wind')) or ((var == 'Flux') and (model_type == 'pv')):
            data_var[var + '_prev'] = x
            data_var[var] = x
            data_var[var + '_next'] = x
        else:
            data_var[var] = x
        data_var['dates'] = x

    fname = os.path.join(path_nwp_project, nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
    if os.path.exists(fname):
        nwps = joblib.load(fname)

        for date in p_dates:
            try:
                nwp = nwps[date]
                if len(nwp['lat'].shape) == 1:
                    nwp['lat'] = nwp['lat'][:, np.newaxis]
                if len(nwp['long'].shape) == 1:
                    nwp['long'] = nwp['long'][np.newaxis, :]
                lats = (np.where((nwp['lat'][:, 0] >= areas[0][0]) & (nwp['lat'][:, 0] <= areas[1][0])))[0]
                longs = (np.where((nwp['long'][0, :] >= areas[0][1]) & (nwp['long'][0, :] <= areas[1][1])))[0]
                break
            except:
                continue
        try:
            for date in p_dates:

                nwp = nwps[date]
                date = pd.to_datetime(date, format='%d%m%y%H%M')
                nwp_prev = nwps[(date - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                nwp_next = nwps[(date + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
                    data_var['dates'] = np.hstack((data_var['dates'], date))
                    x_2d = np.array([])
                    for var in sorted(variables):
                        if ((var == 'WS') and (model_type == 'wind')) or ((var == 'Flux') and (model_type == 'pv')):
                            data_var[var + '_prev'] = stack_2d(data_var[var + '_prev'],
                                                               nwp_prev[var][np.ix_(lats, longs)], compress)
                            data_var[var] = stack_2d(data_var[var], nwp[var][np.ix_(lats, longs)], compress)
                            data_var[var + '_next'] = stack_2d(data_var[var + '_next'],
                                                               nwp_next[var][np.ix_(lats, longs)], compress)
                            x_2d = stack_2d(x_2d, nwp_prev[var][np.ix_(lats, longs)], compress)
                            x_2d = stack_2d(x_2d, nwp[var][np.ix_(lats, longs)], compress)
                            x_2d = stack_2d(x_2d, nwp_next[var][np.ix_(lats, longs)], compress)
                        else:
                            data_var[var] = stack_2d(data_var[var], nwp[var][np.ix_(lats, longs)], compress)
                            x_2d = stack_2d(x_2d, nwp[var][np.ix_(lats, longs)], compress)
                    x_3d = stack_2d(x_3d, x_2d, False)
        except:
            pass
        print(t.strftime('%d%m%y%H%M'), ' extracted')
    return (data_var, x_3d, t.strftime('%d%m%y%H%M'))


class DatasetCreatorPCA:

    def __init__(self, project, data=None, n_jobs=1, test=False, dates=None):
        self.data = data
        self.is_for_test = test
        self.project_name = project['_id']
        self.static_data = project['static_data']
        self.path_nwp_project = self.static_data['pathnwp']
        self.path_data = self.static_data['path_data']
        self.areas = self.static_data['areas']
        self.area_group = self.static_data['area_group']
        self.nwp_model = self.static_data['NWP_model']
        self.nwp_resolution = self.static_data['NWP_resolution']
        self.location = self.static_data['location']
        if self.nwp_resolution == 0.05:
            self.compress = True
        else:
            self.compress = False
        self.n_jobs = n_jobs
        self.variables = self.static_data['data_variables']
        if self.data is not None:
            self.dates = self.check_dates()
        elif dates is not None:
            self.dates = dates

        self.logger = create_logger(logger_name=f"log_{self.static_data['project_group']}",
                                    abs_path=self.path_nwp_project,
                                    logger_path=f"log_{self.static_data['project_group']}.log", write_type='a')

    def check_dates(self):
        # Extract dates of power measurements.
        start_date = pd.to_datetime(self.data.index[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(self.data.index[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)
        data_dates = pd.to_datetime(np.unique(self.data.index.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]
        self.logger.info('Dates is checked. Number of time samples %s', str(len(dates)))
        return pd.DatetimeIndex(dates)

    def make_dataset_res(self):

        nwp_3d_pickle = 'nwps_3d_test.pickle' if self.is_for_test else 'nwps_3d.pickle'
        dataset_cnn_pickle = 'dataset_cnn_test.pickle' if self.is_for_test else 'dataset_cnn.pickle'

        nwp_3d_pickle = os.path.join(self.path_data, nwp_3d_pickle)
        dataset_cnn_pickle = os.path.join(self.path_data, dataset_cnn_pickle)

        if not (os.path.exists(nwp_3d_pickle) and os.path.exists(dataset_cnn_pickle)):
            data, x_3d = self.get_3d_dataset()
        else:
            data = joblib.load(nwp_3d_pickle)
            x_3d = joblib.load(dataset_cnn_pickle)  # FIXME: unused variable

        data_path = self.path_data

        if not isinstance(self.areas, dict):
            self.dataset_for_single_farm(data, data_path)
        else:
            dates_stack = []
            for t in self.dates:
                p_dates = pd.date_range(t + pd.DateOffset(hours=25), t + pd.DateOffset(hours=48), freq='H')
                dates = [dt.strftime('%d%m%y%H%M') for dt in p_dates if dt in self.data.index]
                dates_stack.append(dates)
            flag = False
            for i, p_dates in enumerate(dates_stack):
                t = self.dates[i]
                fname = os.path.join(self.path_nwp_project, self.nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
                if os.path.exists(fname):
                    nwps = joblib.load(fname)

                    for date in p_dates:
                        try:
                            nwp = nwps[date]
                            if len(nwp['lat'].shape) == 1:
                                nwp['lat'] = nwp['lat'][:, np.newaxis]
                            if len(nwp['long'].shape) == 1:
                                nwp['long'] = nwp['long'][np.newaxis, :]
                            lats = (np.where((nwp['lat'][:, 0] >= self.area_group[0][0]) & (
                                    nwp['lat'][:, 0] <= self.area_group[1][0])))[0]
                            longs = (np.where((nwp['long'][0, :] >= self.area_group[0][1]) & (
                                    nwp['long'][0, :] <= self.area_group[1][1])))[0]
                            lats_group = nwp['lat'][lats]
                            longs_group = nwp['long'][:, longs]
                            flag = True
                            break
                        except:
                            continue
                if flag:
                    break

            self.dataset_for_multiple_farms(data, self.areas, lats_group, longs_group)

    def get_3d_dataset(self):

        dates_stack = []
        for t in self.dates:
            p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47),
                                    freq='H')  # 47 hours: 00:00 -> 23:00
            dates = [dt.strftime('%d%m%y%H%M') for dt in p_dates if dt in self.data.index]
            dates_stack.append(dates)  # For each date we have prediction append the next 47 hours

        area = self.area_group if isinstance(self.areas, dict) else self.areas
        nwp_daily = Parallel(n_jobs=self.n_jobs)(
            delayed(stack_daily_nwps)(self.dates[i], p_dates, self.path_nwp_project,
                                      self.nwp_model, area, self.variables,
                                      self.compress, self.static_data['type'])
            for i, p_dates in enumerate(dates_stack))

        x = np.array([])
        data_var = dict()
        for var in self.variables:
            if (var == 'WS' and self.static_data['type'] == 'wind') or \
                    (var == 'Flux' and self.static_data['type'] == 'pv'):
                data_var[var + '_prev'] = x
                data_var[var] = x
                data_var[var + '_next'] = x
            else:
                data_var[var] = x
            data_var['dates'] = x
        x_3d = np.array([])

        for arrays in nwp_daily:
            nwp = arrays[0]
            x_2d = arrays[1]
            if x_2d.shape[0] != 0:
                for var in nwp.keys():
                    if var != 'dates':
                        data_var[var] = stack_3d(data_var[var], nwp[var])
                    else:
                        data_var[var] = np.hstack((data_var[var], nwp[var]))
                x_3d = stack_3d(x_3d, x_2d)
                self.logger.info('NWP data stacked for date %s', arrays[2])
        if self.is_for_test:
            joblib.dump(data_var, os.path.join(self.path_data, 'nwps_3d_test.pickle'))
            joblib.dump(x_3d, os.path.join(self.path_data, 'dataset_cnn_test.pickle'))
        else:
            joblib.dump(data_var, os.path.join(self.path_data, 'nwps_3d.pickle'))
            joblib.dump(x_3d, os.path.join(self.path_data, 'dataset_cnn.pickle'))
        self.logger.info('NWP stacked data saved')
        return data_var, x_3d

    def train_pca(self, data, components, level):
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        param_grid = [{
            "gamma": np.logspace(-3, 0, 20),
        }]

        kpca = KernelPCA(n_components=components, fit_inverse_transform=True, n_jobs=self.n_jobs)
        grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer, n_jobs=self.n_jobs)
        grid_search.fit(data_scaled)
        kpca = grid_search.best_estimator_
        fname = os.path.join(self.path_data, 'kpca_' + level + '.pickle')
        joblib.dump({'scaler': scaler, 'kpca': kpca}, fname)

    def pca_transform(self, data, components, level):

        fname = os.path.join(self.path_data, 'kpca_' + level + '.pickle')
        if not os.path.exists(fname):
            self.train_pca(data, components, level)
        models = joblib.load(fname)
        data_scaled = models['scaler'].transform(data)
        data_compress = models['kpca'].transform(data_scaled)

        return data_compress

    def dataset_for_single_farm(self, data, data_path):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                X0 = np.transpose(data[var + '_prev'], [0, 2, 1])
                X0_level0 = X0[:, 2, 2]

                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                X2 = np.transpose(data[var + '_next'], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col = ['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X0_level0.reshape(-1, 1), X2_level0.reshape(-1, 1), X1_level1.reshape(-1, 1), X1_level3d,
                               X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

            elif var in {'WD', 'Cloud'}:
                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1.reshape(-1, 1), X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                X2 = np.transpose(data[var], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'
                var_sort = 'tp' if var == 'Temperature' else 'ws'
                col = [var_name]

                X = X2_level0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            else:
                continue
        dataset_X = dataset_X
        dataset_y = self.data.loc[dataset_X.index].to_frame()
        dataset_y.columns = ['target']
        if self.is_for_test:
            ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
            columns = dataset_X.columns[ind]
            dataset_X = dataset_X[columns]
            dataset_X.to_csv(os.path.join(self.path_data, 'dataset_X_test.csv'))
            dataset_y.to_csv(os.path.join(self.path_data, 'dataset_y_test.csv'))
            self.logger.info('Successfully dataset created for Evaluation for %s', self.project_name)
        else:
            corr = []
            for f in range(dataset_X.shape[1]):
                corr.append(np.abs(np.corrcoef(dataset_X.values[:, f], dataset_y.values.ravel())[1, 0]))
            ind = np.argsort(np.array(corr))[::-1]
            columns = dataset_X.columns[ind]
            dataset_X = dataset_X[columns]
            joblib.dump(ind, os.path.join(data_path, 'dataset_columns_order.pickle'))

            dataset_X.to_csv(os.path.join(self.path_data, 'dataset_X.csv'))
            dataset_y.to_csv(os.path.join(self.path_data, 'dataset_y.csv'))
            self.logger.info('Successfully dataset created for training for %s', self.project_name)

    def dataset_for_multiple_farms(self, data, areas, lats_group, longs_group):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            for area_name, area in areas.items():
                if len(area) > 1:
                    lats = (np.where((lats_group[:, 0] >= area[0, 0]) & (lats_group[:, 0] <= area[1, 0])))[0]
                    longs = (np.where((longs_group[0, :] >= area[0, 1]) & (longs_group[0, :] <= area[1, 1])))[0]
                else:
                    lats = (np.where((lats_group[:, 0] >= area[0]) & (lats_group[:, 0] <= area[2])))[0]
                    longs = (np.where((longs_group[0, :] >= area[1]) & (longs_group[0, :] <= area[3])))[0]
                if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                        (var == 'Flux') and (self.static_data['type'] == 'pv')):
                    X0 = data[var + '_prev'][:, lats, :][:, :, longs]
                    X0 = X0.reshape(-1, X0.shape[1] * X0.shape[2])

                    level = var + '_prev_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X0_compressed = self.pca_transform(X0, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X2 = data[var + '_next'][:, lats, :][:, :, longs]
                    X2 = X2.reshape(-1, X2.shape[1] * X2.shape[2])

                    level = var + '_next_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X2_compressed = self.pca_transform(X2, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    var_sort = 'fl_' + area_name if var == 'Flux' else 'ws_' + area_name

                    col = ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col += ['n_' + var_name + '.' + str(i) for i in range(3)]
                    col += [var_name + '.' + str(i) for i in range(9)]

                    X = np.hstack((X0_compressed, X2_compressed, X1_compressed))
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

                elif var in {'WD', 'Cloud'}:
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    var_sort = 'cl_' + area_name if var == 'Cloud' else 'wd_' + area_name

                    col = [var_name + '.' + str(i) for i in range(9)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    var_sort = 'tp_' + area_name if var == 'Temperature' else 'ws_' + area_name
                    col = [var_name + '.' + str(i) for i in range(3)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                else:
                    continue

        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                col = []
                col_p = []
                col_n = []
                for area_name, area in areas.items():
                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                    col_p += ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col_n += ['n_' + var_name + '.' + str(i) for i in range(3)]
                var_name = 'flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
                var_name = 'p_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_p].mean(axis=1)
                var_name = 'n_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_n].mean(axis=1)

            elif var in {'WD', 'Cloud'}:
                col = []
                for area_name, area in areas.items():
                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                var_name = 'cloud' if var == 'Cloud' else 'direction'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                col = []
                for area_name, area in areas.items():
                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(3)]
                var_name = 'Temp' if var == 'Temperature' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)

        dataset_y = self.data.loc[dataset_X.index].to_frame()
        dataset_y.columns = ['target']
        if self.is_for_test:
            ind = joblib.load(os.path.join(self.path_data, 'dataset_columns_order.pickle'))
            columns = dataset_X.columns[ind]
            dataset_X = dataset_X[columns]
            dataset_X.to_csv(os.path.join(self.path_data, 'dataset_X_test.csv'))
            dataset_y.to_csv(os.path.join(self.path_data, 'dataset_y_test.csv'))
            self.logger.info('Successfully dataset created for Evaluation for %s', self.project_name)
        else:
            corr = []
            for f in range(dataset_X.shape[1]):
                corr.append(np.abs(np.corrcoef(dataset_X.values[:, f], dataset_y.values.ravel())[1, 0]))
            ind = np.argsort(np.array(corr))[::-1]
            columns = dataset_X.columns[ind]
            dataset_X = dataset_X[columns]
            joblib.dump(ind, os.path.join(self.path_data, 'dataset_columns_order.pickle'))

            dataset_X.to_csv(os.path.join(self.path_data, 'dataset_X.csv'))
            dataset_y.to_csv(os.path.join(self.path_data, 'dataset_y.csv'))
            self.logger.info('Successfully dataset created for training for %s', self.project_name)

    def make_dataset_res_offline(self, utc=False):
        def datetime_exists_in_tz(dt, tz):
            try:
                dt.tz_localize(tz)
                return True
            except:
                return False

        data, X_3d = self.get_3d_dataset_offline(utc)

        if not isinstance(self.areas, dict):
            X = self.dataset_for_single_farm_offline(data)
        else:
            dates_stack = []
            for t in self.dates:
                pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H')
                dates = [dt.strftime('%d%m%y%H%M') for dt in pdates]
                dates_stack.append(dates)
            flag = False
            for i, pdates in enumerate(dates_stack):
                t = self.dates[i]
                fname = os.path.join(self.path_nwp_project, self.nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
                if os.path.exists(fname):
                    nwps = joblib.load(fname)

                    for date in pdates:
                        try:
                            nwp = nwps[date]
                            if len(nwp['lat'].shape) == 1:
                                nwp['lat'] = nwp['lat'][:, np.newaxis]
                            if len(nwp['long'].shape) == 1:
                                nwp['long'] = nwp['long'][np.newaxis, :]
                            lats = (np.where((nwp['lat'][:, 0] >= self.area_group[0][0]) & (
                                    nwp['lat'][:, 0] <= self.area_group[1][0])))[0]
                            longs = (np.where((nwp['long'][0, :] >= self.area_group[0][1]) & (
                                    nwp['long'][0, :] <= self.area_group[1][1])))[0]
                            lats_group = nwp['lat'][lats]
                            longs_group = nwp['long'][:, longs]
                            flag = True
                            break
                        except:
                            continue
                if flag:
                    break

            X = self.dataset_for_multiple_farms_offline(data, self.areas, lats_group, longs_group)
        return X, X_3d

    def get_3d_dataset_offline(self, utc):
        def datetime_exists_in_tz(dt, tz):
            try:
                dt.tz_localize(tz)
                return True
            except:
                return False

        dates_stack = []
        for dt in self.dates:
            if utc:
                pdates = pd.date_range(dt + pd.DateOffset(hours=25), dt + pd.DateOffset(hours=48), freq='H')
                dates = [t.strftime('%d%m%y%H%M') for t in pdates]
                dates_stack.append(dates)
            else:
                pdates = pd.date_range(dt + pd.DateOffset(hours=25), dt + pd.DateOffset(hours=48), freq='H')
                indices = [i for i, t in enumerate(pdates) if datetime_exists_in_tz(t, tz=timezone('Europe/Athens'))]
                pdates = pdates[indices]
                pdates = pdates.tz_localize(timezone('Europe/Athens'))
                pdates = pdates.tz_convert(timezone('UTC'))
                dates = [dt.strftime('%d%m%y%H%M') for dt in pdates]
                dates_stack.append(dates)

        if not isinstance(self.areas, dict):
            nwp = stack_daily_nwps(self.dates[0], dates_stack[0], self.path_nwp_project, self.nwp_model, self.areas,
                                   self.variables, self.compress, self.static_data['type'])
            nwp_daily = Parallel(n_jobs=self.n_jobs)(
                delayed(stack_daily_nwps)(self.dates[i], pdates, self.path_nwp_project, self.nwp_model,
                                          self.areas, self.variables, self.compress, self.static_data['type'])
                for i, pdates in enumerate(dates_stack))
        else:
            nwp = stack_daily_nwps(self.dates[0], dates_stack[0], self.path_nwp_project, self.nwp_model,
                                   self.area_group,
                                   self.variables, self.compress, self.static_data['type'])

            nwp_daily = Parallel(n_jobs=self.n_jobs)(
                delayed(stack_daily_nwps)(self.dates[i], pdates, self.path_nwp_project, self.nwp_model,
                                          self.area_group, self.variables, self.compress, self.static_data['type'])
                for i, pdates in enumerate(dates_stack))

        X = np.array([])
        data_var = dict()
        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                data_var[var + '_prev'] = X
                data_var[var] = X
                data_var[var + '_next'] = X
            else:
                data_var[var] = X
            data_var['dates'] = X
        X_3d = np.array([])
        for arrays in nwp_daily:
            nwp = arrays[0]
            x_2d = arrays[1]
            if x_2d.shape[0] != 0:
                for var in nwp.keys():
                    if var != 'dates':
                        data_var[var] = stack_3d(data_var[var], nwp[var])
                    else:
                        data_var[var] = np.hstack((data_var[var], nwp[var]))
                X_3d = stack_3d(X_3d, x_2d)
                self.logger.info('NWP data stacked for date %s', arrays[2])

        return data_var, X_3d

    def dataset_for_single_farm_offline(self, data):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                X0 = np.transpose(data[var + '_prev'], [0, 2, 1])
                X0_level0 = X0[:, 2, 2]

                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                X2 = np.transpose(data[var + '_next'], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col = ['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X0_level0.reshape(-1, 1), X2_level0.reshape(-1, 1), X1_level1.reshape(-1, 1), X1_level3d,
                               X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

            elif var in {'WD', 'Cloud'}:
                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1.reshape(-1, 1), X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                X2 = np.transpose(data[var], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'
                var_sort = 'tp' if var == 'Temperature' else 'ws'
                col = [var_name]

                X = X2_level0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            else:
                continue
        ind = joblib.load(os.path.join(self.path_data, 'dataset_columns_order.pickle'))
        columns = dataset_X.columns[ind]
        dataset_X = dataset_X[columns]

        return dataset_X

    def dataset_for_multiple_farms_offline(self, data, areas, lats_group, longs_group):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            for area_name, area in areas.items():
                if len(area) > 1:
                    lats = (np.where((lats_group[:, 0] >= area[0, 0]) & (lats_group[:, 0] <= area[1, 0])))[0]
                    longs = (np.where((longs_group[0, :] >= area[0, 1]) & (longs_group[0, :] <= area[1, 1])))[0]
                else:
                    lats = (np.where((lats_group[:, 0] >= area[0]) & (lats_group[:, 0] <= area[2])))[0]
                    longs = (np.where((longs_group[0, :] >= area[1]) & (longs_group[0, :] <= area[3])))[0]
                if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                        (var == 'Flux') and (self.static_data['type'] == 'pv')):
                    X0 = data[var + '_prev'][:, lats, :][:, :, longs]
                    X0 = X0.reshape(-1, X0.shape[1] * X0.shape[2])

                    level = var + '_prev_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X0_compressed = self.pca_transform(X0, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X2 = data[var + '_next'][:, lats, :][:, :, longs]
                    X2 = X2.reshape(-1, X2.shape[1] * X2.shape[2])

                    level = var + '_next_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X2_compressed = self.pca_transform(X2, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    var_sort = 'fl_' + area_name if var == 'Flux' else 'ws_' + area_name

                    col = ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col += ['n_' + var_name + '.' + str(i) for i in range(3)]
                    col += [var_name + '.' + str(i) for i in range(9)]

                    X = np.hstack((X0_compressed, X2_compressed, X1_compressed))
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

                elif var in {'WD', 'Cloud'}:
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    var_sort = 'cl_' + area_name if var == 'Cloud' else 'wd_' + area_name

                    col = [var_name + '.' + str(i) for i in range(9)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    var_sort = 'tp_' + area_name if var == 'Temperature' else 'ws_' + area_name
                    col = [var_name + '.' + str(i) for i in range(3)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                else:
                    continue

        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                col = []
                col_p = []
                col_n = []
                for area_name, area in areas.items():
                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                    col_p += ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col_n += ['n_' + var_name + '.' + str(i) for i in range(3)]
                var_name = 'flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
                var_name = 'p_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_p].mean(axis=1)
                var_name = 'n_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_n].mean(axis=1)

            elif var in {'WD', 'Cloud'}:
                col = []
                for area_name, area in areas.items():
                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                var_name = 'cloud' if var == 'Cloud' else 'direction'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                col = []
                for area_name, area in areas.items():
                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(3)]
                var_name = 'Temp' if var == 'Temperature' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
        ind = joblib.load(os.path.join(self.path_data, 'dataset_columns_order.pickle'))
        columns = dataset_X.columns[ind]
        dataset_X = dataset_X[columns]

        return dataset_X

    def make_dataset_res_online(self, utc=False):
        def datetime_exists_in_tz(dt, tz):
            try:
                dt.tz_localize(tz)
                return True
            except:
                return False

        data, X_3d = self.get_3d_dataset_online(utc)

        if not isinstance(self.areas, dict):
            X = self.dataset_for_single_farm_online(data)
        else:
            dates_stack = []
            for t in self.dates:
                pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H')
                dates = [dt.strftime('%d%m%y%H%M') for dt in pdates]
                dates_stack.append(dates)
            flag = False
            for i, pdates in enumerate(dates_stack):
                t = self.dates[i]
                fname = os.path.join(self.path_nwp_project, self.nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
                if os.path.exists(fname):
                    nwps = joblib.load(fname)

                    for date in pdates:
                        try:
                            nwp = nwps[date]
                            if len(nwp['lat'].shape) == 1:
                                nwp['lat'] = nwp['lat'][:, np.newaxis]
                            if len(nwp['long'].shape) == 1:
                                nwp['long'] = nwp['long'][np.newaxis, :]
                            lats = (np.where((nwp['lat'][:, 0] >= self.area_group[0][0]) & (
                                    nwp['lat'][:, 0] <= self.area_group[1][0])))[0]
                            longs = (np.where((nwp['long'][0, :] >= self.area_group[0][1]) & (
                                    nwp['long'][0, :] <= self.area_group[1][1])))[0]
                            lats_group = nwp['lat'][lats]
                            longs_group = nwp['long'][:, longs]
                            flag = True
                            break
                        except:
                            continue
                if flag:
                    break

            X = self.dataset_for_multiple_farms_online(data, self.areas, lats_group, longs_group)
        return X, X_3d

    def get_3d_dataset_online(self, utc):
        def datetime_exists_in_tz(dt, tz):
            try:
                dt.tz_localize(tz)
                return True
            except:
                return False

        dates_stack = []
        if utc:
            pdates = pd.date_range(self.dates[-1] + pd.DateOffset(hours=25), self.dates[-1] + pd.DateOffset(hours=48),
                                   freq='H')
            dates = [dt.strftime('%d%m%y%H%M') for dt in pdates if dt in self.data.index]
            dates_stack.append(dates)
        else:
            pdates = pd.date_range(self.dates[-1] + pd.DateOffset(hours=25), self.dates[-1] + pd.DateOffset(hours=48),
                                   freq='H')
            indices = [i for i, t in enumerate(pdates) if datetime_exists_in_tz(t, tz=timezone('Europe/Athens'))]
            pdates = pdates[indices]
            pdates = pdates.tz_localize(timezone('Europe/Athens'))
            pdates = pdates.tz_convert(timezone('UTC'))
            dates = [dt.strftime('%d%m%y%H%M') for dt in pdates]
            dates_stack.append(dates)

        if not isinstance(self.areas, dict):
            arrays = stack_daily_nwps(self.dates[-1], dates_stack[0], self.path_nwp_project, self.nwp_model, self.areas,
                                      self.variables, self.compress, self.static_data['type'])

        else:
            arrays = stack_daily_nwps(self.dates[-1], dates_stack[0], self.path_nwp_project, self.nwp_model,
                                      self.area_group,
                                      self.variables, self.compress, self.static_data['type'])

        X = np.array([])
        data_var = dict()
        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                data_var[var + '_prev'] = X
                data_var[var] = X
                data_var[var + '_next'] = X
            else:
                data_var[var] = X
            data_var['dates'] = X
        X_3d = np.array([])

        nwp = arrays[0]
        x_2d = arrays[1]
        if x_2d.shape[0] != 0:
            for var in nwp.keys():
                if var != 'dates':
                    data_var[var] = stack_3d(data_var[var], nwp[var])
                else:
                    data_var[var] = np.hstack((data_var[var], nwp[var]))
            X_3d = stack_3d(X_3d, x_2d)
            self.logger.info('NWP data stacked for date %s', arrays[2])
        return data_var, X_3d

    def dataset_for_single_farm_online(self, data):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                X0 = np.transpose(data[var + '_prev'], [0, 2, 1])
                X0_level0 = X0[:, 2, 2]

                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                X2 = np.transpose(data[var + '_next'], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col = ['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X0_level0.reshape(-1, 1), X2_level0.reshape(-1, 1), X1_level1.reshape(-1, 1), X1_level3d,
                               X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

            elif var in {'WD', 'Cloud'}:
                X1 = np.transpose(data[var], [0, 2, 1])
                X1_level1 = X1[:, 2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3d = self.pca_transform(X1_level3d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level3u = self.pca_transform(X1_level3u, 2, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4d = self.pca_transform(X1_level4d, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[:, indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                self.logger.info('Begin PCA training for %s', level)
                X1_level4u = self.pca_transform(X1_level4u, 3, level)
                self.logger.info('Successfully PCA transform for %s', level)

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(2)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1.reshape(-1, 1), X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                X2 = np.transpose(data[var], [0, 2, 1])
                X2_level0 = X2[:, 2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'
                var_sort = 'tp' if var == 'Temperature' else 'ws'
                col = [var_name]

                X = X2_level0
                dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
            else:
                continue
        dataset_X = dataset_X
        ind = joblib.load(os.path.join(self.path_data, 'dataset_columns_order.pickle'))
        columns = dataset_X.columns[ind]
        dataset_X = dataset_X[columns]

        self.logger.info('Successfully dataset created for training for %s', self.project_name)
        return dataset_X

    def dataset_for_multiple_farms_online(self, data, areas, lats_group, longs_group):
        dataset_X = pd.DataFrame()
        if self.static_data['type'] == 'pv':
            hours = [dt.hour for dt in data['dates']]
            months = [dt.month for dt in data['dates']]
            dataset_X = pd.concat(
                [dataset_X, pd.DataFrame(np.stack([hours, months]).T, index=data['dates'], columns=['hour', 'month'])])
        for var in self.variables:
            for area_name, area in areas.items():
                if len(area) > 1:
                    lats = (np.where((lats_group[:, 0] >= area[0, 0]) & (lats_group[:, 0] <= area[1, 0])))[0]
                    longs = (np.where((longs_group[0, :] >= area[0, 1]) & (longs_group[0, :] <= area[1, 1])))[0]
                else:
                    lats = (np.where((lats_group[:, 0] >= area[0]) & (lats_group[:, 0] <= area[2])))[0]
                if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                        (var == 'Flux') and (self.static_data['type'] == 'pv')):
                    X0 = data[var + '_prev'][:, lats, :][:, :, longs]
                    X0 = X0.reshape(-1, X0.shape[1] * X0.shape[2])

                    level = var + '_prev_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X0_compressed = self.pca_transform(X0, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    X2 = data[var + '_next'][:, lats, :][:, :, longs]
                    X2 = X2.reshape(-1, X2.shape[1] * X2.shape[2])

                    level = var + '_next_' + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X2_compressed = self.pca_transform(X2, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    var_sort = 'fl_' + area_name if var == 'Flux' else 'ws_' + area_name

                    col = ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col += ['n_' + var_name + '.' + str(i) for i in range(3)]
                    col += [var_name + '.' + str(i) for i in range(9)]

                    X = np.hstack((X0_compressed, X2_compressed, X1_compressed))
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)

                elif var in {'WD', 'Cloud'}:
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 9, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    var_sort = 'cl_' + area_name if var == 'Cloud' else 'wd_' + area_name

                    col = [var_name + '.' + str(i) for i in range(9)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                    X1 = data[var][:, lats, :][:, :, longs]
                    X1 = X1.reshape(-1, X1.shape[1] * X1.shape[2])

                    level = var + area_name
                    self.logger.info('Begin PCA training for %s', level)
                    X1_compressed = self.pca_transform(X1, 3, level)
                    self.logger.info('Successfully PCA transform for %s', level)

                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    var_sort = 'tp_' + area_name if var == 'Temperature' else 'ws_' + area_name
                    col = [var_name + '.' + str(i) for i in range(3)]

                    X = X1_compressed
                    dataset_X = pd.concat([dataset_X, pd.DataFrame(X, index=data['dates'], columns=col)], axis=1)
                else:
                    continue

        for var in self.variables:
            if ((var == 'WS') and (self.static_data['type'] == 'wind')) or (
                    (var == 'Flux') and (self.static_data['type'] == 'pv')):
                col = []
                col_p = []
                col_n = []
                for area_name, area in areas.items():
                    var_name = 'flux_' + area_name if var == 'Flux' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                    col_p += ['p_' + var_name + '.' + str(i) for i in range(3)]
                    col_n += ['n_' + var_name + '.' + str(i) for i in range(3)]
                var_name = 'flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
                var_name = 'p_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_p].mean(axis=1)
                var_name = 'n_flux' if var == 'Flux' else 'wind'
                dataset_X[var_name] = dataset_X[col_n].mean(axis=1)

            elif var in {'WD', 'Cloud'}:
                col = []
                for area_name, area in areas.items():
                    var_name = 'cloud_' + area_name if var == 'Cloud' else 'direction_' + area_name
                    col += [var_name + '.' + str(i) for i in range(9)]
                var_name = 'cloud' if var == 'Cloud' else 'direction'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (self.static_data['type'] == 'pv')):
                col = []
                for area_name, area in areas.items():
                    var_name = 'Temp_' + area_name if var == 'Temperature' else 'wind_' + area_name
                    col += [var_name + '.' + str(i) for i in range(3)]
                var_name = 'Temp' if var == 'Temperature' else 'wind'
                dataset_X[var_name] = dataset_X[col].mean(axis=1)

        ind = joblib.load(os.path.join(self.path_data, 'dataset_columns_order.pickle'))
        columns = dataset_X.columns[ind]
        dataset_X = dataset_X[columns]

        self.logger.info('Successfully dataset created for training for %s', self.project_name)
        return dataset_X
