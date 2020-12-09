import numpy as np
import pandas as pd
import joblib, os, logging
from joblib import Parallel, delayed
from scipy.interpolate import interp2d
from sklearn.metrics import mean_squared_error

def my_scorer(estimator, X, y=None):
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
    arr_new = np.zeros([int(np.ceil(arr.shape[0]/2)), int(np.ceil(arr.shape[1]/2))])
    for i in range(0, arr.shape[0], 2):
        for j in range(0, arr.shape[1], 2):
            arr_new[int((i+1)/2),int((j+1)/2)] = np.mean(arr[i:i+2, j:j+2])
    return arr_new

def stack_2d(X, sample, compress):
    if compress:
        sample = rescale(sample, 8, 8)

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
    return X


def stack_3d(X, sample):

    if X.shape[0] == 0:
        X = sample
    elif len(sample.shape)!=len(X.shape):
        X = np.vstack((X, sample[np.newaxis]))
    else:
        X = np.vstack((X, sample))
    return X


class dataset_creator_dense():

    def __init__(self, projects_group, projects, data, path_nwp, nwp_model, nwp_resolution, data_variables, njobs=1, test=False):
        self.projects = projects
        self.isfortest = test
        self.projects_group = projects_group
        self.data = data
        self.path_nwp = path_nwp
        self.create_logger()
        self.check_dates()
        self.nwp_model = nwp_model
        self.nwp_resolution = nwp_resolution
        if self.nwp_resolution == 0.05:
            self.compress = True
        else:
            self.compress = False
        self.njobs = njobs
        self.variables = data_variables


    def create_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(os.path.dirname(self.path_nwp), 'log_' + self.projects_group + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def check_dates(self):
        start_date = pd.to_datetime(self.data.index[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(self.data.index[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)
        data_dates = pd.to_datetime(np.unique(self.data.index.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]
        self.logger.info('Dates is checked. Number of time samples %s', str(len(dates)))
        self.dates = pd.DatetimeIndex(dates)

    def check_empty_nwp(self, nwp, nwp_next, nwp_prev, variables):
        flag = True
        for var in variables:
            if nwp[var].shape[0] == 0 and nwp_next[var].shape[0] == 0 and nwp_prev[var].shape[0] == 0:
                flag = False
                break
        return flag

    def stack_daily_nwps(self, t, data, lats, longs, path_nwp, nwp_model, projects, variables, compress):

        X = dict()
        y = dict()
        X_3d = dict()
        fname = os.path.join(path_nwp, nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
        if os.path.exists(fname):
            nwps = joblib.load(fname)

            pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H').strftime(
                '%d%m%y%H%M')
            for project in projects:
                X[project['_id']] = pd.DataFrame()
                y[project['_id']] = pd.DataFrame()
                X_3d[project['_id']] = np.array([])
                areas = project['static_data']['areas']
                if isinstance(areas, list):
                    for date in pdates:
                        try:

                            nwp = nwps[date]
                            date = pd.to_datetime(date, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            if self.check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
                                y[project['_id']] = pd.concat([y[project['_id']], pd.DataFrame(data.loc[date, project['_id']], columns=['target'], index=[date])])
                                inp, inp_cnn = self.create_sample(date, nwp, nwp_prev, nwp_next, lats[project['_id']], longs[project['_id']], project['static_data']['type'])
                                X[project['_id']] = pd.concat([X[project['_id']], inp])
                                X_3d[project['_id']] = stack_2d(X_3d[project['_id']], inp_cnn, False)
                        except:
                            continue
                else:
                    for date in pdates:
                        try:
                            nwp = nwps[date]
                            date = pd.to_datetime(date, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            if self.check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
                                y[project['_id']] = pd.concat(
                                    [y[project['_id']], pd.DataFrame(data.loc[date, project['_id']],
                                                                     columns=['target'], index=[date])])
                                inp, inp_cnn = self.create_sample_country(date, nwp, nwp_prev, nwp_next, lats[project['_id']],
                                                                  longs[project['_id']], project['static_data']['type'])
                                X[project['_id']] = pd.concat([X[project['_id']], inp])
                                X_3d[project['_id']] = stack_2d(X_3d[project['_id']], inp_cnn, False)
                        except:
                            continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return (X, y, X_3d, t.strftime('%d%m%y%H%M'))

    def lats_longs(self):
        lats = dict()
        longs = dict()
        flag = False
        for t in self.dates:
            fname = os.path.join(self.path_nwp, self.nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
            pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=48), freq='H').strftime(
                '%d%m%y%H%M')
            if os.path.exists(fname):
                nwps = joblib.load(fname)
                for date in pdates:
                    try:
                        nwp = nwps[date]
                        flag = True
                        break
                    except:
                        continue
            if flag:
                break
        if len(nwp['lat'].shape) == 1:
            nwp['lat'] = nwp['lat'][:, np.newaxis]
        if len(nwp['long'].shape) == 1:
            nwp['long'] = nwp['long'][np.newaxis, :]

        for project in self.projects:
            areas = project['static_data']['areas']
            if isinstance(areas, list):

                lats[project['_id']] = \
                (np.where((nwp['lat'][:, 0] >= areas[0][0]) & (nwp['lat'][:, 0] <= areas[1][0])))[0]
                longs[project['_id']] = \
                (np.where((nwp['long'][0, :] >= areas[0][1]) & (nwp['long'][0, :] <= areas[1][1])))[
                    0]
            else:
                lats[project['_id']] = dict()
                longs[project['_id']] = dict()
                for area in sorted(areas.keys()):
                    lats[project['_id']][area] = \
                    (np.where((nwp['lat'][:, 0] >= areas[area][0][0]) & (nwp['lat'][:, 0] <= areas[area][1][0])))[0]
                    longs[project['_id']][area] = \
                    (np.where((nwp['long'][0, :] >= areas[area][0][1]) & (nwp['long'][0, :] <= areas[area][1][1])))[
                        0]

        return lats, longs

    def make_dataset_res(self):

        lats, longs = self.lats_longs()

        nwp = self.stack_daily_nwps(self.dates[0], self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects, self.variables,
                               self.compress)
        nwp_daily = Parallel(n_jobs=self.njobs)(
            delayed(self.stack_daily_nwps)(t, self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects, self.variables,
                                      self.compress) for t in self.dates)
        X = dict()
        y = dict()
        X_3d = dict()
        for project in self.projects:
            X[project['_id']] = pd.DataFrame()
            y[project['_id']] = pd.DataFrame()
            X_3d[project['_id']] = np.array([])
        for nwp in nwp_daily:
            for project in self.projects:
                if project['_id'] in nwp[2].keys():
                    if nwp[2][project['_id']].shape[0] != 0:
                        X[project['_id']] = pd.concat([X[project['_id']],nwp[0][project['_id']]])
                        y[project['_id']] = pd.concat([y[project['_id']],nwp[1][project['_id']]])
                        X_3d[project['_id']] = stack_3d(X_3d[project['_id']], nwp[2][project['_id']])
        self.logger.info('All Inputs stacked')

        if self.isfortest:
            for project in self.projects:
                data_path = project['static_data']['path_data']

                dataset_X = X[project['_id']]
                dataset_y = y[project['_id']]
                if dataset_y.isna().any().values[0]:
                    dataset_X = dataset_X.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                    if len(X_3d.shape) > 1:
                        X_3d = np.delete(X_3d, np.where(dataset_y.isna())[0], axis=0)
                    dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])
                ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
                columns = dataset_X.columns[ind]
                dataset_X = dataset_X[columns]

                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))
                joblib.dump(X_3d[project['_id']], os.path.join(project['static_data']['path_data'], 'dataset_cnn_test.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])
        else:
            for project in self.projects:
                data_path = project['static_data']['path_data']

                dataset_X = X[project['_id']]
                dataset_y = y[project['_id']]
                if dataset_y.isna().any().values[0]:
                    dataset_X = dataset_X.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                    if len(X_3d.shape) > 1:
                        X_3d = np.delete(X_3d, np.where(dataset_y.isna())[0], axis=0)
                    dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                corr = []
                for f in range(dataset_X.shape[1]):
                    corr.append(np.abs(np.corrcoef(dataset_X.values[:, f], dataset_y.values.ravel())[1, 0]))
                ind = np.argsort(np.array(corr))[::-1]
                columns = dataset_X.columns[ind]
                dataset_X = dataset_X[columns]
                joblib.dump(ind, os.path.join(data_path, 'dataset_columns_order.pickle'))

                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))
                joblib.dump(X_3d[project['_id']], os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])

    def create_sample_country(self, date, nwp, nwp_prev, nwp_next, lats_all, longs_all, model_type):
        inp = pd.DataFrame()
        if model_type == 'pv':
            inp = pd.concat([inp, pd.DataFrame(np.stack([date.hour, date.month]).reshape(-1, 1).T, index=[date],
                                               columns=['hour', 'month'])])
        inp_3d = np.array([])
        for var in self.variables:
            X0 = nwp_prev[var]
            if self.compress:
                X0 = rescale_mean(X0)
                X0 = rescale_mean(X0)
            else:
                X0 = rescale_mean(X0)
            inp_3d = stack_2d(inp_3d, X0, False)
        for var in sorted(self.variables):
            for narea, area in enumerate(sorted(lats_all.keys())):
                lats = lats_all[area]
                longs = longs_all[area]
                if ((var == 'WS') and (model_type =='wind')) or ((var == 'Flux') and (model_type == 'pv')):
                    X0 = nwp_prev[var][np.ix_(lats, longs)].ravel()
                    X0_mean = np.mean(X0)
                    # X0 = np.percentile(X0, [25, 75])


                    X1 = nwp[var][np.ix_(lats, longs)].ravel()
                    X1_mean = np.mean(X1)
                    X1 = np.percentile(X1, [10, 90])


                    X2 = nwp_next[var][np.ix_(lats, longs)].ravel()
                    X2_mean = np.mean(X2)
                    # X2 = np.percentile(X2, [25, 75])


                    var_name = 'flux' if var == 'Flux' else 'wind'
                    var_sort = 'fl' if var == 'Flux' else 'ws'
                    col = [var_name + '.' + str(narea)] + ['p_' + var_name + '.' + str(narea)] + ['n_' + var_name + '.' + str(narea)]

                    col = col + [var_sort + str(i) + '.' + str(narea) for i in range(2)]
                    # col = col + ['p_' + var_sort + str(i)  + '.' + str(narea) for i in range(2)]
                    # col = col + ['n_' + var_sort + str(i)  + '.' + str(narea) for i in range(2)]

                    X = np.hstack((X1_mean, X0_mean, X2_mean, X1))
                    # X = np.hstack((X1_mean, X0_mean, X2_mean, X1, X0, X2))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)

                elif var in {'WD', 'Cloud'}:
                    X1 = nwp[var][np.ix_(lats, longs)].ravel()
                    X1_mean = np.mean(X1)
                    X1 = np.percentile(X1, [10, 90])

                    var_name = 'cloud' if var == 'Cloud' else 'direction'
                    var_sort = 'cl' if var == 'Cloud' else 'wd'
                    col = [var_name + '.' + str(narea)]
                    col = col + [var_sort + str(i)  + '.' + str(narea) for i in range(2)]

                    X = np.hstack((X1_mean, X1, ))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)

                elif (var in {'Temperature'}) or ((var == 'WS') and (model_type=='pv')):
                    X2 = nwp_next[var][np.ix_(lats, longs)].ravel()
                    X2_mean = np.mean(X2)
                    # X2 = np.percentile(X2, [25, 75])


                    var_name = 'Temp' if var == 'Temperature' else 'wind'
                    var_sort = 'tp' if var == 'Temperature' else 'ws'
                    col = [var_name + '.' + str(narea)]
                    # col = col + [var_sort + str(i) + '.' + str(narea) for i in range(2)]

                    X = X2_mean
                    # X = np.hstack((X2_mean, X2))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)
                else:
                    continue
        return inp, inp_3d,

    def create_sample(self, date, nwp, nwp_prev, nwp_next, lats, longs, model_type):
        inp = pd.DataFrame()
        if model_type == 'pv':
            inp = pd.concat([inp, pd.DataFrame(np.stack([date.hour, date.month]).reshape(-1,1).T, index=[date],
                                                                                   columns=['hour', 'month'])])
        inp_3d = np.array([])
        for var in sorted(self.variables):
            if ((var == 'WS') and (model_type =='wind')) or ((var == 'Flux') and (model_type == 'pv')):
                X0 = nwp_prev[var][np.ix_(lats, longs)].T
                if self.compress:
                    X0 = rescale_mean(X0)
                inp_3d = stack_2d(inp_3d, X0, False)
                X0_level0 = X0[2, 2]


                X1 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    X1 = rescale_mean(X1)
                inp_3d = stack_2d(inp_3d, X1, False)
                X1_level1 = X1[2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = np.percentile(X1_level3d, [5, 50, 95])

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = np.mean(X1_level3u)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = np.percentile(X1_level4d, [5, 50, 95])

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = np.percentile(X1_level4u, [5, 50, 95])

                X2 = nwp_next[var][np.ix_(lats, longs)].T
                if self.compress:
                    X2 = rescale_mean(X2)
                inp_3d = stack_2d(inp_3d, X2, False)
                X2_level0 = X2[2, 2]

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'
                col = ['p_' + var_name] + ['n_' + var_name] + [var_name]
                col = col + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                              range(1)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]



                X = np.hstack((X0_level0, X2_level0, X1_level1, X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)
            elif var in {'WD', 'Cloud'}:
                X1 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    X1 = rescale_mean(X1)
                inp_3d = stack_2d(inp_3d, X1, False)
                X1_level1 = X1[2, 2]

                ind = [[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)]
                ind = np.array(ind)
                X1_level3d = np.hstack([X1[indices[0], indices[1]].reshape(-1,1) for indices in ind])
                level = var + '_curr_mid_down'
                X1_level3d = np.percentile(X1_level3d, [5, 50, 95])

                ind = [[2, 3], [3, 2], [3, 3]]
                ind = np.array(ind)
                X1_level3u = np.hstack([X1[indices[0], indices[1]].reshape(-1,1) for indices in ind])
                level = var + '_curr_mid_up'
                X1_level3u = np.mean(X1_level3u)

                ind = [[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)]
                ind = np.array(ind)
                X1_level4d = np.hstack([X1[indices[0], indices[1]].reshape(-1,1) for indices in ind])
                level = var + '_curr_out_down'
                X1_level4d = np.percentile(X1_level4d, [5, 50, 95])

                ind = [[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)]
                ind = np.array(ind)
                X1_level4u = np.hstack([X1[indices[0], indices[1]].reshape(-1,1) for indices in ind])
                level = var + '_curr_out_up'
                X1_level4u = np.percentile(X1_level4u, [5, 50, 95])

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'
                col = [var_name] + [var_sort + '_l1.' + str(i) for i in range(3)] + [var_sort + '_l2.' + str(i) for i in
                                                                                     range(1)]
                col = col + [var_sort + '_l3d.' + str(i) for i in range(3)] + [var_sort + '_l3u.' + str(i) for i in
                                                                               range(3)]

                X = np.hstack((X1_level1, X1_level3d, X1_level3u, X1_level4d
                               , X1_level4u))
                inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)
            elif (var in {'Temperature'}) or ((var == 'WS') and (model_type == 'pv')):
                X2 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    X2 = rescale_mean(X2)
                inp_3d = stack_2d(inp_3d, X2, False)
                X2_level0 = X2[2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'
                var_sort = 'tp' if var == 'Temperature' else 'ws'
                col = [var_name]

                X = X2_level0
                inp = pd.concat([inp, pd.DataFrame(X.reshape(-1,1).T, index=[date], columns=col)], axis=1)
            else:
                continue
        return inp, inp_3d,

    def train_PCA(self, data, components, level):
        pass

    def PCA_transform(self, data, components, level):
        pass

