import logging
import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy.interpolate import interp2d
from workalendar.europe import Portugal, Greece


def rescale(arr, nrows, ncol):
    W, H = arr.shape
    new_W, new_H = (nrows, ncol)
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_H), xrange(new_W))

    return new_arr


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
    elif len(sample.shape) != len(X.shape):
        X = np.vstack((X, sample[np.newaxis]))
    else:
        X = np.vstack((X, sample))
    return X

class st_miguel(Portugal):

    FIXED_HOLIDAYS = Portugal.FIXED_HOLIDAYS + (
        (4, 11, "Dia da Liberdade"),
        (7, 18, "Dia de Portugal"),
        (6, 29, " Dia de S. Pedro"),
    )
    def get_fixed_holidays(self, year):
        days = super().get_fixed_holidays(year)
        return days

    def get_variable_days(self, year):
        days = super().get_variable_days(year)
        if year > 2015 or year < 2013:
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=36), "Santo Cristo"))
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=50), "Pombinha"))
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=64), "Dia do Corpo de Deus"))
        return days

    def get_extras(self, year):
        days = []
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=36))
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=50))
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=64))
        return days

class dataset_creator_load():

    def __init__(self, projects_group, projects, data, path_nwp, nwp_model, nwp_resolution, data_variables, njobs=1,
                 test=False):
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
        handler = logging.FileHandler(
            os.path.join(os.path.dirname(self.path_nwp), 'log_' + self.projects_group + '.log'), 'a')
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

    def check_empty_nwp(self, nwp, variables):
        flag = True
        for var in variables:
            if nwp[var].shape[0] == 0:
                flag = False
                break
        return flag

    def stack_daily_nwps(self, t, data, lats, longs, path_nwp, nwp_model, projects, variables, compress):

        X = dict()
        fname = os.path.join(path_nwp, nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
        if os.path.exists(fname):
            nwps = joblib.load(fname)

            pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H').strftime(
                '%d%m%y%H%M')
            for project in projects:
                X[project['_id']] = pd.DataFrame()
                areas = project['static_data']['areas']
                x = pd.DataFrame()
                for date in pdates:
                    try:

                        nwp = nwps[date]

                        if nwp['lat'].shape[0] == 0:
                            area_group = self.projects[0]['static_data']['area_group']
                            resolution = self.projects[0]['static_data']['NWP_resolution']
                            nwp['lat'] = np.arange(area_group[0][0], area_group[1][0] + resolution / 2,
                                                   resolution).reshape(-1, 1)
                            nwp['long'] = np.arange(area_group[0][1], area_group[1][1] + resolution / 2,
                                                    resolution).reshape(-1, 1).T
                        for var in nwp.keys():
                            if not var in {'lat', 'long'}:
                                if nwp['lat'].shape[0] != nwp[var].shape[0]:
                                    nwp[var] = nwp[var].T

                        if 'WS' in variables and not 'WS' in nwp.keys():
                            if 'Uwind' in nwp.keys() and 'Vwind' in nwp.keys():
                                if nwp['Uwind'].shape[0] > 0 and nwp['Vwind'].shape[0] > 0:
                                    r2d = 45.0 / np.arctan(1.0)
                                    wspeed = np.sqrt(np.square(nwp['Uwind']) + np.square(nwp['Vwind']))
                                    wdir = np.arctan2(nwp['Uwind'], nwp['Vwind']) * r2d + 180
                                    nwp['WS'] = wspeed
                                    nwp['WD'] = wdir
                        if 'Temp' in nwp.keys():
                            nwp['Temperature'] = nwp['Temp']
                            del  nwp['Temp']

                        date = pd.to_datetime(date, format='%d%m%y%H%M')
                        if self.check_empty_nwp(nwp, variables):
                            inp = self.create_sample_nwp(date, nwp, lats[project['_id']], longs[project['_id']])
                            x = pd.concat([x, inp])
                    except:
                        continue
                if x.shape[0] > 0:
                    X[project['_id']] = x
                    cols = ['Temp' + '_' + area for area in lats[project['_id']].keys()]
                    X[project['_id']]['Temp_max'] = x[cols].mean(axis=1).max()
                    X[project['_id']]['Temp_min'] = x[cols].mean(axis=1).min()

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return (X, t.strftime('%d%m%y%H%M'))

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

        if nwp['lat'].shape[0] == 0:
            area_group = self.projects[0]['static_data']['area_group']
            resolution = self.projects[0]['static_data']['NWP_resolution']
            nwp['lat'] = np.arange(area_group[0][0], area_group[1][0] + resolution / 2, resolution).reshape(-1, 1)
            nwp['long'] = np.arange(area_group[0][1], area_group[1][1] + resolution / 2, resolution).reshape(-1, 1).T

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

    def make_dataset_load_short_term(self):
        X = dict()

        project = self.projects[0]

        X[project['_id']] = pd.DataFrame()

        file_nwp = 'weather_data_test.csv'

        X[project['_id']] = pd.read_csv(os.path.join(project['static_data']['path_data'], file_nwp), header=0,
                                            index_col=0, parse_dates=True, dayfirst=True)

        data_path = project['static_data']['path_data']

        predictions = dict()
        predictions[project['_id']] = joblib.load(os.path.join(project['static_data']['path_data']
                                                                   , 'predictions_short_term.pickle'))

        dataset_X, dataset_y, X_3d = self.create_dataset_short_term_eval(X[project['_id']], predictions[project['_id']]
                                                                         , data_path, start_index=9001,
                                                                         test=self.isfortest)
        if dataset_y.isna().any().values[0]:
            dataset_X = dataset_X.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

            if len(X_3d.shape) > 1:
                X_3d = np.delete(X_3d, np.where(dataset_y.isna())[0], axis=0)
            dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

        if dataset_X.isna().any().values[0]:
            dataset_y = dataset_y.drop(dataset_X.index[np.where(dataset_X.isna())[0]])

            if len(X_3d.shape) > 1:
                X_3d = np.delete(X_3d, np.where(dataset_X.isna())[0], axis=0)
            dataset_X = dataset_X.drop(dataset_X.index[np.where(dataset_X.isna())[0]])
        dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv'))
        dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))
        joblib.dump(X_3d, os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle'))


    def make_dataset_load(self):
        X = dict()

        for project in self.projects:
            X[project['_id']] = pd.DataFrame()

        if self.isfortest:
            file_nwp = 'weather_data_test.csv'
        else:
            file_nwp = 'weather_data.csv'
        if not os.path.exists(os.path.join(self.projects[0]['static_data']['path_data'], file_nwp)):

            lats, longs = self.lats_longs()

            nwp = self.stack_daily_nwps(self.dates[-1], self.data, lats, longs, self.path_nwp, self.nwp_model,
                                        self.projects, self.variables,
                                        self.compress)
            nwp_daily = Parallel(n_jobs=self.njobs)(
                delayed(self.stack_daily_nwps)(t, self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
                                               self.variables,
                                               self.compress) for t in self.dates)

            for nwp in nwp_daily:
                for project in self.projects:
                    if nwp[0][project['_id']].shape[0] != 0:
                        X[project['_id']] = pd.concat([X[project['_id']], nwp[0][project['_id']]])

                        self.logger.info('All Inputs stacked for date %s', nwp[1])
            for project in self.projects:
                X[project['_id']].to_csv(os.path.join(project['static_data']['path_data'], file_nwp))
        else:
            for project in self.projects:
                X[project['_id']] = pd.read_csv(os.path.join(project['static_data']['path_data'], file_nwp), header=0,
                                                index_col=0, parse_dates=True, dayfirst=True)
        for project in self.projects:
            data_path = project['static_data']['path_data']

            if self.isfortest:
                if project['static_data']['horizon'] == 'day_ahead':
                    dataset_X, dataset_y, X_3d = self.create_dataset(X[project['_id']], data_path, start_index = 9001,
                                                                 test=self.isfortest)
                elif project['static_data']['horizon'] == 'short-term':
                    dataset_X, dataset_y, X_3d = self.create_dataset_short_term(X[project['_id']], data_path, start_index = 200,
                                                                     test=self.isfortest)
                if dataset_y.isna().any().values[0]:
                    dataset_X = dataset_X.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                    if len(X_3d.shape) > 1:
                        X_3d = np.delete(X_3d, np.where(dataset_y.isna())[0], axis=0)
                    dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])
                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))
                joblib.dump(X_3d, os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])
            else:
                if project['static_data']['horizon'] == 'day_ahead':
                    dataset_X, dataset_y, X_3d = self.create_dataset(X[project['_id']], data_path, start_index=9001,
                                                                 test=self.isfortest)
                elif project['static_data']['horizon'] == 'short-term':
                    dataset_X, dataset_y, X_3d = self.create_dataset_short_term(X[project['_id']], data_path, start_index = 200,
                                                                     test=self.isfortest)
                if dataset_y.isna().any().values[0]:
                    dataset_X = dataset_X.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                    if len(X_3d.shape) > 1:
                        X_3d = np.delete(X_3d, np.where(dataset_y.isna())[0], axis=0)
                    dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])
                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))
                joblib.dump(X_3d, os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])

    def sp_index(self, r):
        ### Is modified
        if self.projects[0]['_id'] == 'St_Miguel':
            cal = st_miguel()
            extra = cal.get_extras(r.year)
            if cal.is_holiday(r, extra):
                sp = 100

            else:
                if r.dayofweek == 6:
                    sp = 50
                else:
                    sp = 0
        else:
            cal = Greece()
            # {'New year','Epiphany','Clean Monday','Independence Day','Good Friday','Easter Saturday','Easter Sunday',
            # 'Easter Monday','Labour Day','Pentecost','Whit Monday','Assumption of Mary to Heaven','Ohi Day',
            # 'Christmas Eve','Christmas Day','Glorifying Mother of God','Last day of year'}
            if cal.is_holiday(r):
                sp = 100

            else:
                if r.dayofweek == 6:
                    sp = 50
                else:
                    sp = 0
        return sp

    def create_dataset(self, nwps, data_path, start_index=9001, test=False):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['hour'] = self.data.index.hour
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]

        col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(columns=['Temp_max']).columns.tolist() + [
            'Temp_month', 'Temp_sp_days']
        col += ['load_' + str(i) for i in range(43)]
        col += ['load_' + str(i) for i in range(43, 51)]
        col += ['Temp_max_' + str(i) for i in range(7)]
        col += ['Temp_min_' + str(i) for i in range(7)]
        col += ['sp_index_' + str(i) for i in range(7)]
        dataset = pd.DataFrame(columns=col)
        target = pd.DataFrame(columns=['target'])
        dataset_3d = np.array([])

        nwps_lstm = nwps.copy(deep=True)
        for var in self.variables:
            if var == 'WS':
                var = 'wind'
            elif var == 'WD':
                var = 'direction'
            elif var == 'Temperature':
                var = 'Temp'
            cols = [col for col in nwps.columns if str.lower(var) in str.lower(col)]
            nwps_lstm[str.lower(var)] = nwps_lstm[cols].mean(axis=1).values
        lags1 = np.hstack(
            [np.arange(48, 75), np.arange(95, 98), 96, 120, 144, np.arange(166, 175), 192, ])
        lags2 = np.hstack(
            [np.arange(8735, 8741), 8760, 8736 + 168])
        lags_days = np.arange(1, 8)

        for date in self.data.index[start_index:]:
            print('Input for ', date)
            date_inp1 = [date - pd.DateOffset(hours=int(l)) for l in lags1]
            date_inp2 = [date - pd.DateOffset(hours=int(l)) for l in lags2]
            date_days = [date - pd.DateOffset(days=int(l)) for l in lags_days]

            try:
                temp_max = nwps[['Temp_max']].loc[date].values
                var_imp = np.hstack((temp_max, self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                     nwps.drop(columns=['Temp_max']).loc[date].values,
                                     np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                     np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))

                col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(
                    columns=['Temp_max']).columns.tolist() + ['Temp_month', 'Temp_sp_days']

                var_unimp = np.hstack((
                    self.data.loc[date_inp1, self.projects[0]['_id']].values,
                    self.data.loc[date_inp2, self.projects[0]['_id']].values,
                    nwps.loc[date_days, 'Temp_max'].values,
                    nwps.loc[date_days, 'Temp_min'].values,
                    [self.sp_index(d) for d in date_days]
                ))
                col += ['load_' + str(i) for i in range(43)]
                col += ['load_' + str(i) for i in range(43, 51)]
                col += ['Temp_max_' + str(i) for i in range(7)]
                col += ['Temp_min_' + str(i) for i in range(7)]
                col += ['sp_index_' + str(i) for i in range(7)]

                temp_max = nwps[['Temp_max']].loc[date].values
                var_3d = np.hstack((np.array([0]),
                                    nwps_lstm[['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min',
                                               'Temp_' + self.projects[0]['_id']]].loc[
                                        date].values,
                                    self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                    np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                    np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))
                for d in date_inp1:
                    temp_max = nwps[['Temp_max']].loc[d].values
                    v = np.hstack(
                        (self.data.loc[d, self.projects[0]['_id']],
                         nwps_lstm[
                             ['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min',
                              'Temp_' + self.projects[0]['_id']]].loc[d].values,
                         self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[d].values,
                         np.power(self.data['month'].loc[d] * temp_max / 12, 3),
                         np.power(self.data['sp_index'].loc[d] * temp_max / 100, 3)))
                    var_3d = np.vstack((var_3d, v))


            except:
                continue
            inp = np.hstack((var_imp, var_unimp))

            inp1 = pd.DataFrame(inp.reshape(-1, 1).T, index=[date], columns=col)
            targ1 = pd.DataFrame(self.data[self.projects[0]['_id']].loc[date], index=[date], columns=['target'])
            if not inp1.isnull().any(axis=1).values and not targ1.isnull().any().values:
                dataset = pd.concat([dataset, inp1])
                target = pd.concat([target, targ1])
                if dataset_3d.shape[0] == 0:
                    dataset_3d = var_3d
                elif len(dataset_3d.shape) == 2:
                    dataset_3d = np.stack((dataset_3d, var_3d))
                else:
                    dataset_3d = np.vstack((dataset_3d, var_3d[np.newaxis, :, :]))
        if not test:
            corr = []
            for f in range(dataset.shape[1]):
                corr.append(np.abs(np.corrcoef(dataset.values[:, f], target.values.ravel())[1, 0]))
            ind = np.argsort(np.array(corr))[::-1]
            columns = dataset.columns[ind]
            dataset = dataset[columns]
            joblib.dump(ind, os.path.join(data_path, 'dataset_columns_order.pickle'))
        else:
            ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
            columns = dataset.columns[ind]
            dataset = dataset[columns]
        return dataset, target, dataset_3d

    def smooth(self, x, window_len=11, window='hanning'):

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def create_dataset_short_term(self, nwps, data_path, start_index=9001, test=False):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['hour'] = self.data.index.hour
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]


        col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(columns=['Temp_max']).columns.tolist() + [
            'Temp_month', 'Temp_sp_days']
        col += ['load_' + str(i) for i in range(30)]
        col += ['Temp_max_' + str(i) for i in range(7)]
        col += ['Temp_min_' + str(i) for i in range(7)]
        col += ['sp_index_' + str(i) for i in range(7)]
        dataset = pd.DataFrame(columns=col)
        target = pd.DataFrame(columns=['target'])
        dataset_3d = np.array([])

        nwps_lstm = nwps.copy(deep=True)
        for var in self.variables:
            if var == 'WS':
                var = 'wind'
            elif var == 'WD':
                var = 'direction'
            elif var == 'Temperature':
                var = 'Temp'
            cols = [col for col in nwps.columns if str.lower(var) in str.lower(col)]
            nwps_lstm[str.lower(var)] = nwps_lstm[cols].mean(axis=1).values

        res = self.create_inp(self.data.index[start_index], nwps, nwps_lstm)
        results=Parallel(n_jobs=self.njobs)(
        delayed(self.create_inp)(t, nwps, nwps_lstm) for t in self.data.index[start_index:])

        for res in results:
            if len(res[0].shape) > 1:
                inp1 = res[0]
                targ1 = res[1]
                var_3d = res[2]
                if not inp1.isnull().any(axis=1).values and not targ1.isnull().any().values:
                    dataset = pd.concat([dataset, inp1])
                    target = pd.concat([target, targ1])
                    if dataset_3d.shape[0] == 0:
                        dataset_3d = var_3d
                    elif len(dataset_3d.shape) == 2:
                        dataset_3d = np.stack((dataset_3d, var_3d))
                    else:
                        dataset_3d = np.vstack((dataset_3d, var_3d[np.newaxis, :, :]))
        if not test:
            corr = []
            for f in range(dataset.shape[1]):
                corr.append(np.abs(np.corrcoef(dataset.values[:, f], target.values.ravel())[1, 0]))
            ind = np.argsort(np.array(corr))[::-1]
            columns = dataset.columns[ind]
            dataset = dataset[columns]
            joblib.dump(ind, os.path.join(data_path, 'dataset_columns_order.pickle'))
        else:
            ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
            columns = dataset.columns[ind]
            dataset = dataset[columns]
        return dataset, target, dataset_3d

    def create_inp(self, date, nwps, nwps_lstm):
        print('Input for ', date)
        lags1 = np.hstack(
            [np.arange(1, 4), np.arange(4, 12), np.arange(22, 26), np.arange(47, 52), np.arange(166, 175), 192, ])

        lags_days = np.arange(1, 8)

        date_inp1 = [date - pd.DateOffset(hours=int(l)) for l in lags1]
        date_days = [date.round('H') - pd.DateOffset(days=int(l)) for l in lags_days]

        try:
            temp_max = nwps[['Temp_max']].loc[date.round('H')].values
            var_imp = np.hstack((temp_max, self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                 nwps.drop(columns=['Temp_max']).loc[date.round('H')].values,
                                 np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                 np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))

            col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(
                columns=['Temp_max']).columns.tolist() + ['Temp_month', 'Temp_sp_days']

            var_unimp = np.hstack((
                self.data.loc[date_inp1, self.projects[0]['_id']].values,
                nwps.loc[date_days, 'Temp_max'].values,
                nwps.loc[date_days, 'Temp_min'].values,
                [self.sp_index(d) for d in date_days]
            ))
            col += ['load_' + str(i) for i in range(30)]
            col += ['Temp_max_' + str(i) for i in range(7)]
            col += ['Temp_min_' + str(i) for i in range(7)]
            col += ['sp_index_' + str(i) for i in range(7)]

            temp_max = nwps[['Temp_max']].loc[date.round('H')].values
            var_3d = np.hstack((np.array([0]),
                                nwps_lstm[['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min',
                                           'Temp_' + self.projects[0]['_id']]].loc[
                                    date.round('H')].values,
                                self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))
            for d in date_inp1:
                temp_max = nwps[['Temp_max']].loc[d.round('H')].values
                v = np.hstack(
                    (self.data.loc[d, self.projects[0]['_id']],
                     nwps_lstm[
                         ['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min', 'Temp_' + self.projects[0]['_id']]].loc[
                         d.round('H')].values,
                     self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[d].values,
                     np.power(self.data['month'].loc[d] * temp_max / 12, 3),
                     np.power(self.data['sp_index'].loc[d] * temp_max / 100, 3)))
                var_3d = np.vstack((var_3d, v))

        except:
            return np.array([]), np.array([]), np.array([])
        inp = np.hstack((var_imp, var_unimp))

        inp1 = pd.DataFrame(inp.reshape(-1, 1).T, index=[date], columns=col)
        targ1 = pd.DataFrame(self.data[self.projects[0]['_id']].loc[date], index=[date], columns=['target'])

        return inp1, targ1, var_3d

    def create_sample_nwp(self, date, nwp, lats, longs):

        inp = pd.DataFrame()
        for var in sorted(self.variables):
            if var in {'WS', 'Flux', 'WD', 'Cloud', 'Temperature'}:
                if isinstance(lats, dict) and isinstance(longs, dict):

                    for area in lats.keys():
                        X0 = nwp[var][np.ix_(lats[area], longs[area])]

                        X = np.mean(X0)

                        if var == 'Flux':
                            var_name = 'flux'
                        elif var == 'WS':
                            var_name = 'wind'
                        elif var == 'Cloud':
                            var_name = 'cloud'
                        elif var == 'Temperature':
                            var_name = 'Temp'
                        else:
                            var_name = 'direction'

                        col = [var_name + '_' + area]

                        inp = pd.concat([inp, pd.DataFrame(X.reshape(-1, 1).T, index=[date], columns=col)], axis=1)
                else:
                    X0 = nwp[var][np.ix_(lats, longs)]

                    X = np.mean(X0)

                    if var == 'Flux':
                        var_name = 'flux'
                    elif var == 'WS':
                        var_name = 'wind'
                    elif var == 'Cloud':
                        var_name = 'cloud'
                    elif var == 'Temperature':
                        var_name = 'Temp'
                    else:
                        var_name = 'direction'

                    col = [var_name]

                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1, 1).T, index=[date], columns=col)], axis=1)
            else:
                continue

        return inp

    def create_dataset_short_term_eval(self, nwps, predictions, data_path, start_index=9001, test=False):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['hour'] = self.data.index.hour
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]

        col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(columns=['Temp_max']).columns.tolist() + [
            'Temp_month', 'Temp_sp_days']
        col += ['load_' + str(i) for i in range(30)]
        col += ['Temp_max_' + str(i) for i in range(7)]
        col += ['Temp_min_' + str(i) for i in range(7)]
        col += ['sp_index_' + str(i) for i in range(7)]
        dataset = pd.DataFrame(columns=col)
        target = pd.DataFrame(columns=['target'])
        dataset_3d = np.array([])

        nwps_lstm = nwps.copy(deep=True)
        for var in self.variables:
            if var == 'WS':
                var = 'wind'
            elif var == 'WD':
                var = 'direction'
            elif var == 'Temperature':
                var = 'Temp'
            cols = [col for col in nwps.columns if str.lower(var) in str.lower(col)]
            nwps_lstm[str.lower(var)] = nwps_lstm[cols].mean(axis=1).values

        res = self.create_inp_eval(self.data.index[start_index], nwps, nwps_lstm, predictions)
        results=Parallel(n_jobs=self.njobs)(
        delayed(self.create_inp_eval)(t, nwps, nwps_lstm, predictions) for t in self.data.index[start_index:])

        for res in results:
            if len(res[0].shape) > 1:
                inp1 = res[0]
                targ1 = res[1]
                var_3d = res[2]
                if not inp1.isnull().any(axis=1).values and not targ1.isnull().any().values:
                    dataset = pd.concat([dataset, inp1])
                    target = pd.concat([target, targ1])
                    if dataset_3d.shape[0] == 0:
                        dataset_3d = var_3d
                    elif len(dataset_3d.shape) == 2:
                        dataset_3d = np.stack((dataset_3d, var_3d))
                    else:
                        dataset_3d = np.vstack((dataset_3d, var_3d[np.newaxis, :, :]))

        ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
        columns = dataset.columns[ind]
        dataset = dataset[columns]
        return dataset, target, dataset_3d

    def create_inp_eval(self, date, nwps, nwps_lstm, predictions):
        lags1 = np.hstack(
            [np.arange(1, 4),  np.arange(4, 12),  np.arange(22, 26), np.arange(47, 52), np.arange(166, 175), 192, ])

        lags_days = np.arange(1, 8)

        timestep = 60

        preds = predictions
        hor = preds.columns[-1] + timestep
        t = date - pd.DateOffset(minutes=hor)
        preds = preds.loc[t].to_frame().T
        dates_pred = [t + pd.DateOffset(minutes=h) for h in preds.columns]
        pred = pd.DataFrame(preds.values.ravel(), index=dates_pred, columns=[self.projects[0]['_id']])
        data_temp = pd.concat([self.data[self.projects[0]['_id']].iloc[np.where(self.data.index < t)].to_frame(), pred])

        print('Input for ', date)
        date_inp1 = [date - pd.DateOffset(hours=int(l)) for l in lags1]
        date_days = [date.round('H') - pd.DateOffset(days=int(l)) for l in lags_days]

        try:
            temp_max = nwps[['Temp_max']].loc[date.round('H')].values
            var_imp = np.hstack((temp_max, self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                 nwps.drop(columns=['Temp_max']).loc[date.round('H')].values,
                                 np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                 np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))

            col = ['Temp', 'hour', 'month', 'sp_index', 'dayweek'] + nwps.drop(
                columns=['Temp_max']).columns.tolist() + ['Temp_month', 'Temp_sp_days']


            var_unimp = np.hstack((
                data_temp.loc[date_inp1, self.projects[0]['_id']].values,
                nwps.loc[date_days, 'Temp_max'].values,
                nwps.loc[date_days, 'Temp_min'].values,
                [self.sp_index(d) for d in date_days]
            ))
            col += ['load_' + str(i) for i in range(30)]
            col += ['Temp_max_' + str(i) for i in range(7)]
            col += ['Temp_min_' + str(i) for i in range(7)]
            col += ['sp_index_' + str(i) for i in range(7)]

            temp_max = nwps[['Temp_max']].loc[date.round('H')].values
            var_3d = np.hstack((np.array([0]),
                                nwps_lstm[['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min', 'Temp_' + self.projects[0]['_id']]].loc[
                                    date.round('H')].values,
                                self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[date].values,
                                np.power(self.data['month'].loc[date] * temp_max / 12, 3),
                                np.power(self.data['sp_index'].loc[date] * temp_max / 100, 3)))
            for d in date_inp1:
                temp_max = nwps[['Temp_max']].loc[d.round('H')].values
                v = np.hstack(
                    (data_temp.loc[d, self.projects[0]['_id']],
                     nwps_lstm[
                         ['cloud', 'wind', 'direction', 'Temp_max', 'Temp_min', 'Temp_' + self.projects[0]['_id']]].loc[d.round('H')].values,
                     self.data[['hour', 'month', 'sp_index', 'dayweek']].loc[d].values,
                     np.power(self.data['month'].loc[d] * temp_max / 12, 3),
                     np.power(self.data['sp_index'].loc[d] * temp_max / 100, 3)))
                var_3d = np.vstack((var_3d, v))

        except:
            return np.array([]), np.array([]), np.array([])
        inp = np.hstack((var_imp, var_unimp))

        inp1 = pd.DataFrame(inp.reshape(-1, 1).T, index=[date], columns=col)
        targ1 = pd.DataFrame(self.data[self.projects[0]['_id']].loc[date], index=[date], columns=['target'])

        return inp1, targ1, var_3d