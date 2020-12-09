import joblib
import logging
import numpy as np
import os
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy.interpolate import interp2d
from workalendar.europe import Greece


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


class dataset_creator_ecmwf():

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
                        date = pd.to_datetime(date, format='%d%m%y%H%M')

                        if self.check_empty_nwp(nwp, variables):
                            inp = self.create_sample_nwp(date, nwp, lats[project['_id']], longs[project['_id']])
                            x = pd.concat([x, inp])
                    except:
                        continue
                if x.shape[0] > 0:
                    X[project['_id']] = x.mean().to_frame().transpose()
                    X[project['_id']]['Temp_max'] = x['Temp'].max()
                    X[project['_id']]['Temp_min'] = x['Temp'].min()
                    X[project['_id']].index = [
                        pd.to_datetime((t + pd.DateOffset(hours=24)).strftime('%d%m%y'), format='%d%m%y')]

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

    def make_dataset_ecmwf(self):
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

            if self.isfortest:
                dataset_X, dataset_y, X_3d = self.create_dataset(X[project['_id']], start_index=372)
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
                dataset_X, dataset_y, X_3d = self.create_dataset(X[project['_id']])
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
        cal = Greece()
        # {'New year','Epiphany','Clean Monday','Independence Day','Good Friday','Easter Saturday','Easter Sunday','Easter Monday','Labour Day','Pentecost','Whit Monday','Assumption of Mary to Heaven','Ohi Day','Christmas Eve'
        # ,'Christmas Day','Glorifying Mother of God','Last day of year'}
        if cal.is_holiday(r):
            sp = 100

        else:
            if r.dayofweek == 6:
                sp = 50
            else:
                sp = 0
        return sp

    def create_dataset(self, nwps, start_index=753):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]
        self.data['sin_month'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['sin_dayweek'] = np.sin(2 * np.pi * self.data['dayweek'] / 7)
        self.data['cos_dayweek'] = np.cos(2 * np.pi * self.data['dayweek'] / 7)

        dataset = pd.DataFrame()
        target = pd.Series(name='target')
        dataset_3d = np.array([])

        lags1 = np.hstack(
            [np.arange(1, 9), 14, 21])

        for date in self.data.index[start_index:]:
            date_inp1 = [date - pd.DateOffset(days=int(l)) for l in lags1]
            date_inp2 = [date - pd.DateOffset(days=364), date - pd.DateOffset(days=364) - pd.DateOffset(days=1),
                         date - pd.DateOffset(days=364) - pd.DateOffset(days=7)]
            try:
                temp = nwps[['Temp']].loc[date].values
                var_imp = np.hstack(
                    (nwps[['flux', 'wind', 'cloud', 'direction', 'Temp_max', 'Temp_min']].loc[date].values,
                     self.data[['month', 'sp_index', 'dayweek']].loc[date].values,
                     temp,
                     np.power(self.data['month'].loc[date] * temp / 12, 3),
                     np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))

                col = ['flux', 'wind', 'cloud', 'direction', 'Temp_max', 'Temp_min', 'month', 'sp_index', 'dayweek',
                       'Temp', 'Temp_month', 'Temp_sp_days']

                var_unimp = np.hstack((
                    self.data.loc[date_inp1, 'Athens_24'].values,
                    self.data.loc[date_inp2, 'Final/Ζητούμενο'].values,
                    nwps.loc[date_inp1, 'Temp_max'].values,
                    nwps.loc[date_inp1, 'Temp_min'].values,
                    nwps.loc[date_inp1, 'Temp'].values,
                    self.data[['sin_month',
                               'cos_month', 'sin_dayweek', 'cos_dayweek']].loc[date].values
                ))
                col += ['Ath24_' + str(i) for i in range(10)]
                col += ['final_' + str(i) for i in range(3)]
                col += ['Temp_max_' + str(i) for i in range(10)]
                col += ['Temp_min_' + str(i) for i in range(10)]
                col += ['Temp_' + str(i) for i in range(10)]
                col += ['sin_month',
                        'cos_month', 'sin_dayweek', 'cos_dayweek']

                temp = nwps[['Temp']].loc[date].values
                var_3d = np.hstack((np.array([0]),
                                    nwps[['flux', 'wind', 'cloud', 'direction', 'Temp_max', 'Temp_min']].loc[
                                        date].values,
                                    self.data[['month', 'sp_index', 'dayweek']].loc[date].values,
                                    temp,
                                    np.power(self.data['month'].loc[date] * temp / 12, 3),
                                    np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                for d in date_inp1:
                    temp = nwps[['Temp']].loc[d].values
                    v = np.hstack(
                        (self.data.loc[d, 'Athens_24'],
                         nwps[['flux', 'wind', 'cloud', 'direction', 'Temp_max', 'Temp_min']].loc[d].values,
                         self.data[['month', 'sp_index', 'dayweek']].loc[d].values,
                         temp,
                         np.power(self.data['month'].loc[date] * temp / 12, 3),
                         np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                    var_3d = np.vstack((var_3d, v))

            except:
                continue
            inp = np.hstack((var_imp, var_unimp))

            inp1 = pd.Series(inp, index=col, name=date)
            targ1 = pd.Series(self.data['Final/Ζητούμενο'].loc[date], index=[date], name='target1')
            if not inp1.isnull().any() and not targ1.isnull().any():
                dataset = dataset.append(inp1)
                target = target.append(targ1)
                if dataset_3d.shape[0] == 0:
                    dataset_3d = var_3d
                elif len(dataset_3d.shape) == 2:
                    dataset_3d = np.stack((dataset_3d, var_3d))
                else:
                    dataset_3d = np.vstack((dataset_3d, var_3d[np.newaxis, :, :]))

        return dataset, target, dataset_3d

    def create_sample_nwp(self, date, nwp, lats, longs):

        inp = pd.DataFrame()
        for var in sorted(self.variables):
            if var in {'WS', 'Flux', 'WD', 'Cloud', 'Temperature'}:
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


class dataset_creator_xmachina():

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

    def make_dataset_xmachina_curr(self):
        X = dict()

        for project in self.projects:
            data_path = project['static_data']['path_data']
            if self.isfortest:
                dataset_X, dataset_y, X_3d = self.create_dataset_curr(data_path, start_index=372, test=self.isfortest)
                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X_test.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y_test.csv'))
                joblib.dump(X_3d, os.path.join(project['static_data']['path_data'], 'dataset_lstm_test.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])
            else:
                dataset_X, dataset_y, X_3d = self.create_dataset_curr(data_path, start_index=372, test=self.isfortest)
                dataset_X.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_X.csv'))
                dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y.csv'))
                joblib.dump(X_3d, os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])

    def make_dataset_xmachina_dayahead(self):
        X = dict()

        for project in self.projects:
            data_path = project['static_data']['path_data']
            if self.isfortest:
                dataset_X, dataset_y, X_3d = self.create_dataset_dayahead(data_path, start_index=372,
                                                                          test=self.isfortest)
                dataset_X.to_csv(os.path.join(data_path, 'dataset_X_test.csv'))
                dataset_y.to_csv(os.path.join(data_path, 'dataset_y_test.csv'))
                joblib.dump(X_3d, os.path.join(data_path, 'dataset_lstm_test.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])
            else:
                dataset_X, dataset_y, X_3d = self.create_dataset_dayahead(data_path, start_index=730,
                                                                          test=self.isfortest)
                dataset_X.to_csv(os.path.join(data_path, 'dataset_X.csv'))
                dataset_y.to_csv(os.path.join(data_path, 'dataset_y.csv'))
                joblib.dump(X_3d, os.path.join(data_path, 'dataset_lstm.pickle'))
                self.logger.info('Datasets saved for project %s', project['_id'])

    def sp_index(self, r):
        ### Is modified
        cal = Greece()
        # {'New year','Epiphany','Clean Monday','Independence Day','Good Friday','Easter Saturday','Easter Sunday','Easter Monday','Labour Day','Pentecost','Whit Monday','Assumption of Mary to Heaven','Ohi Day','Christmas Eve'
        # ,'Christmas Day','Glorifying Mother of God','Last day of year'}
        if cal.is_holiday(r):
            sp = 100

        else:
            if r.dayofweek == 6:
                sp = 50
            else:
                sp = 0
        return sp

    def create_dataset_curr(self, data_path, start_index=753, test=False):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]
        self.data['sin_month'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['sin_dayweek'] = np.sin(2 * np.pi * self.data['dayweek'] / 7)
        self.data['cos_dayweek'] = np.cos(2 * np.pi * self.data['dayweek'] / 7)

        lags1 = np.hstack(
            [np.arange(1, 10), np.arange(14, 16), np.arange(21, 23), ])
        lags2 = [364, 365, 371]

        date = self.data.index[0]
        date_inp1 = [date - pd.DateOffset(days=int(l)) for l in lags1]
        date_inp2 = [date - pd.DateOffset(days=int(l)) for l in lags2]

        col = ['Ath6_0', 'temp_min', 'temp_mean', 'rh', 'precip', 'hdd_h', 'hdd_h2', 'month', 'sp_index', 'dayweek',
               'temp_max',
               'Temp_month', 'Temp_sp_days']
        col += ['Ath24_' + str(i) for i in range(len(date_inp1))]
        col += ['Ath6_' + str(i) for i in range(1, len(date_inp1) + 1)]
        col += ['final_' + str(i) for i in range(len(date_inp2))]
        col += ['customers_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['temp_max_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['hdd_h' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['sp_index' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        dataset = pd.DataFrame(columns=col)
        target = pd.DataFrame(columns=['target'])
        dataset_3d = np.array([])

        for date in self.data.index[start_index:]:
            date_inp1 = [date - pd.DateOffset(days=int(l)) for l in lags1]
            date_inp2 = [date - pd.DateOffset(days=int(l)) for l in lags2]
            temp = self.data[['temp_max']].loc[date].values
            var_imp = np.hstack((self.data.loc[date, 'Athens_6'],
                                 self.data[['temp_min', 'temp_mean',
                                            'rh', 'precip', 'hdd_h', 'hdd_h2']].loc[date].values,
                                 self.data[['month', 'sp_index', 'dayweek']].loc[date].values,
                                 temp,
                                 np.power(self.data['month'].loc[date] * temp / 12, 3),
                                 np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))

            col = ['Ath6_0', 'temp_min', 'temp_mean', 'rh', 'precip', 'hdd_h', 'hdd_h2', 'month', 'sp_index', 'dayweek',
                   'temp_max', 'Temp_month', 'Temp_sp_days']

            var_unimp = np.hstack((
                self.data.loc[date_inp1, 'Athens_24'].values,
                self.data.loc[date_inp1, 'Athens_6'].values,
                self.data.loc[date_inp2, 'Final/Ζητούμενο'].values,
                self.data.loc[date_inp1 + date_inp2, 'Αριθμός Πελατών από 1/1/2017'].values,
                self.data.loc[date_inp1 + date_inp2, 'temp_max'].values,
                self.data.loc[date_inp1 + date_inp2, 'hdd_h'].values,
                self.data.loc[date_inp1 + date_inp2, 'sp_index'].values,

            ))
            col += ['Ath24_' + str(i) for i in range(len(date_inp1))]
            col += ['Ath6_' + str(i) for i in range(1, len(date_inp1) + 1)]
            col += ['final_' + str(i) for i in range(len(date_inp2))]
            col += ['customers_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['temp_max_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['hdd_h' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['sp_index' + str(i) for i in range(len(date_inp1) + len(date_inp2))]

            temp = self.data[['temp_max']].loc[date].values
            var_3d = np.hstack((np.array([0]), self.data.loc[date, 'Athens_6'], self.data[['temp_min', 'temp_mean',
                                                                                           'rh', 'hdd_h',
                                                                                           'hdd_h2']].loc[date].values,
                                temp,
                                self.data.loc[date_inp1[0], 'Αριθμός Πελατών από 1/1/2017'],
                                np.power(self.data['month'].loc[date] * temp / 12, 3),
                                np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
            for d in date_inp1:
                temp = self.data[['temp_max']].loc[d].values
                v = np.hstack(
                    (self.data.loc[d, 'Athens_24'], self.data.loc[d, 'Athens_6'],
                     self.data[['temp_min', 'temp_mean', 'rh', 'hdd_h', 'hdd_h2']].loc[d].values,
                     temp,
                     self.data.loc[d, 'Αριθμός Πελατών από 1/1/2017'],
                     np.power(self.data['month'].loc[date] * temp / 12, 3),
                     np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                var_3d = np.vstack((var_3d, v))
            for d in date_inp2:
                temp = self.data[['temp_max']].loc[d].values
                v = np.hstack(
                    (self.data.loc[d, 'Final/Ζητούμενο'], self.data.loc[d, 'Athens_6'],
                     self.data[['temp_min', 'temp_mean', 'rh', 'hdd_h', 'hdd_h2']].loc[d].values,
                     temp,
                     self.data.loc[d, 'Αριθμός Πελατών από 1/1/2017'],
                     np.power(self.data['month'].loc[date] * temp / 12, 3),
                     np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                var_3d = np.vstack((var_3d, v))

            inp = np.hstack((var_imp, var_unimp))

            inp1 = pd.DataFrame(inp.reshape(-1, 1).T, index=[date], columns=col)
            targ1 = pd.DataFrame(self.data['Final/Ζητούμενο'].loc[date], index=[date], columns=['target'])
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

    def create_dataset_dayahead(self, data_path, start_index=753, test=False):
        self.data['dayweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['sp_index'] = [self.sp_index(d) for d in self.data.index]
        self.data['sin_month'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['sin_dayweek'] = np.sin(2 * np.pi * self.data['dayweek'] / 7)
        self.data['cos_dayweek'] = np.cos(2 * np.pi * self.data['dayweek'] / 7)

        lags1 = np.hstack(
            [np.arange(2, 10), np.arange(14, 16), np.arange(21, 23), ])
        lags2 = [364, 365, 371]

        date = self.data.index[0]
        date_inp1 = [date - pd.DateOffset(days=int(l)) for l in lags1]
        date_inp2 = [date - pd.DateOffset(days=int(l)) for l in lags2]

        col = ['temp_min', 'temp_mean', 'rh', 'precip', 'hdd_h', 'hdd_h2', 'month', 'sp_index', 'dayweek', 'temp_max',
               'Temp_month', 'Temp_sp_days']
        col += ['Ath6_0', 'temp_min_curr', 'temp_mean_curr', 'rh_curr', 'precip_curr', 'hdd_h_curr', 'hdd_h2_curr',
                'month_curr',
                'sp_index_curr', 'dayweek_curr',
                'temp_max_curr', 'Temp_month_curr', 'Temp_sp_days_curr']
        col += ['Ath24_' + str(i) for i in range(len(date_inp1))]
        col += ['Ath6_' + str(i) for i in range(1, len(date_inp1) + 1)]
        col += ['final_' + str(i) for i in range(len(date_inp2))]
        col += ['customers_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['temp_max_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['hdd_h' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        col += ['sp_index' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
        dataset = pd.DataFrame(columns=col)
        target = pd.DataFrame(columns=['target'])
        dataset_3d = np.array([])

        for date in self.data.index[start_index:]:
            date_inp1 = [date - pd.DateOffset(days=int(l)) for l in lags1]
            date_inp2 = [date - pd.DateOffset(days=int(l)) for l in lags2]
            temp = self.data[['pred_temp_max']].loc[date].values
            var_imp1 = np.hstack((
                self.data[['pred_temp_min', 'pred_temp_mean',
                           'pred_rh', 'pred_precip', 'pred_hdd_h', 'pred_hdd_h2']].loc[date].values,
                self.data[['month', 'sp_index', 'dayweek']].loc[date].values,
                temp,
                np.power(self.data['month'].loc[date] * temp / 12, 3),
                np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))

            col = ['temp_min', 'temp_mean', 'rh', 'precip', 'hdd_h', 'hdd_h2', 'month', 'sp_index', 'dayweek',
                   'temp_max', 'Temp_month', 'Temp_sp_days']

            date_curr = date - pd.DateOffset(days=int(1))
            temp = self.data[['pred_temp_max']].loc[date_curr].values
            var_imp = np.hstack((self.data.loc[date_curr, 'Athens_6'],
                                 self.data[['pred_temp_min', 'pred_temp_mean',
                                            'pred_rh', 'pred_precip', 'pred_hdd_h', 'pred_hdd_h2']].loc[
                                     date_curr].values,
                                 self.data[['month', 'sp_index', 'dayweek']].loc[date_curr].values,
                                 temp,
                                 np.power(self.data['month'].loc[date_curr] * temp / 12, 3),
                                 np.power(self.data['sp_index'].loc[date_curr] * temp / 100, 3)))

            col += ['Ath6_0', 'temp_min_curr', 'temp_mean_curr', 'rh_curr', 'precip_curr', 'hdd_h_curr', 'hdd_h2_curr',
                    'month_curr', 'sp_index_curr', 'dayweek_curr',
                    'temp_max_curr', 'Temp_month_curr', 'Temp_sp_days_curr']

            var_unimp = np.hstack((
                self.data.loc[date_inp1, 'Athens_24'].values,
                self.data.loc[date_inp1, 'Athens_6'].values,
                self.data.loc[date_inp2, 'Final/Ζητούμενο'].values,
                self.data.loc[date_inp1 + date_inp2, 'Αριθμός Πελατών από 1/1/2017'].values,
                self.data.loc[date_inp1 + date_inp2, 'temp_max'].values,
                self.data.loc[date_inp1 + date_inp2, 'hdd_h'].values,
                self.data.loc[date_inp1 + date_inp2, 'sp_index'].values,

            ))
            col += ['Ath24_' + str(i) for i in range(len(date_inp1))]
            col += ['Ath6_' + str(i) for i in range(1, len(date_inp1) + 1)]
            col += ['final_' + str(i) for i in range(len(date_inp2))]
            col += ['customers_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['temp_max_' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['hdd_h' + str(i) for i in range(len(date_inp1) + len(date_inp2))]
            col += ['sp_index' + str(i) for i in range(len(date_inp1) + len(date_inp2))]

            temp = self.data[['temp_max']].loc[date].values
            var_3d = np.hstack((np.array([0]), np.array([0]), self.data[['pred_temp_min', 'pred_temp_mean',
                                                                         'pred_rh', 'pred_hdd_h', 'pred_hdd_h2']].loc[
                date].values,
                                temp,
                                self.data.loc[date_inp1[0], 'Αριθμός Πελατών από 1/1/2017'],
                                np.power(self.data['month'].loc[date] * temp / 12, 3),
                                np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))

            date_curr = date - pd.DateOffset(days=int(1))
            temp = self.data[['temp_max']].loc[date_curr].values
            v = np.hstack(
                (np.array([0]), self.data.loc[date_curr, 'Athens_6'], self.data[['pred_temp_min', 'pred_temp_mean',
                                                                                 'pred_rh', 'pred_hdd_h',
                                                                                 'pred_hdd_h2']].loc[date_curr].values,
                 temp,
                 self.data.loc[date_inp1[0], 'Αριθμός Πελατών από 1/1/2017'],
                 np.power(self.data['month'].loc[date_curr] * temp / 12, 3),
                 np.power(self.data['sp_index'].loc[date_curr] * temp / 100, 3)))
            var_3d = np.vstack((var_3d, v))

            for d in date_inp1:
                temp = self.data[['temp_max']].loc[d].values
                v = np.hstack(
                    (self.data.loc[d, 'Athens_24'], self.data.loc[d, 'Athens_6'],
                     self.data[['temp_min', 'temp_mean', 'rh', 'hdd_h', 'hdd_h2']].loc[d].values,
                     temp,
                     self.data.loc[d, 'Αριθμός Πελατών από 1/1/2017'],
                     np.power(self.data['month'].loc[date] * temp / 12, 3),
                     np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                var_3d = np.vstack((var_3d, v))

            for d in date_inp2:
                temp = self.data[['temp_max']].loc[d].values
                v = np.hstack(
                    (self.data.loc[d, 'Final/Ζητούμενο'], self.data.loc[d, 'Athens_6'],
                     self.data[['temp_min', 'temp_mean', 'rh', 'hdd_h', 'hdd_h2']].loc[d].values,
                     temp,
                     self.data.loc[d, 'Αριθμός Πελατών από 1/1/2017'],
                     np.power(self.data['month'].loc[date] * temp / 12, 3),
                     np.power(self.data['sp_index'].loc[date] * temp / 100, 3)))
                var_3d = np.vstack((var_3d, v))

            inp = np.hstack((var_imp1, var_imp, var_unimp))

            inp1 = pd.DataFrame(inp.reshape(-1, 1).T, index=[date], columns=col)
            targ1 = pd.DataFrame(self.data['Final/Ζητούμενο'].loc[date], index=[date], columns=['target'])
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
