import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed

from Fuzzy_clustering.version2.common_utils.logging import create_logger
from Fuzzy_clustering.version2.dataset_manager.common_utils import check_empty_nwp
from Fuzzy_clustering.version2.dataset_manager.common_utils import rescale_mean
from Fuzzy_clustering.version2.dataset_manager.common_utils import stack_2d_dense
from Fuzzy_clustering.version2.dataset_manager.common_utils import stack_3d


class DatasetCreatorDense:

    def __init__(self, projects_group, projects, data, path_nwp, nwp_model, nwp_resolution, data_variables, njobs=1,
                 test=False, dates=None):
        self.projects = projects
        self.is_for_test = test
        self.projects_group = projects_group
        self.data = data
        self.path_nwp = path_nwp
        self.nwp_model = nwp_model
        self.nwp_resolution = nwp_resolution
        self.compress = True if self.nwp_resolution == 0.05 else False

        self.n_jobs = njobs
        self.variables = data_variables

        self.logger = create_logger(logger_name=__name__, abs_path=self.path_nwp,
                                    logger_path=f'log_{self.projects_group}.log', write_type='a')
        if not self.data is None:
            self.dates = self.check_dates()
        elif not dates is None:
            self.dates = dates

    def check_dates(self):
        start_date = pd.to_datetime(self.data.index[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(self.data.index[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)
        data_dates = pd.to_datetime(np.unique(self.data.index.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]
        self.logger.info('Dates are checked. Number of time samples is %s', str(len(dates)))
        return pd.DatetimeIndex(dates)

    def correct_nwps(self, nwp, variables):
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
            del nwp['Temp']

        return nwp
    
    def stack_by_sample(self, t, data, lats, longs, path_nwp, nwp_model, projects, variables, predictions):
        timestep = 60
        x = dict()
        y = dict()
        x_3d = dict()
        file_name = os.path.join(path_nwp, f"{nwp_model}_{t.strftime('%d%m%y')}.pickle")
        if os.path.exists(file_name):
            nwps = joblib.load(file_name)

            for project in projects:
                preds = predictions[project['_id']]
                hor = preds.columns[-1] + timestep
                p_dates = [t + pd.DateOffset(minutes=hor)]
                preds = preds.loc[t].to_frame().T
                dates_pred = [t + pd.DateOffset(minutes=h) for h in preds.columns]
                pred = pd.DataFrame(preds.values.ravel(), index=dates_pred, columns=[project['_id']])
                data_temp = pd.concat([data[project['_id']].iloc[np.where(data.index < t)].to_frame(), pred])

                project_id = project['_id']  # It's the project name, the park's name
                x[project_id] = pd.DataFrame()
                y[project_id] = pd.DataFrame()
                x_3d[project_id] = np.array([])
                areas = project['static_data']['areas']
                if isinstance(areas, list):
                    for date in p_dates:
                        date_nwp = date.round('H').strftime('%d%m%y%H%M')
                        try:
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date_nwp = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                                inp, inp_cnn = self.create_sample(date, nwp, nwp_prev, nwp_next, lats[project_id],
                                                                  longs[project_id], project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data_temp.loc[(date - pd.DateOffset(hours=1))].values
                                    inp['Obs_lag2'] = data_temp.loc[(date - pd.DateOffset(hours=2))].values
                                if not inp.isnull().any(axis=1).values and not np.isnan(data.loc[date, project_id]):
                                    x[project_id] = pd.concat([x[project_id], inp])
                                    x_3d[project_id] = stack_2d_dense(x_3d[project_id], inp_cnn, False)

                                    y[project_id] = pd.concat([y[project_id], pd.DataFrame(data.loc[date, project_id],
                                                                                       columns=['target'],
                                                                                       index=[date])])
                        except Exception:
                            continue
                else:
                    for date in p_dates:
                        try:
                            date_nwp = date.round('H').strftime('%d%m%y%H%M')
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date_nwp = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                                inp, inp_cnn = self.create_sample_country(date, nwp, nwp_prev, nwp_next,
                                                                          lats[project['_id']],
                                                                          longs[project['_id']],
                                                                          project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data_temp.loc[(date - pd.DateOffset(hours=1)), project_id].values
                                    inp['Obs_lag2'] = data_temp.loc[(date - pd.DateOffset(hours=2)), project_id].values
                                if not inp.isnull().any(axis=1).values and not np.isnan(data.loc[date, project_id]):
                                    x[project['_id']] = pd.concat([x[project['_id']], inp])
                                    x_3d[project['_id']] = stack_2d_dense(x_3d[project['_id']], inp_cnn, False)

                                    y[project['_id']] = pd.concat(
                                        [y[project['_id']], pd.DataFrame(data.loc[date, project['_id']],
                                                                         columns=['target'], index=[date])])
                        except Exception:
                            continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
            for project in projects:
                if len(x_3d[project['_id']].shape) == 3:
                    x_3d[project['_id']] = x_3d[project['_id']][np.newaxis, :, :, :]
        return x, y, x_3d, t.strftime('%d%m%y%H%M')

    def stack_daily_nwps(self, t, data, lats, longs, path_nwp, nwp_model, projects, variables):

        x = dict()
        y = dict()
        x_3d = dict()
        file_name = os.path.join(path_nwp, f"{nwp_model}_{t.strftime('%d%m%y')}.pickle")
        if os.path.exists(file_name):
            nwps = joblib.load(file_name)

            for project in projects:
                if project['static_data']['horizon'] == 'day_ahead':
                    p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H')
                else:
                    p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=24), freq='H')

                project_id = project['_id']  # It's the project name, the park's name 
                x[project_id] = pd.DataFrame()
                y[project_id] = pd.DataFrame()
                x_3d[project_id] = np.array([])
                areas = project['static_data']['areas']
                if isinstance(areas, list):
                    for date in p_dates:
                        try:
                            date_nwp = date.round('H').strftime('%d%m%y%H%M')
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date_nwp = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
                                
                                inp, inp_cnn = self.create_sample(date, nwp, nwp_prev, nwp_next, lats[project_id],
                                                                  longs[project_id], project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data.loc[(date - pd.DateOffset(hours=1)), project_id]
                                    inp['Obs_lag2'] = data.loc[(date - pd.DateOffset(hours=2)), project_id]
                                    if not self.is_for_test:
                                        inp['Obs_lag1'] = inp['Obs_lag1'] + np.random.normal(0, 0.05) * inp['Obs_lag1']
                                        inp['Obs_lag2'] = inp['Obs_lag2'] + np.random.normal(0, 0.05) * inp['Obs_lag2']
                                if not inp.isnull().any(axis=1).values and not np.isnan(data.loc[date, project_id]):
                                    x[project_id] = pd.concat([x[project_id], inp])
                                    x_3d[project_id] = stack_2d_dense(x_3d[project_id], inp_cnn, False)

                                    y[project_id] = pd.concat([y[project_id], pd.DataFrame(data.loc[date, project_id],
                                                                                           columns=['target'],
                                                                                           index=[date])])
                        except Exception:
                            continue
                else:
                    for date in p_dates:
                        try:
                            date_nwp = date.round('H').strftime('%d%m%y%H%M')
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):
                                
                                inp, inp_cnn = self.create_sample_country(date, nwp, nwp_prev, nwp_next,
                                                                          lats[project['_id']],
                                                                          longs[project['_id']],
                                                                          project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data.loc[(date - pd.DateOffset(hours=1)), project_id]
                                    inp['Obs_lag2'] = data.loc[(date - pd.DateOffset(hours=2)), project_id]
                                if not inp.isnull().any(axis=1).values and not np.isnan(data.loc[date, project_id]):
                                    x[project['_id']] = pd.concat([x[project['_id']], inp])
                                    x_3d[project['_id']] = stack_2d_dense(x_3d[project['_id']], inp_cnn, False)

                                    y[project['_id']] = pd.concat(
                                        [y[project['_id']], pd.DataFrame(data.loc[date, project['_id']],
                                                                         columns=['target'], index=[date])])
                        except Exception:
                            continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return x, y, x_3d, t.strftime('%d%m%y%H%M')

    def stack_daily_nwps_rabbitmq(self, t, path_nwp, nwp_model, project, variables):

        x = dict()
        x_3d = dict()
        nwps = project['nwp']

        p_dates = pd.date_range(t, t + pd.DateOffset(days=3) - pd.DateOffset(hours=1), freq='H')

        project_id = project['_id']  # It's the project name, the park's name
        x[project_id] = pd.DataFrame()
        x_3d[project_id] = np.array([])
        areas = project['static_data']['areas']
        if isinstance(areas, list):
            for date in p_dates:
                try:
                    date_nwp = date.strftime('%d%m%y%H%M')
                    nwp = nwps[date_nwp]
                    date_nwp = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                    nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                    nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                    if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                        inp, inp_cnn = self.create_sample_rabbitmq(date, nwp, nwp_prev, nwp_next, project['static_data']['type'])
                        x[project_id] = pd.concat([x[project_id], inp])
                        x_3d[project_id] = stack_2d_dense(x_3d[project_id], inp_cnn, False)

                except Exception:
                    continue
        else:
            for date in p_dates:
                try:
                    date_nwp = date.strftime('%d%m%y%H%M')
                    nwp = nwps[date_nwp]
                    date = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                    nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                    nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]

                    if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                        inp, inp_cnn = self.create_sample_country(date, nwp, nwp_prev, nwp_next,
                                                                  lats[project['_id']],
                                                                  longs[project['_id']],
                                                                  project['static_data']['type'])
                        x[project['_id']] = pd.concat([x[project['_id']], inp])
                        x_3d[project['_id']] = stack_2d_dense(x_3d[project['_id']], inp_cnn, False)

                except Exception:
                    continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return x, x_3d, t.strftime('%d%m%y%H%M')

    def stack_daily_nwps_online(self, t, data, lats, longs, path_nwp, nwp_model, projects, variables):

        x = dict()
        x_3d = dict()
        file_name = os.path.join(path_nwp, f"{nwp_model}_{t.strftime('%d%m%y')}.pickle")
        if os.path.exists(file_name):
            nwps = joblib.load(file_name)

            for project in projects:
                if project['static_data']['horizon'] == 'day_ahead':
                    p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H')
                else:
                    p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=24), freq='15min')

                project_id = project['_id']  # It's the project name, the park's name
                x[project_id] = pd.DataFrame()
                x_3d[project_id] = np.array([])
                areas = project['static_data']['areas']
                if isinstance(areas, list):
                    for date in p_dates:
                        try:
                            date_nwp = date.round('H').strftime('%d%m%y%H%M')
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date_nwp = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                                inp, inp_cnn = self.create_sample(date, nwp, nwp_prev, nwp_next, lats[project_id],
                                                                  longs[project_id], project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data.loc[(date - pd.DateOffset(hours=1)), project_id]
                                    inp['Obs_lag2'] = data.loc[(date - pd.DateOffset(hours=2)), project_id]
                                x[project_id] = pd.concat([x[project_id], inp])
                                x_3d[project_id] = stack_2d_dense(x_3d[project_id], inp_cnn, False)

                        except Exception:
                            continue
                else:
                    for date in p_dates:
                        try:
                            date_nwp = date.round('H').strftime('%d%m%y%H%M')
                            nwp = nwps[date_nwp]
                            nwp = self.correct_nwps(nwp, variables)
                            date = pd.to_datetime(date_nwp, format='%d%m%y%H%M')
                            nwp_prev = nwps[(date_nwp - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_next = nwps[(date_nwp + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                            nwp_prev = self.correct_nwps(nwp_prev, variables)
                            nwp_next = self.correct_nwps(nwp_next, variables)
                            if check_empty_nwp(nwp, nwp_next, nwp_prev, variables):

                                inp, inp_cnn = self.create_sample_country(date, nwp, nwp_prev, nwp_next,
                                                                          lats[project['_id']],
                                                                          longs[project['_id']],
                                                                          project['static_data']['type'])
                                if project['static_data']['horizon'] == 'short-term':
                                    inp['Obs_lag1'] = data.loc[(date - pd.DateOffset(hours=1)), project_id]
                                    inp['Obs_lag2'] = data.loc[(date - pd.DateOffset(hours=2)), project_id]

                                x[project['_id']] = pd.concat([x[project['_id']], inp])
                                x_3d[project['_id']] = stack_2d_dense(x_3d[project['_id']], inp_cnn, False)

                        except Exception:
                            continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return x, x_3d, t.strftime('%d%m%y%H%M')

    def get_lats_longs(self):
        lats = dict()
        longs = dict()
        nwp_found = False
        for t in self.dates:  # Try to load at least one file ??
            file_name = os.path.join(self.path_nwp, f"{self.nwp_model}_{t.strftime('%d%m%y')}.pickle")
            p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=48), freq='H').strftime(
                '%d%m%y%H%M')
            if os.path.exists(file_name):

                nwps = joblib.load(file_name)
                for date in p_dates:
                    if date in nwps:
                        nwp = nwps[date]
                        nwp_found = True
                        break
            if nwp_found:
                break
        print(nwp_found)
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

            areas = project['static_data']['areas']  # The final area is a 5x5 grid
            project_id = project['_id']
            lat, long = nwp['lat'], nwp['long']
            if isinstance(areas, list):
                # Is this guaranteed to be 5x5 ? I think yes, because of the resolution. TODO: VERIFY
                lats[project_id] = np.where((lat[:, 0] >= areas[0][0]) & (lat[:, 0] <= areas[1][0]))[0]
                longs[project_id] = np.where((long[0, :] >= areas[0][1]) & (long[0, :] <= areas[1][1]))[0]
            else:
                lats[project_id] = dict()
                longs[project_id] = dict()
                for area in sorted(areas.keys()):
                    lats[project_id][area] = np.where((lat[:, 0] >= areas[0][0]) & (lat[:, 0] <= areas[1][0]))[0]
                    longs[project_id][area] = np.where((long[0, :] >= areas[0][1]) & (long[0, :] <= areas[1][1]))[0]

        return lats, longs

    def make_dataset_res_short_term(self):
        lats, longs = self.get_lats_longs()

        predictions = dict()
        for project in self.projects:
            predictions[project['_id']] = joblib.load(os.path.join(project['static_data']['path_data']
                                                                   , 'predictions_short_term.pickle'))
        nwp = self.stack_by_sample(self.data.index[20], self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
                                           self.variables, predictions)
        nwp_samples = Parallel(n_jobs=self.n_jobs)(
            delayed(self.stack_by_sample)(t, self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
                                           self.variables, predictions) for t in self.data.index[20:])
        x = dict()
        y = dict()
        x_3d = dict()
        for project in self.projects:
            x[project['_id']] = pd.DataFrame()
            y[project['_id']] = pd.DataFrame()
            x_3d[project['_id']] = np.array([])

        for nwp in nwp_samples:
            for project in self.projects:
                if project['_id'] in nwp[2].keys():
                    if nwp[2][project['_id']].shape[0] != 0:
                        x[project['_id']] = pd.concat([x[project['_id']], nwp[0][project['_id']]])
                        y[project['_id']] = pd.concat([y[project['_id']], nwp[1][project['_id']]])
                        x_3d[project['_id']] = stack_3d(x_3d[project['_id']], nwp[2][project['_id']])
        self.logger.info('All Inputs stacked')
        dataset_x_csv = 'dataset_X_test.csv'
        dataset_y_csv = 'dataset_y_test.csv'
        dataset_cnn_pickle = 'dataset_cnn_test.pickle'

        for project in self.projects:
            project_id = project['_id']
            data_path = project['static_data']['path_data']

            dataset_x = x[project_id]
            dataset_y = y[project_id]
            if dataset_y.isna().any().values[0]:
                dataset_x = dataset_x.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                if len(x_3d.shape) > 1:
                    x_3d = np.delete(x_3d, np.where(dataset_y.isna())[0], axis=0)
                dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

            if dataset_x.isna().any().values[0]:
                dataset_y = dataset_y.drop(dataset_x.index[np.where(dataset_x.isna())[0]])

                if len(x_3d.shape) > 1:
                    x_3d = np.delete(x_3d, np.where(dataset_x.isna())[0], axis=0)
                dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])
            index = [d for d in dataset_x.index if d in dataset_y.index]
            dataset_x = dataset_x.loc[index]
            dataset_y = dataset_y.loc[index]

            ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))

            columns = dataset_x.columns[ind]
            dataset_x = dataset_x[columns]

            dataset_x.to_csv(os.path.join(data_path, dataset_x_csv))
            dataset_y.to_csv(os.path.join(data_path, dataset_y_csv))
            joblib.dump(x_3d[project_id], os.path.join(data_path, dataset_cnn_pickle))
            self.logger.info('Datasets saved for project %s', project['_id'])

    def make_dataset_res_rabbitmq(self):

        project = self.projects[0]

        nwp_daily = self.stack_daily_nwps_rabbitmq(self.dates[0], self.path_nwp, self.nwp_model, project,
                                                  self.variables)

        x = nwp_daily[0][project['_id']]
        x_3d = nwp_daily[1][project['_id']]

        project_id = project['_id']
        data_path = project['static_data']['path_data']

        dataset_x = x

        if os.path.exists(os.path.join(data_path, 'dataset_columns_order.pickle')):
            ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))

            columns = dataset_x.columns[ind]
            dataset_x = dataset_x[columns]

        return dataset_x, x_3d


    def make_dataset_res_online(self):

        project = self.projects[0]

        lats, longs = self.get_lats_longs()

        nwp_daily = Parallel(n_jobs=self.n_jobs)(
            delayed(self.stack_daily_nwps_online)(t, self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
                                           self.variables) for t in self.dates)

        x = pd.DataFrame()
        y = pd.DataFrame()
        x_3d = np.array([])
        for nwp in nwp_daily:
            if nwp[1][project['_id']].shape[0] != 0:
                x = pd.concat([x, nwp[0][project['_id']]])
                x_3d = stack_3d(x_3d, nwp[2][project['_id']])

        project_id = project['_id']
        data_path = project['static_data']['path_data']

        dataset_x = x

        ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))

        columns = dataset_x.columns[ind]
        dataset_x = dataset_x[columns]

        return dataset_x, x_3d


    def make_dataset_res(self):
        lats, longs = self.get_lats_longs()

        nwp = self.stack_daily_nwps(self.dates[4], self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
        self.variables)
        nwp_daily = Parallel(n_jobs=self.n_jobs)(
            delayed(self.stack_daily_nwps)(t, self.data, lats, longs, self.path_nwp, self.nwp_model, self.projects,
                                           self.variables) for t in self.dates)
        x = dict()
        y = dict()
        x_3d = dict()
        for project in self.projects:
            x[project['_id']] = pd.DataFrame()
            y[project['_id']] = pd.DataFrame()
            x_3d[project['_id']] = np.array([])
        for nwp in nwp_daily:
            for project in self.projects:
                if project['_id'] in nwp[2].keys():
                    if nwp[2][project['_id']].shape[0] != 0:
                        x[project['_id']] = pd.concat([x[project['_id']], nwp[0][project['_id']]])
                        y[project['_id']] = pd.concat([y[project['_id']], nwp[1][project['_id']]])
                        x_3d[project['_id']] = stack_3d(x_3d[project['_id']], nwp[2][project['_id']])
        self.logger.info('All Inputs stacked')

        dataset_x_csv = 'dataset_X_test.csv' if self.is_for_test else 'dataset_X.csv'
        dataset_y_csv = 'dataset_y_test.csv' if self.is_for_test else 'dataset_y.csv'
        dataset_cnn_pickle = 'dataset_cnn_test.pickle' if self.is_for_test else 'dataset_cnn.pickle'

        for project in self.projects:
            project_id = project['_id']
            data_path = project['static_data']['path_data']

            dataset_x = x[project_id]
            dataset_y = y[project_id]
            if dataset_y.isna().any().values[0]:
                dataset_x = dataset_x.drop(dataset_y.index[np.where(dataset_y.isna())[0]])

                if len(x_3d.shape) > 1:
                    x_3d = np.delete(x_3d, np.where(dataset_y.isna())[0], axis=0)
                dataset_y = dataset_y.drop(dataset_y.index[np.where(dataset_y.isna())[0]])
           
            index = [d for d in dataset_x.index if d in dataset_y.index]
            dataset_x = dataset_x.loc[index]
            dataset_y = dataset_y.loc[index]
            if self.is_for_test:
                ind = joblib.load(os.path.join(data_path, 'dataset_columns_order.pickle'))
            else:  # create the right order of the columns
                corr = []
                for f in range(dataset_x.shape[1]):
                    corr.append(np.abs(np.corrcoef(dataset_x.values[:, f], dataset_y.values.ravel())[1, 0]))
                ind = np.argsort(np.array(corr))[::-1]
                joblib.dump(ind, os.path.join(data_path, 'dataset_columns_order.pickle'))

            columns = dataset_x.columns[ind]
            dataset_x = dataset_x[columns]

            dataset_x.to_csv(os.path.join(data_path, dataset_x_csv))
            dataset_y.to_csv(os.path.join(data_path, dataset_y_csv))
            joblib.dump(x_3d[project_id], os.path.join(data_path, dataset_cnn_pickle))
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
            inp_3d = stack_2d_dense(inp_3d, X0, False)
        for var in sorted(self.variables):
            for narea, area in enumerate(sorted(lats_all.keys())):
                lats = lats_all[area]
                longs = longs_all[area]
                if ((var == 'WS') and (model_type == 'wind')) or ((var == 'Flux') and (model_type == 'pv')):
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
                    col = [var_name + '.' + str(narea)] + ['p_' + var_name + '.' + str(narea)] + [
                        'n_' + var_name + '.' + str(narea)]

                    col = col + [var_sort + str(i) + '.' + str(narea) for i in range(2)]
                    # col = col + ['p_' + var_sort + str(i)  + '.' + str(narea) for i in range(2)]
                    # col = col + ['n_' + var_sort + str(i)  + '.' + str(narea) for i in range(2)]

                    X = np.hstack((X1_mean, X0_mean, X2_mean, X1))
                    # X = np.hstack((X1_mean, X0_mean, X2_mean, X1, X0, X2))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1, 1).T, index=[date], columns=col)], axis=1)

                elif var in {'WD', 'Cloud'}:
                    X1 = nwp[var][np.ix_(lats, longs)].ravel()
                    X1_mean = np.mean(X1)
                    X1 = np.percentile(X1, [10, 90])

                    var_name = 'cloud' if var == 'Cloud' else 'direction'
                    var_sort = 'cl' if var == 'Cloud' else 'wd'
                    col = [var_name + '.' + str(narea)]
                    col = col + [var_sort + str(i) + '.' + str(narea) for i in range(2)]

                    X = np.hstack((X1_mean, X1,))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1, 1).T, index=[date], columns=col)], axis=1)

                elif (var in {'Temperature'}) or ((var == 'WS') and (model_type == 'pv')):
                    X2 = nwp_next[var][np.ix_(lats, longs)].ravel()
                    X2_mean = np.mean(X2)
                    # X2 = np.percentile(X2, [25, 75])

                    var_name = 'Temp' if var == 'Temperature' else 'wind'
                    var_sort = 'tp' if var == 'Temperature' else 'ws'
                    col = [var_name + '.' + str(narea)]
                    # col = col + [var_sort + str(i) + '.' + str(narea) for i in range(2)]

                    X = X2_mean
                    # X = np.hstack((X2_mean, X2))
                    inp = pd.concat([inp, pd.DataFrame(X.reshape(-1, 1).T, index=[date], columns=col)], axis=1)
                else:
                    continue
        return inp, inp_3d,

    def create_sample(self, date, nwp, nwp_prev, nwp_next, lats, longs, model_type):
        inp = pd.DataFrame()
        if model_type == 'pv':
            inp = pd.concat([inp, pd.DataFrame(np.stack([date.hour, date.month]).reshape(-1, 1).T, index=[date],
                                               columns=['hour', 'month'])])
        input_3d = np.array([])

        for var in sorted(self.variables):

            if (var == 'WS' and model_type == 'wind') or (var == 'Flux' and model_type == 'pv'):

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'

                variable_names = []
                variable_values = []

                x0 = nwp_prev[var][np.ix_(lats, longs)].T
                if self.compress:
                    x0 = rescale_mean(x0)
                input_3d = stack_2d_dense(input_3d, x0, False)
                x0_level0 = x0[2, 2]

                variable_values.append(x0_level0)
                variable_names.append(f'p_{var_name}')

                x1 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    x1 = rescale_mean(x1)
                input_3d = stack_2d_dense(input_3d, x1, False)
                x1_level0 = x1[2, 2]

                variable_values.append(x1_level0)
                variable_names.append(f'{var_name}')

                x2 = nwp_next[var][np.ix_(lats, longs)].T
                if self.compress:
                    x2 = rescale_mean(x2)
                input_3d = stack_2d_dense(input_3d, x2, False)
                x2_level0 = x2[2, 2]

                variable_values.append(x2_level0)
                variable_names.append(f'n_{var_name}')

                ind = np.array([[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)])
                x1_curr_mid_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_down = np.percentile(x1_curr_mid_down, [5, 50, 95])

                variable_values.append(x1_curr_mid_down)
                variable_names += [var_sort + '_l1.' + str(i) for i in range(3)]

                ind = np.array([[2, 3], [3, 2], [3, 3]])
                x1_curr_mid_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_up = np.mean(x1_curr_mid_up)

                variable_values.append(x1_curr_mid_up)
                variable_names += [var_sort + '_l2.' + str(i) for i in range(1)]

                ind = np.array([[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)])
                x1_curr_out_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_down = np.percentile(x1_curr_out_down, [5, 50, 95])

                variable_values.append(x1_curr_out_down)
                variable_names += [var_sort + '_l3d.' + str(i) for i in range(3)]

                ind = np.array([[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)])
                x1_curr_out_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_up = np.percentile(x1_curr_out_up, [5, 50, 95])

                variable_values.append(x1_curr_out_up)
                variable_names += [var_sort + '_l3u.' + str(i) for i in range(3)]

                x = np.hstack(variable_values)
                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=variable_names)],
                                axis=1)
            elif var in {'WD', 'Cloud'}:

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'

                variable_names = []
                variable_values = []

                x1 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    x1 = rescale_mean(x1)
                input_3d = stack_2d_dense(input_3d, x1, False)
                x1_level1 = x1[2, 2]
                variable_names.append(var_name)
                variable_values.append(x1_level1)

                ind = np.array([[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)])
                x1_curr_mid_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_down = np.percentile(x1_curr_mid_down, [5, 50, 95])

                variable_names += [var_sort + '_l1.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_mid_down)

                ind = np.array([[2, 3], [3, 2], [3, 3]])
                x1_curr_mid_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_up = np.mean(x1_curr_mid_up)

                variable_names += [var_sort + '_l2.' + str(i) for i in range(1)]
                variable_values.append(x1_curr_mid_up)

                ind = np.array([[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)])
                x1_curr_out_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_down = np.percentile(x1_curr_out_down, [5, 50, 95])

                variable_names += [var_sort + '_l3d.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_out_down)

                ind = np.array([[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)])
                x1_curr_out_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_up = np.percentile(x1_curr_out_up, [5, 50, 95])

                variable_names += [var_sort + '_l3u.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_out_up)

                x = np.hstack(variable_values)
                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=variable_names)],
                                axis=1)

            elif (var in {'Temperature'}) or ((var == 'WS') and (model_type == 'pv')):
                x2 = nwp[var][np.ix_(lats, longs)].T
                if self.compress:
                    x2 = rescale_mean(x2)
                input_3d = stack_2d_dense(input_3d, x2, False)
                x = x2[2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'

                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=[var_name])], axis=1)
            else:
                continue

        return inp, input_3d,


    def create_sample_rabbitmq(self, date, nwp, nwp_prev, nwp_next, model_type):
        inp = pd.DataFrame()
        if model_type == 'pv':
            inp = pd.concat([inp, pd.DataFrame(np.stack([date.hour, date.month]).reshape(-1, 1).T, index=[date],
                                               columns=['hour', 'month'])])
        input_3d = np.array([])

        for var in sorted(self.variables):
            if len(nwp[var].shape)==1:
                l = int(np.sqrt(nwp[var].shape[0]))
                nwp[var] = nwp[var].reshape(l, l)
            if len(nwp_prev[var].shape)==1:
                l = int(np.sqrt(nwp_prev[var].shape[0]))
                nwp_prev[var] = nwp_prev[var].reshape(l, l)
            if len(nwp_next[var].shape)==1:
                l = int(np.sqrt(nwp_next[var].shape[0]))
                nwp_next[var] = nwp_next[var].reshape(l, l)
            if (var == 'WS' and model_type == 'wind') or (var == 'Flux' and model_type == 'pv'):

                var_name = 'flux' if var == 'Flux' else 'wind'
                var_sort = 'fl' if var == 'Flux' else 'ws'

                variable_names = []
                variable_values = []

                x0 = nwp_prev[var].T
                if self.compress:
                    x0 = rescale_mean(x0)
                input_3d = stack_2d_dense(input_3d, x0, False)
                x0_level0 = x0[2, 2]

                variable_values.append(x0_level0)
                variable_names.append(f'p_{var_name}')

                x1 = nwp[var].T
                if self.compress:
                    x1 = rescale_mean(x1)
                input_3d = stack_2d_dense(input_3d, x1, False)
                x1_level0 = x1[2, 2]

                variable_values.append(x1_level0)
                variable_names.append(f'{var_name}')

                x2 = nwp_next[var].T
                if self.compress:
                    x2 = rescale_mean(x2)
                input_3d = stack_2d_dense(input_3d, x2, False)
                x2_level0 = x2[2, 2]

                variable_values.append(x2_level0)
                variable_names.append(f'n_{var_name}')

                ind = np.array([[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)])
                x1_curr_mid_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_down = np.percentile(x1_curr_mid_down, [5, 50, 95])

                variable_values.append(x1_curr_mid_down)
                variable_names += [var_sort + '_l1.' + str(i) for i in range(3)]

                ind = np.array([[2, 3], [3, 2], [3, 3]])
                x1_curr_mid_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_up = np.mean(x1_curr_mid_up)

                variable_values.append(x1_curr_mid_up)
                variable_names += [var_sort + '_l2.' + str(i) for i in range(1)]

                ind = np.array([[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)])
                x1_curr_out_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_down = np.percentile(x1_curr_out_down, [5, 50, 95])

                variable_values.append(x1_curr_out_down)
                variable_names += [var_sort + '_l3d.' + str(i) for i in range(3)]

                ind = np.array([[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)])
                x1_curr_out_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_up = np.percentile(x1_curr_out_up, [5, 50, 95])

                variable_values.append(x1_curr_out_up)
                variable_names += [var_sort + '_l3u.' + str(i) for i in range(3)]

                x = np.hstack(variable_values)
                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=variable_names)],
                                axis=1)
            elif var in {'WD', 'Cloud'}:

                var_name = 'cloud' if var == 'Cloud' else 'direction'
                var_sort = 'cl' if var == 'Cloud' else 'wd'

                variable_names = []
                variable_values = []

                x1 = nwp[var].T
                if self.compress:
                    x1 = rescale_mean(x1)
                input_3d = stack_2d_dense(input_3d, x1, False)
                x1_level1 = x1[2, 2]
                variable_names.append(var_name)
                variable_values.append(x1_level1)

                ind = np.array([[1, j] for j in range(1, 4)] + [[i, 1] for i in range(2, 4)])
                x1_curr_mid_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_down = np.percentile(x1_curr_mid_down, [5, 50, 95])

                variable_names += [var_sort + '_l1.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_mid_down)

                ind = np.array([[2, 3], [3, 2], [3, 3]])
                x1_curr_mid_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_mid_up = np.mean(x1_curr_mid_up)

                variable_names += [var_sort + '_l2.' + str(i) for i in range(1)]
                variable_values.append(x1_curr_mid_up)

                ind = np.array([[0, j] for j in range(5)] + [[i, 0] for i in range(1, 5)])
                x1_curr_out_down = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_down = np.percentile(x1_curr_out_down, [5, 50, 95])

                variable_names += [var_sort + '_l3d.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_out_down)

                ind = np.array([[4, j] for j in range(1, 5)] + [[i, 4] for i in range(1, 4)])
                x1_curr_out_up = np.hstack([x1[indices[0], indices[1]].reshape(-1, 1) for indices in ind])
                x1_curr_out_up = np.percentile(x1_curr_out_up, [5, 50, 95])

                variable_names += [var_sort + '_l3u.' + str(i) for i in range(3)]
                variable_values.append(x1_curr_out_up)

                x = np.hstack(variable_values)
                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=variable_names)],
                                axis=1)

            elif (var in {'Temperature'}) or ((var == 'WS') and (model_type == 'pv')):
                x2 = nwp[var].T
                if self.compress:
                    x2 = rescale_mean(x2)
                input_3d = stack_2d_dense(input_3d, x2, False)
                x = x2[2, 2]

                var_name = 'Temp' if var == 'Temperature' else 'wind'

                inp = pd.concat([inp, pd.DataFrame(x.reshape(-1, 1).T, index=[date], columns=[var_name])], axis=1)
            else:
                continue

        return inp, input_3d,