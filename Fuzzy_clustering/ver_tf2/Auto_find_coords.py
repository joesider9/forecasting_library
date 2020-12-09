import numpy as np
import pandas as pd
import joblib, os, logging
from joblib import Parallel, delayed
from scipy.interpolate import interp2d
from sklearn.metrics import mean_squared_error



def rescale(arr, nrows, ncol):
    W, H = arr.shape
    new_W, new_H = (nrows, ncol)
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_H), xrange(new_W))

    return new_arr
def rescale_mean(arr):
    arr_new = np.zeros([int(arr.shape[0]/2), int(arr.shape[1]/2)])
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


class AutoFindCoords():

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

    def check_empty_nwp(self, nwp,  variables):
        flag = True
        for var in variables:
            if nwp[var].shape[0] == 0:
                flag = False
                break
        return flag

    def stack_daily_nwps(self, t, data, path_nwp, nwp_model, projects, variables):
        y = dict()
        X_3d = dict()
        fname = os.path.join(path_nwp, nwp_model + '_' + t.strftime('%d%m%y') + '.pickle')
        if os.path.exists(fname):
            nwps = joblib.load(fname)

            pdates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H').strftime(
                '%d%m%y%H%M')
            for project in projects:
                y[project['_id']] = pd.DataFrame()
                X_3d[project['_id']] = np.array([])
                areas = project['static_data']['areas']
                if isinstance(areas, list):
                    for date in pdates:
                        try:

                            nwp = nwps[date]
                            date = pd.to_datetime(date, format='%d%m%y%H%M')
                            if self.check_empty_nwp(nwp, variables):
                                y[project['_id']] = pd.concat([y[project['_id']], pd.DataFrame(data.loc[date, project['_id']], columns=['target'], index=[date])])
                                inp_cnn = self.create_sample(date, nwp, project['static_data']['type'])

                                X_3d[project['_id']] = stack_2d(X_3d[project['_id']], inp_cnn, False)
                        except:
                            continue
                else:
                    for date in pdates:
                        try:
                            nwp = nwps[date]
                            date = pd.to_datetime(date, format='%d%m%y%H%M')

                            if self.check_empty_nwp(nwp, variables):
                                y[project['_id']] = pd.concat(
                                    [y[project['_id']], pd.DataFrame(data.loc[date, project['_id']],
                                                                     columns=['target'], index=[date])])
                                inp_cnn = self.create_sample_country(date, nwp, project['static_data']['type'])
                                X_3d[project['_id']] = stack_2d(X_3d[project['_id']], inp_cnn, False)
                        except:
                            continue

            print(t.strftime('%d%m%y%H%M'), ' extracted')
        return (y, X_3d, t.strftime('%d%m%y%H%M'))

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
        self.nwp_lat = nwp['lat']
        self.nwp_long = nwp['long']

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

    def create_area(self, coord, resolution):
        if self.nwp_resolution == 0.05:
            levels = 4
            round_coord = 1
        else:
            levels = 2
            round_coord = 0

        if coord!=None:
            if isinstance(coord, list):
                if len(coord)==2:
                    lat = coord[0]
                    long = coord[1]
                    lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20, resolution)
                    lat1 = lat_range[np.abs(lat_range - lat).argmin()]-self.nwp_resolution/10
                    lat2 = lat_range[np.abs(lat_range - lat).argmin()]+self.nwp_resolution/10


                    long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20, resolution)
                    long1 = long_range[np.abs(long_range - long).argmin()] - self.nwp_resolution / 10
                    long2 = long_range[np.abs(long_range - long).argmin()] + self.nwp_resolution / 10

                    area=[[lat1 - self.nwp_resolution*levels, long1 - self.nwp_resolution*levels],
                                 [lat2 + self.nwp_resolution*levels, long2 + self.nwp_resolution*levels]]
                elif len(coord)==4:
                    area = list(np.array(coord).reshape(2,2))
                else:
                    raise ValueError('Wrong coordinates. Should be point (lat, long) or area [lat1, long1, lat2, long2]')
            elif isinstance(coord, dict):
                area = dict()
                for key, value in coord.items():
                    lat = value[0]
                    long = value[1]
                    lat_range = np.arange(np.around(lat, round_coord) - 20, np.around(lat, round_coord) + 20,
                                          resolution)
                    lat1 = lat_range[np.abs(lat_range - lat).argmin()] - self.nwp_resolution / 10
                    lat2 = lat_range[np.abs(lat_range - lat).argmin()] + self.nwp_resolution / 10

                    long_range = np.arange(np.around(long, round_coord) - 20, np.around(long, round_coord) + 20,
                                           resolution)
                    long1 = long_range[np.abs(long_range - long).argmin()] - self.nwp_resolution / 10
                    long2 = long_range[np.abs(long_range - long).argmin()] + self.nwp_resolution / 10

                    area[key] = [[lat1 - self.nwp_resolution*levels, long1 - self.nwp_resolution*levels],
                                 [lat2 + self.nwp_resolution*levels, long2 + self.nwp_resolution*levels]]
            else:
                raise ValueError('Wrong coordinates. Should be dict or list')
        else:
            area = dict()
        self.logger.info('Areas created succesfully')

        return area

    def make_dataset_res(self):
        lats, longs = self.lats_longs()
        nwp = self.stack_daily_nwps(self.dates[0], self.data, self.path_nwp, self.nwp_model, self.projects, self.variables)
        nwp_daily = Parallel(n_jobs=self.njobs)(
            delayed(self.stack_daily_nwps)(t, self.data, self.path_nwp, self.nwp_model, self.projects, self.variables) for t in self.dates)
        y = dict()
        X_3d = dict()
        for project in self.projects:
            y[project['_id']] = pd.DataFrame()
            X_3d[project['_id']] = np.array([])
        for nwp in nwp_daily:
            for project in self.projects:
                if project['_id'] in nwp[1].keys():
                    if nwp[1][project['_id']].shape[0] != 0:
                        y[project['_id']] = pd.concat([y[project['_id']], nwp[0][project['_id']]])
                        X_3d[project['_id']] = stack_3d(X_3d[project['_id']], nwp[1][project['_id']])
                        self.logger.info('All Inputs stacked for date %s', nwp[2])


        for project in self.projects:
            dataset_y = y[project['_id']]
            dataset_y.to_csv(os.path.join(project['static_data']['path_data'], 'dataset_y_autocoord.csv'))
            joblib.dump(X_3d[project['_id']], os.path.join(project['static_data']['path_data'], 'dataset_autocoord.pickle'))
            self.logger.info('Datasets saved for project %s', project['_id'])

        coord_auto = pd.DataFrame()
        for project in self.projects:
            lats_orig = lats[project['_id']]
            longs_orig = longs[project['_id']]
            x_cnn = X_3d[project['_id']]
            y_cnn = y[project['_id']].values.ravel()
            corr = np.zeros([x_cnn.shape[1], x_cnn.shape[2]])
            for i in range(x_cnn.shape[1]):
                for j in range(x_cnn.shape[2]):
                    corr[i, j] = np.corrcoef(x_cnn[:, i, j], y_cnn)[1, 0]
            corr_new = np.where(corr == corr.max())
            corr_new = [c[0] for c in corr_new]
            corr_new[0] = self.nwp_lat[:, 0][corr_new[0]]
            corr_new[1] = self.nwp_long[0, :][corr_new[1]]
            area_new = self.create_area(corr_new, self.nwp_resolution)
            lats_new = \
                (np.where((self.nwp_lat[:, 0] >= area_new[0][0]) & (self.nwp_lat[:, 0] <= area_new[1][0])))[0]
            longs_new = \
                (np.where((self.nwp_long[0, :] >= area_new[0][1]) & (self.nwp_long[0, :] <= area_new[1][1])))[
                    0]

            if np.mean(corr[np.ix_(lats_new[1:-1],longs_new[1:-1])]) - np.mean(corr[np.ix_(lats_orig[1:-1],longs_orig[1:-1])])< 0.01:
                corr_new = project['static_data']['location']
            else:
                project['static_data']['location'] = corr_new
                project['static_data']['areas'] = area_new

            corr_new = pd.DataFrame(np.array(corr_new)[np.newaxis, :], index=[project['_id']])
            coord_auto = coord_auto.append(corr_new)

        return coord_auto, self.projects

    def create_sample(self, date, nwp, model_type):

        inp_3d = np.array([])
        for var in sorted(self.variables):
            if ((var == 'WS') and (model_type =='wind')) or ((var == 'Flux') and (model_type == 'pv')):

                inp_3d = stack_2d(inp_3d, nwp[var], False)

        return inp_3d

    def train_PCA(self, data, components, level):
        pass

    def PCA_transform(self, data, components, level):
        pass

