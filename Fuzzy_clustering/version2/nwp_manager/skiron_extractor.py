import ftplib
import os

import joblib
import numpy as np
import pandas as pd
import pygrib
from joblib import Parallel
from joblib import delayed

from Fuzzy_clustering.version2.common_utils.logging import create_logger


class SkironExtractor:
    """
    Provides numerical weather predictions (nwp) with a horizon of 72 hours.

    """

    def __init__(self, projects_group, path_nwp, nwp_resolution, path_nwp_group, dates_ts, area_group, n_jobs=1):
        self.path_nwp = path_nwp
        self.path_nwp_group = path_nwp_group
        self.nwp_resolution = nwp_resolution
        self.area = area_group
        self.projects_group = projects_group
        self.n_jobs = n_jobs

        self.logger = create_logger(logger_name=f'log_skiron', abs_path=self.path_nwp_group,
                                    logger_path='log_nwp.log', write_type='a')
        self.dates_ts = self.define_dates(dates_ts)

    def define_dates(self, dates_ts):
        """
         Ignoring the hour the measurements were taken, creates a pd.DatetimeIndex dataframe
         of all the dates, we have a measurement.
        """
        start_date = pd.to_datetime(dates_ts[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(dates_ts[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)

        data_dates = pd.to_datetime(np.unique(dates_ts.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]

        self.logger.info('Dates are checked. Number of time samples %s', str(len(dates)))

        return pd.DatetimeIndex(dates)

    def skiron_download(self, dt):

        with ftplib.FTP('ftp.mg.uoa.gr') as ftp:
            try:
                ftp.login('mfstep', '!lam')
                ftp.set_pasv(True)

            except Exception:
                print('Error in connection to FTP')
            local_dir = self.path_nwp + dt.strftime('%Y') + '/' + dt.strftime('%d%m%y')
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            try:
                for hor in range(76):
                    target_filename = '/forecasts/Skiron/daily/005X005/' + dt.strftime(
                        '%d%m%y') + '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb'
                    self.logger.info('Trying to download nwp file %s',
                                     '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb')
                    local_filename = local_dir + '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                        3) + '.grb'
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'w+b') as f:
                            res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                            count = 0
                            while not res.startswith('226 Transfer complete') and count <= 4:
                                print('Downloaded of file {0} is not compile.'.format(target_filename))
                                os.remove(local_filename)
                                with open(local_filename, 'w+b') as f:
                                    res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                                    self.logger.info('Success to download nwp file %s',
                                                     '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                                                         3) + '.grb')
                                count += 1
                                self.logger.info('Failed to download nwp file %s',
                                                 '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                                                     3) + '.grb')
            except Exception:
                print('Error downloading  {0} '.format(local_filename))
            ftp.quit()

    def extract(self, grb, la1, la2, lo1, lo2):
        nwps = dict()

        try:
            g = grb.message(21) if self.nwp_resolution == 0.05 else grb.message(1)
        except:
            g = grb.message(1)
        u_wind, lat, long = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)

        try:
            g = grb.message(22) if self.nwp_resolution == 0.05 else grb.message(2)
        except:
            g = grb.message(2)
        v_wind = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)[0]

        r2d = 45.0 / np.arctan(1.0)
        w_speed = np.sqrt(np.square(u_wind) + np.square(v_wind))
        w_dir = np.arctan2(u_wind, v_wind) * r2d + 180

        nwps['lat'] = lat
        nwps['long'] = long
        nwps['Uwind'] = u_wind
        nwps['Vwind'] = v_wind
        nwps['WS'] = w_speed
        nwps['WD'] = w_dir

        g = grb.message(3)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Temperature'] = x[0]

        g = grb.message(7)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Precipitation'] = x[0]

        g = grb.message(5)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Cloud'] = x[0]

        g = grb.message(8)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Flux'] = x[0]
        del x  # Why are we calling del? Is it too memory intensive?
        return nwps

    def nwps_extract_for_train(self, t):
        nwps = dict()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=50), freq='H')
        hors = [int(hor) for hor in range(0, 51)]
        for hor, dt in zip(hors, dates):

            type_prefix = 'MFSTEP005_00' if self.nwp_resolution == 0.05 else "MFSTEP_IASA_00"
            file_name = f"{t.strftime('%Y')}/{t.strftime('%d%m%y')}/{type_prefix}{t.strftime('%d%m%y')}_{str(hor).zfill(3)}.grb"
            nwp_path = os.path.join(self.path_nwp, file_name)  # MFSTEP_IASA_00010117_000
            if os.path.exists(nwp_path):
                try:
                    grb = pygrib.open(nwp_path)
                    la1 = self.area[0][0]
                    la2 = self.area[1][0]
                    lo1 = self.area[0][1]
                    lo2 = self.area[1][1]

                    nwps[dt.strftime('%d%m%y%H%M')] = self.extract(grb, la1, la2, lo1, lo2)
                    grb.close()
                    del grb
                    print('nwps extracted from ', nwp_path)
                except Exception:
                    pass
        return t.strftime('%d%m%y'), nwps

    def grib2dict_for_train(self, dates):
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.nwps_extract_for_train)(t) for t in dates)
        for res in results:
            joblib.dump(res[1], os.path.join(self.path_nwp_group, f'skiron_{res[0]}.pickle'))
            print('nwp extracted for', res[0])

            self.logger.info('nwp pickle file created for date %s', res[0])

    def grib2dict_for_train_online(self):
        res = self.nwps_extract_for_train(self.dates_ts)

        joblib.dump(res[1], os.path.join(self.path_nwp_group, f'skiron_{res[0]}.pickle'))
        print('nwp extracted for', res[0])

        self.logger.info('nwp pickle file created for date %s', res[0])

    # The value train is never set to False, any reason to have the if/then statement that?
    def extract_nwps(self, train=True):

        if train:
            dates = []
            for dt in self.dates_ts:
                if not os.path.exists(os.path.join(self.path_nwp_group, f"skiron_{dt.strftime('%d%m%y')}.pickle")):
                    dates.append(dt)
            dates_to_load = pd.DatetimeIndex(dates)
            self.grib2dict_for_train(dates_to_load)
        else:
            self.grib2dict_for_train_online()
        self.logger.info('Nwp pickle file created for all date')
