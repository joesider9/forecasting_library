import os
import ftplib
import pandas as pd
import pygrib, joblib, logging
import numpy as np
from joblib import Parallel, delayed

class skiron_Extractor():
    
    def __init__(self, projects_group, pathnwp, nwp_resolution, path_nwp_group, dates_ts, area_group, njobs=1):
        self.pathnwp = pathnwp
        self.pathnwp_group = path_nwp_group
        self.nwp_resolution = nwp_resolution
        self.area = area_group
        self.projects_group = projects_group
        self.njobs = njobs
        self.create_logger()
        self.dates_ts = self.check_dates(dates_ts)

    def create_logger(self):
        self.logger = logging.getLogger('log_skiron')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(os.path.dirname(self.pathnwp_group), 'log_nwp.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def check_dates(self, dates_ts):
        start_date = pd.to_datetime(dates_ts[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(dates_ts[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)

        data_dates = pd.to_datetime(np.unique(dates_ts.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]

        self.logger.info('Dates is checked. Number of time samples %s', str(len(dates)))

        return pd.DatetimeIndex(dates)

    def skiron_download(self, dt):

        with ftplib.FTP('ftp.mg.uoa.gr') as ftp:
            try:
                ftp.login('mfstep', '!lam')
                ftp.set_pasv(True)
    
            except:
                print('Error in connection to FTP')
            local_dir=self.pathnwp +dt.strftime('%Y')+'/'+ dt.strftime('%d%m%y')
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            try:
                for hor in range(76):
                    target_filename='/forecasts/Skiron/daily/005X005/'+dt.strftime('%d%m%y')+'/MFSTEP005_00'+ dt.strftime('%d%m%y')+'_'+str(hor).zfill(3)+'.grb'
                    self.logger.info('Trying to download nwp file %s', '/MFSTEP005_00'+ dt.strftime('%d%m%y')+'_'+str(hor).zfill(3)+'.grb')
                    local_filename = local_dir + '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb'
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'w+b') as f:
                            res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                            count=0
                            while not res.startswith('226 Transfer complete') and count<=4:
                                print('Downloaded of file {0} is not compile.'.format(target_filename))
                                os.remove(local_filename)
                                with open(local_filename, 'w+b') as f:
                                    res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                                    self.logger.info('Success to download nwp file %s',
                                                     '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                                                         3) + '.grb')
                                count+=1
                                self.logger.info('Failed to download nwp file %s',
                                                 '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                                                     3) + '.grb')
            except:
                print('Error downloading  {0} '.format(local_filename))
            ftp.quit()

    def extract(self, grb, la1,la2,lo1,lo2):
        nwps=dict()
        if self.nwp_resolution == 0.05:
            g = grb.message(21)
            Uwind, lat, long = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
            g = grb.message(22)
            Vwind = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)[0]
        else:
            g = grb.message(1)
            Uwind, lat, long = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
            g = grb.message(2)
            Vwind = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)[0]
        r2d = 45.0 / np.arctan(1.0)
        wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
        wdir = np.arctan2(Uwind, Vwind) * r2d + 180

        nwps['lat'] = lat
        nwps['long'] = long
        nwps['Uwind'] = Uwind
        nwps['Vwind'] = Vwind
        nwps['WS'] = wspeed
        nwps['WD'] = wdir

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
        del x
        return nwps

    def nwps_extract_for_train(self, t):
        nwps = dict()
        dates = pd.date_range(start=t + pd.DateOffset(hours=19), end=t + pd.DateOffset(hours=53), freq='H')
        hors = [int(hor) for hor in range(20, 49)]
        for hor, dt in zip(hors, dates):
            if self.nwp_resolution == 0.05:
                fname = os.path.join(self.pathnwp,
                                     t.strftime('%Y') + '/' + t.strftime('%d%m%y') + '/MFSTEP005_00' + t.strftime(
                                         '%d%m%y') + '_' + str(hor).zfill(3) + '.grb')
            else:
                fname = os.path.join(self.pathnwp,
                                     t.strftime('%Y') + '/MFSTEP_IASA_00' + t.strftime(
                                         '%d%m%y') + '_' + str(hor).zfill(3) + '.grb')
            #MFSTEP_IASA_00010117_000
            if os.path.exists(fname):
                try:
                    grb = pygrib.open(fname)
                    la1 = self.area[0][0]
                    la2 = self.area[1][0]
                    lo1 = self.area[0][1]
                    lo2 = self.area[1][1]

                    nwps[dt.strftime('%d%m%y%H%M')] = self.extract(grb, la1,la2,lo1,lo2)
                    grb.close()
                    del grb
                    print('nwps exrtacted from ', fname)
                except:
                    pass
        return (t.strftime('%d%m%y'), nwps)


    def grib2dict_for_train(self):
        res = self.nwps_extract_for_train(self.dates_ts[0])
        results = Parallel(n_jobs=self.njobs)(delayed(self.nwps_extract_for_train)(t) for t in self.dates_ts)
        for res in results:
            joblib.dump(res[1], os.path.join(self.pathnwp_group, 'skiron_' +res[0] + '.pickle'))
            print('NWPs extracted for', res[0])

            self.logger.info('Nwp pickle file created for date %s', res[0])

    def grib2dict_for_train_online(self):
        res = self.nwps_extract_for_train(self.dates_ts)

        joblib.dump(res[1], os.path.join(self.pathnwp_group, 'skiron_' +res[0] + '.pickle'))
        print('NWPs extracted for', res[0])

        self.logger.info('Nwp pickle file created for date %s', res[0])

    def extract_nwps(self,train=True):

        if train:
            dates = []
            for dt in self.dates_ts:
                if not os.path.exists(os.path.join(self.pathnwp_group, 'skiron_' +dt.strftime('%d%m%y') + '.pickle')):
                    dates.append(dt)
            self.dates_ts = pd.DatetimeIndex(dates)

            self.grib2dict_for_train()
        else:
            self.grib2dict_for_train_online()

