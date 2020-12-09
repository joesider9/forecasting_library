import datetime
import email
import imaplib
import os
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd
from credentials import Credentials
from credentials import JsonFileBackend
from joblib import Parallel
from joblib import delayed

from Fuzzy_clustering.version2.common_utils.logging import create_logger

if sys.platform == 'linux':
    file_cred = '~/filemail.json'
    path_nwp = '/media/smartrue/HHD2/ECMWF'
    import pygrib
else:
    file_cred = 'D:/Dropbox/current_codes/PycharmProjects/forecasting_platform/filemail.json'
    path_nwp = 'D:/Dropbox/ECMWF'
    import cfgrib


class DownLoader:

    def __init__(self, date=None):
        if date is None:
            self.date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')
        else:
            self.date = date
        self.credobj = Credentials([JsonFileBackend(file_cred)])
        fname = str(self.date.year) + '/SIDERT' + self.date.strftime('%m%d') + '00UTC.tgz'
        self.filename = os.path.join(path_nwp, fname)
        self.subject = 'Real Time data ' + self.date.strftime('%Y-%m-%d') + ' 00UTC'

    def download(self):
        try:
            imapSession = imaplib.IMAP4_SSL('imap.gmail.com')
            typ, accountDetails = imapSession.login(self.credobj.load('cred1'), self.credobj.load('cred2'))
            if typ != 'OK':
                raise ConnectionError('cannot connect')

            imapSession.select("ECMWF")
            typ, data = imapSession.search(None, '(SUBJECT "' + self.subject + '")')
            if typ != 'OK':
                raise IOError('cannot read emails')

            # Iterating over all emails
            for msgId in data[0].split():
                typ, messageParts = imapSession.fetch(msgId, '(RFC822)')
                if typ != 'OK':
                    raise IOError('cannot read messages')

                emailBody = messageParts[0][1]
                mail = email.message_from_bytes(emailBody)
                if mail.get_content_maintype() != 'multipart':
                    return
                for part in mail.walk():
                    if part.get_content_maintype() != 'multipart' and part.get('Content-Disposition') is not None:
                        print(self.filename)
                        fp = open(self.filename, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
            imapSession.close()
            imapSession.logout()
        except:
            print('Not able to download all attachments.', self.subject)


class EcmwfExtractor:

    def __init__(self, projects_group, pathnwp, nwp_resolution, path_nwp_group, dates_ts, area_group, njobs=1):
        self.pathnwp = pathnwp
        self.pathnwp_group = path_nwp_group
        self.nwp_resolution = nwp_resolution
        self.area = area_group
        self.projects_group = projects_group
        self.njobs = njobs

        self.logger = create_logger(logger_name='log_ecmwf', abs_path=self.pathnwp_group,
                                    logger_path='log_nwp.log', write_type='a')
        self.dates_ts = dates_ts if isinstance(dates_ts, pd.Timestamp) else self.define_dates(dates_ts)


    def define_dates(self, dates_ts):
        # Extracts the dates the measurements were taken, irrespectively of the hour they were obtained.

        start_date = pd.to_datetime(dates_ts[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(dates_ts[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)

        data_dates = pd.to_datetime(np.unique(dates_ts.strftime('%d%m%y')), format='%d%m%y')
        dates = [d for d in dates if d in data_dates]

        self.logger.info('Dates is checked. Number of time samples %s', str(len(dates)))

        return pd.DatetimeIndex(dates)

    def extract_pygrib1(self, t, file_name):

        # We get 48 hours forecasts. For every date available take the next 47 hourly predictions.

        nwps = dict()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=48), freq='H')
        for dt in dates:
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
        grb = pygrib.open(file_name)
        for i in range(1, grb.messages + 1):
            g = grb.message(i)
            if g.cfVarNameECMF == 'u100':
                var = 'Uwind'
            elif g.cfVarNameECMF == 'v100':
                var = 'Vwind'
            elif g.cfVarNameECMF == 't2m':
                var = 'Temperature'
            elif g.cfVarNameECMF == 'tcc':
                var = 'Cloud'
            elif g.cfVarNameECMF == 'ssrd':
                var = 'Flux'
            dt = dates[g.endStep].strftime('%d%m%y%H%M')
            data, lat, long = g.data()
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt][var] = data
        grb.close()
        del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt]['WS'] = wspeed
            nwps[dt]['WD'] = wdir
        return nwps

    def extract_cfgrib1(self, fname):
        nwps = dict()
        data = cfgrib.open_dataset(fname)
        dates = pd.to_datetime(data.valid_time.data, format='%Y-%m-%d %H:%M:%S').strftime('%d%m%y%H%M')
        Uwind = data.u100.data
        Vwind = data.v100.data
        temp = data.t2m.data
        cloud = data.tcc.data
        flux = data.ssrd.data
        lat = data.latitude.data
        long = data.longitude.data
        r2d = 45.0 / np.arctan(1.0)
        wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
        wdir = np.arctan2(Uwind, Vwind) * r2d + 180
        for i, dt in enumerate(dates):
            nwps[dt] = dict()
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt]['Uwind'] = Uwind[i]
            nwps[dt]['Vwind'] = Vwind[i]
            nwps[dt]['WS'] = wspeed[i]
            nwps[dt]['WD'] = wdir[i]
            nwps[dt]['Temperature'] = temp[i]
            nwps[dt]['Cloud'] = cloud[i]
            nwps[dt]['Flux'] = flux[i]

        return nwps

    def extract_pygrib2(self, t, fname):
        path_extract = os.path.join(self.pathnwp, 'extract/' + t.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        tar = tarfile.open(fname)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=48), freq='H')
        nwps = dict()
        for i, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + t.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + t.strftime('%m%d') + '0000' + t.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt]['WS'] = wspeed
            nwps[dt]['WD'] = wdir
        return nwps

    def extract_cfgrib2(self, t, fname):
        path_extract = os.path.join(self.pathnwp, 'extract/' + t.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        tar = tarfile.open(fname)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=48), freq='H')
        nwps = dict()
        for i, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + t.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + t.strftime('%m%d') + '0000' + t.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue

            data = cfgrib.open_dataset(file)
            Uwind = data.u100.data
            Vwind = data.v100.data
            temp = data.t2m.data
            cloud = data.tcc.data
            flux = data.ssrd.data
            lat = data.latitude.data
            long = data.longitude.data
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180

            nwp = dict()
            nwp['lat'] = lat
            nwp['long'] = long
            nwp['Uwind'] = Uwind
            nwp['Vwind'] = Vwind
            nwp['WS'] = wspeed
            nwp['WD'] = wdir
            nwp['Temperature'] = temp
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp

        return nwps

    def extract_pygrib3(self, t, fname):

        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=48), freq='H')
        nwps = dict()
        for i, dt in enumerate(dates):
            file = os.path.join(fname,
                                'E_H6S' + t.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(fname, 'E_H6S' + t.strftime('%m%d') + '0000' + t.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt.strftime('%d%m%y%H%M')]['WS'] = wspeed
            nwps[dt.strftime('%d%m%y%H%M')]['WD'] = wdir
        return nwps

    def extract_cfgrib3(self, t, fname):

        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=48), freq='H')
        nwps = dict()
        for i, dt in enumerate(dates):
            file = os.path.join(fname, 'H6S' + t.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(dt.hour).zfill(
                2) + '001')
            if not os.path.exists(file):
                file = os.path.join(fname, 'H6S' + t.strftime('%m%d') + '0000' + t.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            data = cfgrib.open_dataset(file)
            Uwind = data.u100.data
            Vwind = data.v100.data
            temp = data.t2m.data
            cloud = data.tcc.data
            flux = data.ssrd.data
            lat = data.latitude.data
            long = data.longitude.data
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180

            nwp = dict()
            nwp['lat'] = lat
            nwp['long'] = long
            nwp['Uwind'] = Uwind
            nwp['Vwind'] = Vwind
            nwp['WS'] = wspeed
            nwp['WD'] = wdir
            nwp['Temperature'] = temp
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp

        return nwps

    def nwps_extract_for_train(self, t):
        file_name1 = os.path.join(self.pathnwp, f"{t.strftime('%Y')}/Sider2_{t.strftime('%Y%m%d')}.grib")
        file_name2 = os.path.join(self.pathnwp, t.strftime('%Y') + '/SIDERT' + t.strftime('%m%d') + '00UTC.tgz')
        file_name3 = os.path.join(self.pathnwp, t.strftime('%Y') + '/H6S' + t.strftime('%m%d') + '0000/')

        nwps = dict()
        if os.path.exists(file_name1):
            nwps = self.extract_pygrib1(t, file_name1) if sys.platform == 'linux' else self.extract_cfgrib1(file_name1)
        elif os.path.exists(file_name2):
            nwps = self.extract_pygrib2(t, file_name2) if sys.platform == 'linux' else self.extract_cfgrib2(file_name2)
        elif os.path.exists(file_name3):
            nwps = self.extract_pygrib3(t, file_name3) if sys.platform == 'linux' else self.extract_cfgrib3(file_name3)

        print('Extracted date ', t.strftime('%d%m%y'))
        return t.strftime('%d%m%y'), nwps

    def grib2dict_for_train(self):
        results = Parallel(n_jobs=self.njobs)(delayed(self.nwps_extract_for_train)(t) for t in self.dates_ts)
        for res in results:
            date, nwps = res[0], res[1]
            joblib.dump(nwps, os.path.join(self.pathnwp_group, f'ecmwf_{date}.pickle'))
            print('Nwp pickle file created for date %s', res[0])

    def grib2dict_for_train_online(self):
        res = self.nwps_extract_for_train(self.dates_ts)

        joblib.dump(res[1], os.path.join(self.pathnwp_group, 'ecmwf_' + res[0] + '.pickle'))
        print('NWPs extracted for', res[0])

    def extract_nwps(self, train=True):

        if train:
            dates = []
            for dt in self.dates_ts:
                if not os.path.exists(os.path.join(self.pathnwp_group, 'ecmwf_' + dt.strftime('%d%m%y') + '.pickle')):
                    dates.append(dt)
            self.dates_ts = pd.DatetimeIndex(dates)

            self.grib2dict_for_train()
        else:
            self.grib2dict_for_train_online()
        self.logger.info('Nwp pickle file created for all date')
