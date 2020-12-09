import numpy as np
import pandas as pd
import joblib, os


class dataset_creator():

    def __init__(self, project, data, njobs=1):
        self.data = data
        self.dates_ts = self.check_dates(data.index)
        self.project_name= project['_id']
        self.static_data = project['static_data']
        self.path_nwp_project = self.static_data['pathnwp']
        self.areas = self.static_data['areas']
        self.nwp_model = self.static_data['NWP_model']
        self.njobs = njobs
        if self.static_data['type'] == 'pv':
            self.variables = ['Cloud', 'Flux', 'Temperature']
        elif self.static_data['type'] == 'wind':
            self.variables = ['WS', 'WD']
        else:
            self.variables = []

    def check_dates(self, dates):
        start_date = pd.to_datetime(dates[0].strftime('%d%m%y'), format='%d%m%y')
        end_date = pd.to_datetime(dates[-1].strftime('%d%m%y'), format='%d%m%y')
        dates = pd.date_range(start_date, end_date)
        return dates

    def stack_2d(self, X, sample):
        if len(sample.shape)==3:
            if X.shape[0] == 0:
                X = sample
            elif len(X.shape) == 3:
                X = np.stack((X, sample))
            else:
                X = np.vstack((X, sample[np.newaxis, :, :, :]))
        elif len(sample.shape)==2:
            if X.shape[0] == 0:
                X = sample
            elif len(X.shape) == 2:
                X = np.stack((X, sample))
            else:
                X = np.vstack((X, sample[np.newaxis, :, :]))
        return X

    def get_3d_dataset(self):

        X = np.array([])
        data_var = dict()
        for var in self.variables:
            if var in {'WS', 'Flux'}:
                data_var[var+'_prev'] = X
                data_var[var] = X
                data_var[var+'_next'] = X
            else:
                data_var[var] = X
            data_var['dates'] = X

        for t in self.dates_ts:
            nwps = joblib.load(
                os.path.join(self.path_nwp_project, self.nwp_model + '_' + t.strftime('%d%m%y') + '.pickle'))
            pdates = pd.date_range(t + pd.DateOffset(hours=25), t + pd.DateOffset(hours=48), freq='H').strftime(
                '%d%m%y%H%M')
            for date in pdates:
                try:
                    nwp = nwps[date]
                    date = pd.to_datetime(date, format='%d%m%y%H%M')
                    nwp_prev = nwps[(date - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
                    nwp_next = nwps[(date + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]

                    for var in self.variables:
                        if var in {'WS', 'Flux'}:
                            data_var[var + '_prev'] = self.stack_2d(data_var[var + '_prev'], nwp_prev[var])
                            data_var[var] = self.stack_2d(data_var[var], nwp['WS'])
                            data_var[var + '_next'] = self.stack_2d(data_var[var + '_next'], nwp_next[var])
                        else:
                            data_var[var] = self.stack_2d(data_var[var], nwp[var])
                    data_var['dates'] = np.vstack((data_var['dates'], date))
                except:
                    continue


        return data_var

    def make_training_set_crossbow_pv(data, areas, pathnwp, columns, project_name):


        data_upd = pd.DataFrame(columns=columns)
#     target_upd = pd.Series(name='target')
#     dates_ts = pd.date_range(start=data.index[0], end=data.index[-1])
#     for date in dates_ts:
#
#         pred_dates = pd.date_range(start=date+pd.DateOffset(hours=24), end=date+pd.DateOffset(hours=47), freq='H')
#
#         if os.path.exists(os.path.join(pathnwp, 'nwp' + date.strftime('%d%m%y') + '.h5')):
#             with h5py.File(os.path.join(pathnwp, 'nwp' + date.strftime('%d%m%y') + '.h5'), 'r') as nwpfile:
#
#                 for pdate in pred_dates:
#                     try:
#                         nwp_imp=np.array([])
#                         nwp_unimp=np.array([])
#                         if project_name=='bulgaria_pv':
#                             for area in areas.keys():
#                                 flux = nwpfile[pdate.strftime('%d%m%y%H%M') +'/'+ area +'/Flux'].value
#                                 cloud = nwpfile[pdate.strftime('%d%m%y%H%M') +'/'+ area +'/Cloud'].value
#                                 flux1 = nwpfile[(pdate+pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') +'/'+ area +'/Flux'].value[2]
#                                 flux0 = nwpfile[(pdate-pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') +'/'+ area +'/Flux'].value
#                                 temp = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/Temperature'].value[2]/ 300
#                                 nwp_imp=np.hstack((nwp_imp,[flux[2],cloud[2]]))
#                                 nwp_unimp=np.hstack((nwp_unimp,np.hstack((flux[[0,1,3,4]],flux0[[0,1,3,4]],cloud[[0, 1, 3, 4]],flux1,temp))))
#                         else:
#                             area=project_name
#                             flux = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/Flux'].value
#                             cloud = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/Cloud'].value
#                             flux1 = nwpfile[
#                                 (pdate + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') + '/' + area + '/Flux'].value[2]
#                             flux0 = nwpfile[
#                                 (pdate - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') + '/' + area + '/Flux'].value[2]
#                             temp = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/Temperature'].value[2] / 300
#                             nwp_imp = np.hstack((nwp_imp, [pdate.hour,flux[2], cloud[2]]))
#                             nwp_unimp = np.hstack((nwp_unimp, np.hstack(
#                                 (flux[[0, 1, 3, 4]], flux0, cloud[[0, 1, 3, 4]], flux1, temp,pdate.month))))
#
#                         inp=np.hstack((nwp_imp,nwp_unimp))
#
#
#                         inp1 = pd.Series(inp, index=columns, name=pdate)
#                         targ1 = pd.Series(data.loc[pdate].values, index=[pdate], name='target1')
#                         if not inp1.isnull().any() and not targ1.isnull().any():
#                             data_upd = data_upd.append(inp1)
#                             target_upd = target_upd.append(targ1)
#                     except:
#                         continue
#                 nwpfile.close()
#     return data_upd, target_upd
#     def make_training_set_crossbow_pv(self):
#
#
#         data_upd = pd.DataFrame(columns=columns)
#         target_upd = pd.Series(name='target')
#
#
#         for t in self.dates_ts:
#             nwps = joblib.load(os.path.join(self.path_nwp_project, self.nwp_model + '_' +t.strftime('%d%m%y') + '.pickle'))
#             pdates = pd.date_range(t + pd.DateOffset(hours=25), t + pd.DateOffset(hours=48), freq='H').strftime(
#                 '%d%m%y%H%M')
#             for date in pdates:
#                 nwp = nwps[date]
#                 date = pd.to_datetime(date, format='%d%m%y%H%M')
#                 nwp_prev = nwps[(date - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
#                 nwp_next = nwps[(date + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M')]
#
#                 try:
#                     nwp_imp = np.array([])
#                     nwp_unimp = np.array([])
#                     if isinstance(self.areas, dict):
#                         for area in self.areas.keys():
#                             wind = np.percentile(nwp[area]['Uwind'], [5, 25, 50, 75, 95])
#                             direction = np.percentile(nwp[area]['Vwind'], [5, 25, 50, 75, 95])
#                             wind1 = np.percentile(nwp_next[area]['Uwind'], [5, 25, 50, 75, 95])[2]
#                             wind0 = np.percentile(nwp_prev[area]['Uwind'], [5, 25, 50, 75, 95])
#                             nwp_imp = np.hstack((nwp_imp, [wind[2], direction[2], wind0[2]]))
#                             nwp_unimp = np.hstack((nwp_unimp, np.hstack(
#                                 (wind[[0, 1, 3, 4]], wind0[[0, 1, 3, 4]], direction[[0, 1, 3, 4]], wind1))))
#                     else:
#                         wind = np.percentile(nwp['Uwind'], [5, 25, 50, 75, 95])
#                         direction = np.percentile(nwp['Vwind'], [5, 25, 50, 75, 95])
#                         wind1 = np.percentile(nwp_next['Uwind'], [5, 25, 50, 75, 95])[2]
#                         wind0 = np.percentile(nwp_prev['Uwind'], [5, 25, 50, 75, 95])
#
#                         nwp_imp = np.hstack((nwp_imp, [wind[2], direction[2], wind0[2]]))
#                         nwp_unimp = np.hstack((nwp_unimp, np.hstack(
#                             (wind[[0, 1, 3, 4]], wind0[[0, 1, 3, 4]], direction[[0, 1, 3, 4]], wind1))))
#
#                     inp = np.hstack((nwp_imp, nwp_unimp))
#
#                     inp1 = pd.Series(inp, index=columns, name=date)
#                     targ1 = pd.Series(self.data.loc[date].values, index=[date], name='target1')
#                     if not inp1.isnull().any() and not targ1.isnull().any():
#                         data_upd = data_upd.append(inp1)
#                         target_upd = target_upd.append(targ1)
#                 except:
#                     continue
#
#         return data_upd, target_upd
#
# def make_training_set_wind(data, pathnwp, project_name):
#
#     data_upd = np.array([])
#     target_upd = pd.DataFrame(columns=data.columns)
#     dates_ts = pd.date_range(start=pd.to_datetime(data.index[0].strftime('%d%m%y'),format='%d%m%y'), end=data.index[-1])
#     for date in dates_ts:
#
#         pred_dates = pd.date_range(start=date+pd.DateOffset(hours=24), end=date+pd.DateOffset(hours=47), freq='H')
#
#         if os.path.exists(os.path.join(pathnwp, 'nwp' + date.strftime('%d%m%y') + '.h5')):
#             try:
#                 with h5py.File(os.path.join(pathnwp, 'nwp' + date.strftime('%d%m%y') + '.h5'), 'r') as nwpfile:
#
#                     for pdate in pred_dates:
#
#                             nwp_imp=np.array([])
#                             nwp_unimp=np.array([])
#
#                             area=project_name
#                             wind = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/WS'].value
#                             direction = nwpfile[pdate.strftime('%d%m%y%H%M') + '/' + area + '/WD'].value
#                             # wind1 = nwpfile[(pdate + pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') + '/' + area + '/WS'].value.reshape(5,6)
#                             # wind0 = nwpfile[(pdate - pd.DateOffset(hours=1)).strftime('%d%m%y%H%M') + '/' + area + '/WS'].value.reshape(5,6)
#                             inp1=np.stack((wind,direction))
#                             targ1 = data.loc[pdate]
#                             if not targ1.isnull().any():
#                                 if data_upd.shape[0]==0:
#                                     data_upd = inp1
#                                 elif len(data_upd.shape)==3:
#                                     data_upd = np.stack((data_upd, inp1))
#                                 else:
#                                     data_upd = np.vstack((data_upd,inp1[np.newaxis,:,:,:]))
#                                 target_upd = target_upd.append(targ1)
#
#                 nwpfile.close()
#             except:
#                 continue
#     return data_upd, target_upd.values
#
