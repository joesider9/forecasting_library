import copy
import difflib
import logging
import os

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Fuzzy_clustering.ver_tf2.Sklearn_models_TL import sklearn_model_tl
from Fuzzy_clustering.ver_tf2.Sklearn_models_optuna import sklearn_model
from Fuzzy_clustering.ver_tf2.utils_for_forecast import split_continuous_ind


class ProjectLinker():

    def __init__(self, projects_group):
        self.project_col = []
        self.projects = []
        self.X = dict()
        self.y = dict()
        self.coord = []
        for project in projects_group:
            self.static_data = project['static_data']
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                        or os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')):
                    self.projects.append(project)
                    self.project_col.append(project['_id'])
                    self.X[project['_id']] = pd.read_csv(
                        os.path.join(project['static_data']['path_data'], 'dataset_X.csv'), index_col=0,
                        header=0, parse_dates=True, dayfirst=True)
                    self.y[project['_id']] = pd.read_csv(
                        os.path.join(project['static_data']['path_data'], 'dataset_y.csv'), index_col=0,
                        header=0, parse_dates=True, dayfirst=True)
                    self.var_imp = project['static_data']['clustering']['var_lin']
                    self.coord.append([project['_id']] + project['static_data']['location'])
                    self.country = project['static_data']['projects_group']
        if len(self.project_col) < 2:
            raise FileNotFoundError('Dataset creator seems to not have been executed yet. Run a Dataset creator first')
        self.coord = pd.DataFrame(self.coord, columns=['location', 'Latitude', 'Longitude'])
        self.create_logger()

    def create_logger(self):
        self.logger = logging.getLogger('log_' + self.country + '.log')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']),
                                                   'log_' + self.country + '.log'), 'a')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def plot_map(self):
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        gdf = geopandas.GeoDataFrame(
            self.coord, geometry=geopandas.points_from_xy(self.coord.Longitude, self.coord.Latitude))
        name = difflib.get_close_matches(self.country, world.name.to_list())[0]
        ax = world[world.name == name].plot(
            color='white', edgecolor='black')
        # We can now plot our ``GeoDataFrame``.
        gdf.plot(ax=ax, color='red', alpha=0.5, markersize=10, figsize=[100, 50])
        gdf.apply(lambda x: ax.annotate(s=x.location, xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'Map_Country.png'))

    def compute_correlation(self):
        self.logger.info('Start compute linear correlations')
        self.corr_X = pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.corr_X_imp = pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.corr_y = pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.best_lag = pd.DataFrame(index=self.project_col, columns=self.project_col)
        for project1 in self.project_col:
            for project2 in self.project_col:
                df1 = self.y[project1].copy(deep=True)
                df2 = self.y[project2].copy(deep=True)

                dates = [dt for dt in df1.index if dt in df2.index]
                df1 = df1.loc[dates]
                df2 = df2.loc[dates]

                max_corr = df1.corrwith(df2, axis=0).mean()
                best_lag = 0
                if max_corr < 1:
                    for lag in range(-4, 5):
                        if lag != 0:
                            corr = df1.corrwith(df2.shift(lag), axis=0).mean()
                            if corr > max_corr:
                                max_corr = corr
                                best_lag = lag
                self.corr_y.loc[project1, project2] = max_corr

                df1 = self.X[project1].copy(deep=True)
                df2 = self.X[project2].copy(deep=True)

                for var in ['hour', 'month']:
                    if var in df1.columns:
                        df1 = df1.drop(columns=[var])
                    if var in df2.columns:
                        df2 = df2.drop(columns=[var])
                dates = [dt for dt in df1.index if dt in df2.index]
                df1 = df1.loc[dates]
                df2 = df2.loc[dates]

                if best_lag == 0:
                    max_corr = df1.corrwith(df2, axis=0).mean()
                else:
                    max_corr = df1.corrwith(df2.shift(best_lag), axis=0).mean()

                self.corr_X.loc[project1, project2] = max_corr

                df1 = self.X[project1].copy(deep=True)
                df2 = self.X[project2].copy(deep=True)

                dates = [dt for dt in df1.index if dt in df2.index]
                df1 = df1.loc[dates]
                df2 = df2.loc[dates]

                df1 = df1[self.var_imp]
                df2 = df2[self.var_imp]

                for var in ['hour', 'month']:
                    if var in df1.columns:
                        df1 = df1.drop(columns=[var])
                    if var in df2.columns:
                        df2 = df2.drop(columns=[var])
                dates = [dt for dt in df1.index if dt in df2.index]
                df1 = df1.loc[dates]
                df2 = df2.loc[dates]

                if best_lag == 0:
                    max_corr = df1.corrwith(df2, axis=0).mean()
                else:
                    max_corr = df1.corrwith(df2.shift(best_lag), axis=0).mean()

                self.corr_X_imp.loc[project1, project2] = max_corr
                self.best_lag.loc[project1, project2] = best_lag
                self.logger.info('Correlations between %s and %s computed', project1, project2)

    def linked_with_regression(self):
        self.corr_reg = pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.logger.info('Start computing correlations with regression')
        for project1 in self.projects:
            self.logger.info('Start train svm regressor for project %s', project1['_id'])
            path_regressor = project1['static_data']['path_data'] + '/Corr_regressor'
            rated = project1['static_data']['rated']
            X = self.X[project1['_id']].copy(deep=True)
            y = self.y[project1['_id']].copy(deep=True)
            sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
            scale_y = MinMaxScaler(feature_range=(0, 1)).fit(y.values)
            X = sc.transform(X.values)
            y = scale_y.transform(y.values)
            cvs = []
            cvs_ind = []
            for _ in range(3):
                mask_train1, mask_val1 = split_continuous_ind(X, y, test_size=0.15, random_state=42)
                X_train = X[mask_train1]
                y_train = y[mask_train1]
                X_test1 = X[mask_val1]
                y_test1 = y[mask_val1]
                mask_train2, mask_val2 = split_continuous_ind(X_train, y_train, test_size=0.15)
                X_val = X_train[mask_val2]
                y_val = y_train[mask_val2]
                X_train = X_train[mask_train2]
                y_train = y_train[mask_train2]

                cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                cvs_ind.append([mask_train1, mask_val1, mask_train2, mask_val2])
            model1 = sklearn_model(self.static_data, path_regressor, rated, 'svm', project1['static_data']['njobs'])
            _ = model1.train(cvs, n_trials=400)
            self.corr_reg.loc[project1['_id'], project1['_id']] = model1.acc_test
            self.logger.info('Error computed for svm regressor for project %s', project1['_id'])
            for project2 in self.projects:
                path_regressor = project2['static_data']['path_data'] + '/Corr_regressor'
                rated = project1['static_data']['rated']
                if project1['_id'] != project2['_id']:
                    X = self.X[project2['_id']].copy(deep=True)
                    y = self.y[project2['_id']].copy(deep=True)
                    sc = MinMaxScaler(feature_range=(0, 1)).fit(X.values)
                    scale_y = MinMaxScaler(feature_range=(0, 1)).fit(y.values)
                    X = sc.transform(X.values)
                    y = scale_y.transform(y.values)
                    cvs = []
                    for i in range(3):
                        X_train = X[cvs_ind[i][0]]
                        y_train = y[cvs_ind[i][0]]
                        X_test1 = X[cvs_ind[i][1]]
                        y_test1 = y[cvs_ind[i][1]]
                        X_val = X_train[cvs_ind[i][3]]
                        y_val = y_train[cvs_ind[i][3]]
                        X_train = X_train[cvs_ind[i][2]]
                        y_train = y_train[cvs_ind[i][2]]

                        cvs.append([X_train, y_train, X_val, y_val, X_test1, y_test1])
                    model2 = sklearn_model_tl(self.static_data, path_regressor, rated, 'svm',
                                              project2['static_data']['njobs'])
                    _ = model2.train(cvs, model1.best_params)
                    self.corr_reg.loc[project1['_id'], project2['_id']] = model2.acc_test
                    self.logger.info(
                        'Error computed for svm regressor traine with transfer learning from project %s to project %s',
                        project1['_id'], project2['_id'])
        self.logger.info('Successfully computing correlations with regression')

    def compute_relations(self):
        self.compute_correlation()
        # self.linked_with_regression()
        self.plot_map()
        self.corr_X.to_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv'))
        self.corr_X_imp.to_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_imp_X.csv'))
        self.corr_y.to_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'))
        # self.corr_reg.to_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_with_regression.csv'))

    def find_relations(self, corr_X_thres=0.85, corr_y_thres=0.9):
        if self.projects[0]['static_data']['type'] == 'wind':
            corr_X_thres = 0.8
            corr_y_thres = 0.8
        # self.plot_map()
        if (not os.path.exists(
                os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv')) and
                not os.path.exists(
                    os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'))):
            self.compute_relations()
        self.corr_X = pd.read_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv'),
            index_col=[0], header=[0])
        self.corr_X_imp = pd.read_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_imp_X.csv'),
            index_col=[0], header=[0])
        self.corr_y = pd.read_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'), index_col=[0],
            header=[0])
        # self.corr_reg = pd.read_csv(
        #     os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_with_regression.csv'), index_col=[0], header=[0])

        corr_X = self.corr_X > corr_X_thres
        corr_y = self.corr_y > corr_y_thres
        corr = pd.DataFrame(corr_X.values.astype('int') + corr_y.values.astype('int'), index=corr_X.index,
                            columns=corr_X.columns)
        corr_values = pd.DataFrame(self.corr_X.values + self.corr_y.values, index=corr_X.index,
                                   columns=corr_X.columns)
        project_groups = dict()
        for project in corr.index:

            if len(project_groups) == 0:
                project_groups[project] = corr.columns[corr.loc[project] == 2].tolist()
            else:
                groups = copy.deepcopy(project_groups)
                next_group = corr.columns[corr.loc[project] == 2].tolist()
                for main_project, group in project_groups.items():
                    flag1 = True
                    flag2 = True
                    if len(groups[main_project]) >= len(next_group):
                        flag2 = False
                        for gr in next_group:
                            if gr not in groups[main_project]:
                                flag1 = False
                                break
                        if flag1 == True:
                            break
                    else:
                        flag1 = False
                        for gr in groups[main_project]:
                            if gr not in next_group:
                                flag2 = False
                                break
                        if flag2 == True:
                            del groups[main_project]
                            continue
                        else:
                            flag2 = False
                    next_group1 = copy.deepcopy(next_group)
                    for prj in next_group1:
                        if prj in group:
                            if corr_values.loc[project, prj] > corr_values.loc[main_project, prj]:
                                groups[main_project].remove(prj)
                            else:
                                next_group.remove(prj)
                if len(next_group) > 0 and flag1 == False:
                    groups[project] = next_group
                project_groups = copy.deepcopy(groups)
        self.logger.info(project_groups)
        return project_groups
