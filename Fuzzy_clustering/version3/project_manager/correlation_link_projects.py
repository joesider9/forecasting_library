import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os, difflib, logging, copy

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
                    self.var_imp = [v for v in project['static_data']['clustering']['var_imp'].keys()]
                    self.coord.append([project['_id']] + project['static_data']['location'])
                    self.country = project['static_data']['projects_group']
        if len(self.project_col) < 2:
            raise FileNotFoundError('Dataset creator seems to not have been executed yet. Run a Dataset creator first')
        self.coord = pd.DataFrame(self.coord, columns=['location', 'Latitude', 'Longitude'])
        self.create_logger()

    def create_logger(self):
        self.logger = logging.getLogger('log_' + self.country + '.log')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'log_' + self.country + '.log'), 'a')
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
        gdf.plot(ax=ax, color='red', alpha=0.5, markersize=10, figsize=[100,50])
        gdf.apply(lambda x: ax.annotate(s=x.location, xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'Map_Country.png'))

    def compute_correlation(self):
        self.logger.info('Start compute linear correlations')
        self.corr_X= pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.corr_X_imp= pd.DataFrame(index=self.project_col, columns=self.project_col)
        self.corr_y= pd.DataFrame(index=self.project_col, columns=self.project_col)
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

    def compute_relations(self):
        self.compute_correlation()
        #self.plot_map()
        self.corr_X.to_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv'))
        self.corr_X_imp.to_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_imp_X.csv'))
        self.corr_y.to_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'))

    def find_relations(self, corr_X_thres=0.85, corr_y_thres=0.9):
        if self.projects[0]['static_data']['type']=='wind':
            corr_X_thres = 0.8
            corr_y_thres = 0.8
        # self.plot_map()
        if (not os.path.exists(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv')) and
            not os.path.exists(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'))):
            self.compute_relations()
        self.corr_X = pd.read_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_all_X.csv'), index_col=[0], header=[0])
        self.corr_X_imp = pd.read_csv(
            os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_imp_X.csv'), index_col=[0], header=[0])
        self.corr_y = pd.read_csv(os.path.join(os.path.dirname(self.projects[0]['static_data']['path_project']), 'corr_y.csv'), index_col=[0], header=[0])
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
                project_groups[project] = corr.columns[corr.loc[project]==2].tolist()
            else:
                groups = copy.deepcopy(project_groups)
                next_group = corr.columns[corr.loc[project]==2].tolist()
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
                            if corr_values.loc[project, prj]>corr_values.loc[main_project, prj]:
                                groups[main_project].remove(prj)
                            else:
                                next_group.remove(prj)
                if len(next_group)>0 and flag1 == False:
                    groups[project] = next_group
                project_groups = copy.deepcopy(groups)
        self.logger.info(project_groups)
        return project_groups
