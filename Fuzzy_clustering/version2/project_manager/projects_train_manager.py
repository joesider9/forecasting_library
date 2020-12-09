import logging, sys

from multiprocessing import Process

from Fuzzy_clustering.version2.model_manager.data_manage_clusters import cluster_object
from Fuzzy_clustering.version2.model_manager.models_predict_manager import ModelPredictManager
from Fuzzy_clustering.version2.model_manager.models_train_manager import ModelTrainManager
from Fuzzy_clustering.version2.project_manager.create_threads import *
from Fuzzy_clustering.version2.project_manager.create_threads_tl import *


class ProjectsTrainManager:
    def __init__(self, static_data):
        self.static_data = static_data
        self.TasksCreator = TaskCreator(static_data)
        self.TasksCreator_TL = TaskCreator_TL(static_data)
        self.nwp_model = static_data['NWP_model']
        self.nwp_resolution = static_data['NWP_resolution']
        self.project_owner = static_data['project_owner']
        self.projects_group = static_data['projects_group']
        self.area_group = static_data['area_group']
        self.version_group = static_data['version_group']
        self.version_model = static_data['version_model']
        self.data_variables = static_data['data_variables']
        self.model_type = self.static_data['type']
        self.sys_folder = self.static_data['sys_folder']
        self.path_nwp = self.static_data['path_nwp']
        self.path_group = self.static_data['path_group']
        self.path_nwp_group = self.static_data['path_nwp_group']
        self.methods = [method for method in static_data['project_methods'].keys() if
                        static_data['project_methods'][method] == True]
        self.group_static_data = joblib.load(os.path.join(self.path_group, 'static_data_projects.pickle'))
        self.create_logger()

    def fit(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(self.static_data['ngpus'])])
        projects, projects_TL = self.collect_projects()

        ncpus = int(self.static_data['njobs'])
        joblib.dump(ncpus, os.path.join(self.path_group, 'total_cpus.pickle'))
        cpu_status = ncpus
        joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(self.path_group, 'gpu_status.pickle'))

        # Fuzzy clustering threads
        responses = []
        for project in projects:
            self.logger.info('Train Fuzzy model of %s', project.static_data['_id'])
            res = fuzzy_thread(project.static_data)
            responses.append(res)

        for res in responses:
            if res[0] not in {'Done'}:
                raise RuntimeError('Fuzzy Clustering fails %s', res[1])
        # Transfer learning of fuzzy models
        for project in projects_TL:
            From_model_path = project.static_data['tl_project']['static_data']['path_fuzzy_models']
            To_model_path = project.static_data['path_fuzzy_models']

            for filename in glob.glob(os.path.join(From_model_path, '*.*')):
                shutil.copy(filename, To_model_path)
            fuzzy_model = FuzzyManager(project.static_data)
            if fuzzy_model.istrained == True:
                fuzzy_model.save()
            else:
                raise RuntimeError('Check the transfer Fuzzy model for %s', project.static_data['_id'])

        # Create cluster regressors and write their data
        for project in projects:
            self.logger.info('Create clusters of %s', project.static_data['_id'])
            project.create_clusters_and_data()
        for project in projects_TL:
            self.logger.info('Create clusters of %s', project.static_data['_id'])
            project.create_clusters_and_data()

        results_fs = Parallel(n_jobs=ncpus)(delayed(FS_thread)(project.static_data['_id'], cluster)
                                            for project in projects for cluster_name, cluster in
                                            project.clusters.items())
        # results_fs = []
        # for project in projects:
        #     for cluster_name, cluster in project.clusters.items():
        #         if not self.static_data['sklearn']['fs_method'] == '':
        #             results_fs.append(FS_thread(project.static_data['_id'], cluster))
        if len(results_fs) > 0:
            for res in results_fs:
                if res[0] not in {'Done'}:
                    raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                clust_obj = cluster_object(cluster.static_data, cluster_name)
                clust_obj.istrained = False
                clust_obj.save(cluster.cluster_dir)
                cluster = clust_obj
            project.save()
        procs = []
        procs.append(Process(target=train_on_cpus, args=(projects, self.static_data, self.methods, self.path_group)))
        procs.append(Process(target=train_on_gpus, args=(projects, self.static_data, self.methods, self.path_group)))
        for p in procs:
            p.daemon = False
            p.start()
        for p in procs:
            p.join()

        self.train_combine_models(projects)

    def train_combine_models(self, projects):

        tasks_comb = self.TasksCreator.create_tasks_stage_for_combine(projects)
        if len(tasks_comb) > 0:
            self.logger.info('Train Combine cluster models')
            results_comb = train_stage_combine(tasks_comb)
            for res in results_comb:
                if res[0] not in {'Done'}:
                    raise RuntimeError('Combine model training fails for cluster %s of project %s', res[1],
                                       res[2])

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                clust_obj = cluster_object(cluster.static_data, cluster_name)
                clust_obj.istrained = True
                clust_obj.save(cluster.cluster_dir)
                cluster.istrained = True
            project.save()

        tasks_comb_model = self.TasksCreator.create_tasks_stage_model_combine(projects)
        if len(tasks_comb_model) > 0:
            self.logger.info('Train overall Combine models')
            results_comb_model = train_stage_modelcombine(tasks_comb_model)
            for res in results_comb_model:
                project = res[1]
                if res[0] not in {'Done'}:
                    raise RuntimeError('Combine model training fails of project %s', project.static_data['_id'])
                else:
                    self.logger.info('%s trained successfully', project.static_data['_id'])
                    project.istrained = True
                    project.save()

    def train_proba(self):
        projects, projects_TL = self.collect_projects()
        projects += projects_TL
        for project in projects:
            model_manager = ModelPredictManager(project.static_data)
            predictions = model_manager.predict_offline()
        njobs_mlp = self.static_data['MLP']['njobs']
        tasks_proba_stage1 = self.TasksCreator.create_tasks_proba_stage1(projects)
        if len(tasks_proba_stage1) > 0:
            results_proba_stage1 = train_proba(tasks_proba_stage1, njobs_mlp)

            result_1st_stage_proba_pd = pd.DataFrame(results_proba_stage1,
                                                     columns=['acc', 'project', 'test', 'method'])
            result_1st_stage_proba_pd = result_1st_stage_proba_pd.iloc[
                result_1st_stage_proba_pd.groupby(by=['method', 'project']).agg(
                    {'acc': 'idxmin'}).values.ravel()]

            result_1st_stage_proba_pd.to_csv(os.path.join(self.path_group, 'results_proba_models_best.csv'))
            self.save_proba_models('MLP', result_1st_stage_proba_pd, projects)

    def save_proba_models(self, method, results, projects):
        for project in projects:
            test = results['test'].iloc[np.where((method == results['method']).any() and (
                    project.static_data['_id'] == results['project']).any())[0]].values[0]
            model = proba_model_manager(project.static_data, {'test': test})
            for filename in glob.glob(os.path.join(model.test_dir, '*.*')):
                shutil.copy(filename, model.model_dir)

    def save_deep_models(self, method, results, projects):
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                test = results['test'].iloc[np.where((method == results['method']).any() and (
                        project.static_data['_id'] == results['project']).any()
                                                     and (cluster_name == results['cluster']).any())[0]].values[0]
                model = model3d_manager(project.static_data, cluster, method, {'test': test})
                for filename in glob.glob(os.path.join(model.test_dir, '*.*')):
                    shutil.copy(filename, model.model_dir)

    def create_logger(self):
        self.logger = logging.getLogger('ProjectTrainManager_' + self.model_type)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_group, 'log_' + self.projects_group + '.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)

    def collect_projects(self):
        projects = []
        projects_TL = []
        for project in self.group_static_data:
            if not 'path_group' in project['static_data'].keys():
                project['static_data']['path_group'] = self.path_group
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelTrainManager(project['static_data']['path_model'])
                # if project_model.istrained == False:
                project_model.init(project['static_data'], self.data_variables)
                if self.model_type in {'wind', 'pv'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_cnn.pickle')):
                        if project['static_data']['transfer_learning'] == False:
                            projects.append(project_model)
                        else:
                            projects_TL.append(project_model)
                    else:
                        raise ValueError('Cannot find project ', project['_id'], ' datasets')

                elif self.model_type in {'load'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle')):
                        projects.append(project_model)

                elif self.model_type in {'fa'}:
                    if os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_X.csv')) \
                            and os.path.exists(os.path.join(project['static_data']['path_data'], 'dataset_y.csv')) \
                            and os.path.exists(
                        os.path.join(project['static_data']['path_data'], 'dataset_lstm.pickle')):
                        projects.append(project_model)
                else:
                    raise ValueError('Cannot recognize model type')
        return projects, projects_TL

    def clear_backup_projects(self):

        for project in self.group_static_data:
            if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                project_model = ModelTrainManager(project['static_data']['path_model'])
                if project_model.istrained == True:
                    project_model.init(project['static_data'], self.data_variables)
                    self.clear(project_model)
                    self.backup(project_model)
                else:
                    raise ValueError('Cannot recognize model type')

    def clear(self, project):
        paths = []
        for filename in os.listdir(project.path_model):
            p = os.path.join(project.path_model, filename)
            if os.path.isdir(p):
                if 'test' in filename:
                    paths.append(p)
                for filename1 in os.listdir(p):
                    p1 = os.path.join(p, filename1)
                    if os.path.isdir(p1):
                        if 'test' in filename1:
                            paths.append(p1)
                        for filename2 in os.listdir(p1):
                            p2 = os.path.join(p1, filename2)
                            if os.path.isdir(p2):
                                if 'test' in filename2:
                                    paths.append(p2)
                                for filename3 in os.listdir(p2):
                                    p3 = os.path.join(p2, filename3)
                                    if os.path.isdir(p3):
                                        if 'test' in filename3:
                                            paths.append(p3)
        for path in paths:
            shutil.rmtree(path)

    def backup(self, project):
        shutil.rmtree(project.static_data['path_backup'])
        shutil.copytree(project.static_data['path_model'], project.static_data['path_backup'])
        paths = []
        for name in ['data', 'DATA']:
            for filename in os.listdir(project.static_data['path_backup']):
                p = os.path.join(project.static_data['path_backup'], filename)
                if os.path.isdir(p):
                    if name in filename:
                        paths.append(p)
                    for filename1 in os.listdir(p):
                        p1 = os.path.join(p, filename1)
                        if os.path.isdir(p1):
                            if name in filename1:
                                paths.append(p1)
                            for filename2 in os.listdir(p1):
                                p2 = os.path.join(p1, filename2)
                                if os.path.isdir(p2):
                                    if name in filename2:
                                        paths.append(p2)
                                    for filename3 in os.listdir(p2):
                                        p3 = os.path.join(p2, filename3)
                                        if os.path.isdir(p3):
                                            if name in filename3:
                                                paths.append(p3)
        for path in paths:
            shutil.rmtree(path)

    def train_TL(self, projects):
        # Train in parallel deep_models and Feature Selection
        njobs_cnn_3d = self.static_data['CNN']['njobs_3d']
        njobs_lstm = self.static_data['LSTM']['njobs']
        tasks_fs, tasks_3d_stage1 = self.TasksCreator_TL.create_tasks_TL_3d_fs(projects, njobs_cnn_3d, njobs_lstm)
        self.logger.info('Train feature selection and Model_3d 1st stage for Transfer Learning Models')
        results_stage1 = train_TL_stage1([tasks_fs, tasks_3d_stage1])
        result_stage_fs = results_stage1[0]
        result_1st_stage_3d = results_stage1[1][0]
        result_1st_stage_3d_pd = pd.DataFrame(result_1st_stage_3d,
                                              columns=['acc', 'cluster', 'project', 'test', 'method'])
        result_1st_stage_3d_pd = result_1st_stage_3d_pd.iloc[
            result_1st_stage_3d_pd.groupby(by=['method', 'project', 'cluster']).agg({'acc': 'idxmin'}).values.ravel()]
        for res in result_stage_fs:
            if res[0] not in {'Done'}:
                raise RuntimeError('Feature selection fails cluster %s of project %s for Transfer Learning Models',
                                   res[1], res[2])

        if 'CNN' in result_1st_stage_3d_pd['method'].to_list():
            self.save_deep_models('CNN', result_1st_stage_3d_pd, projects)
        if 'LSTM' in result_1st_stage_3d_pd['method'].to_list():
            self.save_deep_models('LSTM', result_1st_stage_3d_pd, projects)

        self.logger.info('Train RBFols and RBFNN 1st stage')
        if ('RBF_ALL_CNN' in self.methods) or ('RBF_ALL' in self.methods):
            njobs_rbf = self.static_data['RBF']['njobs']
            tasks_rbf_ols, task_rbf_stage1 = self.TasksCreator_TL.create_tasks_TL_stage_for_rbfs(projects, njobs_rbf)
            if len(tasks_rbf_ols) > 0:
                result_stage_rbfols = train_stage_rbfols_TL(tasks_rbf_ols, njobs_rbf)
                for res in result_stage_rbfols:
                    if res[0] not in {'Done'}:
                        raise RuntimeError('RBF_OLS training fails cluster %s of project %s', res[1], res[2])
            if len(task_rbf_stage1) > 0:
                result_1st_stage_rbf = train_3d(task_rbf_stage1)[0]
                result_1st_stage_rbf_pd = pd.DataFrame(result_1st_stage_rbf,
                                                       columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_1st_stage_rbf_pd = result_1st_stage_rbf_pd.iloc[
                    result_1st_stage_rbf_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                result_1st_stage_rbf_pd.to_csv(os.path.join(self.path_group, 'results_RBF_models_best.csv'))
                self.save_deep_models('RBFNN', result_1st_stage_rbf_pd, projects)

        if ('RBF_ALL_CNN' in self.methods):
            self.logger.info('Train RBF-CNN 1st stage')
            njobs_cnn = self.static_data['CNN']['njobs']
            tasks_rbfcnn_stage1 = self.TasksCreator_TL.create_tasks_TL_rbfcnn_stage1(projects, njobs_cnn)
            if len(tasks_rbfcnn_stage1) > 0:
                results_rbfcnn_stage1 = train_3d(tasks_rbfcnn_stage1)[0]

                result_1st_stage_rbfcnn_pd = pd.DataFrame(results_rbfcnn_stage1,
                                                          columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_1st_stage_rbfcnn_pd = result_1st_stage_rbfcnn_pd.iloc[
                    result_1st_stage_rbfcnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                result_1st_stage_rbfcnn_pd.to_csv(os.path.join(self.path_group, 'results_3d_models_best.csv'))
                self.save_deep_models('RBF-CNN', result_1st_stage_rbfcnn_pd, projects)

        self.logger.info('Train SKlearn models')
        sklearn_methods = [method for method in self.methods if method in {'SVM', 'NUSVM', 'MLP', 'RF', 'XGB'}]
        if len(sklearn_methods):
            tasks_sk = self.TasksCreator_TL.create_tasks_TL_stage_for_sklearn(projects, sklearn_methods)
            if len(tasks_sk) > 0:
                results_sk = train_stage_sklearn_TL(tasks_sk, self.static_data['sklearn']['njobs'])
                for res in results_sk:
                    if res[0] not in {'Done'}:
                        raise RuntimeError('SKLearn %s model training fails cluster %s of project %s', res[3], res[1],
                                           res[2])

        tasks_comb = self.TasksCreator.create_tasks_stage_for_combine(projects)
        if len(tasks_comb) > 0:
            self.logger.info('Train Combine cluster models')
            results_comb = train_stage_combine(tasks_comb)
            for res in results_comb:
                if res[0] not in {'Done'}:
                    raise RuntimeError('Combine model training fails for cluster %s of project %s', res[1],
                                       res[2])

        tasks_comb_model = self.TasksCreator.create_tasks_stage_model_combine(projects)
        if len(tasks_comb_model) > 0:
            self.logger.info('Train overall Combine models')
            results_comb_model = train_stage_modelcombine(tasks_comb_model)
            for res in results_comb_model:
                project = res[1]
                if res[0] not in {'Done'}:
                    raise RuntimeError('Combine model training fails of project %s', project.static_data['_id'])
                else:
                    self.logger.info('%s trained successfully', project.static_data['_id'])
                    project.istrained = True
                    project.save()
