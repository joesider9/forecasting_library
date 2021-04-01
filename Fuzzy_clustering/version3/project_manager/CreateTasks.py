import copy

import numpy as np


class TaskCreator():
    def __init__(self, static_data):
        self.static_data = static_data

    def create_tasks_stage_model_combine(self, projects):
        # Train in parallel SKLEARN models
        tasks_comb_ols = []
        for project in projects:
            task = {'project': project}
            tasks_comb_ols.append(task)
        return tasks_comb_ols

    def create_tasks_stage_for_combine(self, projects):
        # Train in parallel SKLEARN models
        tasks_comb_ols = []
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    task = {'project': project.static_data['_id'], 'static_data': project.static_data, 'cluster': cluster.cluster_name}
                    tasks_comb_ols.append(task)
        return tasks_comb_ols

    def create_tasks_stage_for_sklearn(self, projects, sklearn_methods):
        # Train in parallel SKLEARN models
        tasks_sk_ols = []
        for method in sklearn_methods:
            for project in projects:
                for cluster_name, cluster in project.clusters.items():
                    if cluster.is_trained == False:
                        task = {'project': project.static_data['_id'], 'static_data': project.static_data,
                                'cluster': cluster.cluster_name, 'method': method,
                                'optimize_method': project.static_data['sklearn']['optimizer']}
                        tasks_sk_ols.append(task)
        return tasks_sk_ols

    def create_tasks_rbfcnn_stage2(self, result_1st_stage_rbf_pd, tasks_rbf_stage1):
        tasks = []
        for task_ind in tasks_rbf_stage1:
            ind = np.where((result_1st_stage_rbf_pd['method'] == task_ind['method']).any()
                           and (result_1st_stage_rbf_pd['project'] == task_ind['project']).any()
                           and (result_1st_stage_rbf_pd['cluster'] == task_ind['cluster'].cluster_name).any())[0]
            if task_ind['params']['test'] == result_1st_stage_rbf_pd['test'].iloc[ind].values[0]:
                tasks.append(task_ind)
        gpu = np.tile(np.arange(self.static_data['ngpus']), int(2 * len(tasks) / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        i = 0
        task_rbfcnn_stage2 = []

        for task_ind in tasks:
            task1 = copy.deepcopy(task_ind)

            task1['params']['h_size'] = [512, 128]
            task1['params']['test'] = 5
            task1['params']['gpu'] = gpu[i]
            task_rbfcnn_stage2.append(task1)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0

            task2 = copy.deepcopy(task_ind)
            task2['params']['h_size'] = [1024, 256]
            task2['params']['test'] = 6
            task1['params']['gpu'] = gpu[i]
            task_rbfcnn_stage2.append(task2)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0

        return task_rbfcnn_stage2

    def create_tasks_rbfcnn_stage1(self, projects):
        # Train in parallel RBF-CNN

        task_rbfcnn_stage1 = dict()
        gpu = 0
        task_count = 0
        task_rbfcnn_stage1 = []

        ntasks = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                ntasks += 4
        gpu = np.tile(np.arange(self.static_data['ngpus']), int(ntasks / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        i = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.is_trained == False:
                    if ('RBF_ALL_CNN' in cluster.methods):
                        lr = project.static_data['CNN']['learning_rate']
                        h_size = project.static_data['CNN']['h_size']

                        task = {'method': 'RBF-CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'trial': 0, 'pool_size': [2, 1], 'kernels': [2, 4],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_rbfcnn_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'RBF-CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 2, 'trial': 0, 'pool_size': [2, 1], 'kernels': [4, 2],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_rbfcnn_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'RBF-CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 3, 'trial': 3, 'pool_size': [1, 2, 2], 'kernels': [2, 4, 4],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_rbfcnn_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'RBF-CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 4, 'trial': 3, 'pool_size': [1, 2, 2], 'kernels': [3, 2, 2],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_rbfcnn_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0

        return task_rbfcnn_stage1

    def create_tasks_3d_stage2(self, result_1st_stage_3d_pd, tasks_3d_stage1):
        tasks = []
        for task_ind in tasks_3d_stage1:
            ind = np.where((result_1st_stage_3d_pd['method'] == task_ind['method']).any()
                           and (result_1st_stage_3d_pd['project'] == task_ind['project']).any()
                           and (result_1st_stage_3d_pd['cluster'] == task_ind['cluster'].cluster_name).any())[0]
            if task_ind['params']['test'] == result_1st_stage_3d_pd['test'].iloc[ind].values[0]:
                tasks.append(task_ind)

        gpu = np.tile(np.arange(self.static_data['ngpus']), int(2 * len(tasks) / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        task_count = 0
        task_3d_stage2 = []
        i = 0
        for task_ind in tasks:
            task1 = copy.deepcopy(task_ind)
            if task1['method'] == 'LSTM':

                task1['params']['lr'] = 1e-4
                task1['params']['test'] = 3
                task1['params']['gpu'] = gpu[i]
                task_3d_stage2.append(task1)

                i += 1
                # if gpu == self.static_data['ngpus']:
                #     gpu = 0

                task2 = copy.deepcopy(task_ind)
                task2['params']['lr'] = 1e-5
                task2['params']['test'] = 4
                task1['params']['gpu'] = gpu[i]
                task_3d_stage2.append(task2)

                i += 1
                # if gpu == self.static_data['ngpus']:
                #     gpu = 0
            else:
                task1['params']['h_size'] = [512, 128]
                task1['params']['test'] = 5
                task1['params']['gpu'] = gpu[i]
                task_3d_stage2.append(task1)

                i += 1
                # if gpu == self.static_data['ngpus']:
                #     gpu = 0

                task2 = copy.deepcopy(task_ind)
                task2['params']['h_size'] = [1024, 256]
                task2['params']['test'] = 6
                task2['params']['gpu'] = gpu[i]
                task_3d_stage2.append(task2)

                i += 1
                # if gpu == self.static_data['ngpus']:
                #     gpu = 0

        return task_3d_stage2

    def create_tasks_stage_rbf_lr(self, result_1st_stage_rbf_pd, task_rbf_stage1):
        tasks = []
        for task_ind in task_rbf_stage1:
            ind = np.where((result_1st_stage_rbf_pd['method'] == task_ind['method']).any()
                           and (result_1st_stage_rbf_pd['project'] == task_ind['project']).any()
                           and (result_1st_stage_rbf_pd['cluster'] == task_ind['cluster'].cluster_name).any())[0]
            if task_ind['params']['test'] == result_1st_stage_rbf_pd['test'].iloc[ind].values[0]:
                tasks.append(task_ind)

        gpu = np.tile(np.arange(self.static_data['ngpus']), int(2 * len(tasks) / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)

        i = 0
        task_rbf_stage_fn = []

        for task_ind in tasks:
            task1 = copy.deepcopy(task_ind)
            task1['params']['lr'] = 1e-3
            task1['params']['test'] = 13
            task1['params']['gpu'] = gpu[i]
            task_rbf_stage_fn.append(task1)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0

            task2 = copy.deepcopy(task_ind)
            task2['params']['lr'] = 1e-4
            task2['params']['test'] = 14
            task2['params']['gpu'] = gpu[i]
            task_rbf_stage_fn.append(task2)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0
        return task_rbf_stage_fn

    def create_tasks_stage_rbf_ft(self, result_1st_stage_rbf_pd, task_rbf_stage1):
        tasks = []
        for task_ind in task_rbf_stage1:
            ind = np.where((result_1st_stage_rbf_pd['method'] == task_ind['method']).any()
                           and (result_1st_stage_rbf_pd['project'] == task_ind['project']).any()
                           and (result_1st_stage_rbf_pd['cluster'] == task_ind['cluster'].cluster_name).any())[0]
            if task_ind['params']['test'] == result_1st_stage_rbf_pd['test'].iloc[ind].values[0]:
                tasks.append(task_ind)
        i = 0
        task_rbf_stage1 = []
        gpu = np.tile(np.arange(self.static_data['ngpus']), int(2 * len(tasks) / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)

        for task_ind in tasks:
            task1 = copy.deepcopy(task_ind)
            task1['params']['num_centr'] = task_ind['params']['num_centr'] - 2
            task1['params']['test'] = 11
            task1['params']['gpu'] = gpu[i]
            task_rbf_stage1.append(task1)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0

            task2 = copy.deepcopy(task_ind)
            task2['params']['num_centr'] = task_ind['params']['num_centr'] + 2
            task2['params']['test'] = 12
            task2['params']['gpu'] = gpu[i]
            task_rbf_stage1.append(task2)

            i += 1
            # if gpu == self.static_data['ngpus']:
            #     gpu = 0

        return task_rbf_stage1

    def create_tasks_stage_for_rbfs(self, projects):

        gpu = 0
        task_rbf_stage1 = []
        ntasks = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                ntasks += 11
        gpu = np.tile(np.arange(self.static_data['ngpus']), int(ntasks / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        i = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.is_trained == False:
                    if ('RBF_ALL_CNN' in cluster.methods) or ('RBF_ALL' in cluster.methods):
                        lr = project.static_data['RBF']['learning_rate']
                        for i, num_centr in enumerate([8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 52]):
                            task = {'method': 'RBFNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                    'static_data': project.static_data,
                                    'params': {'test': i, 'num_centr': num_centr, 'lr': lr,
                                               'gpu': gpu[i]}}
                            task_rbf_stage1.append(task)

                            i += 1
                            # if gpu == project.static_data['ngpus']:
                            #     gpu = 0
        return task_rbf_stage1

    def create_tasks_3d_stage1(self, projects):
        # Train in parallel deep_models and Feature Selection
        task_3d_stage1 = dict()
        ntasks = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if 'LSTM' in cluster.methods:
                    ntasks += 2
                if 'CNN' in cluster.methods:
                    ntasks += 4

        gpu = np.tile(np.arange(self.static_data['ngpus']), int(ntasks / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        i = 0
        task_count = 0
        task_3d_stage1 = []

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.is_trained == False:
                    if 'LSTM' in cluster.methods:
                        min_units = project.static_data['LSTM']['units']
                        lr = project.static_data['LSTM']['learning_rate']

                        task = {'method': 'LSTM', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'trial': 2, 'units': [2 * min_units, 1024, min_units], 'lr': lr,
                                           'gpu': gpu[i]}}
                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'LSTM', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 2, 'trial': 3, 'units': [2 * min_units, 1024, min_units, 1024],
                                           'lr': lr,
                                           'gpu': gpu[i]}}

                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.is_trained == False:
                    if 'CNN' in cluster.methods:
                        lr = project.static_data['CNN']['learning_rate']
                        h_size = project.static_data['CNN']['h_size']

                        task = {'method': 'CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'trial': 0, 'pool_size': [2, 1], 'kernels': [2, 4],
                                           'h_size': h_size,
                                           'lr': lr, 'gpu': gpu[i]}}

                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 2, 'trial': 0, 'pool_size': [2, 1], 'kernels': [4, 2],
                                           'h_size': h_size,
                                           'lr': lr, 'gpu': gpu[i]}}

                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 3, 'trial': 3, 'pool_size': [1, 2, 2], 'kernels': [2, 4, 4],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0
                        task = {'method': 'CNN', 'project': project.static_data['_id'], 'cluster': cluster.cluster_name,
                                'static_data': project.static_data,
                                'params': {'test': 4, 'trial': 3, 'pool_size': [1, 2, 2], 'kernels': [3, 2, 2],
                                           'h_size': h_size, 'lr': lr, 'gpu': gpu[i]}}

                        task_3d_stage1.append(task)

                        i += 1
                        # if gpu == project.static_data['ngpus']:
                        #     gpu = 0

        return task_3d_stage1

    def create_tasks_proba_stage1(self, projects):
        # Train in parallel proba models
        ntasks = 0
        for project in projects:
            ntasks += 4

        gpu = np.tile(np.arange(self.static_data['ngpus']), int(ntasks / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        task_count = 0
        task_3d_stage1 = []
        i = 0
        for project in projects:
            lr = project.static_data['MLP']['learning_rate']

            task = {'method': 'MLP', 'project': project.static_data['_id'],
                    'static_data': project.static_data,
                    'params': {'test': 1, 'trial': 0, 'units': [20], 'lr': lr, 'act_func': 'tanh',
                               'gpu': gpu[i]}}
            task_3d_stage1.append(task)

            i += 1
            # if gpu == project.static_data['ngpus']:
            #     gpu = 0
            task = {'method': 'MLP', 'project': project.static_data['_id'],
                    'static_data': project.static_data,
                    'params': {'test': 2, 'trial': 0, 'units': [100], 'lr': lr, 'act_func': 'elu',
                               'gpu': gpu[i]}}

            task_3d_stage1.append(task)

            i += 1
            # if gpu == project.static_data['ngpus']:
            #     gpu = 0

            task = {'method': 'MLP', 'project': project.static_data['_id'],
                    'static_data': project.static_data,
                    'params': {'test': 3, 'trial': 1, 'units': [100, 20], 'lr': lr, 'act_func': 'tanh',
                               'gpu': gpu[i]}}
            task_3d_stage1.append(task)

            i += 1
            # if gpu == project.static_data['ngpus']:
            #     gpu = 0
            task = {'method': 'MLP', 'project': project.static_data['_id'],
                    'static_data': project.static_data,
                    'params': {'test': 4, 'trial': 1, 'units': [250, 80], 'lr': lr, 'act_func': 'elu',
                               'gpu': gpu[i]}}

            task_3d_stage1.append(task)

            i += 1
            # if gpu == project.static_data['ngpus']:
            #     gpu = 0

        return task_3d_stage1

    def create_tasks_MLP_stage1(self, projects):
        # Train in parallel proba models
        ntasks = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                ntasks += 6

        lr = project.static_data['MLP']['learning_rate']

        gpu = np.tile(np.arange(self.static_data['ngpus']), int(ntasks / self.static_data['ngpus']) + 1)
        np.random.shuffle(gpu)
        i = 0
        task_3d_stage1 = []

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.is_trained == False:
                    lr = project.static_data['MLP']['learning_rate']

                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 1, 'trial': 0, 'units': [100], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}
                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0
                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 2, 'trial': 0, 'units': [250], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}

                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0

                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 3, 'trial': 1, 'units': [100, 20], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}
                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0
                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 4, 'trial': 1, 'units': [250, 80], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}

                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0

                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 5, 'trial': 2, 'units': [200, 100, 20], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}
                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0
                    task = {'method': 'MLP', 'project': project.static_data['_id'],
                            'static_data': project.static_data,
                            'params': {'test': 6, 'trial': 2, 'units': [512, 256, 64], 'lr': lr, 'act_func': 'elu',
                                       'gpu': gpu[i]}}

                    task_3d_stage1.append(task)

                    i += 1
                    # if gpu == project.static_data['ngpus']:
                    #     gpu = 0

        return task_3d_stage1
