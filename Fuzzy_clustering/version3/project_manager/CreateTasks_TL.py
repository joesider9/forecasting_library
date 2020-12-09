import numpy as np
import pandas as pd
import copy


class TaskCreator_TL():
    def __init__(self, static_data):
        self.static_data = static_data

    def create_tasks_TL_stage_for_sklearn(self, projects, sklearn_methods):
        # Train in parallel SKLEARN models
        tasks_sk_ols = []
        for method in sklearn_methods:
            for project in projects:
                for cluster_name, cluster in project.clusters.items():
                    if cluster.istrained == False:
                        task = {'project': project.static_data['_id'], 'static_data': project.static_data,
                                'cluster': cluster, 'method': method,
                                'optimize_method': project.static_data['sklearn']['optimizer']}
                        tasks_sk_ols.append(task)
        return tasks_sk_ols

    def create_tasks_TL_rbfcnn_stage1(self, projects, njobs_cnn):
        # Train in parallel RBF-CNN

        task_rbfcnn_stage1 = dict()
        gpu = 0
        task_count = 0
        task_rbfcnn_stage1['task' + str(task_count)] = dict()
        for n in range(self.static_data['ngpus']):
            task_rbfcnn_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []

        def task_check(task_rbfcnn_stage1, task_count, njobs, ngpus):
            flag = 0
            for n in range(ngpus):
                if len(task_rbfcnn_stage1['task' + str(task_count)]['/device:GPU:' + str(n)]) >= njobs:
                    flag += 1
            if flag == ngpus:
                task_count += 1
                task_rbfcnn_stage1['task' + str(task_count)] = dict()
                for n in range(project.static_data['ngpus']):
                    task_rbfcnn_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []
            return task_rbfcnn_stage1, task_count

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    if ('RBF_ALL_CNN' in cluster.methods):

                        task = {'method': 'RBF-CNN', 'project': project.static_data['_id'], 'cluster': cluster,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'gpu': gpu}}

                        task_rbfcnn_stage1['task' + str(task_count)]['/device:GPU:' + str(gpu)].append(task)

                        task_rbfcnn_stage1, task_count = task_check(task_rbfcnn_stage1, task_count, njobs_cnn,
                                                                    project.static_data['ngpus'])

                        gpu += 1
                        if gpu == project.static_data['ngpus']:
                            gpu = 0

        return task_rbfcnn_stage1

    def create_tasks_TL_stage_for_rbfs(self, projects, njobs_rbf):
        # Train in parallel deep_models and Feature Selection
        tasks_rbf_ols = []
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    task = {'project': project.static_data['_id'], 'static_data': project.static_data, 'cluster': cluster}
                    tasks_rbf_ols.append(task)

        task_rbf_stage1 = dict()
        gpu = 0
        task_count = 0
        task_rbf_stage1['task' + str(task_count)] = dict()
        for n in range(self.static_data['ngpus']):
            task_rbf_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []

        def task_check(task_rbf_stage1, task_count, njobs, ngpus):
            flag = 0
            for n in range(ngpus):
                if len(task_rbf_stage1['task' + str(task_count)]['/device:GPU:' + str(n)]) >= njobs:
                    flag += 1
            if flag == ngpus:
                task_count += 1
                task_rbf_stage1['task' + str(task_count)] = dict()
                for n in range(project.static_data['ngpus']):
                    task_rbf_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []
            return task_rbf_stage1, task_count

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    if ('RBF_ALL_CNN' in cluster.methods) or ('RBF_ALL' in cluster.methods):

                        task = {'method': 'RBFNN', 'project': project.static_data['_id'], 'cluster': cluster,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'gpu': gpu}}
                        task_rbf_stage1['task' + str(task_count)]['/device:GPU:' + str(gpu)].append(task)

                        task_rbf_stage1, task_count = task_check(task_rbf_stage1, task_count, njobs_rbf,
                                                                 project.static_data['ngpus'])

                        gpu += 1
                        if gpu == project.static_data['ngpus']:
                            gpu = 0
        return tasks_rbf_ols, task_rbf_stage1

    def create_tasks_TL_3d_fs(self, projects, njobs_cnn, njobs_lstm):
        # Train in parallel deep_models and Feature Selection
        tasks_fs = []
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    task = {'project': project.static_data['_id'], 'cluster': cluster}
                    tasks_fs.append(task)

        task_3d_stage1 = dict()
        gpu = 0
        task_count = 0
        task_3d_stage1['task' + str(task_count)] = dict()
        for n in range(self.static_data['ngpus']):
            task_3d_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []

        def task_check(task_3d_stage1, task_count, njobs, ngpus):
            flag = 0
            for n in range(ngpus):
                if len(task_3d_stage1['task' + str(task_count)]['/device:GPU:' + str(n)]) >= njobs:
                    flag += 1
            if flag == ngpus:
                task_count += 1
                task_3d_stage1['task' + str(task_count)] = dict()
                for n in range(project.static_data['ngpus']):
                    task_3d_stage1['task' + str(task_count)]['/device:GPU:' + str(n)] = []
            return task_3d_stage1, task_count

        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    if 'LSTM' in cluster.methods:
                        min_units = np.maximum(cluster.D, 64)
                        lr = project.static_data['LSTM']['learning_rate']

                        task = {'method': 'LSTM', 'project': project.static_data['_id'], 'cluster': cluster,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'gpu': gpu}}
                        task_3d_stage1['task' + str(task_count)]['/device:GPU:' + str(gpu)].append(task)

                        task_3d_stage1, task_count = task_check(task_3d_stage1, task_count, njobs_lstm,
                                                                project.static_data['ngpus'])

                        gpu += 1
                        if gpu == project.static_data['ngpus']:
                            gpu = 0

        gpu = 0
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if cluster.istrained == False:
                    if 'CNN' in cluster.methods:

                        task = {'method': 'CNN', 'project': project.static_data['_id'], 'cluster': cluster,
                                'static_data': project.static_data,
                                'params': {'test': 1, 'gpu': gpu}}

                        task_3d_stage1['task' + str(task_count)]['/device:GPU:' + str(gpu)].append(task)

                        task_3d_stage1, task_count = task_check(task_3d_stage1, task_count, njobs_cnn,
                                                                project.static_data['ngpus'])

                        gpu += 1
                        if gpu == project.static_data['ngpus']:
                            gpu = 0

        return tasks_fs, task_3d_stage1