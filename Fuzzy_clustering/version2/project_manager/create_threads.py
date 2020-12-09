import glob
import joblib
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed

from Fuzzy_clustering.version2.combine_clusters_manager.combine_cluster_manager import ClusterCombiner
from Fuzzy_clustering.version2.combine_model_manager.combine_model_manager import CombineModelManager
from Fuzzy_clustering.version2.deep_models.model_3d_manager import model3d_manager
from Fuzzy_clustering.version2.feature_selection_manager.fs_manager import FeatSelManager
from Fuzzy_clustering.version2.fuzzy_clustering_manager.train_fuzzy_manager import FuzzyManager
from Fuzzy_clustering.version2.probabilistic_manager.proba_model_manager import proba_model_manager
from Fuzzy_clustering.version2.project_manager.create_tasks import TaskCreator
from Fuzzy_clustering.version2.rbf_ols_manager.rbf_ols_manager import RBFOLS_Manager
from Fuzzy_clustering.version2.sklearn_models.sklearn_manager import SKLearn_Manager


def fuzzy_thread(static_data):
    fuzzy_model = FuzzyManager(static_data)
    if fuzzy_model.istrained == False:
        return fuzzy_model.train_fuzzy_clustering(), static_data['_id']
    else:
        return ('Done', static_data['_id'])


def FS_thread(project_id, cluster):
    FS_model = FeatSelManager(cluster)
    if FS_model.istrained == False:
        return FS_model.fit(), cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id


def FS_check(project_id, cluster):
    FS_model = FeatSelManager(cluster)
    if FS_model.istrained == False:
        return 'Untrained', cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id


def RBF_check(project_id, static_data, cluster):
    rbf_model = RBFOLS_Manager(static_data, cluster)
    if rbf_model.istrained == False:
        return 'Untrained', cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id


def Model3dThread(project_id, static_data, cluster, method, params):
    model = model3d_manager(static_data, cluster, method, params)
    if model.istrained == False:
        acc = model.fit()
    else:
        acc = model.acc

    return acc, cluster.cluster_name, project_id, params['test'], method


def RBFOLS_thread(project_id, static_data, cluster):
    rbf_model = RBFOLS_Manager(static_data, cluster)
    if rbf_model.istrained == False:
        return rbf_model.fit(), cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id


def GPU_thread(tasks, njobs):
    # pool = mp.Pool(processes=njobs)
    # result = [pool.apply_async(Model3dThread, args=(task['project'], task['static_data'],
    #                                                               task['cluster'], task['method'], task['params'])) for
    #                                                                task in tasks]
    # results = [p.get() for p in result]
    # pool.close()
    # pool.terminate()
    # pool.join()
    results = Parallel(n_jobs=njobs)(delayed(Model3dThread)(task['project'], task['static_data'],
                                                            task['cluster'], task['method'], task['params']) for
                                     task in tasks)
    return results


def GPU_thread_parallel(tasks, njobs):
    pool = mp.Pool(processes=njobs)
    result = [pool.apply_async(Model3dThread, args=(task['project'], task['static_data'],
                                                    task['cluster'], task['method'], task['params'])) for
              task in tasks]
    results = [p.get() for p in result]
    pool.close()
    pool.terminate()
    pool.join()
    # results = Parallel(n_jobs=njobs)(delayed(Model3dThread)(task['project'], task['static_data'],
    #                                                               task['cluster'], task['method'], task['params']) for
    #                                                                task in tasks)
    return results


def ProbaThread(project_id, static_data, params):
    method = 'MLP'
    model = proba_model_manager(static_data, params)
    if model.istrained == False:
        acc = model.fit()
    else:
        acc = model.acc

    return acc, project_id, params['test'], method


def GPU_thread_proba_parallel(tasks, njobs):
    pool = mp.Pool(processes=njobs)
    result = [pool.apply_async(ProbaThread, args=(task['project'], task['static_data'], task['params'])) for
              task in tasks]
    results = [p.get() for p in result]
    pool.close()
    pool.terminate()
    pool.join()
    # results = Parallel(n_jobs=njobs)(delayed(ProbaThread)(task['project'], task['static_data'],
    #                                                               task['cluster'], task['method'], task['params']) for
    #                                                                task in tasks)
    return results


def GPU_thread_proba(tasks, njobs):
    results = Parallel(n_jobs=njobs)(delayed(ProbaThread)(task['project'], task['static_data'],
                                                          task['params']) for task in tasks)
    return results


def train_3d(tasks_3d, njobs):
    results = []
    tasks_gpu = dict()
    for task in tasks_3d:
        gpu_name = '/device:GPU:' + str(task['params']['gpu'])
        if not gpu_name in tasks_gpu:
            tasks_gpu[gpu_name] = []
        tasks_gpu[gpu_name].append(task)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(len(tasks_gpu))])
    if len(tasks_gpu) > 1:
        results1 = Parallel(n_jobs=len(tasks_gpu))(
            delayed(GPU_thread_parallel)(tasks_on_gpu, njobs) for tasks_on_gpu in tasks_gpu.values())

        results = [res for res_gpu in results1 for res in res_gpu]

    else:
        for gpu_name, tasks_on_gpu in tasks_gpu.items():
            results = GPU_thread(tasks_on_gpu, njobs)

    return results


def train_proba(tasks_3d, njobs):
    results = []
    tasks_gpu = dict()
    for task in tasks_3d:
        gpu_name = '/device:GPU:' + str(task['params']['gpu'])
        if not gpu_name in tasks_gpu:
            tasks_gpu[gpu_name] = []
        tasks_gpu[gpu_name].append(task)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(len(tasks_gpu))])
    if len(tasks_gpu) > 1:
        results1 = Parallel(n_jobs=len(tasks_gpu))(
            delayed(GPU_thread_proba_parallel)(tasks_on_gpu, njobs) for tasks_on_gpu in tasks_gpu.values())

        results = [res for res_gpu in results1 for res in res_gpu]

    else:
        for gpu_name, tasks_on_gpu in tasks_gpu.items():
            results = GPU_thread_proba(tasks_on_gpu, njobs)
    return results


def SKlearn_thread(project_id, static_data, cluster, method, optimize_method):
    sk_model = SKLearn_Manager(static_data, cluster, method, optimize_method)
    if sk_model.istrained == False:
        return sk_model.fit(), cluster.cluster_name, project_id
    else:
        return ['Done', cluster.cluster_name, project_id, method]


def Combine_thread(project_id, static_data, cluster):
    comb_cluster = ClusterCombiner(static_data, cluster)
    if comb_cluster.istrained == False:
        return comb_cluster.train(), cluster.cluster_name, project_id
    else:
        return ['Done', cluster.cluster_name, project_id]


def Combine_all_thread(project):
    comb_model = CombineModelManager(project)
    if comb_model.istrained == False:
        return comb_model.train(), project
    else:
        return ['Done', project]


def train_stage_combine(tasks_comb):
    results = []
    for task in tasks_comb:
        res = Combine_thread(task['project'], task['static_data'], task['cluster'])
        results.append(res)
    return results


def train_stage_modelcombine(tasks_modelcomb):
    results = []
    for task in tasks_modelcomb:
        res = Combine_all_thread(task['project'])
        results.append(res)
    return results


def train_on_cpus(projects, static_data, methods, path_group):
    ncpus = int(static_data['njobs'])
    joblib.dump(ncpus, os.path.join(path_group, 'total_cpus.pickle'))
    cpu_status = 0
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

    TasksCreator = TaskCreator(static_data)
    if static_data['sklearn']['fs_method'] == '':
        print('Feature selection is disabled')
    else:
        print('Feature selection starts')

    time.sleep(240)

    # if sys.platform != 'linux':
    #     mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    #     if mem<15:
    #         time.sleep(900)

    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

    # results_fs = []
    # for project in projects:
    #     for cluster_name, cluster in project.clusters.items():
    #         if not static_data['sklearn']['fs_method'] == '':
    #             cluster.static_data['njobs_feat_sel'] = njobs
    #             results_fs.append(FS_thread(project.static_data['_id'], cluster))
    # if len(results_fs) > 0:
    #     for res in results_fs:
    #         if res[0] not in {'Done'}:
    #             raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])
    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))
    # if sys.platform != 'linux':
    #     mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    #     if mem<15:
    #         time.sleep(900)
    result_rbfols = []
    if ('RBF_ALL_CNN' in methods) or ('RBF_ALL' in methods):
        print('Training RBFols starts')
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                print(cpu_status)
                print(gpu_status)
                print(cluster_name)
                cluster.static_data['sklearn']['njobs'] = njobs
                project.static_data['sklearn']['njobs'] = njobs
                result_rbfols.append(RBFOLS_thread(project.static_data['_id'], project.static_data, cluster))
    if len(result_rbfols) > 0:
        for res in result_rbfols:
            if res[0] not in {'Done'}:
                raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])

    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))
    # if sys.platform != 'linux':
    #     mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    #     if mem<15:
    #         time.sleep(900)
    result_sklearn = []
    sklearn_methods = [method for method in methods if method in {'SVM', 'NUSVM', 'MLP', 'RF', 'XGB', 'elasticnet'}]
    if len(sklearn_methods) > 0:
        print('Training SKlearn models starts')
        print(cpu_status)
        print(gpu_status)
        for project in projects:
            project.static_data['sklearn']['njobs'] = njobs
            for cluster_name, cluster in project.clusters.items():
                cluster.static_data['sklearn']['njobs'] = njobs
        tasks_sk = TasksCreator.create_tasks_stage_for_sklearn(projects, sklearn_methods)
        if len(tasks_sk) > 0:
            for task in tasks_sk:
                result_sklearn.append(
                    SKlearn_thread(task['project'], task['static_data'], task['cluster'], task['method'],
                                   task['optimize_method']))
        else:
            raise RuntimeError('Cannot create tasks for SKlearn models')
    if len(result_sklearn) > 0:
        for res in result_sklearn:
            if res[0] not in {'Done'}:
                raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])
    cpu_status = 0
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))


def train_on_gpus(projects, static_data, methods, path_group):
    TasksCreator = TaskCreator(static_data)
    ngpus = static_data['ngpus']
    ncpus = static_data['njobs']
    gpu_status = 0
    joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
    result_3d = []
    if ('LSTM' in methods) or ('CNN' in methods):
        if ('LSTM' in methods):
            njobs = static_data['LSTM']['njobs']
        else:
            njobs = static_data['CNN']['njobs_3d']

        gpu_status = int(ngpus * njobs)
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        while True:
            try:
                cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
                break
            except:
                time.sleep(30)
                continue

        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        print('3d models cnn or lstm 1st stage starts')
        tasks_3d_stage1 = TasksCreator.create_tasks_3d_stage1(projects)
        if len(tasks_3d_stage1) > 0:
            print(cpu_status)
            print(gpu_status)
            result_3d = train_3d(tasks_3d_stage1, njobs)
            if len(result_3d) > 0:
                print('3d models cnn or lstm 2nd stage starts')
                result_3d_pd = pd.DataFrame(result_3d,
                                            columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_3d_pd = result_3d_pd.iloc[
                    result_3d_pd.groupby(by=['method', 'project', 'cluster']).agg({'acc': 'idxmin'}).values.ravel()]

                tasks_3d_stage2 = TasksCreator.create_tasks_3d_stage2(result_3d_pd, tasks_3d_stage1)
                tasks_3d_stage1 += tasks_3d_stage2

                cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
                while (cpu_status + gpu_status) > ncpus:
                    time.sleep(30)
                    cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
                print(cpu_status)
                print(gpu_status)
                if len(tasks_3d_stage2) > 0:
                    result_3d += train_3d(tasks_3d_stage2, njobs)
                else:
                    raise RuntimeError('1st stage 3d Models cnn or lstm failed')
                result_3d_2nd_pd = pd.DataFrame(result_3d,
                                                columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_3d_2nd_pd.to_csv(os.path.join(path_group, 'results_3d_models_all.csv'))
                result_3d_2nd_pd = result_3d_2nd_pd.iloc[
                    result_3d_2nd_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                result_3d_2nd_pd.to_csv(os.path.join(path_group, 'results_3d_models_best.csv'))

                if 'CNN' in result_3d_2nd_pd['method'].to_list():
                    save_deep_models('CNN', result_3d_2nd_pd, projects)
                if 'LSTM' in result_3d_2nd_pd['method'].to_list():
                    save_deep_models('LSTM', result_3d_2nd_pd, projects)
                print('Training of Models 3d ends succesfully')
    gpu_status = 0
    joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
    while True:
        flag = 1
        results_fs = []
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                if not static_data['sklearn']['fs_method'] == '':
                    results_fs.append(FS_check(project.static_data['_id'], cluster))
        for res in results_fs:

            if res[0] not in {'Done'}:
                print(res)
                print('failed')
                flag *= 0
            else:
                print(res)
                print('succeed')
                flag *= 1
        print(flag)
        if flag == 0:
            time.sleep(300)
        else:
            break

    result_rbfnn = []
    if ('RBF_ALL_CNN' in methods) or ('RBF_ALL' in methods):
        njobs = static_data['RBF']['njobs']
        gpu_status = ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        task_rbf_stage1 = TasksCreator.create_tasks_stage_for_rbfs(projects)
        print('Training RBFNN 1st stage starts')
        print(ngpus)
        print(ncpus)
        print(cpu_status)
        print(gpu_status)
        if len(task_rbf_stage1) > 0:
            result_rbfnn = train_3d(task_rbf_stage1, njobs)
            if len(result_rbfnn) > 0:
                if static_data['RBF']['Fine_tuning']:
                    print('Train RBFNN Fine tunining stage')
                    result_rbfnn_pd = pd.DataFrame(result_rbfnn,
                                                   columns=['acc', 'cluster', 'project', 'test', 'method'])
                    result_rbfnn_pd = result_rbfnn_pd.iloc[
                        result_rbfnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                            {'acc': 'idxmin'}).values.ravel()]

                    tasks_rbf_stage2 = TasksCreator.create_tasks_stage_rbf_ft(result_rbfnn_pd, task_rbf_stage1)
                    task_rbf_stage1 += tasks_rbf_stage2
                    if len(tasks_rbf_stage2):
                        result_rbfnn += train_3d(tasks_rbf_stage2, njobs)
                    else:
                        raise RuntimeError('1st stage 3d Models cnn or lstm failed')
                print('Train RBFNN Final stage')
                result_rbfnn_pd = pd.DataFrame(result_rbfnn,
                                               columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_rbfnn_pd = result_rbfnn_pd.iloc[
                    result_rbfnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                print('Train RBFNN 3rd stage')
                task_rbf_stage_3 = TasksCreator.create_tasks_stage_rbf_lr(result_rbfnn_pd, task_rbf_stage1)
                task_rbf_stage1 += task_rbf_stage_3
                print(cpu_status)
                print(gpu_status)
                if len(task_rbf_stage_3):
                    result_rbfnn += train_3d(task_rbf_stage_3, njobs)
                else:
                    raise RuntimeError('1st stage 3d Models cnn or lstm failed')
                result_rbfnn_pd = pd.DataFrame(result_rbfnn,
                                               columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_rbfnn_pd.to_csv(os.path.join(path_group, 'results_RBF_models_all.csv'))
                result_rbfnn_pd = result_rbfnn_pd.iloc[
                    result_rbfnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                result_rbfnn_pd.to_csv(os.path.join(path_group, 'results_RBF_models_best.csv'))
                save_deep_models('RBFNN', result_rbfnn_pd, projects)
                print('Training of RBFNNs ends succesfully')
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        while True:
            flag = 1
            results_rbf = []
            for project in projects:
                for cluster_name, cluster in project.clusters.items():
                    results_rbf.append(RBF_check(project.static_data['_id'], project.static_data, cluster))
            for res in results_rbf:

                if res[0] not in {'Done'}:
                    print(res)
                    print('failed')
                    flag *= 0
                else:
                    print(res)
                    print('succeed')
                    flag *= 1
            print(flag)
            if flag == 0:
                time.sleep(900)
            else:
                break
    result_rbf_cnn = []
    cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
    for project in projects:
        project.static_data['cpu_status'] = cpu_status
    if ('RBF_ALL_CNN' in methods):
        print('Train RBF-CNN 1st stage')
        njobs = static_data['CNN']['njobs']
        gpu_status = 4 * ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        print(cpu_status)
        print(gpu_status)
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        tasks_rbfcnn_stage1 = TasksCreator.create_tasks_rbfcnn_stage1(projects)
        if len(tasks_rbfcnn_stage1) > 0:
            result_rbf_cnn = train_3d(tasks_rbfcnn_stage1, njobs)
            if len(result_rbfnn) > 0:
                print('Train RBF-CNN 2nd stage')
                result_rbf_cnn_pd = pd.DataFrame(result_rbf_cnn,
                                                 columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_rbf_cnn_pd = result_rbf_cnn_pd.iloc[
                    result_rbf_cnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                tasks_rbfcnn_stage2 = TasksCreator.create_tasks_rbfcnn_stage2(result_rbf_cnn_pd,
                                                                              tasks_rbfcnn_stage1)
                tasks_rbfcnn_stage1 += tasks_rbfcnn_stage2
                if len(tasks_rbfcnn_stage2) > 0:
                    result_rbf_cnn += train_3d(tasks_rbfcnn_stage2, njobs)
                else:
                    raise RuntimeError('Cannot create tasks for RBF-CNN models')

                result_rbf_cnn_pd = pd.DataFrame(result_rbf_cnn,
                                                 columns=['acc', 'cluster', 'project', 'test', 'method'])
                result_rbf_cnn_pd.to_csv(os.path.join(path_group, 'results_RBF_CNN_models_all.csv'))
                result_rbf_cnn_pd = result_rbf_cnn_pd.iloc[
                    result_rbf_cnn_pd.groupby(by=['method', 'project', 'cluster']).agg(
                        {'acc': 'idxmin'}).values.ravel()]
                result_rbf_cnn_pd.to_csv(os.path.join(path_group, 'results_RBF_CNN_models_best.csv'))
                save_deep_models('RBF-CNN', result_rbf_cnn_pd, projects)
                print('Training of RBF-CNNs ends succesfully')

    result_mlp_3d = []
    cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
    for project in projects:
        project.static_data['cpu_status'] = cpu_status
    if ('MLP_3D' in methods):
        print('Training MLP_3D 1st stage starts')
        njobs = static_data['MLP']['njobs']
        gpu_status = 4 * ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        tasks_mlp = TasksCreator.create_tasks_MLP_stage1(projects)
        if len(tasks_mlp) > 0:
            result_mlp_3d = train_3d(tasks_mlp, njobs)
        else:
            raise RuntimeError('Cannot create tasks for MLP_3d models')
        if len(result_mlp_3d) > 0:
            result_mlp_3d_pd = pd.DataFrame(result_mlp_3d,
                                            columns=['acc', 'cluster', 'project', 'test', 'method'])
            result_mlp_3d_pd.to_csv(os.path.join(path_group, 'results_MLP_models_all.csv'))
            result_mlp_3d_pd = result_mlp_3d_pd.iloc[
                result_mlp_3d_pd.groupby(by=['method', 'project', 'cluster']).agg(
                    {'acc': 'idxmin'}).values.ravel()]
            result_mlp_3d_pd.to_csv(os.path.join(path_group, 'results_MLP_models_best.csv'))

            save_deep_models('MLP', result_mlp_3d_pd, projects)
            print('Training MLP_3D ends succesfully')

    gpu_status = 0
    joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

    print(ngpus)
    print(ncpus)
    print(cpu_status)
    print(gpu_status)


def save_deep_models(method, results, projects):
    for project in projects:
        for cluster_name, cluster in project.clusters.items():
            test = results['test'].iloc[np.where((method == results['method']).any() and (
                    project.static_data['_id'] == results['project']).any()
                                                 and (cluster_name == results['cluster']).any())[0]].values[0]
            model = model3d_manager(project.static_data, cluster, method, {'test': test})
            for filename in glob.glob(os.path.join(model.test_dir, '*.*')):
                print(filename)
                shutil.copy(filename, model.model_dir)
