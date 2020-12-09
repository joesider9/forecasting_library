import os, glob, shutil, joblib, time, sys, psutil,copy
from joblib import Parallel, delayed
import multiprocessing as mp
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
from rabbitmq_rpc.client import RPCClient
from Fuzzy_clustering.version3.project_manager.CreateTasks import TaskCreator
from Fuzzy_clustering.version3.project_manager.CreateTasks_TL import TaskCreator_TL
from Fuzzy_clustering.version3.project_manager.Model_3d_object import model3d_object
from Fuzzy_clustering.version3.project_manager.FS_object import FeatSelobject
from Fuzzy_clustering.version3.project_manager.RBF_ols_object import RBFOLS_manager_object


RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))



def FS_check(project_id, cluster):
    FS_model = FeatSelobject(cluster)
    if FS_model.istrained==False:
        return 'Untrained', cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id

def RBF_check(project_id, static_data, cluster):
    rbf_model = RBFOLS_manager_object(static_data, cluster)
    if rbf_model.istrained == False:
        return 'Untrained', cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id

def train_3d(client, tasks_3d):
    results = []
    for task in tasks_3d:
        static_data = copy.deepcopy(task['static_data'])
        static_data['cluster_name'] = task['cluster']
        static_data['params'] = task['params']
        static_data['method'] = task['method']
        results.append(client.call_deep_manager(static_data))
    return results

def train_proba(client, tasks_3d):
    results = []
    for task in tasks_3d:
        static_data = copy.deepcopy(task['static_data'])
        static_data['params'] = task['params']
        static_data['method'] = task['method']
        results.append(client.call_proba_manager(static_data))
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

    time.sleep(900)
    # if sys.platform != 'linux':
    #     mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    #     if mem<10:
    #         time.sleep(900)

    gpu_status=joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

    results_fs = []
    client_fs = RPCClient(queue_name='FeatSelmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)
    for project in projects:
        for cluster_name, cluster in project.clusters.items():
            if not static_data['sklearn']['fs_method'] == '':
                project.static_data['njobs_feat_sel'] = njobs
                static_data = copy.deepcopy(project.static_data)
                static_data['cluster_name'] = cluster.cluster_name
                results_fs.append(client_fs.call_FeatSelmanager(static_data))

    if len(results_fs) > 0:
        for res in results_fs:
            if res[0] not in {'Done'}:
                raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])

    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status=joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))
    # if sys.platform != 'linux':
    #     mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    #     if mem<10:
    #         time.sleep(900)
    result_rbfols = []
    if ('RBF_ALL_CNN' in methods) or ('RBF_ALL' in methods):
        print('Training RBFols starts')
        client_rbfols = RPCClient(queue_name='RBFOLSmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)
        for project in projects:
            for cluster_name, cluster in project.clusters.items():
                project.static_data['sklearn']['njobs'] = njobs
                static_data = copy.deepcopy(project.static_data)
                static_data['cluster_name'] = cluster.cluster_name
                result_rbfols.append(client_rbfols.call_RBFOLSmanager(static_data))

    if len(result_rbfols) > 0:
        for res in result_rbfols:
            if res[0] not in {'Done'}:
                raise RuntimeError('Feature selection fails cluster %s of project %s', res[1], res[2])

    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

    result_sklearn = []
    sklearn_methods = [method for method in methods if method in {'SVM', 'NUSVM', 'MLP', 'RF', 'XGB'}]
    if len(sklearn_methods) > 0:
        print('Training SKlearn models starts')
        for project in projects:
            project.static_data['sklearn']['njobs'] = njobs
            for cluster_name, cluster in project.clusters.items():
                cluster.static_data['sklearn']['njobs'] = njobs
        tasks_sk = TasksCreator.create_tasks_stage_for_sklearn(projects, sklearn_methods)
        if len(tasks_sk) > 0:
            client_sk = RPCClient(queue_name='SKlearnmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)
            for task in tasks_sk:
                static_data = copy.deepcopy(task['static_data'])
                static_data['cluster_name'] = task['cluster']
                static_data['optimize_method'] = task['optimize_method']
                static_data['method'] = task['method']
                result_sklearn.append(client_sk.call_SKlearnmanager(static_data))
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
    intra_op = static_data['intra_op']
    gpu_status = 0
    joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
    result_3d = []
    if ('LSTM' in methods) or ('CNN' in methods):
        if ('LSTM' in methods):
            njobs = static_data['LSTM']['njobs']
            client_3d = RPCClient(queue_name='LSTMmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)
        else:
            njobs = static_data['CNN']['njobs_3d']
            client_3d = RPCClient(queue_name='CNNmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)

        gpu_status = intra_op * ngpus*njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        while True:
            try:
                cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
                break
            except:
                time.sleep(30)
                continue

        while (cpu_status+gpu_status)>ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))

        print('3d models cnn or lstm 1st stage starts')
        tasks_3d_stage1 = TasksCreator.create_tasks_3d_stage1(projects)
        if len(tasks_3d_stage1) > 0:

            result_3d = train_3d(client_3d, tasks_3d_stage1)

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

                if len(tasks_3d_stage2) > 0:
                    result_3d += train_3d(client_3d, tasks_3d_stage2)
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
        if flag==0:
            time.sleep(300)
        else:
            break

    result_rbfnn = []
    if ('RBF_ALL_CNN' in methods) or ('RBF_ALL' in methods):
        client_3d = RPCClient(queue_name='RBFNNmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)
        njobs = static_data['RBF']['njobs']
        gpu_status = ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        task_rbf_stage1 = TasksCreator.create_tasks_stage_for_rbfs(projects)
        print('Training RBFNN 1st stage starts')
        if len(task_rbf_stage1) > 0:

            result_rbfnn = train_3d(client_3d, task_rbf_stage1)
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
                        result_rbfnn += train_3d(client_3d, tasks_rbf_stage2)
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
                if len(task_rbf_stage_3):
                    result_rbfnn += train_3d(client_3d, task_rbf_stage_3)
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
            if flag == 0:
                time.sleep(900)
            else:
                break
    result_rbf_cnn = []
    cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
    for project in projects:
        project.static_data['cpu_status'] = cpu_status
    if ('RBF_ALL_CNN' in methods):
        client_3d = RPCClient(queue_name='RBF_CNN_manager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)
        print('Train RBF-CNN 1st stage')
        njobs = static_data['CNN']['njobs']
        gpu_status =  intra_op * ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        tasks_rbfcnn_stage1 = TasksCreator.create_tasks_rbfcnn_stage1(projects)
        if len(tasks_rbfcnn_stage1) > 0:
            result_rbf_cnn = train_3d(client_3d, tasks_rbfcnn_stage1)
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
                    result_rbf_cnn += train_3d(client_3d, tasks_rbfcnn_stage2)
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
        client_3d = RPCClient(queue_name='MLPmanager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT,
                                      threaded=False)
        print('Training MLP_3D 1st stage starts')
        njobs = static_data['MLP']['njobs']
        gpu_status =  intra_op * ngpus * njobs
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
        cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        while (cpu_status + gpu_status) > ncpus:
            time.sleep(30)
            cpu_status = joblib.load(os.path.join(path_group, 'cpu_status.pickle'))
        tasks_mlp = TasksCreator.create_tasks_MLP_stage1(projects)
        if len(tasks_mlp) > 0:
            result_mlp_3d = train_3d(client_3d, tasks_mlp)
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


def save_deep_models(method, results, projects):
    for project in projects:
        for cluster_name, cluster in project.clusters.items():
            test = results['test'].iloc[np.where((method == results['method']).any() and (
                    project.static_data['_id'] == results['project']).any()
                                                 and (cluster_name == results['cluster']).any())[0]].values[0]
            model = model3d_object(project.static_data, cluster, method, {'test': test})
            for filename in glob.glob(os.path.join(model.test_dir, '*.*')):
                print(filename)
                shutil.copy(filename, model.model_dir)