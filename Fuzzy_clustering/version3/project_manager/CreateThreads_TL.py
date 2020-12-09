import os
from joblib import Parallel, delayed
from Fuzzy_clustering.version3.project_manager.CreateTasks_TL import TaskCreator_TL
from Fuzzy_clustering.version3.project_manager.FS_object import FeatSelobject
from Fuzzy_clustering.version3.project_manager.Model_3d_object import model3d_object
from Fuzzy_clustering.version3.project_manager.rbf_ols_manager.rbf_ols_manager import RBFOLS_Manager
from Fuzzy_clustering.version3.project_manager.sklearn_models.sklearn_manager import SKLearn_Manager

def FS_TL_thread(project_id, cluster):
    FS_model = FeatSelManager(cluster)
    if FS_model.istrained==False:
        return FS_model.fit_TL(), cluster.cluster_name, project_id
    else:
        return 'Done', cluster.cluster_name, project_id

def Model3dThread_TL(project_id, static_data, cluster, method, params):
    model = model3d_manager(static_data, cluster, method, params)
    if model.istrained==False:
        acc = model.fit_TL()
    else:
        acc = model.acc

    return acc, cluster.cluster_name, project_id, params['test'], method

def RBFOLS_thread_TL(project_id, static_data, cluster):
    rbf_model = RBFOLS_Manager(static_data, cluster)
    if rbf_model.istrained==False:
        return rbf_model.fit_TL(), cluster.cluster_name, project_id
    else:
        return ['Done', cluster.cluster_name, project_id]

def train_stage_TL_fs(tasks_fs):
    results = []
    for task in tasks_fs:
        results.append(FS_TL_thread(task['project'], task['cluster']))
    return results

def GPU_thread_TL(tasks,njobs):
    results = Parallel(n_jobs=njobs)(delayed(Model3dThread_TL)(task['project'], task['static_data'],
                                                                  task['cluster'], task['method'], task['params']) for
                                                                   task in tasks)
    return results
def thread_TL_stage1(proc, tasks):
    if proc == 0:
        results = train_stage_TL_fs(tasks[0])
    else:
        results = train_3d_TL(tasks[1])
    return results

def train_TL_stage1(tasks):
    results = Parallel(n_jobs=2)(delayed(thread_TL_stage1)(proc, tasks) for proc in range(2))

    return results

def train_3d_TL(tasks_3d):
    tasks = dict()
    njobs = 0
    for ntask in tasks_3d.keys():
        task = tasks_3d[ntask]
        for gpu in task.keys():
            if gpu not in tasks.keys():
                tasks[gpu] = []
            if len(task[gpu])>0:
                tasks[gpu] += task[gpu]
            if njobs == 0:
                njobs = len(task[gpu])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(n) for n in range(len(tasks))])
    results = Parallel(n_jobs=len(tasks))(delayed(GPU_thread_TL)(task, njobs) for i, task in tasks.items())
    return results

def train_stage_rbfols_TL(tasks_ols, njobs):

    results = Parallel(n_jobs=njobs)(delayed(RBFOLS_thread_TL)(task['project'], task['static_data'], task['cluster']) for task in tasks_ols)

    return results

def SKlearn_thread_TL(project_id, static_data, cluster, method, optimize_method):
    sk_model = SKLearn_Manager(static_data, cluster, method, optimize_method)
    if sk_model.istrained == False:
        return sk_model.fit_TL(), cluster.cluster_name, project_id
    else:
        return ['Done', cluster.cluster_name, project_id, method]

def train_stage_sklearn_TL(tasks_sk, njobs):

    results = Parallel(n_jobs=njobs)(
        delayed(SKlearn_thread_TL)(task['project'], task['static_data'], task['cluster'], task['method'], task['optimize_method']) for task in tasks_sk)

    return results
