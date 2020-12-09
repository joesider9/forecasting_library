import pika, uuid, time, json, os
import numpy as np
from Fuzzy_clustering.version3.project_manager.projects_data_manager import ProjectsDataManager
from Fuzzy_clustering.version3.project_manager.Projects_Train_Manager import ProjectsTrainManager
from Fuzzy_clustering.version3.project_manager.Project_Eval_Manager import ProjectsEvalManager
from Fuzzy_clustering.version3.rabbitmq_rpc.server import RPCServer

RABBIT_MQ_HOST = os.getenv('RABBIT_MQ_HOST')
RABBIT_MQ_PASS = os.getenv('RABBIT_MQ_PASS')
RABBIT_MQ_PORT = int(os.getenv('RABBIT_MQ_PORT'))

server = RPCServer(queue_name='manager', host=RABBIT_MQ_HOST, port=RABBIT_MQ_PORT, threaded=False)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer) or isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.str) or isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, bool):
            return bool(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            print(obj)
            raise TypeError('Object is not JSON serializable')

@server.consumer()
def manager(static_data):
    prepare_data(static_data)
    train_project(static_data)
    eval_project(static_data)
    backup_project(static_data)
    return 'Done'
def prepare_data(static_data):

    project_data_manager = ProjectsDataManager(static_data)
    nwp_response = project_data_manager.nwp_extractor()

    data_response = project_data_manager.create_datasets(test=False)

    TL_response = project_data_manager.create_projects_relations()

    if hasattr(project_data_manager.projects, 'data_eval'):
        nwp_response_test = project_data_manager.nwp_extractor(test=True)
        data_responsetest = project_data_manager.create_datasets(test=True)



def train_project(static_data):
    project_train_manager = ProjectsTrainManager(static_data)
    project_train_manager.fit()

def eval_project(static_data):
    project_eval_manager = ProjectsEvalManager(static_data)
    project_eval_manager.evaluate_all()

def backup_project(static_data):
    project_backup_manager = ProjectsTrainManager(static_data)
    project_backup_manager.clear_backup_projects()




if __name__=='__main__':
    server.run()
