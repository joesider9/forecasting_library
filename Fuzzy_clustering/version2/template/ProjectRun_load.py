from Fuzzy_clustering.version2.project_manager.project_eval_manager import ProjectsEvalManager
from Fuzzy_clustering.version2.project_manager.projects_data_manager import ProjectsDataManager
from Fuzzy_clustering.version2.project_manager.projects_train_manager import ProjectsTrainManager
from util_database_load import write_database
import sys, joblib

def prepare_data():
    static_data = write_database()

    project_data_manager = ProjectsDataManager(static_data, is_test=False)

    nwp_response = project_data_manager.nwp_extractor()
    if nwp_response == 'Done':
        data_response = project_data_manager.create_datasets()
    else:
        raise RuntimeError('Something was going wrong with NWP extractor')

    if data_response == 'Done':
        project_data_manager.create_projects_relations()
    else:
        raise RuntimeError('Something was going wrong with data manager')

    if hasattr(project_data_manager, 'data_eval'):
        project_data_manager = ProjectsDataManager(static_data, is_test=True)
        nwp_response = project_data_manager.nwp_extractor()
        if nwp_response == 'Done':
            _ = project_data_manager.create_datasets()
        else:
            raise RuntimeError('Something was going wrong with NWP extractor on evaluation')


def train_project():
    static_data = write_database()

    project_train_manager = ProjectsTrainManager(static_data)
    # for _ in range(3):
        # try:
    project_train_manager.fit()
        # except:
        #     e = sys.exc_info()
        #     print(e[0])
        #     joblib.dump(e, 'error.pickle')
        #     continue

def eval_short_term_project():
    static_data = write_database()

    project_eval_manager = ProjectsEvalManager(static_data)
    project_eval_manager.eval_short_term(horizon = 4, best_method='bcp_average')

def eval_project():
    static_data = write_database()

    project_eval_manager = ProjectsEvalManager(static_data)
    project_eval_manager.evaluate()


def backup_project():
    static_data = write_database()

    project_backup_manager = ProjectsTrainManager(static_data)
    project_backup_manager.clear_backup_projects()


if __name__ == '__main__':
    # prepare_data()
    # train_project()
    # eval_project()
    eval_short_term_project()
    backup_project()