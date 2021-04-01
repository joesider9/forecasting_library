from Fuzzy_clustering.version2.project_managers.project_eval_manager import ProjectsEvalManager
from Fuzzy_clustering.version2.project_managers.projects_data_manager import ProjectsDataManager
from Fuzzy_clustering.version2.project_managers.projects_train_manager import ProjectsTrainManager
from Fuzzy_clustering.version2.template.constants import *
from Fuzzy_clustering.version2.template.util_database_timos import write_database


def prepare_data():
    static_data = write_database()
    project_data_manager = ProjectsDataManager(static_data, is_test=False)

    nwp_response = project_data_manager.nwp_extractor()
    if nwp_response == DONE:
        data_response = project_data_manager.create_datasets()
    else:
        raise RuntimeError('Something was going wrong with nwp extractor')

    if data_response == DONE:
        project_data_manager.create_projects_relations()
    else:
        raise RuntimeError('Something was going wrong with data manager')

    if hasattr(project_data_manager, 'data_eval'):
        project_data_manager.is_test = True
        nwp_response = project_data_manager.nwp_extractor()
        if nwp_response == DONE:
            nwp_response = project_data_manager.create_datasets()
            if nwp_response != DONE:
                raise RuntimeError('Something was going wrong with on evaluation dataset creator')
        else:
            raise RuntimeError('Something was going wrong with nwp extractor on evaluation')
    print("Data is prepared, training can start")


def train_project():
    static_data = write_database()
    project_train_manager = ProjectsTrainManager(static_data)
    project_train_manager.fit()


def eval_project():
    static_data = write_database()
    project_eval_manager = ProjectsEvalManager(static_data)
    project_eval_manager.evaluate()


def backup_project():
    static_data = write_database()

    project_backup_manager = ProjectsTrainManager(static_data)
    project_backup_manager.clear_backup_projects()


if __name__ == '__main__':
    prepare_data()
    train_project()
    eval_project()
    backup_project()
