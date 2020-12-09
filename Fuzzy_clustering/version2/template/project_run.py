from Fuzzy_clustering.version2.project_manager.project_eval_manager import ProjectsEvalManager
from Fuzzy_clustering.version2.project_manager.projects_data_manager import ProjectsDataManager
from Fuzzy_clustering.version2.project_manager.projects_train_manager import ProjectsTrainManager
from Fuzzy_clustering.version2.template.util_database import write_database


def prepare_data():
    static_data = write_database()

    project_data_manager = ProjectsDataManager(static_data)

    nwp_response = project_data_manager.nwp_extractor(test=False)
    if nwp_response == 'Done':
        data_response = project_data_manager.create_datasets(test=False)
    else:
        raise RuntimeError('Something was going wrong with NWP extractor')

    if data_response == 'Done':
        project_data_manager.create_projects_relations()
    else:
        raise RuntimeError('Something was going wrong with data manager')

    if hasattr(project_data_manager, 'data_eval'):
        nwp_response = project_data_manager.nwp_extractor(test=True)
        if nwp_response == 'Done':
            _ = project_data_manager.create_datasets(test=True)
        else:
            raise RuntimeError('Something was going wrong with NWP extractor on evaluation')


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