from contextlib import contextmanager
from timeit import default_timer  # for timing

from Fuzzy_clustering.version2.common_utils.logging import create_logger
from Fuzzy_clustering.version2.dataset_manager.dataset_creator import DatasetCreator
from Fuzzy_clustering.version2.nwp_manager.nwp_extractor import NwpExtractor
from Fuzzy_clustering.version2.project_manager.correlation_link_projects import ProjectLinker
from Fuzzy_clustering.version2.project_manager.project_group_init import ProjectGroupInit


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


class ProjectsDataManager(object):

    def __init__(self, static_data, is_test: bool):
        self.is_test = is_test
        self.static_data = static_data
        self.projects = ProjectGroupInit(self.static_data)
        self.projects.initialize()
        self.model_type = self.projects.model_type
        self.path_group = self.projects.path_group
        self.projects_group = self.projects.projects_group
        self.data = self.projects.data
        if not hasattr(self, 'data_eval') and hasattr(self.projects, 'data_eval'):
            self.data_eval = self.projects.data_eval
        self.data_variables = self.projects.data_variables
        # List of dicts. Each dict has information about a different project
        self.group_static_data = self.projects.group_static_data

        self.logger = create_logger(logger_name=f'ProjectsDataManager_{self.model_type}', write_type='a',
                                    abs_path=self.path_group, logger_path=f'log_{self.projects_group}.log')

    def nwp_extractor(self):
        nwp_extractor = NwpExtractor(self.static_data)
        return nwp_extractor.extract(self.data_eval) if self.is_test else nwp_extractor.extract(self.data)

    def create_datasets(self):
        dataset_creator = DatasetCreator(self.static_data, self.group_static_data, self.is_test)
        data_input = self.projects.data_eval if self.is_test else self.data
        return dataset_creator.create_datasets(data_input)

    def create_short_term_datasets(self, data_input):
        dataset_creator = DatasetCreator(self.static_data, self.group_static_data, None)
        return dataset_creator.create_datasets(data_input)

    def create_projects_relations(self):

        if self.static_data['enable_transfer_learning'] and len(self.group_static_data) > 1:
            self.logger.info('Create projects relations')
            transfer_learning_linker = ProjectLinker(self.group_static_data)
            self.training_project_groups = transfer_learning_linker.find_relations()

            projects = dict()
            for project in self.group_static_data:
                if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                    projects[project['_id']] = project

            count_projects = 0
            for project_name, project in projects.items():
                if project_name not in self.training_project_groups.keys():
                    for main_project, group in self.training_project_groups.items():
                        if project_name in group:
                            project['static_data']['transfer_learning'] = True
                            project['static_data']['tl_project'] = projects[main_project]
                            count_projects += 1
                else:
                    count_projects += 1

            if len(projects) != count_projects:
                raise RuntimeError('Some projects does not include in a transfer learning project group')

            for project in self.group_static_data:
                if project['_id'] != project['static_data']['projects_group'] + '_' + project['static_data']['type']:
                    project['static_data']['transfer_learning'] = projects[project['_id']]['static_data'][
                        'transfer_learning']
                    project['static_data']['tl_project'] = projects[project['_id']]['static_data']['tl_project']
