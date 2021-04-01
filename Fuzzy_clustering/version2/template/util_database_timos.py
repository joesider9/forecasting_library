import os

from Fuzzy_clustering.version2.template.constants import *
from Fuzzy_clustering.version2.template.default_project import get_fuzzy_features
from Fuzzy_clustering.version2.template.project_config import projects

project = 'Petalas'


def write_database():
    project_config = projects[project]
    file_name = os.path.basename(project_config[DATA_FILE_NAME])

    # dictionary containing details about the fuzzy clusters to be created
    fuzzy_dict = get_fuzzy_features(file_name)
    project_config.update(fuzzy_dict)

    # dictionary containing all information necessary for different model training
    model_details = {
        SKLEARN: {N_JOBS: project_config[N_JOBS_SKLEARN],
                  FS_STATUS: project_config[FS_STATUS],
                  FEATURE_SELECTION_METHOD: project_config[FEATURE_SELECTION_METHOD],
                  SKLEARN_OPTIMIZER: project_config[SKLEARN_OPTIMIZER],
                  DIMENSIONALITY_REDUCTION: project_config[DIMENSIONALITY_REDUCTION],
                  POP: project_config[POP_SK],
                  GEN: project_config[GEN_SK],
                  },
        CLUSTERING: {N_JOBS: project_config[N_JOBS_FUZZY],
                     IS_FUZZY: project_config[IS_FUZZY],
                     N_CLUSTERS: project_config[N_CLUSTERS],
                     THRESH_ACT: project_config[THRESH_ACT],
                     THRESH_SPLIT: project_config[THRESH_SPLIT],
                     TRAIN_ONLINE: project_config[TRAIN_ONLINE],
                     IS_CLUSTERING_TRAINED: project_config[IS_CLUSTERING_TRAINED],
                     VAR_IMP: project_config[VAR_IMP],
                     VAR_LIN: project_config[VAR_LIN],
                     VAR_NON_REG: project_config[VAR_NON_REG],
                     CLUSTER_FILE: project_config[CLUSTER_FILE],
                     ADD_RULES_INDIVIDUAL: project_config[ADD_RULES_INDIVIDUAL],
                     IMPORT_EXTERNAL_RULES: project_config[IMPORT_EXTERNAL_RULES],
                     POP: project_config[POP_FZ],
                     GEN: project_config[GEN_FZ],
                     },
        RBF: {MAX_ITERATIONS: project_config[MAX_ITERATIONS],
              LEARNING_RATE: project_config[LEARNING_RATE],
              MEAN_VAR: project_config[MEAN_VAR],
              STD_VAR: project_config[STD_VAR],
              N_JOBS: project_config[N_JOBS_RBF_NN],
              FINE_TUNING: project_config[FINE_TUNING],
              POP: project_config[POP_RBF],
              GEN: project_config[GEN_RBF],
              },
        CNN: {
            FILTERS: project_config[FILTERS],
            POOL_SIZE: project_config[POOL_SIZE],
            H_SIZE: project_config[H_SIZE],
            MAX_ITERATIONS: project_config[CNN_MAX_ITERATIONS],
            LEARNING_RATE: project_config[CNN_LEARNING_RATE],
            N_JOBS_CNN_3D: project_config[N_JOBS_CNN_3D],
            N_JOBS: project_config[N_JOBS_CNN],
        },
        LSTM: {
            UNITS: project_config[UNITS],
            HOLD_PROB: project_config[HOLD_PROB],
            MAX_ITERATIONS: project_config[LSTM_MAX_ITERATIONS],
            LEARNING_RATE: project_config[LSTM_LEARNING_RATE],
            N_JOBS: project_config[N_JOBS_LSTM],
        },
        MLP: {
            HOLD_PROB: project_config[HOLD_PROB],
            MAX_ITERATIONS: project_config[MLP_MAX_ITERATIONS],
            LEARNING_RATE: project_config[MLP_LEARNING_RATE],
            N_JOBS: project_config[N_JOBS_MLP],
        }}
    project_config.update(model_details)

    # dictionary containing the nwp, the input data and save paths
    path_dict = define_folder_names(project_config)
    project_config.update(path_dict)

    return project_config


def define_folder_names(project_config):
    model_type = project_config[MODEL_TYPE]
    sys_folder = project_config[SYS_FOLDER]

    if project_config[NWP_MODEL] == SKIRON:
        if project_config[NWP_RESOLUTION] == 0.05:
            path_nwp = os.path.join(project_config[NWP_FOLDER], SKIRON)
        elif project_config[NWP_RESOLUTION] == 0.1:
            path_nwp = os.path.join(project_config[NWP_FOLDER], SKIRON_LOW)
        else:
            raise ValueError("Wrong spatial resolution")
    elif project_config[NWP_MODEL] == ECMWF:
        path_nwp = os.path.join(project_config[NWP_FOLDER], ECMWF)
    elif project_config[NWP_MODEL] == GFS:
        path_nwp = os.path.join(project_config[NWP_FOLDER], GFS)
    else:
        path_nwp = None

    path_prefix = f'{sys_folder}{project_config[PROJECT_OWNER]}/{project_config[PROJECTS_GROUP]}_ver{project_config[VERSION_GROUP]}'

    path_group = f'{path_prefix}/{model_type}'
    if not os.path.exists(path_group):
        os.makedirs(path_group)

    path_nwp_group = f'{path_prefix}/{NWP}'
    if not os.path.exists(path_nwp_group):
        os.makedirs(path_nwp_group)

    paths = {
        SYS_FOLDER: sys_folder,
        PATH_NWP: path_nwp,
        PATH_GROUP: path_group,
        PATH_NWP_GROUP: path_nwp_group,
    }
    return paths
