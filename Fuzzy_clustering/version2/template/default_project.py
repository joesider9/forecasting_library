from Fuzzy_clustering.version2.template.constants import *

"""
Define a set of default parameters for each project. These values can be overridden 
by new values. If no new value, is provided the following will be used. 
"""
default_project_values = {
    DOCKER: False,
    COMPRESS_DATA: DENSE,
    VERSION_GROUP: 0,
    VERSION_MODEL: 0,
    N_JOBS: 2,
    N_JOBS_FUZZY: 2,
    N_JOBS_FEAT_SEL: 2,
    N_JOBS_RBF_OLS: 2,
    N_JOBS_RBF_NN: 2,
    N_JOBS_SKLEARN: 2,
    N_JOBS_LSTM: 2,
    N_JOBS_CNN_3D: 2,
    N_JOBS_CNN: 1,
    N_JOBS_MLP: 2,
    INNER_JOBS_FEAT_SEL: 1,
    INTRA_OP: 2,
    N_GPUS: 1,
    TRAIN_SPLIT: 0.7,
    TRANSFER_LEARNING: False,
    # If true then EvolutionaryAlgorithmSearchCV() will be run to select best hyperparameters
    HYPER_PARAMETER_TUNING: False,
    RESAMPLING: True,  # ONLY FOR THE COMBINATION
    TL_PROJECT: '',
    EVALUATION: None,
    FEATURE_SELECTION_METHOD: LINEAR_SEARCH,
    DIMENSIONALITY_REDUCTION: False,
    COMBINE_METHODS: [RLS, BCP, MLP, BAYESIAN_RIDGE, ELASTIC_NET, RIDGE, AVERAGE],
    EXCLUDE_METHOD_COMBINE: [],
    IS_GLOBAL: False,
    IS_FUZZY: True,
    CLUSTER_FILE: BEST_FUZZY_PICKLE,
    IS_CLUSTERING_TRAINED: False,
    IS_PROBABILISTIC: False,
    PROJECT_METHODS: {RBF_ALL_CNN: False, RBF_ALL: False, CNN: False, LSTM: False, MLP_3D: False,
                      SVM: False, NU_SVM: False, MLP: False, RF: False, XGB: False, ELASTIC_NET: False},
    SKLEARN_OPTIMIZER: DEAP,
    RECREATE_DATASETS: False,
    RECREATE_NWP_FILES: False,
    WEATHER_IN_DATA: False,
    ADD_RULES_INDIVIDUAL: False,
    ENABLE_TRANSFER_LEARNING: False,
    FS_STATUS: NOTOK,
    RESAMPLING_THRESH: 200,  # TODO: What is resampling thresh???
    ΝUM_GEN: 300,
    IMPORT_EXTERNAL_RULES: [],
    N_CLUSTERS: 200,
    RESAMPLING_ON_VAR: [[CLOUD, 0], [FLUX, 1], [TEMPERATURE, 4], [WIND, 5]],
    CHECK_FUZZY_MODELS: False,

    TRAIN_ONLINE: False,
    GEN_FZ: 20,
    POP_FZ: 50,
    GEN_SK: 20,
    POP_SK: 50,
    THRESH_SPLIT: 0.75,
    THRESH_ACT: 0.01,
    DAYS_OFFSET: 372,

    # RBF NN
    GEN_RBF: 100,
    POP_RBF: 50,
    POPULATION: 50,
    MAX_ITERATIONS: 60000,
    LEARNING_RATE: 0.5e-4,
    MEAN_VAR: 0.005,
    STD_VAR: 0.005,
    FINE_TUNING: False,

    # CNN
    FILTERS: 24,
    HOLD_PROB: 1,
    POOL_SIZE: [2, 1],
    H_SIZE: [2048, 512],
    CNN_MAX_ITERATIONS: 30000,
    CNN_LEARNING_RATE: 1e-4,

    # LSTM
    LSTM_MAX_ITERATIONS: 30000,  # If an iteration is an epoch, 30_000 epochs are simply too many.
    LSTM_LEARNING_RATE: 1e-4,
    UNITS: 24,
    MLP_MAX_ITERATIONS: 30000,
    MLP_LEARNING_RATE: 1e-4,

}

# Make sure each project has all the necessary fields
ALL_FIELDS = {N_JOBS, N_JOBS_FUZZY, N_JOBS_FEAT_SEL, HYPER_PARAMETER_TUNING, INNER_JOBS_FEAT_SEL, N_JOBS_RBF_OLS,
              N_JOBS_RBF_NN, N_JOBS_SKLEARN, N_JOBS_LSTM, N_JOBS_CNN_3D, N_JOBS_CNN, N_JOBS_MLP, INTRA_OP, N_GPUS,
              TRAIN_SPLIT, TRANSFER_LEARNING,
              RESAMPLING, TL_PROJECT, FEATURE_SELECTION_METHOD, DIMENSIONALITY_REDUCTION, COMBINE_METHODS,
              EXCLUDE_METHOD_COMBINE, IS_GLOBAL, IS_FUZZY, CLUSTER_FILE, IS_CLUSTERING_TRAINED, IS_PROBABILISTIC,
              PROJECT_METHODS, SKLEARN_OPTIMIZER, RECREATE_DATASETS, RECREATE_NWP_FILES, WEATHER_IN_DATA,
              ADD_RULES_INDIVIDUAL, ENABLE_TRANSFER_LEARNING, FS_STATUS, RESAMPLING_THRESH, IMPORT_EXTERNAL_RULES,
              N_CLUSTERS, RESAMPLING_ON_VAR, CHECK_FUZZY_MODELS, TRAIN_ONLINE, GEN_FZ, POP_FZ, GEN_SK, POP_SK,
              THRESH_SPLIT, THRESH_ACT, GEN_RBF, POP_RBF, POPULATION, MAX_ITERATIONS, LEARNING_RATE, MEAN_VAR, STD_VAR,
              FINE_TUNING, FILTERS, UNITS, HOLD_PROB, POOL_SIZE, H_SIZE, CNN_MAX_ITERATIONS, CNN_LEARNING_RATE,
              LSTM_MAX_ITERATIONS, LSTM_LEARNING_RATE, MLP_MAX_ITERATIONS, MLP_LEARNING_RATE, DOCKER, PROJECT_OWNER,
              PROJECTS_GROUP, DATA_FILE_NAME, NWP_FOLDER, SYS_FOLDER, AREA_GROUP, DAYS_OFFSET, NWP_MODEL,
              NWP_RESOLUTION, EVALUATION, FEATURE_SELECTION, DENSE_LAYER_DIM, COMPRESS_DATA, VERSION_GROUP,
              VERSION_MODEL, ΝUM_GEN}


def get_fuzzy_features(file_name):
    if LOAD in file_name:
        model_type = LOAD
    elif PV in file_name:
        model_type = PV
    elif WIND in file_name:
        model_type = WIND
    elif GAS in file_name:
        model_type = GAS
    else:
        raise IOError('Wrong data file name. Use one of load_ts.csv, wind_ts.csv, pv_ts.csv')

    var_imp = {}
    var_lin = []
    var_non_reg = []
    variables = []

    if model_type == PV:
        var_imp = [{HOUR: [{FLUX: {MFS: 1}}, {FLUX: {MFS: 2}}, {FLUX: {MFS: 2}}, {FLUX: {MFS: 3}}, {FLUX: {MFS: 4}},
                           {FLUX: {MFS: 3}}, {FLUX: {MFS: 2}}, {FLUX: {MFS: 2}}, {FLUX: {MFS: 1}}]}]
        var_lin = [FLUX, CLOUD]

    elif model_type == WIND:
        var_imp = [{DIRECTION: [{WIND: {MFS: 3}}, {WIND: {MFS: 3}},
                                {WIND: {MFS: 3}}, {WIND: {MFS: 3}},
                                ]}]
        var_lin = [WIND, DIRECTION, P_WIND]

    elif model_type == LOAD:
        # split first based on sp_index

        var_imp = [

            {SP_INDEX: [{HOUR: {MFS: 2}, MONTH: {MFS: 1}},
                        # {HOUR: {MFS: 2}, MONTH: {MFS: 2}},
                        # {HOUR: {MFS: 2}, MONTH: {MFS: 1}}
                        ]},
            # {SP_INDEX: [{HOUR: {MFS: 4}, MONTH: {MFS: 6}},
            #             {HOUR: {MFS: 2}, MONTH: {MFS: 2}},
            #             {HOUR: {MFS: 1}, MONTH: {MFS: 2}}
            #             ]},
            # {SP_INDEX: [{HOUR: {MFS: 4}, TEMPERATURE: {MFS: 6}},
            #             {HOUR: {MFS: 2}, TEMPERATURE: {MFS: 2}},
            #             {HOUR: {MFS: 1}, TEMPERATURE: {MFS: 2}}
            #             ]}
        ]

        # TODO: What are those var_lin - Are not present in the membership functions
        var_lin = [MONTH, SP_INDEX, DAY_WEEK, TEMPERATURE] + [f'{LOAD}_{k}' for k in range(17, 25)]
    elif model_type == GAS:
        var_imp = [{SP_INDEX: [{MONTH: {MFS: 4}},
                               {MONTH: {MFS: 1}},
                               ]},
                   {SP_INDEX: [{TEMP_MAX: {MFS: 6}},
                               {TEMP_MAX: {MFS: 1}}]},
                   {SP_INDEX: [{MONTH: {MFS: 2}, TEMP_MAX: {MFS: 2}},
                               {MONTH: {MFS: 1}, TEMP_MAX: {MFS: 1}}, ]}
                   ]
        var_lin = [MONTH, SP_INDEX, DAY_WEEK, TEMP_MAX, HDD_H2] + ['Ath24_' + str(i) for i in range(10)]

    if model_type == PV:
        variables = [CLOUD, FLUX, TEMPERATURE, WS]

    elif model_type == WIND:
        variables = [WS, WD, TEMPERATURE]

    elif model_type == LOAD:
        variables = [CLOUD, FLUX, WS, WD, TEMPERATURE]

    elif model_type == GAS:
        variables = [CLOUD, FLUX, WS, WD, TEMPERATURE]

    return {VAR_IMP: var_imp, VAR_LIN: var_lin, VAR_NON_REG: var_non_reg, DATA_VARIABLES: variables,
            MODEL_TYPE: model_type}
