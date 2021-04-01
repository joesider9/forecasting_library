import os

import Fuzzy_clustering.version2.template.config_timos as cg


def write_database():
    static_data = {'Docker': cg.Docker,
                   'data_file_name': cg.data_file_name,
                   'project_owner': cg.project_owner,
                   'projects_group': cg.projects_group,
                   'area_group': cg.area_group,
                   'version_group': cg.version_group,
                   'version_model': cg.version_model,
                   'type': cg.model_type,
                   'Evaluation_start': cg.Evaluation,
                   'project_methods': cg.project_methods,
                   'combine_methods': cg.combine_methods,
                   'is_probabilistic': cg.is_probabilistic,
                   'resampling': cg.resampling,
                   'resampling_thres': cg.resampling_thres,
                   'resampling_on_var': cg.resampling_on_var,
                   'transfer_learning': cg.transfer_learning,
                   'tl_project': cg.tl_project,
                   'NWP_model': cg.NWP_model,
                   'NWP_resolution': cg.NWP_resolution,
                   'weather_in_data': cg.weather_in_data,
                   'compress_data': cg.compress_data,
                   'data_variables': cg.variables,
                   'recreate_nwp_files': cg.RECREATE_NWP_FILES,
                   'recreate_datasets': cg.RECREATE_DATASETS,
                   'enable_transfer_learning': cg.ENABLE_TRANSFER_LEARNING,
                   'exclude_method_combine': cg.exclude_method_combine,
                   'check_fuzzy_models': cg.check_fuzzy_models,
                   'njobs': cg.njobs,
                   'ngpus': cg.ngpus,
                   'njobs_feat_sel': cg.njobs_feat_sel,
                   'inner_jobs_feat_sel': cg.inner_jobs_feat_sel,
                   'intra_op': cg.intra_op,
                   'train_online': cg.clustering_train_online,
                   'sklearn': {'njobs': cg.njobs_sklearn,
                               'fs_status': cg.fs_status,
                               'fs_method': cg.feature_selection_method,
                               'optimizer': cg.Sklearn_optimizer,
                               'dimentional_reduction': cg.dimentional_reduction,
                               'pop': cg.pop_sk,
                               'gen': cg.gen_sk},
                   'is_Global': cg.is_Global,
                   'clustering': {'njobs': cg.njobs_fuzzy,
                                  'is_Fuzzy': cg.is_Fuzzy,
                                  'n_clusters': cg.n_clusters,
                                  'thres_act': cg.thres_act,
                                  'thres_split': cg.thres_split,
                                  'clustering_train_online': cg.clustering_train_online,
                                  'is_clustering_trained': cg.is_clustering_trained,
                                  'var_imp': cg.var_imp,
                                  'var_lin': cg.var_lin,
                                  'var_nonreg': cg.var_nonreg,
                                  'cluster_file': cg.cluster_file,
                                  'add_rules_indvidual': cg.add_rules_indvidual,
                                  'import_external_rules': cg.import_external_rules,
                                  'pop': cg.pop_fz,
                                  'gen': cg.gen_fz,
                                  },
                   'RBF': {'max_iterations': cg.max_iterations,
                           'learning_rate': cg.learning_rate,
                           'mean_var': cg.mean_var,
                           'std_var': cg.std_var,
                           'njobs': cg.njobs_rbfnn,
                           'Fine_tuning': cg.fine_tuning,
                           'pop': cg.pop_rbf,
                           'gen': cg.gen_rbf,
                           },
                   'CNN': {
                       'filters': cg.filters,
                       'pool_size': cg.pool_size,
                       'h_size': cg.h_size,
                       'max_iterations': cg.cnn_max_iterations,
                       'learning_rate': cg.cnn_learning_rate,
                       'njobs_3d': cg.njobs_cnn_3d,
                       'njobs': cg.njobs_cnn,
                   },
                   'LSTM': {
                       'filters': cg.filters,
                       'pool_size': cg.pool_size,
                       'h_size': cg.h_size,
                       'max_iterations': cg.cnn_max_iterations,
                       'learning_rate': cg.cnn_learning_rate,
                       'njobs': cg.njobs_lstm,
                   },
                   'MLP': {
                       'hold_prob': cg.hold_prob,
                       'max_iterations': cg.mlp_max_iterations,
                       'learning_rate': cg.mlp_learning_rate,
                       'njobs': cg.njobs_mlp,
                   }
                   }
    paths = define_folder_names()
    static_data.update(paths)
    return static_data


def define_folder_names():
    model_type = cg.model_type
    sys_folder = cg.sys_folder

    if cg.NWP_model == 'skiron' and cg.NWP_resolution == 0.05:
        path_nwp = os.path.join(cg.nwp_folder, 'SKIRON')
    elif cg.NWP_model == 'skiron' and cg.NWP_resolution == 0.1:
        path_nwp = os.path.join(cg.nwp_folder, 'SKIRON_low')
    elif cg.NWP_model == 'ecmwf':
        path_nwp = os.path.join(cg.nwp_folder, 'ECMWF')
    else:
        path_nwp = None

    path_group = sys_folder + cg.project_owner + '/' + cg.projects_group + '_ver' + str(
        cg.version_group) + '/' + model_type
    if not os.path.exists(path_group):
        os.makedirs(path_group)
    path_nwp_group = sys_folder + cg.project_owner + '/' + cg.projects_group + '_ver' + str(
        cg.version_group) + '/nwp'
    if not os.path.exists(path_nwp_group):
        os.makedirs(path_nwp_group)

    paths = {
        'sys_folder': sys_folder,
        'path_nwp': path_nwp,
        'path_group': path_group,
        'path_nwp_group': path_nwp_group,
    }
    return paths
