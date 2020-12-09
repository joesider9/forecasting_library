import pymongo
import Fuzzy_clustering.ver_tf2.config as cg
import pandas as pd
import os


def write_database(use_db=False):

    static_data = {'data_file_name':cg.data_file_name,
            'project_owner': cg.project_owner,
            'projects_group': cg.projects_group,
            'area_group':cg.area_group,
            'version':cg.version,
            'Evaluation_start': cg.Evaluation,
            'project_methods': cg.project_methods,
            'combine_methods':cg.combine_methods,
            'resampling': cg.resampling,
            'resampling_thres': cg.resampling_thres,
            'resampling_on_var':cg.resampling_on_var,
            'AUTO_COORDS_FIND': cg.AUTO_COORDS_FIND,
            'url': cg.url,
            'port': cg.port,
            'transfer_learning': cg.transfer_learning,
            'tl_project': cg.tl_project,
            'NWP_model':cg.NWP_model,
            'NWP_resolution':cg.NWP_resolution,
            'compress_data' : cg.compress_data,
            'data_variables':cg.variables,
            'recreate_nwp_files': cg.RECREATE_NWP_FILES,
            'recreate_datasets': cg.RECREATE_DATASETS,
            'enable_transfer_learning':cg.ENABLE_TRANSFER_LEARNING,
            'exclude_method_combine':cg.exclude_method_combine,
            'check_fuzzy_models': cg.check_fuzzy_models,
            'njobs': cg.njobs,
            'train_online': cg.clustering_train_online,
            'sklearn': {'njobs': cg.njobs, 'fs_status':cg.fs_status, 'fs_method':cg.feature_selection_method},
            'clustering':{'is_Fuzzy':cg.is_Fuzzy,
                          'n_clusters':cg.n_clusters,
                          'thres_act':cg.thres_act,
                          'thres_split':cg.thres_split,
                          'clustering_train_online':cg.clustering_train_online,
                          'is_clustering_trained':cg.is_clustering_trained,
                          'var_imp':cg.var_imp,
                          'var_lin':cg.var_lin,
                          'var_nonreg':cg.var_nonreg,
                          'cluster_file':cg.cluster_file,
                          'add_rules_indvidual':cg.add_rules_indvidual,
                          'import_external_rules':cg.import_external_rules},
            'RBF': {'max_iterations': cg.max_iterations,
                    'learning_rate': cg.learning_rate,
                    'mean_var': cg.mean_var,
                    'std_var': cg.std_var,
                    'njobs': cg.njobs,
                    'Fine_tuning': cg.fine_tuning,
                    'gpus': cg.gpus},
            'CNN': {
                'filters': cg.filters,
                'pool_size': cg.pool_size,
                'h_size': cg.h_size,
                'max_iterations': cg.cnn_max_iterations,
                'learning_rate': cg.cnn_learning_rate,
                'njobs': cg.njobs,
                'CNN_path_temp':cg.cnn_path_temp,
                'gpus': cg.gpus

            }
            }
    if use_db:
        project_dict = dict()
        project_dict['static_data'] = static_data

        project_dict['data'] = {}
        project_dict['data']['obs'] = []
        project_dict['data']['pred'] = []
        project_dict['data']['extra'] = []
        project_dict['_id'] = cg.project_name

        for data_type in cg.data.keys():
            for data_name in cg.data[data_type]:
                data = pd.read_csv(os.path.join(cg.country_data, data_name + '.csv'), header=[0], index_col=0,
                                   parse_dates=True)
                data = data[cg.project_name].to_frame()
                for d, o in data.iterrows():
                    try:
                        obs_dict = {
                            '_id': d.strftime('%Y%m%d%H%M'),
                            'data': str(o.values[0])
                        }
                        project_dict['data'][data_type].append(obs_dict)
                    except:
                        continue

        myclient = pymongo.MongoClient("mongodb://" + cg.url + ":" + cg.port + "/")
        project_db = myclient[cg.project_owner]
        if cg.project_name not in project_db.collection_names():
            proj_col = project_db[cg.project_name]
            proj_col.insert_one(project_dict)
        else:
            proj_col = project_db[cg.project_name]
            proj_col.update_one({'_id': cg.project_name}, {
                '$set': {'static_data': project_dict['static_data'], 'data.obs': project_dict['data']['obs'],
                         'data.extra': project_dict['data']['extra']}})

    return static_data