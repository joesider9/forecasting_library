import pickle, joblib, os

class cluster_object():
    def __init__(self, static_data, clust):
        self.istrained = False
        self.cluster_name = clust
        self.cluster_dir = os.path.join(static_data['path_model'], 'Regressor_layer/' + clust)
        self.static_data = static_data
        self.model_type = static_data['type']
        self.methods = [method for method in static_data['project_methods'].keys() if static_data['project_methods'][method]==True]
        self.combine_methods = static_data['combine_methods']
        self.rated = static_data['rated']
        self.n_jobs = static_data['njobs']
        self.var_lin = static_data['clustering']['var_lin']
        self.data_dir = os.path.join(self.cluster_dir, 'data')
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']

        try:
            self.load(self.cluster_dir)
        except:
            raise ImportError('Cannot find cluster ', clust)


    def load(self, cluster_dir):
        if os.path.exists(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'model_' + self.cluster_name +'.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict={}
                for k in tmp_dict.keys():
                    tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open rule model %s', self.cluster_name)
        else:
            raise ImportError('Cannot find rule model %s', self.cluster_name)


