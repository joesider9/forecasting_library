import os

import joblib


class SKLearn_Manager(object):
    def __init__(self, static_data, cluster, method, optimizer):
        self.static_data = static_data
        self.path_group = self.static_data['path_group']
        self.optimize_method = optimizer
        self.method = method
        self.istrained = False
        self.njobs = static_data['sklearn']['njobs']
        self.rated = static_data['rated']
        self.models = dict()
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.cluster_name = cluster.cluster_name
        self.sk_models_dir = os.path.join(self.cluster_dir, 'SKLearn')
        self.model_dir = os.path.join(self.sk_models_dir, str.upper(method))
        if not os.path.exists(self.sk_models_dir):
            os.makedirs(self.sk_models_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load()
        except:
            pass

    def load_data(self):
        data_path = self.data_dir
        cvs = joblib.load(os.path.join(data_path, 'cvs.pickle'))
        return cvs

    def fit(self):
        if self.optimize_method == 'deap':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_deap import sklearn_model
        elif self.optimize_method == 'optuna':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_optuna import sklearn_model
        elif self.optimize_method == 'skopt':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_skopt import sklearn_model
        else:
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_grid import sklearn_model

        if self.istrained == False:
            cvs = self.load_data()
            model_sklearn = sklearn_model(self.static_data, self.sk_models_dir, self.rated, self.method, self.njobs,
                                          path_group=self.path_group)
            if model_sklearn.istrained == False:
                print('Train ', self.method, ' ', self.cluster_name)
                self.models[self.method] = model_sklearn.train(cvs)
            else:
                self.models[self.method] = model_sklearn.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def fit_TL(self):
        if self.optimize_method == 'deap':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_deap import sklearn_model
        elif self.optimize_method == 'optuna':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_optuna import sklearn_model
        elif self.optimize_method == 'skopt':
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_skopt import sklearn_model
        else:
            from Fuzzy_clustering.version2.sklearn_models.sklearn_models_grid import sklearn_model
        static_data_tl = self.static_data['tl_project']['static_data']
        cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)
        model_sklearn_TL = sklearn_model(static_data_tl, cluster_dir_tl, static_data_tl['rated'], self.method,
                                         self.njobs)
        if self.istrained == False:
            cvs = self.load_data()
            model_sklearn = sklearn_model(self.static_data, self.sk_models_dir, self.rated, self.method, self.njobs)
            if model_sklearn.istrained == False:
                self.models[self.method] = model_sklearn.train_TL(cvs, model_sklearn_TL.best_params)
            else:
                self.models[self.method] = model_sklearn.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'SKlearnManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'SKlearnManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open SKlearn model')
        else:
            raise ImportError('Cannot find SKlearn model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'cluster_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.model_dir, 'SKlearnManager.pickle'))
