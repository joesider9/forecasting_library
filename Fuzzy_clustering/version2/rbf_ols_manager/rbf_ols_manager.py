import os

import joblib

from Fuzzy_clustering.version2.rbf_ols_manager.rbf_ols import rbf_ols_module


class RbfOlsManager(object):
    def __init__(self, static_data, cluster):
        self.static_data = static_data
        self.istrained = False
        self.njobs = cluster.static_data['sklearn']['njobs']
        self.models = dict()
        self.cluster_name = cluster.cluster_name
        self.data_dir = cluster.data_dir
        self.cluster_dir = cluster.cluster_dir
        self.model_dir = os.path.join(self.cluster_dir, 'RBF_OLS')
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
        if self.istrained == False:
            cvs = self.load_data()
            model_rbf_ols = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'],
                                           self.static_data['sklearn']['njobs'], GA=False
                                           , path_group=self.static_data['path_group'])
            model_rbf_ga = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'],
                                          self.static_data['sklearn']['njobs'], GA=True
                                          , path_group=self.static_data['path_group'])
            if model_rbf_ols.istrained == False:
                max_samples = 1500
                print('Train RBFOLS ', self.cluster_name)
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf(cvs, max_samples=max_samples)

            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False:
                max_samples = 1500

                print('Train GA-RBF ', self.cluster_name)
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf(cvs, max_samples=max_samples)

            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def fit_TL(self):
        if self.istrained == False:
            static_data_tl = self.static_data['tl_project']['static_data']
            cluster_dir_tl = os.path.join(static_data_tl['path_model'], 'Regressor_layer/' + self.cluster_name)

            model_rbf_ols_TL = rbf_ols_module(static_data_tl, cluster_dir_tl, static_data_tl['rated'],
                                              self.static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga_TL = rbf_ols_module(static_data_tl, cluster_dir_tl, static_data_tl['rated'],
                                             self.static_data['sklearn']['njobs'], GA=True)
            cvs = self.load_data()
            model_rbf_ols = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'],
                                           self.static_data['sklearn']['njobs'], GA=False)
            model_rbf_ga = rbf_ols_module(self.static_data, self.model_dir, self.static_data['rated'],
                                          self.static_data['sklearn']['njobs'], GA=True)
            if model_rbf_ols.istrained == False:
                self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf_TL(cvs, model_rbf_ols_TL.models)
            else:
                self.models['RBF_OLS'] = model_rbf_ols.to_dict()
            if model_rbf_ga.istrained == False:
                self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf_TL(cvs, model_rbf_ga_TL.models)
            else:
                self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()
            self.istrained = True
            self.save()
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'RBFolsManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.model_dir, 'RBFolsManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'model_dir', 'cluster_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.model_dir, 'RBFolsManager.pickle'))
