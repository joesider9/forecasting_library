import os
import pickle

class Model_object():

    def __init__(self, path_model):
        self.istrained = False
        self.intialized = False
        self.clusters_created = True
        self.path_model = path_model
        try:
            self.load()
        except:
            pass

    def init(self, static_data, data_variables):
        self.data_variables = data_variables
        self.static_data = static_data
        self.thres_split = static_data['clustering']['thres_split']
        self.thres_act = static_data['clustering']['thres_act']
        self.n_clusters = static_data['clustering']['n_clusters']
        self.rated = static_data['rated']
        self.var_imp = static_data['clustering']['var_imp']
        self.var_lin = static_data['clustering']['var_lin']
        self.var_nonreg = static_data['clustering']['var_nonreg']

    def load(self):
        if os.path.exists(os.path.join(self.path_model, 'manager' + '.pickle')):
            try:
                f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                if 'path_model' in tmp_dict.keys():
                    del tmp_dict['path_model']
                self.__dict__.update(tmp_dict)
            except:
                raise ValueError('Cannot find model for %s', self.path_model)
        else:
            raise ValueError('Cannot find model for %s', self.path_model)

    def save(self):
        f = open(os.path.join(self.path_model, 'manager' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger','db', 'path_model', 'static_data','thres_act','thres_split','use_db']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()