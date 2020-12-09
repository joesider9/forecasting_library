import os, pickle

class FuzzyManager():
    def __init__(self,static_data):
        self.istrained = False
        self.static_data = static_data
        self.rated = static_data['rated']
        self.fuzzy_path = self.static_data['path_fuzzy_models']
        self.fuzzy_file = self.static_data['clustering']['cluster_file']
        self.model_type = self.static_data['type']
        try:
            self.load()
        except:
            pass


    def load(self):
        if os.path.exists(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict={}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'fuzzy_file', 'fuzzy_path', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open rule fuzzy model')
        else:
            raise ImportError('Cannot find rule fuzzy model')


    def save(self):
        f = open(os.path.join(self.fuzzy_path, 'fuzzy_model.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'fuzzy_file', 'fuzzy_path', 'n_jobs']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()