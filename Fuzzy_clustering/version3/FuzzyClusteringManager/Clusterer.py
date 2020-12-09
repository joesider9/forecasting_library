
import numpy as np
import pandas as pd
import os, copy
import joblib, pickle
import skfuzzy as fuzz

class clusterer(object):

    def __init__(self, static_data):
        self.istrained = False
        self.train_online = static_data['train_online']
        self.add_individual_rules = static_data['clustering']['add_rules_indvidual']
        self.import_external_rules = static_data['clustering']['import_external_rules']
        self.njobs = static_data['clustering']['njobs']
        self.resampling = static_data['resampling']
        self.path_fuzzy = static_data['path_fuzzy_models']
        self.file_fuzzy = static_data['clustering']['cluster_file']
        self.type = static_data['type']

        self.static_data = static_data
        try:
            self.load()
        except:
            pass


    def compute_activations(self, X):
        if not hasattr(self, 'best_fuzzy_model'):
            self.best_fuzzy_model = joblib.load(os.path.join(self.path_fuzzy, self.file_fuzzy))
        self.rules = self.best_fuzzy_model['rules']
        activations = pd.DataFrame(index=X.index, columns=[i for i in sorted(self.rules.keys())])
        var_del=[]
        for rule in sorted(self.rules.keys()):
            act = []
            for mf in self.rules[rule]:
                if mf['var_name'] not in X.columns:
                    var_names = [c for c in X.columns if mf['var_name'] in c]
                    X[mf['var_name']] = X[var_names].mean(axis=1)
                    var_del.append(mf['var_name'])
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
                if not 'p' in mf.keys():
                    mf['p'] = 2
            activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / mf['p'])
        if len(var_del)>0:
            X = X.drop(columns=var_del)
        return activations

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict={}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')
#