
import numpy as np
import pandas as pd
import os, copy
import joblib
import skfuzzy as fuzz

class clusterer(object):

    def __init__(self, fuzzy_path, fuzzy_file, type):
        self.fuzzy_file = os.path.join(fuzzy_path, fuzzy_file)
        fmodel = joblib.load(self.fuzzy_file)
        self.rules = fmodel['rules']
        if type == 'pv':
            self.p = 4
        elif type == 'wind':
            self.p = 3
        elif type == 'load':
            self.p = 4
        elif type == 'fa':
            self.p = 3

    def compute_activations(self, X):
        activations = pd.DataFrame(index=X.index, columns=[i for i in sorted(self.rules.keys())])
        var_del = []
        for rule in sorted(self.rules.keys()):
            act = []
            for mf in self.rules[rule]:
                if mf['var_name'] not in X.columns:
                    var_names = [c for c in X.columns if mf['var_name'] in c]
                    X[mf['var_name']] = X[var_names].mean(axis=1)
                    var_del.append(mf['var_name'])
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
            activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / self.p)
        if len(var_del) > 0:
            X = X.drop(columns=var_del)
        return activations
