import os, pickle


class rbf_ols_module(object):
    def __init__(self, cluster_dir, rated, njobs, GA=False, path_group = None):
        self.cluster = [p1 for p1 in cluster_dir.split('/') if ('rule' in p1) or ('global' in p1)][0]
        self.path_group = path_group
        self.njobs = njobs
        self.rated = rated
        self.GA=GA
        self.istrained = False
        if GA==False:
            self.model_dir = os.path.join(cluster_dir, 'RBF_OLS')
        else:
            self.model_dir = os.path.join(cluster_dir, 'GA_RBF_OLS')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def load(self, pathname):
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        # creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        # toolbox = base.Toolbox()
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'rbf_ols' + '.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'rbf_ols' + '.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RBFNN model')
        else:
            raise ImportError('Cannot find RBFNN model')
