import os, pickle
import numpy as np

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

        try:
            self.load(self.model_dir)
        except:
            raise ImportError('Cannot find RBF model %s', self.cluster)

    def compute_metrics(self, pred, y, rated):
        if rated== None:
            rated=y.ravel()
        else:
            rated=1
        err=np.abs(pred.ravel()-y.ravel())/rated
        sse=np.sum(np.square(pred.ravel()-y.ravel()))
        rms=np.sqrt(np.mean(np.square(err)))
        mae=np.mean(err)
        mse = sse/y.shape[0]

        return [sse, rms, mae, mse]

    def predict(self, x):
        if len(x.shape)==1:
            x=x.reshape(1,-1)
        self.load(self.model_dir)
        model = self.models
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 2.))
        v = np.matmul(v,model['W'][:-1])+model['W'][-1]
        pred = v
        pred[np.where(pred < 0)] = 0
        return pred

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

