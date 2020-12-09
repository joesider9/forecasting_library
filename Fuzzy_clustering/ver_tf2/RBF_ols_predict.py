import numpy as np
import joblib, os, pickle
from deap import base, creator, tools, algorithms
from itertools import repeat
from collections import Sequence
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class rbf_ols_predict(object):
    def __init__(self, cluster_dir, rated, njobs, GA=False):
        self.cluster = os.path.basename(cluster_dir)
        self.njobs = 2 * njobs
        self.rated = rated
        self.GA=GA
        self.istrained = False
        if GA==False:
            self.cluster_dir = os.path.join(cluster_dir, 'RBF_OLS')
            self.model_dir = os.path.join(self.cluster_dir, 'model')
        else:
            self.cluster_dir = os.path.join(cluster_dir, 'GA_RBF_OLS')
            self.model_dir = os.path.join(self.cluster_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass



    def compute_metrics(self, pred, y, rated):
        if rated== None:
            rated=y.ravel()
        else:
            rated=20
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
        pred=[]
        for model in self.models:
            v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
            v = np.sqrt((v ** 2.).sum(-1))
            v = np.exp(-(v ** 2.))
            v = np.matmul(v,model['W'][:-1])+model['W'][-1]
            pred.append(v)
        return np.mean(np.array(pred), axis=0)

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
                del tmp_dict['cluster_dir'], tmp_dict[
                    'model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RBFNN model')
        else:
            raise ImportError('Cannot find RBFNN model')