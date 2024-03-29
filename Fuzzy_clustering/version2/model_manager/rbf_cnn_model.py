import os

from Fuzzy_clustering.version2.model_manager.deep_models import Model3d
from Fuzzy_clustering.version2.rbf_ols_manager.rbf_ols import rbf_ols_module


class RbfCnnModel(object):
    def __init__(self, static_data, cluster, cnn=False):
        self.static_data = static_data
        self.cluster_dir = cluster.cluster_dir
        self.model_dir_rbf_ols = os.path.join(self.cluster_dir, 'RBF_OLS')
        self.model_rbf_ols = rbf_ols_module(self.static_data, self.model_dir_rbf_ols, self.static_data['rated'],
                                            self.static_data['sklearn']['njobs'],
                                            GA=False)
        self.model_rbf_ga = rbf_ols_module(self.static_data, self.model_dir_rbf_ols, self.static_data['rated'],
                                           self.static_data['sklearn']['njobs'],
                                           GA=True)
        self.model_rbf_nn = Model3d(self.static_data, cluster, 'RBFNN')

        if self.model_rbf_nn.istrained == False:
            raise ImportError('Cannot found RBFNN model for cluster %s of project %s', cluster.cluster_name,
                              static_data['_id'])
        if self.model_rbf_nn.istrained == False:
            raise ImportError('Cannot found RBF_OLS model for cluster %s of project %s', cluster.cluster_name,
                              static_data['_id'])
        if self.model_rbf_nn.istrained == False:
            raise ImportError('Cannot found RBF_GA_OLS model for cluster %s of project %s', cluster.cluster_name,
                              static_data['_id'])
        if cnn:
            self.model_cnn = Model3d(self.static_data, cluster, 'RBF-CNN')
            if self.model_cnn.istrained == False:
                raise ImportError('Cannot found RBF_CNN model for cluster %s of project %s', cluster.cluster_name,
                                  static_data['_id'])
