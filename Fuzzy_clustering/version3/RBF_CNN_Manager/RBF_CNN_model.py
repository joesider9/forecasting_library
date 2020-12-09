import os
from Fuzzy_clustering.version3.RBF_CNN_Manager.RBF_ols_object import rbf_ols_module
from Fuzzy_clustering.version3.RBF_CNN_Manager.Model_3d_object import model3d_object

class RBF_CNN_model(object):
    def __init__(self, static_data, cluster, cnn=False):
        self.static_data = static_data
        self.cluster_dir = cluster.cluster_dir
        self.cnn = cnn
        self.model_dir_rbfols = os.path.join(self.cluster_dir, 'RBF_OLS')
        self.model_rbf_ols = rbf_ols_module(self.model_dir_rbfols, self.static_data['rated'], self.static_data['sklearn']['njobs'],
                                       GA=False)
        self.model_rbf_ga = rbf_ols_module(self.model_dir_rbfols, self.static_data['rated'], self.static_data['sklearn']['njobs'],
                                      GA=True)
        self.model_rbfnn = model3d_object(self.static_data, cluster, 'RBFNN')

        if self.model_rbfnn.istrained==False:
            raise ImportError('Cannot found RBFNN model for cluster %s of project %s', cluster.cluster_name, static_data['_id'])
        if self.model_rbf_ols.istrained==False:
            raise ImportError('Cannot found RBF_OLS model for cluster %s of project %s', cluster.cluster_name, static_data['_id'])
        if self.model_rbf_ga.istrained==False:
            raise ImportError('Cannot found RBF_GA_OLS model for cluster %s of project %s', cluster.cluster_name, static_data['_id'])

        if cnn:
            self.model_cnn = model3d_object(self.static_data, cluster, 'RBF-CNN')
            if self.model_cnn.istrained == False:
                raise ImportError('Cannot found RBF_CNN model for cluster %s of project %s', cluster.cluster_name,
                                  static_data['_id'])