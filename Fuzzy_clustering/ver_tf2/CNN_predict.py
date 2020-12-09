import tensorflow as tf
import numpy as np
from scipy.interpolate import interp2d
import os, joblib
from Fuzzy_clustering.ver_tf2.RBFNN_predict import rbf_model_predict
from Fuzzy_clustering.ver_tf2.RBF_ols_predict import rbf_ols_predict

class CNN_predict():
    def __init__(self, static_data, rated, cluster_dir, rbf_dir):

        self.static_data = static_data['CNN']
        self.static_data_rbf = static_data['RBF']
        self.rated = rated
        self.cluster = os.path.basename(cluster_dir)
        self.istrained = False
        if isinstance(rbf_dir, list):
            self.rbf_method = 'RBF_ALL'
            self.cluster_cnn_dir = os.path.join(cluster_dir, 'RBF_ALL/CNN')
            self.model_dir = os.path.join(self.cluster_cnn_dir, 'model')
            self.rbf = rbf_model_predict(self.static_data_rbf, self.rated, cluster_dir)
            self.rbf.models=[]
            for dir in rbf_dir:
                rbf_method = os.path.basename(dir)
                cluster_rbf_dir = os.path.join(dir, 'model')

                if rbf_method == 'RBFNN':
                    rbf = rbf_model_predict(self.static_data_rbf, self.rated, cluster_dir)

                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[0])
                elif rbf_method == 'RBF_OLS':
                    rbf = rbf_ols_predict(cluster_dir, rated, self.static_data_rbf['njobs'], GA=False)
                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[-1])
                elif rbf_method == 'GA_RBF_OLS':
                    rbf = rbf_ols_predict(cluster_dir, rated, self.static_data_rbf['njobs'], GA=True)
                    try:
                        rbf.load(cluster_rbf_dir)
                    except:
                        raise ImportError('Cannot load RBFNN models')
                    self.rbf.models.append(rbf.models[0])
                else:
                    raise ValueError('Cannot recognize RBF method')

        else:
            self.rbf_method = os.path.basename(rbf_dir)
            cluster_rbf_dir = os.path.join(rbf_dir, 'model')
            self.cluster_cnn_dir = os.path.join(rbf_dir, 'CNN')
            self.model_dir = os.path.join(self.cluster_cnn_dir, 'model')
            if self.rbf_method == 'RBFNN':
                self.rbf = rbf_model_predict(self.static_data_rbf, self.rated, cluster_dir)
            elif self.rbf_method == 'RBF_OLS':
                self.rbf = rbf_ols_predict(cluster_dir, rated, self.static_data_rbf['njobs'], GA=False)
            elif self.rbf_method == 'GA_RBF_OLS':
                self.rbf = rbf_ols_predict(cluster_dir, rated, self.static_data_rbf['njobs'], GA=True)
            else:
                raise ValueError('Cannot recognize RBF method')
            try:
                self.rbf.load(cluster_rbf_dir)
            except:
                raise ImportError('Cannot load RBFNN models')


        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.temp_dir = os.path.join(self.static_data['CNN_path_temp'], 'temp')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        try:
            self.load(self.model_dir)
            self.istrained = True
        except:
            pass


    def rbf_map(self, X, num_centr, centroids, radius):
        hmap_list = []
        s = X.shape
        d1 = np.transpose(np.tile(np.expand_dims(X, axis=0), [num_centr, 1, 1]), [1, 0, 2]) - np.tile(
            np.expand_dims(centroids, axis=0), [s[0], 1, 1])
        d = np.sqrt(np.power(np.multiply(d1, np.tile(np.expand_dims(radius, axis=0), [s[0], 1, 1])), 2))
        phi = np.exp((-1) * np.power(d, 2))

        return np.transpose(phi,[1, 0, 2])

    def rescale(self,arr, nrows, ncol):
        W, H = arr.shape
        new_W, new_H = (nrows, ncol)
        xrange = lambda x: np.linspace(0, 1, x)

        f = interp2d(xrange(H), xrange(W), arr, kind="linear")
        new_arr = f(xrange(new_H), xrange(new_W))

        return new_arr

    def create_inputs(self, X_train):
        self.N, self.D = X_train.shape


        H = []

        self.depth = len(self.rbf.models)

        self.num_centr=0

        for i in range(self.depth):
            if self.rbf.models[i]['centroids'].shape[0]>self.num_centr:
                self.num_centr = self.rbf.models[i]['centroids'].shape[0]

        for i in range(self.depth):
            if len(self.rbf.models[i]['Radius'].shape) == 1:
                self.rbf.models[i]['Radius'] = np.tile(self.rbf.models[i]['Radius'].reshape(1, -1), [self.num_centr,1])

            if self.rbf.models[i]['centroids'].shape[0] < self.num_centr:
                centroids=self.rescale(self.rbf.models[i]['centroids'], self.num_centr, self.D)
            else:
                centroids = self.rbf.models[i]['centroids']
            if np.isscalar(self.rbf.models[i]['Radius']):
                Radius = self.rbf.models[i]['Radius']
            elif self.rbf.models[i]['Radius'].shape[0] == self.num_centr:
                Radius = self.rbf.models[i]['Radius']
            elif self.rbf.models[i]['Radius'].shape[0] < self.num_centr:
                Radius = self.rescale(self.rbf.models[i]['Radius'], self.num_centr, self.D)
            else:
                raise ValueError('Unkown shape')
            H.append(np.transpose(self.rbf_map(X_train, self.num_centr, centroids, Radius), [1, 2, 0]))
            H[i] = np.array(H[i])
            H[i] = H[i].reshape(-1, self.D * self.num_centr)
            sc=self.scale_cnn[i]
            H[i] = sc.transform(H[i].reshape(-1, self.D * self.num_centr))
            H[i] = np.nan_to_num(H[i].reshape(-1, self.D, self.num_centr))

        H = np.transpose(np.stack(H), [1, 2, 3, 0])


        return H

    def init_weights(self, init_w):
        init_random_dist = tf.convert_to_tensor(init_w)
        return tf.Variable(init_random_dist)

    def init_bias(self, init_b):
        init_bias_vals = tf.convert_to_tensor(init_b)
        return tf.Variable(init_bias_vals)

    def normal_full_layer(self,input_layer, init_w, init_b):

        W = self.init_weights(init_w)
        b = self.init_bias(init_b)
        return tf.add(tf.matmul(input_layer, W), b, name='prediction'), W, b

    def build_graph(self, x1, best_weights, kernels, h_size, hold_prob, filters):

        with tf.name_scope("build_cnn") as scope:
            H = tf.reshape(x1, [-1, self.D, self.num_centr, self.depth],name='reshape_1')

            convo_1= tf.keras.layers.Conv2D(filters=int(filters),
                                  kernel_size=kernels,
                                  padding="same",
                                  name='cnn1',
                                  # kernel_initializer=lambda shape, dtype: best_weights['build_cnn/cnn1/kernel:0'],
                                  # bias_initializer=lambda  shape, dtype: best_weights['build_cnn/cnn1/bias:0'],
                                  activation=tf.nn.elu)

            convo_1_pool = tf.keras.layers.AveragePooling2D(pool_size=self.static_data['pool_size'], strides=1, name='pool1')
            cnn_output=convo_1_pool(convo_1(H))
            full_one_dropout = tf.nn.dropout(cnn_output, keep_prob=hold_prob)
            shape = full_one_dropout.get_shape().as_list()
            s = shape[1] * shape[2] * shape[3]
            convo_2_flat = tf.reshape(full_one_dropout, [-1, s])

            full_layer_one = tf.keras.layers.Dense(units=h_size[0],activation=tf.nn.elu, name='dense1')
            # , kernel_initializer = lambda x: best_weights['build_cnn/dense1/kernel:0'],
            # bias_initializer = lambda x: best_weights['build_cnn/dense1/bias:0'],
            full_layer_two = tf.keras.layers.Dense(units=h_size[1], activation=tf.nn.elu, name='dense2')
            # , kernel_initializer = lambda x: best_weights['build_cnn/dense2/kernel:0'],
            # bias_initializer = lambda x: best_weights['build_cnn/dense2/bias:0']
            full_two_dropout = tf.nn.dropout(full_layer_one(convo_2_flat), keep_prob=hold_prob)
            dense_output = tf.nn.dropout(full_layer_two(full_two_dropout), keep_prob=hold_prob)

            convo_1.set_weights([best_weights['build_cnn/cnn1/kernel:0'], best_weights['build_cnn/cnn1/bias:0']])
            full_layer_one.set_weights([best_weights['build_cnn/dense1/kernel:0'], best_weights['build_cnn/dense1/bias:0']])
            full_layer_two.set_weights([best_weights['build_cnn/dense2/kernel:0'], best_weights['build_cnn/dense2/bias:0']])

            y_pred, W, b = self.normal_full_layer(dense_output, best_weights['build_cnn/Variable:0'],best_weights['build_cnn/Variable_1:0'] )
            weights = convo_1.trainable_weights + full_layer_one.trainable_weights + full_layer_two.trainable_weights + [W, b]
        return y_pred, weights, convo_1, full_layer_one, full_layer_two

    def predict(self, X):
        if self.istrained:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

            if hasattr(self.model, 'filters'):
                filters = self.model['filters']
            else:
                filters = int(self.static_data['filters'])
            kernels = self.model['kernels']
            h_size = self.model['h_size']
            best_weights = self.model['best_weights']
            H = self.create_inputs(X)

            tf.compat.v1.reset_default_graph()
            graph_cnn = tf.Graph()
            with graph_cnn.as_default():
                with tf.device("/cpu:0"):
                    x1 = tf.compat.v1.placeholder('float', shape=[None, self.D, self.num_centr, self.depth], name='input_data')
                    hold_prob = tf.compat.v1.placeholder(tf.float32, name='drop')
                with tf.device("/cpu:0"):
                    y_pred_, weights, convo_1, full_layer_one, full_layer_two = self.build_graph(x1, best_weights, kernels, h_size, hold_prob, filters)



            config_tf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config_tf.gpu_options.allow_growth = True

            with tf.compat.v1.Session(graph=graph_cnn, config=config_tf) as sess:
                print('Open an rbf-cnn network with %s' % self.num_centr)

                sess.run(tf.compat.v1.global_variables_initializer())

                convo_1.set_weights([best_weights['build_cnn/cnn1/kernel:0'], best_weights['build_cnn/cnn1/bias:0']])
                full_layer_one.set_weights(
                    [best_weights['build_cnn/dense1/kernel:0'], best_weights['build_cnn/dense1/bias:0']])
                full_layer_two.set_weights(
                    [best_weights['build_cnn/dense2/kernel:0'], best_weights['build_cnn/dense2/bias:0']])

                y_pred, weights_run= sess.run([y_pred_, weights],
                                 feed_dict={x1: H, hold_prob:1})


                sess.close()
        else:
            raise ModuleNotFoundError("Error on prediction of %s cluster. The model CNN seems not properly trained", self.cluster)

        return y_pred

    def load(self, pathname):
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'cnn' + '.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(cluster_dir, 'cnn' + '.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')