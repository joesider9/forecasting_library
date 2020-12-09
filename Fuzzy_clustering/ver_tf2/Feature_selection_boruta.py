import numpy as np
import pandas as pd
import logging, os
from Fuzzy_clustering.ver_tf2.Sklearn_models_deap import sklearn_model
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from boruta import BorutaPy
# from util_database import write_database

class FS(object):
    def __init__(self, model_path, njobs):
        self.njobs=njobs
        self.log_dir=os.path.join(model_path, 'FS/boruta')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def fit(self, cvs):
        logger = logging.getLogger('log_fs_boruta.log')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, 'log_fs_boruta.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        print()
        print('Training the model (Fitting to the training data) ')
        logger.info('Training the feature extraction ')
        X = np.vstack((cvs[0][0], cvs[0][2], cvs[0][4]))

        if len(cvs[0][1].shape) == 1 and len(cvs[0][5].shape) == 1:
            y = np.hstack((cvs[0][1], cvs[0][3], cvs[0][5]))
        else:
            y = np.vstack((cvs[0][1], cvs[0][3], cvs[0][5])).ravel()
        self.D, self.N = X.shape

        regressor = sklearn_model(self.log_dir, 1, 'rf', self.njobs)
        if regressor.istrained==False:
            regressor.train(cvs)

        # Update classifier parameters
        estimator=regressor.model
        estimator.set_params(n_jobs=-1)
        self.init_params=[regressor.best_params]
        # Define steps
        step1 = {'Constant Features': {'frac_constant_values': 0.999}}

        step2 = {'Correlated Features': {'correlation_threshold': 0.999}}

        step3 = {'Relevant Features': {'cv': 3,
                                       'estimator': estimator,
                                        'n_estimators': 500,
                                        'max_iter': 20,
                                        'verbose': 0,
                                        'random_state': 42}}

        step4 = {'RFECV Features': {'cv': 3,
                                    'estimator': estimator,
                                    'step': 1,
                                    'scoring': 'neg_root_mean_squared_error',
                                    'verbose': 50}}

        # Place steps in a list in the order you want them execute it
        steps = [step1, step2, step3]
        columns = ['other_' + str(i) for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)
        # Initialize FeatureSelector()
        fs = FeatureSelector()

        # Apply feature selection methods in the order they appear in steps
        fs.fit(X_df, y.ravel(), steps)
        features = [i for i in range(len(X_df.columns)) if X_df.columns[i] in fs.selected_features]

        # Get selected features
        self.features = np.array(features)

        # logger.info('best score %s', str(best_score))
        logger.info('Number of variables %s', str(self.features.shape[0]))
        logger.info('Finish the feature extraction ')
        return features

class FeatureSelector:

    def __init__(self):
        self.rfecv = None
        self.selected_features = None

    def fit(self, X, y, steps=[]):

        for step in steps:
            available_methods = ['Constant Features', 'Correlated Features', 'Relevant Features', 'RFECV Features']

            for key, value in step.items():
                if key not in available_methods:
                    print(f'{key} is not a valid key!')
                    print(f'Only these are available: {available_methods}')
                    print(f'Redefine the key in this dict/step: {step}')
                    print('Now exiting function!')
                    return None

        # Get the order the methods are going to be applied
        method_order = [[*step][0] for step in steps]

        # Get methods
        ordered_methods = self.get_methods(method_order)

        # Initiate empty list of labels to drop
        drop_features = []

        # Temporary features
        X_temp = X.copy()

        for method_label in method_order:
            # Get method
            method = ordered_methods[method_label]

            # Get method parameters
            for step in steps:
                if method_label in step.keys():
                    params = step[method_label]

            # Determine features to drop
            if method_label in ['Constant Features', 'Correlated Features']:
                # Message to user
                print(f'Removing {method_label}')
                drop_features_temp = method(X_temp, **params)
                print(drop_features_temp)
                print('')

                # Append features to drop list
                drop_features = drop_features + drop_features_temp

                # Update feature matrix
                X_temp = X.drop(columns=drop_features, axis=1)

            elif method_label in ['Relevant Features']:
                print('Selecting relevant features')
                relevant_features_temp = method(X_temp, y, params)
                print(relevant_features_temp)
                print('')

                # Update feature matrix
                X_temp = X[relevant_features_temp]


            elif method_label in ['RFECV Features']:
                if X_temp.shape[1]>4:
                    print('Selecting RFECV features')
                    rfecv_features_temp, feature_selector = method(X_temp, y, params)
                    print(rfecv_features_temp)
                    print('')

                    # Save fitted rfecv
                    self.rfecv = feature_selector

                    # Update feature matrix
                    if len(rfecv_features_temp)>=4:
                        X_temp = X[rfecv_features_temp]

        # Save selected features
        self.selected_features = list(X_temp.columns)

        # Message to user
        message = 'Done selecting features'

        return (print(message))

    def transform(self, X):

        if self.selected_features == None:
            message = 'You first need to use the fit() method to determine the selected features!'
            return (print(message))
        else:
            # Get selected features
            X_selected = X[self.selected_features]
            return X_selected

    def get_methods(self, method_order):

        # Return feature selection methods in the order specified:
        ordered_methods = {}

        for method_label in method_order:

            if method_label == 'Constant Features':
                ordered_methods.update({method_label: constant_features})

            elif method_label == 'Correlated Features':
                ordered_methods.update({method_label: correlated_features})

            elif method_label == 'Relevant Features':
                ordered_methods.update({method_label: relevant_features})

            elif method_label == 'RFECV Features':
                ordered_methods.update({method_label: rfecv_features})

        return ordered_methods


def constant_features(X, frac_constant_values=0.90):


    # Get number of rows in X
    num_rows = X.shape[0]

    # Get column labels
    allLabels = X.columns.tolist()

    # Make a dict to store the fraction describing the value that occurs the most
    constant_per_feature = {label: X[label].value_counts().iloc[0] / num_rows for label in allLabels}

    # Determine the features that contain a fraction of missing values greater than
    # the specified threshold
    labels = [label for label in allLabels if constant_per_feature[label] > frac_constant_values]

    return labels


def correlated_features(X, correlation_threshold=0.90):

    # Make correlation matrix
    corr_matrix = X.corr(method="spearman").abs()

    # Select upper triangle of matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than correlation_threshold
    labels = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    return labels


def relevant_features(X, y, params):


    # Unpack params
    if 'cv' in params:
        cv = params['cv']
    else:
        cv = 5

    # Remove cv key from params so we can use with BorutaPy
    del params['cv']

    # Initiate variables
    feature_labels = list(X.columns)
    selected_features_mask = np.ones(len(feature_labels))
    counter = 0

    # Get K-folds indices
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)

    # Initiate progress bar
    status.printProgressBar(counter, cv, prefix='Progress:', suffix='Complete', length=50)

    # K-fold cross validation
    for train_index, val_index in kf.split(X):
        # Get train fold data
        X_train_fold = X.iloc[train_index, :]
        y_train_fold = y[train_index]

        # Define Boruta feature selection method
        feat_selector = BorutaPy(**params)

        # Find all relevant features
        feat_selector.fit(X_train_fold.values, y_train_fold)

        # Boruta selected feature mask
        selected_features_temp = feat_selector.support_

        # Update selected relevant features
        selected_features_mask = selected_features_mask + selected_features_temp

        # Update progress bar
        counter += 1
        status.printProgressBar(counter, cv, prefix='Progress:', suffix='Complete', length=50)

    # Boruta selected feature labels
    labels = [feature_labels[ii] for ii in range(len(feature_labels)) if selected_features_mask[ii] >= 3]
    if len(labels)<4:
        labels = [feature_labels[ii] for ii in range(len(feature_labels)) if selected_features_mask[ii] > 0]
    if len(labels) < 4:
        labels = labels + [feature_labels[ii] for ii in range(4)]
    return labels


def rfecv_features(X, y, rfecv_params):


    # Initialize RFECV object
    feature_selector = RFECV(**rfecv_params)

    # Fit RFECV
    feature_selector.fit(X, y)

    # Get selected features
    feature_labels = X.columns

    # Get selected features
    labels = feature_labels[feature_selector.support_].tolist()

    return labels, feature_selector


class status:
    """  Report progress of process. """

    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar

        Parameters
        ----------
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)

        Examples
        --------
        from time import sleep
        # A List of Items
        items = list(range(0, 57))
        l = len(items)

        # Initial call to print 0% progress
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i, item in enumerate(items):
            # Do stuff...
            sleep(0.1)
            # Update Progress Bar
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

        References
        ----------
        Original Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        """

        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

#
# def test_boruta(cvs, X_test1,  y_test1, cluster_dir):
#
#     logger = logging.getLogger('log_rbf_cnn_test.log')
#     logger.setLevel(logging.INFO)
#     handler = logging.FileHandler(os.path.join(cluster_dir, 'log_rbf_cnn_test.log'), 'a')
#     handler.setLevel(logging.INFO)
#
#     # create a logging format
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     # add the handlers to the logger
#     logger.addHandler(handler)
#
#     rated = None
#
#     static_data = write_database()
#
#     logger.info('Boruta Evaluation')
#     logger.info('/n')
#     method = 'svm'
#     model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
#     if model_sklearn.istrained == True:
#         model_sklearn.istrained = False
#     model_sklearn.train(cvs)
#     pred = model_sklearn.predict(X_test1)
#
#     metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
#     logger.info('before feature selection metrics')
#     logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
#
#
#     fs = FS(cluster_dir, static_data['sklearn']['njobs'])
#     features = fs.fit(cvs)
#     logger.info('Number of variables %s', str(features.shape[0]))
#     for i in range(3):
#         cvs[i][0] = cvs[i][0][:,features]
#         cvs[i][2] = cvs[i][2][:,features]
#         cvs[i][4] = cvs[i][4][:,features]
#
#     method = 'svm'
#     model_sklearn = sklearn_model(cluster_dir, rated, method, static_data['sklearn']['njobs'])
#     if model_sklearn.istrained == True:
#         model_sklearn.istrained = False
#     model_sklearn.train(cvs)
#     pred = model_sklearn.predict(X_test1[:,features])
#
#     metrics_svm = model_sklearn.compute_metrics(pred, y_test1, rated)
#     logger.info('After feature selection metrics')
#     logger.info('sse, %s rms %s, mae %s, mse %s', *metrics_svm)
#
