import logging
import multiprocessing as mp
import os
import pickle
import random
import warnings
from collections import Sequence
from itertools import repeat

import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=RuntimeWarning)
import joblib
from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def rbf_optim_single(net, X, y, X_val, y_val, X_test, y_test, width, keep, rated):
    err_train = net.train_ols(X, y, X_val, y_val, X_test, y_test, rated, gw=width, keep=keep, verbose=False)
    pred_test = net.predict(X_test)
    pred_val = net.predict(X_val)
    sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
    sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
    metrics = np.array([0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
                        0.6 * mse_test + 0.4 * mse_val])
    return (metrics, {'centroids': net.model['centroids'], 'Radius': net.model['Radius'], 'W': net.model['W'],
                      'acc': 0.6 * mae_test + 0.4 * mae_val})


class rbf_ols():
    def __init__(self):
        self.model = dict()

    def predict(self, x):
        return self._predict(self.model, x)

    def _predict(self, model, x):
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 2.))
        v = np.matmul(v, model['W'][:-1]) + model['W'][-1]
        return v

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return sse, rms, mae, mse

    def _distance(self, obj_new, obj_old, obj_max, obj_min):
        if np.any(np.isinf(obj_old)):
            obj_old = obj_new.copy()
            obj_max = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
            obj_min = obj_new.copy()
        d = 0
        for i in range(obj_new.shape[0]):
            if obj_max[i] < obj_new[i]:
                obj_max[i] = obj_new[i]
            if obj_min[i] > obj_new[i]:
                obj_min[i] = obj_new[i]

            d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        if d < 0:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def train_ols(self, I, O, X_val, y_val, X_test, y_test, rated, gw=1.0, keep=3, verbose=False):

        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        model = dict()
        model_temp = dict()
        remains = 0
        mse_best = np.inf
        k = np.sqrt(-np.log(0.5)) / gw
        m, d = O.shape
        d *= m
        idx = np.arange(m)
        P = np.exp(-(np.sqrt((((I[np.newaxis, :] - I[:, np.newaxis]) * k) ** 2.0).sum(-1))) ** 2.0)
        G = np.array(P)
        D = (O * O).sum(0)
        e = (((np.dot(P.T, O) ** 2.) / ((P * P).sum(0)[:, np.newaxis] * D)) ** 2.).sum(1)
        next = e.argmax()
        used = np.array([next])
        idx = np.delete(idx, next)
        W = P[:, next, np.newaxis]
        P = np.delete(P, next, 1)
        G1 = G[:, used]
        v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * k
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 2.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
        except:
            out_layer = np.inf * np.ones(v.shape[1] + 1).reshape(-1, 1)
        centers = I[used]
        ibias = k
        model_temp['centroids'] = I[used]
        model_temp['W'] = out_layer
        model_temp['Radius'] = k

        pred_test = self._predict(model_temp, X_test)
        pred_val = self._predict(model_temp, X_val)
        sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
        metrics = np.array(
            [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
             0.6 * mse_test + 0.4 * mse_val])

        flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
        if not flag:
            remains += 1
        else:
            remains = 0
            centers = I[used]
            ibias = k
        while remains < keep and P.shape[1] > 0 and used.shape[0] <= 48:
            if verbose:
                print(0.6 * mae_test + 0.4 * mae_val, m - P.shape[1])
            wj = W[:, -1:]
            a = np.dot(wj.T, P) / np.dot(wj.T, wj)
            P = P - wj * a
            not_zero = np.ones((P.shape[1])) * np.finfo(np.float64).eps
            e = (((np.dot(P.T, O) ** 2.) / ((P * P).sum(0)[:, np.newaxis] * D + not_zero)) ** 2.).sum(1)
            next = e.argmax()
            W = np.append(W, P[:, next, np.newaxis], axis=1)
            used = np.append(used, idx[next])
            P = np.delete(P, next, 1)
            idx = np.delete(idx, next)
            v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * ibias
            v = np.sqrt((v ** 2.).sum(-1))
            v = np.exp(-(v ** 2.))
            try:
                out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            except:
                out_layer = np.inf * np.ones(v.shape[1] + 1).reshape(-1, 1)
            model_temp['centroids'] = I[used]
            model_temp['W'] = out_layer
            model_temp['Radius'] = k
            pred_test = self._predict(model_temp, X_test)
            pred_val = self._predict(model_temp, X_val)
            sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
            sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
            metrics = np.array(
                [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
                 0.6 * mse_test + 0.4 * mse_val])

            flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
            if not flag:
                remains += 1
            else:
                remains = 0
                centers = I[used]
                ibias = k
        if centers.shape[0] < keep:
            centers = I[used]
            ibias = k

        if verbose:
            print(0.6 * mae_test + 0.4 * mae_val, m - P.shape[1])
        model['centroids'] = centers
        model['Radius'] = ibias
        x = np.vstack((I, X_val, X_test))
        y = np.vstack((O, y_val, y_test))
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 2.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), y, rcond=None)[0]
        except:
            out_layer = np.inf * np.ones(v.shape[1] + 1).reshape(-1, 1)
        model['W'] = out_layer
        pred_test = self._predict(model, X_test)
        pred_val = self._predict(model, X_val)
        sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
        metrics = np.array(
            [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
             0.6 * mse_test + 0.4 * mse_val])
        self.model = model
        self.ncenters = model['centroids'].shape[0]
        self.err = 0.6 * mae_test + 0.4 * mae_val
        return metrics


class rbf_ols_module(object):
    def __init__(self, static_data, cluster_dir, rated, njobs, GA=False, path_group=None):
        self.static_data = static_data
        self.cluster = [p1 for p1 in cluster_dir.split('/') if ('rule' in p1) or ('global' in p1)][0].split('\\')[0]
        self.path_group = path_group
        self.njobs = njobs
        self.rated = rated
        self.GA = GA
        self.istrained = False
        if GA == False:
            self.model_dir = os.path.join(cluster_dir, 'RBF_OLS')
        else:
            self.model_dir = os.path.join(cluster_dir, 'GA_RBF_OLS')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

    def compute_metrics(self, pred, y, rated):
        if rated == None:
            rated = y.ravel()
        else:
            rated = 1
        err = np.abs(pred.ravel() - y.ravel()) / rated
        sse = np.sum(np.square(pred.ravel() - y.ravel()))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return [sse, rms, mae, mse]

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        self.load(self.model_dir)
        model = self.models
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 2.))
        v = np.matmul(v, model['W'][:-1]) + model['W'][-1]
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

    def save(self, pathname):
        f = open(os.path.join(pathname, 'rbf_ols' + '.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

    def _distance(self, obj_new, obj_old, obj_max, obj_min):
        if np.any(np.isinf(obj_old)):
            obj_old = obj_new.copy()
            obj_max = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
            obj_min = obj_new.copy()
        d = 0
        for i in range(obj_new.shape[0]):
            if obj_max[i] < obj_new[i]:
                obj_max[i] = obj_new[i]
            if obj_min[i] > obj_new[i]:
                obj_min[i] = obj_new[i]

            d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        if d < 0:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def optimize_rbf(self, cvs, max_samples=3000):
        X = cvs[0][0]
        y = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)
        if X.shape[0] > max_samples:
            ind = np.random.randint(low=0, high=X.shape[0], size=max_samples)
            X = X[ind]
            y = y[ind]
        if self.GA == True:
            with threadpool_limits(limits=1):
                self.optimize_with_deep(X, y, X_val, y_val, X_test, y_test, self.rated)
        else:
            with threadpool_limits(limits=1):
                self.optimize_common_width(X, y, X_val, y_val, X_test, y_test, self.rated)
        return self.to_dict()

    def optimize_rbf_TL(self, cvs, models, max_samples=3000):
        X = cvs[0][0]
        y = cvs[0][1].reshape(-1, 1)
        X_val = cvs[0][2]
        y_val = cvs[0][3].reshape(-1, 1)
        X_test = cvs[0][4]
        y_test = cvs[0][5].reshape(-1, 1)
        if X.shape[0] > max_samples:
            ind = np.random.randint(low=0, high=X.shape[0], size=max_samples)
            X = X[ind]
            y = y[ind]
        if self.GA == True:
            self.optimize_with_deep_TL(X, y, X_val, y_val, X_test, y_test, self.rated, models)
        else:
            self.optimize_common_width_TL(X, y, X_val, y_val, X_test, y_test, self.rated, models)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        return dict

    def optimize_common_width(self, X, y, X_val, y_val, X_test, y_test, rated):
        logger = logging.getLogger('optimize_common_width')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info('RBF single width optimization running for rule %s....', self.cluster)
        print('RBF single width optimization running for rule ', self.cluster, ' .....')
        net = rbf_ols()
        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        widths = np.arange(0.2, 12.8, 0.02)

        ncpus = joblib.load(os.path.join(self.path_group, 'total_cpus.pickle'))
        gpu_status = joblib.load(os.path.join(self.path_group, 'gpu_status.pickle'))

        njobs = int(ncpus - gpu_status)
        cpu_status = njobs
        joblib.dump(cpu_status, os.path.join(self.path_group, 'cpu_status.pickle'))

        pool = mp.Pool(njobs)
        result = [pool.apply_async(rbf_optim_single, args=(net, X, y, X_val, y_val, X_test, y_test, w, 4, rated)) for w
                  in widths]

        results = [p.get() for p in result]
        pool.close()
        pool.terminate()
        pool.join()

        # results = Parallel(n_jobs=self.njobs)(
        #     delayed(rbf_optim_single)(net, X, y, X_val, y_val, X_test, y_test, w, 4, rated) for
        #     w in widths)

        res = dict()
        for p in results:
            flag, obj_old, obj_max, obj_min = self._distance(p[0], obj_old, obj_max, obj_min)
            res[str(p[1]['Radius'])] = p[1]['acc']
            if flag:
                self.models = p[1]

        res = pd.DataFrame.from_dict(res, orient='index')
        res.to_csv(os.path.join(self.model_dir, 'single_rbf_res.csv'))
        p = self.models
        logger.info('Best model Radius %s centers %s acc %s', p['Radius'], p['centroids'].shape[0], p['acc'])
        self.istrained = True
        self.save(self.model_dir)
        logger.info('\n')

    def optimize_common_width_TL(self, X, y, X_val, y_val, X_test, y_test, rated, models):

        print('RBF single width optimization running for rule ', self.cluster, ' .....')
        net = rbf_ols()

        results = rbf_optim_single(net, X, y, X_val, y_val, X_test, y_test, models['models']['Radius'], 4, rated)

        self.models = results[1]

        self.istrained = True
        self.save(self.model_dir)

    def optimize_with_deep_TL(self, X, y, X_val, y_val, X_test, y_test, rated, models):

        net = rbf_ols()

        err_train = net.train_ols(X, y, X_val, y_val, X_test, y_test, rated, gw=models['models']['Radius'], keep=4,
                                  verbose=True)

        pred_test = net.predict(X_test)
        pred_val = net.predict(X_val)
        sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
        self.models = ({'centroids': net.model['centroids'], 'Radius': net.model['Radius'], 'W': net.model['W'],
                        'acc': float(0.6 * mae_test + 0.4 * mae_val)})

        self.istrained = True
        self.save(self.model_dir)

    def optimize_with_deep(self, X, y, X_val, y_val, X_test, y_test, rated):
        logger = logging.getLogger('optimize_with_deep')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster + '.log'), 'a')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info('GA rbf running for rule %s....', self.cluster)
        print('GA rbf running for rule ', self.cluster, '....')

        def checkBounds(mn, mx):
            def decorator(func):
                def wrappper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for i in range(len(child)):
                            if child[i] > mx:
                                child[i] = mx
                            elif child[i] < mn:
                                child[i] = mn
                    return offspring

                return wrappper

            return decorator

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        # pool = mp.Pool(self.njobs)
        # toolbox.register("map", pool.map)
        toolbox.register("attribute1", random.uniform, 0.005, 5.5)
        toolbox.register("individual1", tools.initRepeat, creator.Individual, toolbox.attribute1, n=X.shape[1])
        toolbox.register("attribute2", random.uniform, 0.001, 12)
        toolbox.register("individual2", tools.initRepeat, creator.Individual, toolbox.attribute2, n=X.shape[1])
        toolbox.register("population", tools.initCycle, list, (toolbox.individual1, toolbox.individual2), n=50)

        toolbox.register("mate", cx_fun, alpha=0.5)
        toolbox.register("mutate", mut_fun, mu=0, sigma=.5, eta=0.8, low=0.0001, up=12, indpb=0.8)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", rbf_optim)
        hof = tools.ParetoFront(lambda x, y: (x == y).all())
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)
        toolbox.decorate("mate", checkBounds(0.0001, 12))
        toolbox.decorate("mutate", checkBounds(0.0001, 12))
        net = rbf_ols()
        population, logbook = optimize(toolbox.population(), toolbox, 40, 60, net, X, y, X_val, y_val, X_test, y_test,
                                       rated, self.njobs, self.path_group, cxpb=0.6, mutpb=0.4,
                                       ngen=64, stats=stats, halloffame=hof, verbose=False)
        logbook = pd.DataFrame(logbook)
        logbook.to_csv(os.path.join(self.model_dir, 'GA_results.csv'))

        self.best_pop = [np.array(p) for p in population]

        fits = np.array([p1.fitness.getValues() for p1 in population])
        fits /= np.nanmax(fits, axis=0)
        fits = np.nansum(fits, axis=1)
        fits[fits == 0] = np.inf
        best_ind = fits.argmin()
        population[best_ind].fitness.setValues([np.inf, np.inf, np.inf, np.inf])

        net = rbf_ols()

        err_train = net.train_ols(X, y, X_val, y_val, X_test, y_test, rated, gw=np.array(population[best_ind]), keep=4,
                                  verbose=True)

        pred_test = net.predict(X_test)
        pred_val = net.predict(X_val)
        sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
        logger.info('Best model centers %s acc %s', net.model['centroids'].shape[0],
                    float(0.6 * mae_test + 0.4 * mae_val))
        self.models = ({'centroids': net.model['centroids'], 'Radius': net.model['Radius'], 'W': net.model['W'],
                        'acc': float(0.6 * mae_test + 0.4 * mae_val)})

        self.istrained = True
        self.save(self.model_dir)
        logger.info('\n')


def cx_fun(ind1, ind2, alpha):
    if random.random() > 0.5:
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    else:
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            gamma = (1. + 2. * alpha) * random.random() - alpha
            ind1[i] = (1. - gamma) * x1 + gamma * x2
            ind2[i] = gamma * x1 + (1. - gamma) * x2

    return ind1, ind2


def mut_fun(individual, mu, sigma, eta, low, up, indpb):
    if random.random() > 0.5:

        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)
    else:
        size = len(individual)
        if not isinstance(low, Sequence):
            low = repeat(low, size)
        elif len(low) < size:
            raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
        if not isinstance(up, Sequence):
            up = repeat(up, size)
        elif len(up) < size:
            raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

        for i, xl, xu in zip(range(size), low, up):
            if random.random() <= indpb:
                x = individual[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    if xy < 0:
                        xy = 1e-6
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    if xy < 0:
                        xy = 1e-6
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                individual[i] = x
    return individual,


def rbf_optim(individual, net, X, y, X_val, y_val, X_test, y_test, rated):
    err_train = net.train_ols(X, y, X_val, y_val, X_test, y_test, rated, gw=np.array(individual), keep=4, verbose=False)

    pred_test = net.predict(X_test)
    pred_val = net.predict(X_val)
    sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
    sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
    metrics = np.array([0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
                        0.6 * mse_test + 0.4 * mse_val])
    return (metrics)


def optimize(population, toolbox, mu, lambda_, net, X, y, X_val, y_val, X_test, y_test, rated, njobs, path_group, cxpb,
             mutpb, ngen,
             stats=None, halloffame=None, verbose=__debug__):
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # fitnesses=[]
    # for individual in invalid_ind:
    #     m=rbf_optim(individual, X, y, X_val, y_val, X_test, y_test)
    #     fitnesses.append(m)

    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

    pool = mp.Pool(njobs)
    results = [
        pool.apply_async(rbf_optim, args=(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated))
        for individual in invalid_ind]
    fitnesses = [p.get() for p in results]
    pool.close()
    pool.terminate()
    pool.join()
    # fitnesses = Parallel(n_jobs=njobs)(delayed(rbf_optim)(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated) for
    #                                    individual in invalid_ind)

    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    # Gather all the fitnesses in one list and compute the stats
    fits = np.array([ind.fitness.values for ind in population])

    maximums = np.nanmax(fits, axis=0)
    minimums = np.nanmin(fits, axis=0)
    logbook.header = ['gen', 'nevals'] + ['Max_mae:', 'Min_mae:', 'Max_rms:', 'Min_rms:']

    record = {'Max_mae:': maximums[0], 'Min_mae:': minimums[0], 'Max_rms:': maximums[2], 'Min_rms:': minimums[2]}
    print('GA rbf running generation 0')
    print(record)

    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    with elapsed_timer() as eval_elapsed:
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
            gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

            njobs = int(ncpus - gpu_status)
            cpu_status = njobs
            joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

            pool = mp.Pool(njobs)
            results = [pool.apply_async(rbf_optim,
                                        args=(
                                        np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated))
                       for individual in invalid_ind]
            fitnesses = [p.get() for p in results]
            pool.close()
            pool.terminate()
            pool.join()
            # fitnesses = Parallel(n_jobs=njobs)(
            #     delayed(rbf_optim)(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated) for
            #     individual in invalid_ind)
            # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fits = np.array([ind.fitness.values for ind in population])

            maximums = np.nanmax(fits, axis=0)
            minimums = np.nanmin(fits, axis=0)
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = {'Max_mae:': maximums[0], 'Min_mae:': minimums[0], 'Max_rms:': maximums[2],
                      'Min_rms:': minimums[2]}
            print('GA rbf running generation ', str(gen))
            print(record)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            if eval_elapsed() > 1800:
                break
    return population, logbook
