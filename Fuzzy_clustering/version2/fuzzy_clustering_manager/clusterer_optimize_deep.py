import copy
import os
import pickle
import random
import warnings
from collections import Sequence
from itertools import repeat

import joblib
import logging
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from joblib import Parallel
from joblib import delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_EVALUATIONS = 30000
JOBS = 5
POPULATION_SIZE = 50


class cluster_optimize():

    def __init__(self, static_data):
        self.istrained = False
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
        logger = logging.getLogger('log_fuzzy.log')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.path_fuzzy, 'log_fuzzy.log'), 'w')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
        self.logger = logger

    def create_mfs(self, model_mfs, var_name, num_mf, old_num_mf):
        mfs = []
        var_range = [-0.005, 1.005]
        if var_name in {'hdd_h', 'temp_max', 'flux', 'wind', 'temp', 'Temp', 'load', 'power'}:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
            std = 1.25 * var_range[1] / num_mf
            for i in range(num_mf):
                mfs.append({'name': 'mf_' + var_name + str(old_num_mf + i),
                            'var_name': var_name,
                            'prange': std,
                            'type': 'gauss',
                            'param': [mean[i], std / 2],
                            'universe': np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                            'func': fuzz.gaussmf(np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                                                 mean[i], std)})
        elif var_name in {'sp_index', 'dayweek', 'cloud', 'hour', 'month', 'direction', 'sp_days'}:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
            std = 1.25 * var_range[1] / num_mf
            std1 = 1.125 * var_range[1] / (num_mf)
            for i in range(num_mf):
                param = [mean[i] - std, mean[i] - std1, mean[i] + std1, mean[i] + std]
                mfs.append({'name': 'mf_' + var_name + str(old_num_mf + i),
                            'var_name': var_name,
                            'prange': std,
                            'type': 'trap',
                            'param': param,
                            'universe': np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001),
                            'func': fuzz.trapmf(np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001, ),
                                                param)})
        else:
            raise NameError('MF type not recognize')
        model_mfs[var_name] = mfs
        return model_mfs

    def run(self, X_train, y_train, X_test, y_test, rated, num_samples=500, n_ratio=0.4, ngen=300):
        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
        y_test = scale_y.inverse_transform(y_test.values).ravel()
        if rated is None:
            rated = y_test
        else:
            rated = rated
        self.n_ratio = n_ratio

        if not os.path.exists(os.path.join(self.path_fuzzy, 'models.pickle')):
            fuzzy_models = []
            self.p_list = []
            for case in self.static_data['clustering']['var_imp']:
                self.base_name = [v for v in sorted(case.keys())][0]
                self.var_names = [v for v in sorted(case[self.base_name][0].keys())]
                self.num_base_mfs = len(case[self.base_name])
                self.base_mfs = dict()
                self.base_mfs = self.create_mfs(self.base_mfs, self.base_name, self.num_base_mfs, 0)

                fuzzy_model = dict()
                fuzzy_model['mfs'] = dict()
                for n in range(len(self.base_mfs[self.base_name])):
                    fuzzy_model['mfs'][self.base_name + str(n)] = dict()
                    fuzzy_model['mfs'][self.base_name + str(n)][self.base_name] = [self.base_mfs[self.base_name][n]]

                for var in self.var_names:
                    old_num_mf = 0
                    for n, base_case in enumerate(case[self.base_name]):
                        n_mf = base_case[var]['mfs']
                        fuzzy_model['mfs'][self.base_name + str(n)] = self.create_mfs(
                            fuzzy_model['mfs'][self.base_name + str(n)]
                            , var, n_mf, old_num_mf)
                        old_num_mf += n_mf

                self.var_lin = self.static_data['clustering']['var_lin']

                self.p = len(self.var_names) + 1
                self.p_list.append(self.p)
                var_del = []
                for var in self.var_names + [self.base_name]:
                    if var not in X_train.columns:
                        var_names = [c for c in X_train.columns if var in c]
                        X_train[var] = X_train[var_names].mean(axis=1)
                        X_test[var] = X_test[var_names].mean(axis=1)
                        var_del.append(var)
                    if var not in self.var_lin:
                        self.var_lin.append(var)
                lin_models = ElasticNetCV(cv=5).fit(X_train[self.var_lin].values, y_train.values.ravel())
                preds = scale_y.inverse_transform(
                    lin_models.predict(X_test[self.var_lin].values).reshape(-1, 1)).ravel()

                err = (preds.ravel() - y_test) / rated

                self.rms_before = np.sum(np.square(err))
                self.mae_before = np.mean(np.abs(err))
                print('rms = %s', self.rms_before)
                print('mae = %s', self.mae_before)
                self.logger.info("Objective before train: %s", self.mae_before)
                problem = cluster_problem(fuzzy_model['mfs'], X_train[self.var_lin], self.p, rated, self.resampling,
                                          self.add_individual_rules, self.logger, self.njobs, num_samples=num_samples,
                                          n_ratio=self.n_ratio)

                problem.run(X_train[self.var_lin], y_train, X_test[self.var_lin], y_test, scale_y, fuzzy_model['mfs'],
                            75, 100, ngen=ngen)

                fuzzy_model = problem.fmodel
                self.logger.info("Objective after train: %s", str(fuzzy_model['result']))
                fuzzy_model['p'] = self.p
                fuzzy_models.append(fuzzy_model)
            self.fuzzy_models = fuzzy_models
            joblib.dump(fuzzy_models, os.path.join(self.path_fuzzy, 'models.pickle'))
        else:
            self.fuzzy_models = joblib.load(os.path.join(self.path_fuzzy, 'models.pickle'))
        fmodel = dict()
        fmodel['mfs'] = dict()
        fmodel['rules'] = dict()
        fmodel['result'] = []
        num = 0
        for fuzzy_model in self.fuzzy_models:
            i = 0
            for var in fuzzy_model['mfs'].keys():
                fmodel['mfs'][var] = fuzzy_model['mfs'][var]
            for rule in fuzzy_model['rules']:
                fmodel['rules']['rule.' + str(num + i)] = fuzzy_model['rules'][rule]
                for mf in range(len(fmodel['rules']['rule.' + str(num + i)])):
                    fmodel['rules']['rule.' + str(num + i)][mf]['p'] = fuzzy_model['p']
                i += 1
            fmodel['result'].append(fuzzy_model['result'])
            num += len(fuzzy_model['rules'])

        self.best_fuzzy_model = copy.deepcopy(fmodel)
        joblib.dump(self.best_fuzzy_model, os.path.join(self.path_fuzzy, self.file_fuzzy))
        if 'horizon' in self.import_external_rules:
            self.compact_external_mfs()
        if len(var_del) > 0:
            X_train = X_train.drop(columns=var_del)
            X_test = X_test.drop(columns=var_del)
        self.istrained = True
        self.save()

    def compact_external_mfs(self):
        self.fuzzy_file = os.path.join(self.path_fuzzy, self.file_fuzzy)
        fmodel = joblib.load(self.fuzzy_file)
        type_mf = 'horizon'
        var_name = 'horizon'
        params = [
            [0.5, 0.9, 1.1, 1.5],
            [1.5, 1.9, 2.1, 2.5],
            [2.5, 2.9, 3.1, 3.5],
            [3.5, 3.9, 4.1, 4.5],
            [4.5, 4.9, 5.1, 5.5],
            [5.5, 5.9, 6.1, 6.5],
            [6.5, 6.9, 7.1, 7.5],
            [7.5, 7.9, 8.1, 8.5],
            [8.5, 8.9, 12.1, 15.5],
            [12, 13.2, 22.1, 27.5],
            [22.1, 25.2, 36.1, 42.5],
            [38.1, 42.2, 48.1, 52.5],
        ]
        mfs = []
        i = 0
        for param in params:
            mfs.append({'name': 'mf_' + type_mf + str(i),
                        'var_name': var_name,
                        'type': 'trap',
                        'param': param,
                        'universe': np.arange(0, 49, .01),
                        'func': fuzz.trapmf(np.arange(0, 49, .01), param)})
            i += 1
        fmodel['mfs']['horizon'] = mfs
        i = 0
        rules = dict()
        for mf in mfs:
            for rule in fmodel['rules']:
                rules['rule.' + str(i)] = fmodel['rules'][rule] + [mf]
                i += 1

        fmodel['rules'] = rules
        joblib.dump(fmodel, os.path.join(self.path_fuzzy, self.file_fuzzy))

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict = {}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')

    def save(self):
        f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()


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
    if random.random() > 0.75:

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

                x = x + delta_q * (xu - xl) / 2
                x = min(max(x, xl), xu)
                individual[i] = x
    return individual,


def checkBounds(mn, mx):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > mx[i]:
                        child[i] = mx[i]
                    elif child[i] < mn[i]:
                        child[i] = mn[i]
            return offspring

        return wrappper

    return decorator


class cluster_problem():

    def __init__(self, mfs, X, p, rated, resampling, add_individual_rules, logger, njobs, num_samples=500, n_ratio=0.4):
        self.logger = logger
        self.njobs = njobs
        self.resampling = resampling
        self.add_individual_rules = add_individual_rules
        self.num_samples = num_samples
        self.n_ratio = n_ratio
        self.mfs = mfs
        self.rated = rated
        self.p = p
        self.rules = dict()
        for base_case in self.mfs.keys():
            self.rules = self.create_rules(self.rules, self.mfs[base_case])
        x = []
        self.lower_bound = []
        self.upper_bound = []
        self.sigma = []
        self.index_constrains = []
        self.number_of_constraints = 0
        for rule_name, rule in sorted(self.rules.items()):
            for mf in rule:
                param = mf['param']
                xrange = [mf['universe'][0], mf['universe'][-1]]
                prange = mf['prange']
                x = x + param
                if len(param) == 2:
                    self.index_constrains.append(np.arange(len(x) - 2, len(x)))
                    self.number_of_constraints = self.number_of_constraints + 3

                    lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                    up = param[0] + prange if (param[0] + prange) < xrange[1] else xrange[1]

                    self.lower_bound.extend([lo, 0.0001])
                    self.upper_bound.extend([up, prange])
                    self.sigma.extend([prange, prange])
                elif len(param) == 4:
                    self.index_constrains.append(np.arange(len(x) - 4, len(x)))
                    self.number_of_constraints = self.number_of_constraints + 7
                    for i in param:
                        lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                        up = param[3] + prange if (param[3] + prange) < xrange[1] else xrange[1]
                        self.lower_bound.append(lo)
                        self.upper_bound.append(up)
                        self.sigma.append(prange)
        self.number_of_variables = len(x)
        self.number_of_objectives = 2
        self.x = x
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()

        attributes = []
        for i in range(self.number_of_variables):
            self.toolbox.register("attribute" + str(i), random.gauss, self.lower_bound[i], self.upper_bound[i])
            attributes.append(self.toolbox.__getattribute__("attribute" + str(i)))

        self.toolbox.register("individual1", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual1, n=100)

        self.toolbox.register("mate", cx_fun, alpha=0.5)
        self.toolbox.register("mutate", mut_fun, mu=0, sigma=self.sigma, eta=0.8, low=self.lower_bound,
                              up=self.upper_bound, indpb=0.6)
        self.toolbox.register("select", tools.selTournament, tournsize=4)
        self.toolbox.register("evaluate", evaluate)
        self.hof = tools.ParetoFront(lambda x, y: (x == y).all())
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("Avg", np.mean)
        self.stats.register("Std", np.std)
        self.stats.register("Min", np.min)
        self.stats.register("Max", np.max)
        self.toolbox.decorate("mate", checkBounds(self.lower_bound, self.upper_bound))
        self.toolbox.decorate("mutate", checkBounds(self.lower_bound, self.upper_bound))

    def create_rules(self, final_rules, model_mfs):

        rules = []
        for mf in sorted(model_mfs.keys()):
            if len(rules) == 0:
                for f in model_mfs[mf]:
                    rules.append([f])
            else:
                new_rules = []
                for rule in rules:
                    for f in model_mfs[mf]:
                        new_rules.append(rule + [f])
                rules = new_rules

        if self.add_individual_rules:
            for mf in sorted(model_mfs.keys()):
                for f in model_mfs[mf]:
                    rules.append([f])

        n_old_rules = len(final_rules)
        for i in range(len(rules)):
            final_rules['rule.' + str(n_old_rules + i)] = rules[i]

        return final_rules

    def run(self, X, y, X_test, y_test, scale_y, mfs, mu, lambda_, cxpb=0.6, mutpb=0.4, ngen=300):
        perf = np.inf
        front_best = None
        rules = copy.deepcopy(self.rules)
        self.population = self.toolbox.population()

        param_ind = creator.Individual(self.x)
        self.population.pop()
        self.population.insert(len(self.population), param_ind)
        i = 0
        while i < 0.5 * len(self.population):
            param_ind = mut_fun(self.x, 0, self.sigma, 0.8, self.lower_bound, self.upper_bound, 0.6)
            param_ind = creator.Individual(param_ind[0])
            self.population.pop(i)
            self.population.insert(i, param_ind)
            i += 1
        assert lambda_ >= mu, "lambda must be greater or equal to mu."

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        rules = copy.deepcopy(self.rules)
        fit1 = evaluate(np.array(invalid_ind[-1]).ravel(),
                        X, y, X_test, y_test, scale_y, self.rated,
                        mfs, rules, self.p, self.resampling, self.num_samples, self.n_ratio)
        print('initial candidate error ', fit1[1])
        rules = copy.deepcopy(self.rules)
        fitnesses = Parallel(n_jobs=self.njobs)(delayed(evaluate)(np.array(individual).ravel(),
                                                                  X, y, X_test, y_test, scale_y, self.rated,
                                                                  mfs, rules, self.p, self.resampling, self.num_samples,
                                                                  self.n_ratio) for individual in invalid_ind)

        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(self.population)

        self.logbook = tools.Logbook()
        # Gather all the fitnesses in one list and compute the stats
        fits = np.array([ind.fitness.values for ind in self.population])

        maximums = np.nanmax(fits, axis=0)
        minimums = np.nanmin(fits, axis=0)
        self.logbook.header = ['gen', 'nevals'] + ['Max_sse:', 'Min_sse:', 'Max_mae:', 'Min_mae:']
        self.logger.info('Iter: %s, Max_sse: %s, Min_mae: %s', 0, *minimums)
        record = {'Max_sse:': maximums[0], 'Min_sse:': minimums[0], 'Max_mae:': maximums[1], 'Min_mae:': minimums[1]}
        print('GA rbf running generation 0')
        print(record)

        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)

        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            rules = copy.deepcopy(self.rules)
            offspring = algorithms.varOr(self.population, self.toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=self.njobs)(delayed(evaluate)(np.array(individual).ravel(),
                                                                      X, y, X_test, y_test, scale_y, self.rated,
                                                                      mfs, rules, self.p, self.resampling,
                                                                      self.num_samples, self.n_ratio) for
                                                    individual in invalid_ind)
            # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fits = np.array([ind.fitness.values for ind in self.population])

            maximums = np.nanmax(fits, axis=0)
            minimums = np.nanmin(fits, axis=0)
            # Update the hall of fame with the generated individuals
            if self.hof is not None:
                self.hof.update(self.population)

            # Select the next generation population
            self.population[:] = self.toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = {'Max_sse:': maximums[0], 'Min_sse:': minimums[0], 'Max_mae:': maximums[1],
                      'Min_mae:': minimums[1]}

            print('GA rbf running generation ', str(gen))
            print(record)
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            front = self.population
            for i in range(len(front)):
                if front[i].fitness.getValues()[0] < perf:
                    front_best = front[i]
                    perf = front[i].fitness.getValues()[0]
        self.logger.info('Iter: %s, Max_sse: %s, Min_mae: %s', str(gen), *minimums)
        self.fmodel = self.evaluate(np.array(front_best).ravel(), X, y, X_test, y_test, scale_y, self.rated,
                                    mfs, self.rules, self.p, self.resampling)

    def evaluate(self, x, X, y, X_test, y_test, scale_y, rated, mfs, rules, p, resampling):

        # print(solution.variables)
        i = 0
        for rule_name, rule in sorted(rules.items()):
            for mf in rule:
                if mf['type'] == 'gauss':
                    mf['param'] = x[i:i + 2]
                    mf['func'] = fuzz.gaussmf(mf['universe'],
                                              mf['param'][0],
                                              np.abs(mf['param'][1]))
                    i += 2
                elif mf['type'] == 'trap':
                    mf['param'] = sorted(x[i:i + 4])
                    mf['func'] = fuzz.trapmf(mf['universe'], mf['param'])
                    i += 4

        activations = pd.DataFrame(index=X.index, columns=[rule for rule in sorted(rules.keys())])
        for rule in sorted(rules.keys()):
            act = []
            for mf in rules[rule]:
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
            activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

        lin_models = dict()
        remove_null_rules = []
        total = 0
        for rule in sorted(activations.columns):
            indices = activations[rule].index[activations[rule] >= 0.01].tolist()

            if len(indices) > self.num_samples and len(indices) < self.n_ratio * X.shape[0]:
                X1 = X.loc[indices].values
                y1 = y.loc[indices].values

                lin_models[rule] = LinearRegression().fit(X1, y1.ravel())

            else:
                act = activations.loc[indices].copy(deep=True)
                act = act.drop(columns=[rule])
                if not act.isnull().all(axis=1).any():
                    del rules[rule]
                else:
                    raise ValueError('Cannot remove rule ', rule)
            print(len(indices))
            self.logger.info("Number of samples of rule %s is %s", rule, len(indices))
            total += len(indices)
        print(total)
        self.logger.info("Number of samples of dataset with %s is %s", X.shape[0], total)

        activations_test = pd.DataFrame(index=X_test.index,
                                        columns=[rule for rule in sorted(rules.keys())])
        for rule in sorted(rules.keys()):
            act = []
            for mf in rules[rule]:
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X_test[mf['var_name']]))
            activations_test[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

        preds = pd.DataFrame(index=X_test.index, columns=sorted(lin_models.keys()))
        for rule in sorted(rules.keys()):
            indices = activations_test[rule].index[activations_test[rule] >= 0.01].tolist()
            if len(indices) != 0:
                X1 = X_test.loc[indices].values
                preds.loc[indices, rule] = scale_y.inverse_transform(
                    lin_models[rule].predict(X1).reshape(-1, 1)).ravel()

        pred = preds.mean(axis=1)
        # pred.name='target'
        # pred=pred.to_frame()
        err = (pred.values.ravel() - y_test.ravel()) / rated

        self.objectives = [np.sum(np.square(err)), np.mean(np.abs(err))]
        self.rules = rules
        self.mfs = mfs
        fmodel = dict()
        fmodel['mfs'] = self.mfs
        fmodel['rules'] = self.rules
        fmodel['result'] = self.objectives[1]
        print('Error = ', self.objectives[1])
        return fmodel


def evaluate(x, X, y, X_test, y_test, scale_y, rated, mfs, rules, p, resampling, num_samples=500, n_ratio=0.33):
    # print(solution.variables)
    i = 0
    for rule_name, rule in sorted(rules.items()):
        for mf in rule:
            if mf['type'] == 'gauss':
                mf['param'] = x[i:i + 2]
                mf['func'] = fuzz.gaussmf(mf['universe'],
                                          mf['param'][0],
                                          np.abs(mf['param'][1]))
                i += 2
            elif mf['type'] == 'trap':
                mf['param'] = sorted(x[i:i + 4])
                mf['func'] = fuzz.trapmf(mf['universe'], mf['param'])
                i += 4

    activations = pd.DataFrame(index=X.index, columns=[rule for rule in rules.keys()])
    for rule in sorted(rules.keys()):
        act = []
        for mf in rules[rule]:
            act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
        activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

    lin_models = dict()
    for rule in activations.columns:
        indices = activations[rule].index[activations[rule] >= 0.01].tolist()
        if len(indices) > num_samples and len(indices) < n_ratio * X.shape[0]:
            X1 = X.loc[indices].values
            y1 = y.loc[indices].values
            # if resampling:
            #     if X1.shape[0] < 300:
            #         X1, y1 = resampling_fun(X1,y1)
            lin_models[rule] = LinearRegression().fit(X1, y1.ravel())
        else:
            del rules[rule]
    activations_test = pd.DataFrame(index=X_test.index,
                                    columns=[rule for rule in rules.keys()])
    for rule in sorted(rules.keys()):
        act = []
        for mf in rules[rule]:
            act.append(fuzz.interp_membership(mf['universe'], mf['func'], X_test[mf['var_name']]))
        activations_test[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

    preds = pd.DataFrame(index=X_test.index, columns=sorted(rules.keys()))
    for rule in sorted(rules.keys()):
        indices = activations_test[rule].index[activations_test[rule] >= 0.01].tolist()
        if len(indices) != 0:
            X1 = X_test.loc[indices].values
            preds.loc[indices, rule] = scale_y.inverse_transform(lin_models[rule].predict(X1).reshape(-1, 1)).ravel()

    pred = preds.mean(axis=1)
    pred[pred.isnull()] = 1e+15
    # pred.name='target'
    # pred=pred.to_frame()
    err = (pred.values.ravel() - y_test.ravel()) / rated

    objectives = [np.sum(np.square(err)), np.mean(np.abs(err))]
    return objectives


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
        var_del = []
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
        if len(var_del) > 0:
            X = X.drop(columns=var_del)
        return activations

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict = {}
                for k in tmp_dict.keys():
                    if k not in ['logger', 'static_data', 'data_dir', 'cluster_dir', 'n_jobs']:
                        tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')
#
# if __name__ == '__main__':
#     import sys
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     if sys.platform == 'linux':
#         sys_folder = '/media/smartrue/HHD1/George/models/'
#     else:
#         sys_folder = 'D:/models/'
#     project_name = 'APE_net_ver2'
#     project_country = 'APE_net_ver2'
#     project_owner = '4cast_models'
#     path_project = sys_folder + project_owner + '/' + project_country + '/' + project_name
#     cluster_dir = path_project +'/Regressor_layer/rule.12'
#     data_dir = path_project + '/Regressor_layer/rule.12/data'
#     # logger = logging.getLogger(__name__)
#     # logger.setLevel(logging.INFO)
#     # handler = logging.FileHandler(os.path.join(cluster_dir, 'log_rbf_cnn_test.log'), 'a')
#     # handler.setLevel(logging.INFO)
#     #
#     # # create a logging format
#     # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # handler.setFormatter(formatter)
#     #
#     # # add the handlers to the logger
#     # logger.addHandler(handler)
#
#     rated = None
#
#     static_data = write_database()
#     X = pd.read_csv(os.path.join(static_data['path_data'], 'training_inputs.csv'), index_col=0,
#                     parse_dates=True, dayfirst=True)
#     y = pd.read_csv(os.path.join(static_data['path_data'], 'training_target.csv'), index_col=0,
#                     header=None,
#                     names=['target'], parse_dates=True, dayfirst=True)
#     X_train = X.loc[X.index <= pd.to_datetime('2019-01-01 00:00')]
#     X_test = X.loc[X.index > pd.to_datetime('2019-01-01 00:00')]
#     y_train = y.loc[y.index <= pd.to_datetime('2019-01-01 00:00')]
#     y_test = y.loc[y.index > pd.to_datetime('2019-01-01 00:00')]
#
#     optimizer = cluster_optimize(static_data)
#     optimizer.run(X_train, y_train, X_test, y_test, rated)
