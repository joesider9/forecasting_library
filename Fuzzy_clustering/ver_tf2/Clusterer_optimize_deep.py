import numpy as np
import pandas as pd
import os, copy
import joblib
import skfuzzy as fuzz
import difflib, random
from deap import base, creator, tools, algorithms
from itertools import repeat
from collections import Sequence
import re
import logging
from sklearn.linear_model import LinearRegression
from Fuzzy_clustering.ver_tf2.imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, SMOTE,ADASYN
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_EVALUATIONS = 30000
JOBS = 5
POPULATION_SIZE = 50


class cluster_optimize():

    def __init__(self, static_data):
        self.static_data = static_data
        self.train_online = static_data['train_online']
        self.add_individual_rules = static_data['clustering']['add_rules_indvidual']
        self.import_external_rules = static_data['clustering']['import_external_rules']
        self.njobs=2*static_data['RBF']['njobs']
        self.resampling = static_data['resampling']
        self.path_fuzzy = static_data['path_fuzzy_models']
        self.file_fuzzy = static_data['clustering']['cluster_file']
        self.type = static_data['type']

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

    def create_mfs(self, model_mfs, type_mf, num_mf, var_range, var_name):
        mfs = []
        if type_mf in {'hdd_h2', 'temp_max', 'flux', 'wind', 'temp', 'Temp', 'load', 'power'}:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
            std = var_range[1] / num_mf
            for i in range(num_mf):
                mfs.append({'name': 'mf_' + type_mf + str(i),
                            'var_name': var_name,
                            'type': 'gauss',
                            'param': [mean[i], 1.25 * std],
                            'universe': np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                            'func': fuzz.gaussmf(np.arange(var_range[0] - std - .01, var_range[1] + std + .01, .001),
                                                 mean[i], std)})
        elif type_mf in {'sp_index', 'dayweek', 'cloud', 'hour', 'month', 'direction', 'sp_days'}:
            mean = np.linspace(var_range[0], var_range[1], num=num_mf)
            std = var_range[1] / num_mf
            std1 = var_range[1] / (2 * num_mf)
            for i in range(num_mf):
                param = [mean[i] - 1.5 * std, mean[i] - 1.25 * std1, mean[i] + 1.25 * std1, mean[i] + 1.5 * std]
                mfs.append({'name': 'mf_' + type_mf + str(i),
                            'var_name': var_name,
                            'type': 'trap',
                            'param': param,
                            'universe': np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001),
                            'func': fuzz.trapmf(np.arange(var_range[0] - .01 - std, var_range[1] + std + .01, .001, ),
                                                param)})
        else:
            raise NameError('MF type not recognize')
        model_mfs[var_name] = mfs
        return model_mfs


    def create_rules(self, model_mfs):

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
        ## Uncomment when you want to create rules without combine variables
        if self.add_individual_rules:
            for mf in sorted(model_mfs.keys()):
                for f in model_mfs[mf]:
                    rules.append([f])
        final_rules = dict()
        for i in range(len(rules)):
            final_rules['rule.' + str(i)] = rules[i]

        return final_rules



    def run(self, X_train, y_train, X_test, y_test, rated, num_samples=200):

        if rated is None:
            rated = y_test.values.ravel()
        else:
            rated = 20
        self.var_names = [v for v in sorted(self.static_data['clustering']['var_imp'].keys())]
        self.p = len(self.var_names)
        self.range_mfs = dict()
        self.num_mfs = dict()
        for k, v in self.static_data['clustering']['var_imp'].items():
            self.range_mfs[k] = v['range']
            self.num_mfs[k] = [v['mfs']]
        self.var_lin = self.static_data['clustering']['var_lin']
        if self.train_online:
            try:
                fuzzy_model = joblib.load(os.path.join(self.path_fuzzy, self.file_fuzzy))
                self.var_names = [var for var in self.var_names if var not in sorted(fuzzy_model['mfs'].keys())]
            except:
                fuzzy_model = dict()
                fuzzy_model['mfs'] = dict()
        else:
            fuzzy_model = dict()
            fuzzy_model['mfs'] = dict()

        for fuzzy_var_name in self.var_names:
            for n in self.num_mfs[fuzzy_var_name]:
                fuzzy_model['mfs'] = self.create_mfs(fuzzy_model['mfs'], fuzzy_var_name, n,
                                                     self.range_mfs[fuzzy_var_name], fuzzy_var_name)

        var_del=[]
        for var in self.var_names:
            if var not in X_train.columns:
                var_names = [c for c in X_train.columns if var in c]
                X_train[var] = X_train[var_names].mean(axis=1)
                X_test[var] = X_test[var_names].mean(axis=1)
                var_del.append(var)
            if var not in self.var_lin:
                self.var_lin.append(var)

        lin_models = LinearRegression().fit(X_train[self.var_lin].values, y_train.values.ravel())
        preds = lin_models.predict(X_test[self.var_lin].values).ravel()

        err = (preds - y_test.values.ravel()) / rated

        rms = np.sum(np.square(err))
        mae = np.mean(np.abs(err))
        print('rms = %s', rms)
        print('mae = %s', mae)
        self.logger.info("Objective before train: %s", mae)
        problem = cluster_problem(fuzzy_model['mfs'], X_train[self.var_lin], y_train, X_test[self.var_lin],
                                  y_test, self.p, rated, self.resampling, self.add_individual_rules, self.logger, self.njobs, num_samples=num_samples)
        problem.run(50, 50)

        fuzzy_model = problem.fmodel

        self.logger.info("Objective after train: %s", str(fuzzy_model['result']))

        best_fuzzy_model = copy.deepcopy(fuzzy_model)
        joblib.dump(best_fuzzy_model, os.path.join(self.path_fuzzy, self.file_fuzzy))
        if 'horizon' in self.import_external_rules:
            self.compact_external_mfs()
        if len(var_del) > 0:
            X_train = X_train.drop(columns=var_del)
            X_test = X_test.drop(columns=var_del)


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


def cx_fun(ind1,ind2, alpha):
    if random.random()>0.5:
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
                    if xy<0:
                        xy=1e-6
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    if xy<0:
                        xy=1e-6
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl)
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

    def __init__(self, mfs, X, y, X_test, y_test, p, rated, resampling, add_individual_rules, logger, njobs, num_samples=500, n_ratio=0.6):

        self.logger = logger
        self.njobs=njobs
        self.resampling = resampling
        self.add_individual_rules = add_individual_rules
        self.num_samples = num_samples
        self.n_ratio = n_ratio
        self.rules = self.create_rules(mfs)
        self.mfs = mfs
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.p = p
        if rated is None:
            self.rated = self.y_test.values.ravel()
        else:
            self.rated = rated
        x = []
        self.lower_bound = []
        self.upper_bound = []
        self.sigma = []
        self.index_constrains = []
        self.number_of_constraints = 0
        for var in sorted(self.mfs.keys()):
            for mf in self.mfs[var]:
                param = mf['param']
                xrange = [mf['universe'][0], mf['universe'][-1]]
                prange = (xrange[1] - xrange[0]) / 5
                x = x + param
                if len(param) == 2:
                    self.index_constrains.append(np.arange(len(x) - 2, len(x)))
                    self.number_of_constraints = self.number_of_constraints + 3

                    lo = param[0] - prange if (param[0] - prange)<-0.05 else -0.05
                    up = param[0] + prange if (param[0] + prange)>1.05 else 1.05

                    self.lower_bound.extend([lo, 0.01])
                    self.upper_bound.extend([up, prange])
                    self.sigma.extend([X[var].std() / 3, X[var].std() / 3])
                elif len(param) == 4:
                    self.index_constrains.append(np.arange(len(x) - 4, len(x)))
                    self.number_of_constraints = self.number_of_constraints + 7
                    for i in param:
                        lo = param[0] - prange if (param[0] - prange) < -0.05 else -0.05
                        up = param[3] + prange if (param[3] + prange) > 1.05 else 1.05
                        self.lower_bound.append(lo)
                        self.upper_bound.append(up)
                        self.sigma.append(X[var].std() / 3)
        self.number_of_variables = len(x)
        self.number_of_objectives = 2
        self.x = x
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()

        attributes=[]
        for i in range(self.number_of_variables):
            self.toolbox.register("attribute"+str(i), random.gauss, self.lower_bound[i], self.upper_bound[i])
            attributes.append(self.toolbox.__getattribute__("attribute"+str(i)))

        self.toolbox.register("individual1", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual1, n=100)

        self.toolbox.register("mate", cx_fun, alpha=0.05)
        self.toolbox.register("mutate", mut_fun, mu=0, sigma=self.sigma, eta=0.8, low=self.lower_bound, up=self.upper_bound, indpb=0.6)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", evaluate)
        self.hof = tools.ParetoFront(lambda x, y: (x == y).all())
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("Avg", np.mean)
        self.stats.register("Std", np.std)
        self.stats.register("Min", np.min)
        self.stats.register("Max", np.max)
        self.toolbox.decorate("mate", checkBounds(self.lower_bound, self.upper_bound))
        self.toolbox.decorate("mutate", checkBounds(self.lower_bound, self.upper_bound))

    def create_rules(self, model_mfs):

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
        final_rules = dict()
        for i in range(len(rules)):
            final_rules['rule.' + str(i)] = rules[i]

        return final_rules
    def run(self, mu, lambda_, cxpb=0.6, mutpb=0.4, ngen=300):

        self.population=self.toolbox.population()
        param_ind = creator.Individual(self.x)
        self.population.pop()
        self.population.insert(0, param_ind)
        assert lambda_ >= mu, "lambda must be greater or equal to mu."

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]

        fit1 = evaluate(np.array(invalid_ind[0]).ravel(),
                                                          self.X, self.y, self.X_test, self.y_test, self.rated,
                                                          self.mfs, self.rules, self.p, self.resampling, self.num_samples, self.n_ratio)
        fitnesses = Parallel(n_jobs=self.njobs)(delayed(evaluate)(np.array(individual).ravel(),
                                                          self.X, self.y, self.X_test, self.y_test, self.rated,
                                                          self.mfs, self.rules, self.p, self.resampling, self.num_samples, self.n_ratio) for individual in invalid_ind)


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
            offspring = algorithms.varOr(self.population, self.toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=self.njobs)(delayed(evaluate)(np.array(individual).ravel(),
                                                              self.X, self.y, self.X_test, self.y_test, self.rated,
                                                              self.mfs, self.rules, self.p, self.resampling, self.num_samples) for
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
        self.logger.info('Iter: %s, Max_sse: %s, Min_mae: %s', str(gen), *minimums)
        front = self.population
        perf = np.inf
        best = 0
        for i in range(len(front)):
            if front[i].fitness.getValues()[0] < perf:
                best = i
                perf = front[i].fitness.getValues()[0]

        self.fmodel = self.evaluate(np.array(front[best]).ravel(), self.X, self.y, self.X_test, self.y_test, self.rated,
                                                          self.mfs, self.rules, self.p, self.resampling)
    def evaluate(self, x, X, y, X_test, y_test, rated, mfs, rules, p, resampling):

        # print(solution.variables)
        i = 0
        for var in sorted(mfs.keys()):
            for mf in range(len(mfs[var])):
                if mfs[var][mf]['type'] == 'gauss':
                    mfs[var][mf]['param'] = x[i:i + 2]
                    mfs[var][mf]['func'] = fuzz.gaussmf(mfs[var][mf]['universe'],
                                                             mfs[var][mf]['param'][0],
                                                             np.abs(mfs[var][mf]['param'][1]))
                    i += 2
                elif mfs[var][mf]['type'] == 'trap':
                    mfs[var][mf]['param'] = sorted(x[i:i + 4])
                    mfs[var][mf]['func'] = fuzz.trapmf(mfs[var][mf]['universe'], mfs[var][mf]['param'])
                    i += 4

        for r in sorted(rules.keys()):
            for i in range(len(rules[r])):
                ind = int(re.sub("\D", "", rules[r][i]['name']))
                rules[r][i]['param'] = mfs[rules[r][i]['var_name']][ind]['param']
                rules[r][i]['func'] = mfs[rules[r][i]['var_name']][ind]['func']

        activations = pd.DataFrame(index=X.index, columns=[rule for rule in sorted(rules.keys())])
        for rule in sorted(rules.keys()):
            act = []
            for mf in rules[rule]:
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
            activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

        lin_models = dict()
        remove_null_rules = []

        for rule in sorted(activations.columns):
            indices = activations[rule].index[activations[rule] >= 0.01].tolist()

            if len(indices) > self.num_samples and len(indices) < self.n_ratio * X.shape[0] :
                X1 = X.loc[indices].values
                y1 = y.loc[indices].values

                lin_models[rule] = LinearRegression().fit(X1, y1.ravel())
            elif len(indices) > 0:
                lin_models[rule] = 'null'
                remove_null_rules.append(rule)
            else:
                lin_models[rule] = None
                remove_null_rules.append(rule)


        total = 0
        for rule in sorted(rules.keys()):
            indices = activations[rule].index[activations[rule] >= 0.01].tolist()
            act = activations.loc[indices].copy(deep=True)
            act = act.drop(columns=[rule])
            if len(indices) <= self.num_samples and len(indices) < self.n_ratio * X.shape[0] and not act.isnull().all(axis=1).any():
                del rules[rule]
                del lin_models[rule]
            else:
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
                y1 = y_test.loc[indices].values
                if (lin_models[rule] != 'null' and not lin_models[rule] is None):
                    preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()
                elif lin_models[rule] == 'null':
                    preds.loc[indices, rule] = 1e+15
                if isinstance(rated, float) or isinstance(rated, int):
                    err = (preds.loc[indices, rule].values.ravel() - y1.ravel()) / rated
                else:
                    err = (preds.loc[indices, rule].values.ravel() - y1.ravel()) / y1.ravel()
                self.logger.info("MAE of rule %s is %s", rule, np.mean(np.abs(err)))
        pred = preds.mean(axis=1)
        pred[pred.isnull()] = 1e+15
        # pred.name='target'
        # pred=pred.to_frame()
        err = (pred.values.ravel() - y_test.values.ravel()) / rated

        self.objectives = [np.sum(np.square(err)),np.mean(np.abs(err))]
        self.rules = rules
        self.mfs = mfs
        fmodel = dict()
        fmodel['mfs'] = self.mfs
        fmodel['rules'] = self.rules
        fmodel['result'] = self.objectives[1]
        print('Error = ', self.objectives[1])
        return fmodel

def resampling_fun(X, y, random_state=42):
    flag = False
    Std = 0.01
    while (flag == False and Std <= 1):
        try:
            std = np.maximum(Std * np.std(y), 0.2)
            yy = np.digitize(y, np.arange(np.min(y), np.max(y), std), right=True)
            bins = np.arange(np.min(y), np.max(y), std)
            bins = bins[(np.bincount(yy.ravel()) >= 2)[:-1]]
            yy = np.digitize(y, bins, right=True)
            # if Std==0.01 and np.max(yy)!=0:
            #     strategy = {cl:int(100*X.shape[0]/np.max(yy)) for cl in np.unique(yy)}
            # else:
            strategy = "auto"
            if np.unique(yy).shape[0]==1:
                X2 = X
                yy2 = y
                return X2, yy2
            if np.any(np.bincount(yy.ravel())<2):
                for cl in np.where(np.bincount(yy.ravel())<2)[0]:
                    X = X[np.where(yy!=cl)[0]]
                    y = y[np.where(yy!=cl)[0]]
                    yy = yy[np.where(yy!=cl)[0]]

            sm = ADASYN(sampling_strategy=strategy, random_state=random_state, n_neighbors=np.min(np.bincount(yy.ravel()) - 1))


            try:
                X2, yy2 = sm.fit_resample(X, yy.ravel())
            except:
                pass


            X2[np.where(X2<0)] = 0
            yy2 = bins[yy2 - 1]
            flag = True
        except:
            Std *= 10

    if flag == True:
        return X2, yy2
    else:
        raise RuntimeError('Cannot make resampling ')

def evaluate(x, X, y, X_test, y_test, rated, mfs, rules, p, resampling, num_samples=200, n_ratio=0.6):

    # print(solution.variables)
    i = 0
    for var in sorted(mfs.keys()):
        for mf in range(len(mfs[var])):
            if mfs[var][mf]['type'] == 'gauss':
                mfs[var][mf]['param'] = x[i:i + 2]
                mfs[var][mf]['func'] = fuzz.gaussmf(mfs[var][mf]['universe'],
                                                         mfs[var][mf]['param'][0],
                                                         np.abs(mfs[var][mf]['param'][1]))
                i += 2
            elif mfs[var][mf]['type'] == 'trap':
                mfs[var][mf]['param'] = sorted(x[i:i + 4])
                mfs[var][mf]['func'] = fuzz.trapmf(mfs[var][mf]['universe'], mfs[var][mf]['param'])
                i += 4

    for r in sorted(rules.keys()):
        for i in range(len(rules[r])):
            ind = int(re.sub("\D", "", rules[r][i]['name']))
            rules[r][i]['param'] = mfs[rules[r][i]['var_name']][ind]['param']
            rules[r][i]['func'] = mfs[rules[r][i]['var_name']][ind]['func']

    activations = pd.DataFrame(index=X.index, columns=['rule.' + str(i) for i in range(len(rules))])
    for rule in sorted(rules.keys()):
        act = []
        for mf in rules[rule]:
            act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
        activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / p)

    lin_models = dict()
    for rule in activations.columns:
        indices = activations[rule].index[activations[rule] >= 0.01].tolist()
        if len(indices) > num_samples and len(indices) < n_ratio * X.shape[0] :
            X1 = X.loc[indices].values
            y1 = y.loc[indices].values
            # if resampling:
            #     if X1.shape[0] < 300:
            #         X1, y1 = resampling_fun(X1,y1)
            lin_models[rule] = LinearRegression().fit(X1, y1.ravel())
        elif len(indices) > 0:
            lin_models[rule] = 'null'
        else:
            lin_models[rule] = None
    for rule in sorted(rules.keys()):
        indices = activations[rule].index[activations[rule] >= 0.01].tolist()
        if len(indices) <= num_samples and len(indices) < n_ratio * X.shape[0] and len(indices) < n_ratio * X.shape[0] :
            del rules[rule]
            del lin_models[rule]
    activations_test = pd.DataFrame(index=X_test.index,
                                    columns=['rule.' + str(i) for i in sorted(rules.keys())])
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
            if (lin_models[rule] != 'null' and not lin_models[rule] is None):
                preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()
            elif lin_models[rule] == 'null':
                preds.loc[indices, rule] = 1e+15
    pred = preds.mean(axis=1)
    pred[pred.isnull()] = 1e+15
    # pred.name='target'
    # pred=pred.to_frame()
    err = (pred.values.ravel() - y_test.values.ravel()) / rated

    objectives = [np.sum(np.square(err)),np.mean(np.abs(err))]
    return objectives



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
        var_del=[]
        for rule in sorted(self.rules.keys()):
            act = []
            for mf in self.rules[rule]:
                if mf['var_name'] not in X.columns:
                    var_names = [c for c in X.columns if mf['var_name'] in c]
                    X[mf['var_name']] = X[var_names].mean(axis=1)
                    var_del.append(mf['var_name'])
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], X[mf['var_name']]))
            activations[rule] = np.power(np.prod(np.array(act), axis=0), 1 / self.p)
        if len(var_del)>0:
            X = X.drop(columns=var_del)
        return activations

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
