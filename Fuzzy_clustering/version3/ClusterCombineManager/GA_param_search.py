import os
from threadpoolctl import threadpool_limits

import numpy as np
import random, os, joblib
from deap import base, creator, tools
from collections import defaultdict
from sklearn.base import clone, is_classifier
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import BaseSearchCV, check_cv, _check_param_grid
from sklearn.metrics import check_scoring
from sklearn.utils.validation import _num_samples, indexable
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp



def enum(**enums):
    return type('Enum', (), enums)


param_types = enum(Categorical=1, Numerical=2)

def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]
    size = len(offspring)
    fit_old=np.inf

    for ind in offspring:
        if len(ind.fitness.values)>0:
            if ind.fitness.values[0]<fit_old:
                elite =ind
                fit_old = ind.fitness.values[0]

    offspring = []
    for _ in range(size):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    del elite.fitness.values
    offspring.append(elite)
    return offspring

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, njobs=1, inner_jobs=1,
             name_values=(),
             scorer=None, cvs=[], rated = None, iid=None, verbose1=0,
             error_score=None, fit_params=None,
             score_cache=None, estimator=None, path_group = None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # with parallel_backend("threading", inner_max_num_threads=inner_jobs):
    #     with Parallel(n_jobs=njobs, prefer='threads') as parallel:
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fit1 = toolbox.evaluate(np.array(invalid_ind[0]), name_values, X, y, scorer, cvs, iid, fit_params,
    #                         verbose1, error_score, score_cache, estimator)

    # fitnesses = parallel(delayed(toolbox.evaluate)(np.array(individual).ravel()
    #                                    , name_values, X, y, scorer, cvs, iid, fit_params,
    #                                    verbose1, error_score, score_cache, estimator) for
    #                                         individual in invalid_ind)

    ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
    gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

    njobs = int(ncpus - gpu_status)
    cpu_status = njobs
    joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))
    toolbox.evaluate(np.array(invalid_ind[0]), name_values, scorer, cvs, rated, iid, fit_params,
                     verbose1, error_score, score_cache, estimator)
    pool = mp.Pool(njobs)
    results = [pool.apply_async(toolbox.evaluate,
                                args=(np.array(individual), name_values, scorer, cvs, rated, iid, fit_params,
                                      verbose1, error_score, score_cache, estimator))
               for individual in
               invalid_ind]
    fitnesses = [p.get() for p in results]
    pool.close()
    pool.terminate()
    pool.join()

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        ncpus = joblib.load(os.path.join(path_group, 'total_cpus.pickle'))
        gpu_status = joblib.load(os.path.join(path_group, 'gpu_status.pickle'))

        njobs = int(ncpus - gpu_status)
        cpu_status = njobs
        joblib.dump(cpu_status, os.path.join(path_group, 'cpu_status.pickle'))

        pool = mp.Pool(njobs)
        results = [pool.apply_async(toolbox.evaluate,
                                    args=(np.array(individual), name_values, scorer, cvs, rated, iid, fit_params,
                                          verbose1, error_score, score_cache, estimator))
                   for individual in
                   invalid_ind]
        fitnesses = [p.get() for p in results]
        pool.close()
        pool.terminate()
        pool.join()

        # fitnesses = parallel(delayed(toolbox.evaluate)(np.array(individual).ravel()
        #                                                              , name_values, X, y, scorer, cvs, iid, fit_params,
        #                                                              verbose1, error_score, score_cache, estimator) for
        #                                    individual in invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        fits = np.array([ind.fitness.values for ind in population])

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def _get_param_types_maxint(params):
    """
    Returns characteristics of parameters
    :param params: dictionary of pairs
        it must have parameter_name:list of possible values:
        params = {"kernel": ["rbf"],
                 "C"     : [1,2,3,4,5,6,7,8],
                 "gamma" : np.logspace(-9, 9, num=25, base=10)}
    :return: name_values pairs - list of (name,possible_values) tuples for each parameter
             types - list of types for each parameter
             maxints - list of maximum integer for each particular gene in chromosome
    """
    name_values = list(params.items())
    types = []
    for _, possible_values in name_values:
        if isinstance(possible_values[0], float):
            types.append(param_types.Numerical)
        else:
            types.append(param_types.Categorical)
    maxints = [len(possible_values) - 1 for _, possible_values in name_values]
    return name_values, types, maxints


def _initIndividual(pcls, maxints):
    part = pcls(random.randint(0, maxint) for maxint in maxints)
    return part


def _mutIndividual(individual, up, indpb, gene_type=None):
    for i, up, rn in zip(range(len(up)), up, [random.random() for _ in range(len(up))]):
        if rn < indpb:
            individual[i] = random.randint(0, up)
    return individual,


def _cxIndividual(ind1, ind2, indpb, gene_type):
    for i, gt, rn in zip(range(len(ind1)), gene_type, [random.random() for _ in range(len(ind1))]):
        if rn > indpb:
            continue
        if random.random()<=0.5:
            if gt is param_types.Categorical:
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:
                # Case when parameters are numerical
                if ind1[i] <= ind2[i]:
                    ind1[i] = random.randint(ind1[i], ind2[i])
                    ind2[i] = random.randint(ind1[i], ind2[i])
                else:
                    ind1[i] = random.randint(ind2[i], ind1[i])
                    ind2[i] = random.randint(ind2[i], ind1[i])
        else:
            if gt is param_types.Categorical:
                ind1[i] = np.rint(np.mean([ind2[i], ind1[i]]))
                ind2[i] = np.rint(np.mean([ind2[i], ind1[i]]))
            else:
                # Case when parameters are numerical
                if isinstance(ind1[i], int) and isinstance(ind1[i], int):
                    ind1[i] = np.rint(np.mean([ind2[i], ind1[i]]))
                    ind2[i] = np.rint(np.mean([ind2[i], ind1[i]]))
                else:
                    ind1[i] = np.mean([ind2[i], ind1[i]])
                    ind2[i] = np.mean([ind2[i], ind1[i]])

    return ind1, ind2


def _individual_to_params(individual, name_values):
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))

def fit_and_score(model, cvs, params, rated):

    model.set_params(**params)
    rms_val=[]
    rms_test=[]
    for cv in cvs:
        model.fit(cv[0], cv[1].ravel())
        if rated is None:
            ypred=model.predict(cv[2]).ravel()
            rms_val.append(np.sqrt(np.mean(np.square(np.abs(ypred-cv[3].ravel()) / cv[3].ravel()))))
            ypred = model.predict(cv[4]).ravel()
            rms_test.append(np.sqrt(np.mean(np.square(np.abs(ypred-cv[5].ravel()) / cv[5].ravel()))))
        else:
            ypred = model.predict(cv[2]).ravel()
            rms_val.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[3].ravel())))))
            ypred = model.predict(cv[4]).ravel()
            rms_test.append(np.sqrt(np.mean(np.square(np.abs(ypred - cv[5].ravel())))))
        # print(np.mean(np.abs(ypred - cv[5].ravel())))
    return 0.4*np.mean(rms_val)+0.6*np.mean(rms_test)

def _evalFunction(individual, name_values, scorer, cvs, rated, iid, fit_params,
                  verbose, error_score, score_cache, estimator):
    """ Developer Note:
        --------------------
        score_cache was purposefully moved to parameters, and given a dict reference.
        It will be modified in-place by _evalFunction based on it's reference.
        This is to allow for a managed, paralell memoization dict,
        and also for different memoization per instance of EvolutionaryAlgorithmSearchCV.
        Remember that dicts created inside function definitions are presistent between calls,
        So unless it is replaced this function will be memoized each call automatically. """

    parameters = _individual_to_params(individual, name_values)
    estimator.set_params(**parameters)

    score = 0
    n_test = 0

    paramkey = str(individual)
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        score = fit_and_score(estimator, cvs, parameters, rated)
        score_cache[paramkey] = score

    return (score,)


class EvolutionaryAlgorithmSearchCV(BaseSearchCV):
    """Evolutionary search of best hyperparameters, based on Genetic
    Algorithms

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    params : dict or list of dictionaries
        each dictionary must have parameter_name:sorted(list of possible values):
        params = {"kernel": ["rbf"],
                 "C"     : [1,2,3,4,5,6,7,8],
                 "gamma" : np.logspace(-9, 9, num=25, base=10)}
        Notice that Numerical values (floats) must be ordered in ascending or descending
        order.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    population_size : int, default=50
        Population size of genetic algorithm

    gene_mutation_prob : float, default=0.1
        Probability of gene mutation in chromosome

    gene_crossover_prob : float, default=0.5
        Probability of gene swap between two chromosomes

    tournament_size : int, default=3
        Size of tournament for selection stage of genetich algorithm

    generations_number : int, default=10
        Number of generations

    gene_type : list of integers, default=None
        list of types for each parameter, if None - it gets inferred from
        params, if some parameter has list of float values - it becomes param_types.Numerical
        if it has any other type of values - it becomes param_types.Categorical

        For Categorical features crossover operation just swaps corresponding
        genes between chromosomes with probability 'gene_crossover_prob'.

        For Numerical features crossover operation picks random gene between two
        genes of parents. Thus offsprings will have value of particular Numerical
        parameter in range [ind1_parameter, ind2_parameter]. Of course it is correct only
        when parameters of some value is sorted.

    n_jobs : int or map function, default=1
        Number of jobs to run in parallel.
        Also accepts custom parallel map functions from Pool or SCOOP.

    pre_dispatch : int, or string, optional
        Dummy parameter for compatibility with sklearn's GridSearch

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.


    Examples
    --------
        import sklearn.datasets
        import numpy as np
        import random

        data = sklearn.datasets.load_digits()
        X = data["data"]
        y = data["target"]

        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold

        paramgrid = {"kernel": ["rbf"],
                     "C"     : np.logspace(-9, 9, num=25, base=10),
                     "gamma" : np.logspace(-9, 9, num=25, base=10)}

        random.seed(1)

        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                           params=paramgrid,
                                           scoring="accuracy",
                                           cv=StratifiedKFold(n_splits=10),
                                           verbose=1,
                                           population_size=50,
                                           gene_mutation_prob=0.10,
                                           gene_crossover_prob=0.5,
                                           tournament_size=3,
                                           generations_number=10)
        cv.fit(X, y)


    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_: dict
        Dictionary of parameters for the estimator with the best score.

    cv_results_: list of dicts or dict
        Returns a pandas compatable dict or list of dicts with the
        log output of the learner.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    all_history_ : list of deap.tools.History objects, indexed by params (len 1 if params is not a list).
        Use to get the geneology data of the search.

    all_logbooks_: list of the deap.tools.Logbook objects, indexed by params (len 1 if params is not a list).
       With the statistics of the evolution.

    """

    def _run_search(self, evaluate_candidates):
        """
            scikit-learn new version introduce a new abstract function hence we have to implement an anonymous function
        """
        pass

    def __init__(self, estimator, params, rated, scoring=None, cv=4,
                 refit=True, verbose=False, population_size=50,
                 gene_mutation_prob=0.1, gene_crossover_prob=0.5,
                 tournament_size=3, generations_number=10, gene_type=None,
                 n_jobs=1, inner_jobs=1, iid=True, error_score='raise', init_params=[],
                 fit_params={}, path_group = None):
        super(EvolutionaryAlgorithmSearchCV, self).__init__(
            estimator=estimator, scoring=scoring,
            iid=iid, refit=refit, cv=cv, verbose=verbose,
            error_score=error_score)
        self.path_group = path_group
        self.params = params
        self.rated = rated
        self.population_size = population_size
        self.generations_number = generations_number
        self._individual_evals = {}
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.gene_type = gene_type
        self.all_history_, self.all_logbooks_ = [], []
        self._cv_results = None
        self.best_score_ = None
        self.best_params_ = None
        self.score_cache = {}
        self.n_jobs = n_jobs
        self.inner_jobs = inner_jobs
        self.init_params = init_params
        self.fit_params = fit_params
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    @property
    def possible_params(self):
        """ Used when assuming params is a list. """
        return self.params if isinstance(self.params, list) else [self.params]

    @property
    def cv_results_(self):
        if self._cv_results is None:  # This is to cache the answer until updated
            # Populate output and return
            # If not already fit, returns an empty dictionary
            possible_params = self.possible_params  # Pre-load property for use in this function
            out = defaultdict(list)
            for p, gen in enumerate(self.all_history_):
                # Get individuals and indexes, their list of scores,
                # and additionally the name_values for this set of parameters

                idxs, individuals, each_scores = zip(*[(idx, indiv, np.mean(indiv.fitness.values))
                                                       for idx, indiv in list(gen.genealogy_history.items())
                                                       if indiv.fitness.valid and not np.all(
                        np.isnan(indiv.fitness.values))])

                name_values, _, _ = _get_param_types_maxint(possible_params[p])

                # Add to output
                out['param_index'] += [p] * len(idxs)
                out['index'] += idxs
                out['params'] += [_individual_to_params(indiv, name_values)
                                  for indiv in individuals]
                out['mean_test_score'] += [np.nanmean(scores) for scores in each_scores]
                out['std_test_score'] += [np.nanstd(scores) for scores in each_scores]
                out['min_test_score'] += [np.nanmin(scores) for scores in each_scores]
                out['max_test_score'] += [np.nanmax(scores) for scores in each_scores]
                out['nan_test_score?'] += [np.any(np.isnan(scores)) for scores in each_scores]
            self._cv_results = out

        return self._cv_results

    @property
    def best_index_(self):
        """ Returns the absolute index (not the 'index' column) with the best max_score
            from cv_results_. """
        return np.argmax(self.cv_results_['max_test_score'])

    def fit(self, cvs, groups=None, **fit_params):

        self.best_estimator_ = None
        self.best_mem_score_ = float("-inf")
        self.best_mem_params_ = None
        for possible_params in self.possible_params:
            _check_param_grid(possible_params)
            self._fit(cvs, possible_params)
        # if self.refit:
        #     self.best_estimator_ = clone(self.estimator)
        #     self.best_estimator_.set_params(**self.best_mem_params_)
        #     if self.fit_params is not None:
        #         self.best_estimator_.fit(X, y, **self.fit_params)
        #     else:
        #         self.best_estimator_.fit(X, y)

    def _fit(self, cvs, parameter_dict):
        self._cv_results = None  # To indicate to the property the need to update
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        toolbox = base.Toolbox()
        toolbox1 = base.Toolbox()

        name_values, gene_type, maxints = _get_param_types_maxint(parameter_dict)
        if self.gene_type is None:
            self.gene_type = gene_type

        if self.verbose:
            print("Types %s and maxint %s detected" % (self.gene_type, maxints))

        toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        toolbox.register("evaluate", _evalFunction)

        toolbox.register("mate", _cxIndividual, indpb=self.gene_crossover_prob, gene_type=self.gene_type)

        toolbox.register("mutate", _mutIndividual, indpb=self.gene_mutation_prob, up=maxints)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        pop = toolbox.population(n=self.population_size)
        if len(self.init_params)>0:
            for param in self.init_params:
                param_ind = creator.Individual(param)
                pop.pop()
                pop.insert(0,param_ind)
        hof = tools.HallOfFame(1, similar=lambda x, y: (x == y).all())

        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        stats.register("std", np.nanstd)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))
        with threadpool_limits(limits=1):
            pop, logbook = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5,
                                    ngen=self.generations_number, stats=stats,
                                    halloffame=hof, verbose=self.verbose, njobs=self.n_jobs, inner_jobs=self.inner_jobs,
                                    name_values=name_values,
                                    scorer=self.scorer_, cvs=cvs, rated=self.rated, iid=self.iid, verbose1=self.verbose,
                                    error_score=self.error_score, fit_params=self.fit_params,
                                    score_cache=self.score_cache, estimator=self.estimator, path_group = self.path_group)

        # Save History
        self.all_history_.append(hist)
        self.all_logbooks_.append(logbook)
        current_best_score_ = hof[0].fitness.values[0]
        current_best_params_ = _individual_to_params(hof[0], name_values)
        if self.verbose:
            print("Best individual is: %s\nwith fitness: %s" % (
                current_best_params_, current_best_score_))

        if current_best_score_ > self.best_mem_score_:
            self.best_mem_score_ = current_best_score_
            self.best_mem_params_ = current_best_params_

        self.best_score_ = current_best_score_
        self.best_params_ = current_best_params_
