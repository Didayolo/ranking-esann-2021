# Functions used by ranking_esann.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('../../ranky')
import ranky as rk
from glob import glob
import yaml
import itertools as it

# All functions used in this notebook
from utilities import *

def concat(dfs, names):
    """ Concatenate df from the list dfs
        Add a column with the name
    """
    for i in range(len(dfs)):
        dfs[i]['name'] = names[i]
    return pd.concat(dfs)

def winner_presence(r1, r2):
    w1 = r1.idxmax() # r1 winner
    w2 = r2.idxmax() # r2 winner
    return (w1 in r2.index), (w2 in r1.index)

def align(r1, r2):
    r1 = r1[set(r1.index)]
    r2 = r2[set(r2.index)]
    r1 = r1.drop([x for x in r1.index if x not in r2.index])
    r2 = r2.drop([x for x in r2.index if x not in r1.index])
    r1 = r1[~r1.index.duplicated(keep='first')]
    r2 = r2[~r2.index.duplicated(keep='first')]
    return r1, r2

def loo_centrality(m, ranking_function, metric='kendalltau', **kwargs):
    loos = [rk.corr(ranking_function(m.drop(j, axis=1), **kwargs), m[j], method=metric) for j in m.columns]
    return loos

def loo_winner(m, ranking_function, **kwargs):
    loos = [1 - rk.dist(ranking_function(m.drop(j, axis=1), **kwargs), m[j], method='winner_distance') for j in m.columns]
    return loos

def bootstrap_generalization(m, ranking_functions, kw_list, methods, metric, n=10, axis=0, candidate_perturbation=False, **kwargs):
    """ Generalization score with bootstrap.
        ranking_functions: list of function
        kw_list: list of **kwargs dict for ranking function
        method: list of method names
        Metric: kendalltau, spearman, winner_distance
        Axis: Judge axis
        n: number of trials
        candidate_perturbation: If True, bootstrap axis candidate axis
        
        TODO: compare all methods on the same bootstraps
    """
    all_scores = []
    for _ in range(n):
        _m = m
        if candidate_perturbation:
            _m = rk.bootstrap(m, axis=(1 - axis))
        train, test = rk.bootstrap(_m, return_holdout=True, axis=axis)
        scores = []
        for k, ranking_function in enumerate(ranking_functions):
            r = ranking_function(train, **kw_list[k])
            test_score = []
            for i in range(test.shape[axis]): # loop on test set
                score = rk.any_metric(r, np.take(test, [i], axis=axis), method=metric)
                if not np.isnan(score):
                    test_score.append(score)
            scores.append(np.mean(test_score))
        all_scores.append(scores)
    all_scores = pd.DataFrame(all_scores, columns=methods)
    return all_scores

def concordance(m, method='spearman', axis=0, align_arrays=True, winner=False):
    """ Coefficient of concordance between ballots.

    This is a measure of agreement between raters.
    The computation is the mean of the correlation between all possible pairs of judges.

    Args:
        axis: Axis of raters.
        method: spearman, kendalltau, winner_distance
        winner: If True, apply the metric between r1 and r2 only if the winner is present before alignment
    """
    # Idea: Kendall's W - linearly related to spearman between all pairwise
    #if rk.is_dataframe(m):
    #    m = np.array(m)
    idx = range(len(m)) #.shape[axis])
    scores = []
    for pair in it.permutations(idx, 2):
        r1 = m[pair[0]] #np.take(m, pair[0], axis=axis)
        r2 = m[pair[1]] #np.take(m, pair[1], axis=axis)
        # sort and remove missing candidates
        if align_arrays:
            if winner:
                w1, w2 = winner_presence(r1, r2)
            r1, r2 = align(r1, r2)
        if len(r1) > 1: # else do nothing
            if method in ['spearman', 'kendalltau']:
                c, p_value = rk.corr(r1, r2, method=method, return_p_value=True)
            else:
                if winner: # Compute asymmetrical winner distance based on the presence of the winner
                    if w1 and w2:
                        c = rk.dist(r1, r2, method='symmetrical_winner_distance')
                    elif w1:
                        c = rk.dist(r1, r2, method='winner_distance')
                    elif w2:
                        c = rk.dist(r2, r1, method='winner_distance')
                    else: # both winners got removed from the other array by the bootstrap
                        c = None
                else:
                    c = rk.dist(r1, r2, method=method)
            if c is not None:
                scores.append(c)
        #except Exception as e:
        #    print(e)
    return np.mean(scores)

def robustness_task(m, ranking_methods, kw_list, n=10, verbose=False, metric='kendall'):
    # VARIABILITY = TASK
    # use the same bootstraps for all ranking methods
    # therefore ranking_methods is the list of ranking methods and we loop over them
    # metric : kendall, winner_distance
    results = [[] for _ in ranking_methods] #  init rankings
    for i in range(n): # bootstraps loop
        # Use the same bootstrap for all models/methods
        matrix = rk.bootstrap(m, axis=1) # judge axis
        for j in range(len(ranking_methods)): # rankings loops
            results[j].append(rk.rank(ranking_methods[j](matrix, **kw_list[j])))
    # compute concordance
    robs = []
    #robs_ties = []
    for j in range(len(ranking_methods)):
            rankings = np.array(results[j]) # results[metric] is a list of rankings
            # concordance, judge axis: eah row is a ranking
            if metric == 'kendall':
                robs.append(rk.kendall_w_ties(rankings, axis=0))
            elif metric in ['winner_distance', 'symmetrical_winner_distance', 'spearman']:
                robs.append(concordance(rankings, axis=0, align_arrays=False, method=metric))
            else:
                raise Exception('Unknown method: {}'.format(metric))
            # correction for ties
            #robs_ties.append(rk.kendall_w_ties(rankings, axis=0))
    return robs, results

# Replace Kendall W by pairwise Winner distance for winner version

def robustness_candidates(m, ranking_methods, kw_list, n=10, verbose=False, metric='spearman', winner=False):
    # VARIABILITY = CANDIDATES
    # use the same bootstraps for all ranking methods
    # therefore ranking_methods is the list of ranking methods and we loop over them
    results = [[] for _ in ranking_methods] #  init rankings
    for i in range(n): # bootstraps loop
        # Use the same bootstrap for all models/methods
        matrix = rk.bootstrap(m, axis=0) # candidate axis
        for j in range(len(ranking_methods)): # rankings loops
            results[j].append(rk.rank(ranking_methods[j](matrix, **kw_list[j])))
    # compute concordance
    robs = []
    for j in range(len(ranking_methods)):
        #rankings = np.array(results[j]) # results[metric] is a list of rankings
        # concordance, candidate axis: eah row is a ranking
        robs.append(concordance(results[j], axis=0, method=metric, winner=winner))
    return robs, results