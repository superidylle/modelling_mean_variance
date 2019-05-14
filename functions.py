# -*- coding: utf-8  -*-
# @Author: Xingqi Ye
# @Time: 2019-05-15-02

import numpy as np
from scipy.optimize import minimize
from math import *

def maximum_diversification_function(weights,cov,std):

    temp = sum(std * weights)

    return(- temp/np.dot(np.dot(weights, cov), weights))

def maximum_diversification(data, long=1):

    cov = np.cov(data)
    std = np.std(data, axis=1)
    n = cov.shape[0]
    weights = np.ones(n)/n
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = [(0, 0.5) for i in weights]

    if long == 1:
        res = minimize(maximum_diversification_function, x0 = weights, args = (cov,std), method= 'SLSQP',
                       constraints=cons, bounds=bnds, tol = 1e-30)

    else:

        res = minimize(maximum_diversification_function, x0=weights, args=(cov,std), method='SLSQP',
                       constraints=cons, tol= 1e-30)

    return res.x

def global_minimum_variance_function(weights,cov):

    return(np.dot(np.dot(weights,cov),weights))

def global_minimum_variance(data, long=1):

    cov = np.cov(data)
    n = cov.shape[0]
    weights = np.ones(n)/n
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = [(0, 0.5) for i in weights]

    if long == 1:
        res = minimize(global_minimum_variance_function, x0 = weights, args = (cov), method= 'SLSQP',
                       constraints=cons, bounds=bnds, tol = 1e-30)

    else:

        res = minimize(global_minimum_variance_function, x0=weights, args=(cov), method='SLSQP',
                       constraints=cons, tol= 1e-30)

    return res.x


def maximum_sharpe_ratio_function(weights,cov,mu,rf):

    return(-(np.dot(mu, weights)-rf)/sqrt(np.dot(np.dot(weights,cov),weights)))

def maximum_sharpe_ratio(data, long=1):

    rf = 0.02/12
    cov = np.cov(data)
    mu  = np.mean(data, axis = 1)
    n = cov.shape[0]
    weights = np.ones(n)/n
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = [(0, 0.5) for i in weights]

    if long == 1:
        res = minimize(maximum_sharpe_ratio_function, x0 = weights, args = (cov, mu, rf), method= 'SLSQP',
                       constraints=cons, bounds=bnds, tol = 1e-30)

    else:

        res = minimize(maximum_sharpe_ratio_function, x0 = weights, args = (cov, mu, rf), method='SLSQP',
                       constraints=cons, tol= 1e-30)

    return res.x

