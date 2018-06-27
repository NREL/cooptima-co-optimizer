import numpy as np
from scipy.optimize import minimize
from merit_functions import revised_mf as mmf_single
from blend_functions import blend_fancy_vec as blend
from pyDOE import *


def maximize_merit(KK, propvec,  pDB):
    NDIM = len(propvec['COST'])
    ntrials = 30  # number of local optimizations
    bound_list = []
    cons = {'type': 'eq', 'fun': const}
    fbest = np.inf
    startpoints = lhs(NDIM, samples=ntrials, criterion="maximin")
    n0 = np.divide(startpoints, np.asmatrix(np.sum(startpoints, axis=1)).T)
    for ii in range(NDIM):
        bound_pair = (0., 1.)
        bound_list.append(bound_pair)
    for ii in range(ntrials):
        x0 = startpoints[ii, :]  # np.random.rand(NDIM)
        res = minimize(objective, x0, args=(propvec, KK), bounds=bound_list,
                       constraints=cons, tol=1e-6)
        # print(res.nfev)
        xnew = res.x
        fnew = objective(xnew, propvec, KK)
        if fnew < fbest:
            xbest = xnew
            fbest = fnew

    return -fnew  # return negative of best value because maximization


def objective(individual, propvec, Kinp):
    # compute miles function
    this_ron = blend(individual, propvec, 'RON')
    this_s = blend(individual, propvec, 'S')
    this_HoV = blend(individual, propvec, 'HoV')
    this_SL = blend(individual, propvec, 'SL')
    this_AFR = blend(individual, propvec, 'AFR_STOICH')
    this_LFV150 = blend(individual, propvec, 'LFV150')
    this_PMI = blend(individual, propvec, 'PMI')
    cost_f = blend(individual, propvec, 'COST')

    merit_f = mmf_single(RON=this_ron, S=this_s,
                         HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)

    return -merit_f  # maximize merit = minimize -merit


def const(child):  # sum over all fractions = 1 constraint
    sumChild = sum(child[0:22])
    for i in xrange(len(child)):
        child[i] = child[i]/sumChild
    return sum(child)-1
