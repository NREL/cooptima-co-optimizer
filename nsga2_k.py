# -*- coding: utf-8; -*-
"""nsga2_k.py: GA based approach to cost-merit Pareto front identification 
--------------------------------------------------------------------------------
Developed by the NREL Computational Science Center
and LBNL Center for Computational Science and Engineering
Contact: Ray Grout <ray.grout@nrel.gov>

Authors: Ray Grout and Juliane Mueller

N.B.: This approach inspired by the example supplied with DEAP, which 
is licensed under the GNU Lesser General Public License. Since nothing
from DEAP is statically linked to the co-optimizer this should not be 
an infectious license, but eventually we may need to re-implement this 
capability in such a way that we do not infringe on the DEAP license terms.

--------------------------------------------------------------------------------


This file is part of the Co-optimizer, developed as part of the Co-Optimization
of Fuels & Engines (Co-Optima) project sponsored by the U.S. Department of 
Energy (DOE) Office of Energy Efficiency and Renewable Energy (EERE), Bioenergy 
Technologies and Vehicle Technologies Offices. (Optional): Co-Optima is a 
collaborative project of multiple national laboratories initiated to 
simultaneously accelerate the introduction of affordable, scalable, and 
sustainable biofuels and high-efficiency, low-emission vehicle engines.

"""

import array
import random

import numpy
from multiprocessing import Pool
#from scoop import futures

import cooptimizer_input
#from merit_functions import mmf_single
from merit_functions import mmf_single_param
from merit_functions import revised_mf as mmf_single
#from blend_functions import blend_linear_vec as blend
from blend_functions import blend_fancy_vec as blend
from fuelsdb_interface import make_property_vector_all_sample_cost


from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools


def fraction_constraint(individual):
    """Compute the constraint function value (sum over all parameters = 1)"""
    return sum(individual)

def bob_constraint(individual, propvec):
    return sum(individual*propvec['BOB'])


def n_samp_wrap((n,individual,propDB,Kinp)):
    numpy.random.seed(n)
    
    ncomp, spc_names, propvec = make_property_vector_all_sample_cost(propDB)

    this_ron = blend(individual, propvec, 'RON')
    this_s = blend(individual, propvec, 'S')
    this_HoV = blend(individual, propvec, 'HoV')
    #    this_SL = blend(individual, propvec, 'SL')
    this_AFR = blend(individual, propvec, 'AFR_STOICH')
    this_LFV150 = blend(individual, propvec, 'LFV150')
    this_PMI = blend(individual, propvec, 'PMI')
    cost_f = blend(individual, propvec, 'COST')
    #    merit_f = mmf_single(RON=this_ron, S=this_s,
    #                         HoV=this_HoV, SL=this_SL, K=Kinp)
    merit_vec = mmf_single(RON=this_ron, S=this_s,
                         HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)
    cost_vec = cost_f

    
    #cost_vec = numpy.random.rand(1)[0]
    return cost_vec #merit_vec

def eval_mean_var(individual, propDB, Kinp):

    n_samp = 10
    merit_vec = numpy.zeros(n_samp)
    cost_vec = numpy.zeros(n_samp)

    #numpy.random.seed(n)
    #ncomp, spc_names, propvec = make_property_vector_all_sample_cost(propDB)

    pool = Pool()
    pool_res = pool.map_async(n_samp_wrap, ((ns,individual, propDB, Kinp) for ns in range(n_samp)))#range(cooptimizer_input.nsamples)))
    result = pool_res.get()
    

    for ns in range(n_samp):#(cooptimizer_input.nsamples):
        cost_vec[ns] = n_samp_wrap((ns,individual, propDB, Kinp))

        #print(result)
        #print(result[ns])
        #cost_vec[ns]=result[ns]

        #Front.append(result[ns][0])
        #costlist.append(result[ns][1]['COST'])
        #xnames=result[ns][2]

    #print cost_vec
    #lll
    '''
    for ii in range(n_samp):
        ncomp, spc_names, propvec = make_property_vector_all_sample_cost(propDB)

        this_ron = blend(individual, propvec, 'RON')
        this_s = blend(individual, propvec, 'S')
        this_HoV = blend(individual, propvec, 'HoV')
        #    this_SL = blend(individual, propvec, 'SL')
        this_AFR = blend(individual, propvec, 'AFR_STOICH')
        this_LFV150 = blend(individual, propvec, 'LFV150')
        this_PMI = blend(individual, propvec, 'PMI')
        cost_f = blend(individual, propvec, 'COST')

        #    merit_f = mmf_single(RON=this_ron, S=this_s,
        #                         HoV=this_HoV, SL=this_SL, K=Kinp)
        merit_vec[ii] = mmf_single(RON=this_ron, S=this_s,
                         HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)
    '''

    cost_mean = numpy.mean(cost_vec)
    cost_var = numpy.var(cost_vec)    

    #print cost_mean
    #print cost_var

    #kkk
    '''
    g_val = fraction_constraint(individual)

    if (numpy.abs(g_val-1) <= 1.0e-3):
        penalty_mmf = 0
        penalty_cost = 0
    else:  # theoretically, this part of the condition will never be entered
        raise 'Your parameters dont add up to 1 '

    # penalty_cost = 0 if nothing is changed about creation of individuals
    c_p_p = cost_f+penalty_cost
#    c_p_p = 0.0 #take out

    # penalty_mmf = 0 if nothing is changed about creation of individuals
    mmf_p_p = (merit_f+penalty_mmf) #minimize -Merit

    #add here more objective functions as needed -- add 1/-1 in line 125, weights (-1 = minimize, 1 = maximize)
    obj3 = numpy.prod(individual)
    return mmf_p_p, c_p_p, obj3
    '''
    return cost_mean, cost_var


def eval_mo(individual, propvec, Kinp):

    this_ron = blend(individual, propvec, 'RON')
    this_s = blend(individual, propvec, 'S')
    this_HoV = blend(individual, propvec, 'HoV')
    this_SL = blend(individual, propvec, 'SL')
    this_AFR = blend(individual, propvec, 'AFR_STOICH')
    this_LFV150 = blend(individual, propvec, 'LFV150')
    this_PMI = blend(individual, propvec, 'PMI')
    cost_f = blend(individual, propvec, 'COST')

#    merit_f = mmf_single(RON=this_ron, S=this_s,
#                         HoV=this_HoV, SL=this_SL, K=Kinp)
    merit_f = mmf_single(RON=this_ron, S=this_s,
                         HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)


    g_val = fraction_constraint(individual)

    if (numpy.abs(g_val-1) <= 1.0e-3):
        penalty_mmf = 0
        penalty_cost = 0
    else:  # theoretically, this part of the condition will never be entered
        raise 'Your parameters dont add up to 1 '

    # penalty_cost = 0 if nothing is changed about creation of individuals
    c_p_p = cost_f+penalty_cost
#    c_p_p = 0.0 #take out

    # penalty_mmf = 0 if nothing is changed about creation of individuals
    mmf_p_p = -(merit_f+penalty_mmf) #minimize -Merit

    #add here more objective functions as needed -- add 1/-1 in line 125, weights (-1 = minimize, 1 = maximize)
    obj3 = numpy.prod(individual)
    return mmf_p_p, c_p_p#, obj3


def eval_mo2(individual, propvec, Kinp):#, ref_in, sen_in):

    this_ron = blend(individual, propvec, 'RON')
    this_s = blend(individual, propvec, 'S')
    this_HoV = blend(individual, propvec, 'HoV')
    this_SL = blend(individual, propvec, 'SL')
    this_AFR = blend(individual, propvec, 'AFR_STOICH')
    this_LFV150 = blend(individual, propvec, 'LFV150')
    this_PMI = blend(individual, propvec, 'PMI')
    cost_f = blend(individual, propvec, 'COST')

#    merit_f = mmf_single(RON=this_ron, S=this_s,
#                         HoV=this_HoV, SL=this_SL, K=Kinp)
    #merit_f = mmf_single(RON=this_ron, S=this_s,
    #                     HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)

    #this function for uncertainty in fuel components:
    #loop through all ref-sen samples
    n = cooptimizer_input.nsamples
    merit_vec = numpy.zeros(n)

    sen_samples = {}
    ref_samples = {}
    for kk in cooptimizer_input.sen_mean.keys():
        sen_samples[kk] = []
    for kk in cooptimizer_input.ref_mean.keys():
        ref_samples[kk] = []

    nn = 0
    p=0
    sen = {}
    ref = {}
    while nn < n:
        #numpy.random.seed(nn+p)
        # Draw a sample candidate
        for kk in cooptimizer_input.sen_mean.keys():
            if cooptimizer_input.sen_var[kk] ==0:
                sen[kk] =cooptimizer_input.sen_mean[kk]
            else:
                sen[kk] = numpy.random.normal(cooptimizer_input.sen_mean[kk],cooptimizer_input.sen_var[kk])
        for kk in cooptimizer_input.ref_mean.keys():
            if cooptimizer_input.ref_var[kk]==0:
                ref[kk] = cooptimizer_input.ref_mean[kk]
            else:
                ref[kk] = numpy.random.normal(cooptimizer_input.ref_mean[kk],cooptimizer_input.ref_var[kk])
        # Reject samples outside bounds
        if ref['PMI'] < 0.0:
            p=p+1
            continue 
        if ref['S'] < 0.0:
            p=p+1
            continue  
        
        '''
        # If we're good, sotre it and then go on to evaluation
        for kk in cooptimizer_input.sen_mean.keys():
            sen_samples[kk].append(sen[kk])
        for kk in cooptimizer_input.ref_mean.keys():
            ref_samples[kk].append(ref[kk])

        '''
        merit_vec[nn] = mmf_single_param(ref,sen,RON=this_ron, S=this_s, HoV=this_HoV, SL=this_SL,
                     LFV150=this_LFV150, PMI=this_PMI, K=Kinp)

        #merit_f = mmf_single(RON=this_ron, S=this_s,
        #                 HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)
        
        nn += 1
    #print(merit_vec)
    
    merit_mean = numpy.mean(merit_vec)
    merit_var = numpy.var(merit_vec)
    
    '''
    g_val = fraction_constraint(individual)

    if (numpy.abs(g_val-1) <= 1.0e-3):
        penalty_mmf = 0
        penalty_cost = 0
    else:  # theoretically, this part of the condition will never be entered
        raise 'Your parameters dont add up to 1 '

    # penalty_cost = 0 if nothing is changed about creation of individuals
    c_p_p = cost_f+penalty_cost
#    c_p_p = 0.0 #take out

    # penalty_mmf = 0 if nothing is changed about creation of individuals
    mmf_p_p = (merit_f+penalty_mmf) #minimize -Merit

    #add here more objective functions as needed -- add 1/-1 in line 125, weights (-1 = minimize, 1 = maximize)
    obj3 = numpy.prod(individual)
    '''
    #return mmf_p_p, c_p_p#, obj3    
    return merit_mean, merit_var#, cost_f

def uniform(low, up, size=None):
    """generate uniform random numbrs in [0,1] and scale so that sum = 1"""

    # Uniformly generate new points within parameter bounds
    X = [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    sumX = sum(X)  # sum up over all parameters in each individual

    # Divide each individual's parameters by the sum to satisfy
    # sum x = 1 constraint;
    Y = [X[ii] / sumX for ii in range(len(X))]
    return Y


def scale():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                sumChild = sum(child)

                for i in xrange(len(child)):
                    child[i] = child[i]/sumChild

            return offspring
        return wrapper
    return decorator


def nsga2_pareto_K(KK, propvec,  pDB, sen=None, ref=None,seed=None):

    NDIM = len(propvec['COST'])  # Number of parameters

    # Minimize both objectives (min -f(x) if maximization is needed)
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))

    # Individuals in the generation
    creator.create("Individual", array.array, typecode='d',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    BOUND_LOW, BOUND_UP = 0.0, 1.0  # Lower and upper variable bounds
    if not(cooptimizer_input.parallel_nsgaruns):
        pool =Pool()
        toolbox.register("map",pool.map)
    #toolbox.register("map",futures.map)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_mo2, propvec=propvec, Kinp=KK)
    #toolbox.register("evaluate", eval_mo2, propvec=propvec, Kinp=KK)#, ref_in = ref, sen_in = sen )
    #toolbox.register("evaluate", eval_mean_var, propDB=pDB, Kinp=KK)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW,
                     up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)

    toolbox.decorate("mate", scale())
    toolbox.decorate("mutate", scale())

    toolbox.register("select", tools.selNSGA2)

    #  These are parameters that can be adjusted and may change
    #  the algorithm's performance
    #NGEN = 300  # Number of generations
    #MU = 100  # Number of individuals
    NGEN=300
    MU=100
    CXPB = 0.75  # Cross-over probability, [0,1]

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pf = tools.ParetoFront()
    hof = tools.HallOfFame(100)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pf.update(pop)
    hof.update(pop)
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selNSGA2(pop, len(pop))
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        try:
            pf.update(offspring)
        except:
            print([ind.fitness.values for ind in offspring])
            lll

        hof.update(offspring)
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    pop.sort(key=lambda x: x.fitness.values)
    front = numpy.array([ind.fitness.values for ind in pf])
    print("NSGA done; hof: {}".format(pf[0]))
    print("K = {}; Score: {}".format(KK, -eval_mo(pf[0],propvec,KK)[0]))
    print("pv RON = {}".format(propvec['NAME']))
    print('paretofront:',front)
    
    

    return front # front[:, 1], -front[:, 0], front[:,2]

#    return pop, logbook, hof, pf
