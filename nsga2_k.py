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

#from merit_functions import mmf_single
from merit_functions import revised_mf as mmf_single
#from blend_functions import blend_linear_vec as blend
from blend_functions import blend_fancy_vec as blend

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

def eval_mo(individual, propvec, Kinp):

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
    mmf_p_p = -(merit_f+penalty_mmf)

    return mmf_p_p, c_p_p


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


def nsga2_pareto_K(KK, propvec, seed=None):

    NDIM = len(propvec['COST'])  # Number of parameters

    # Minimize both objectives (min -f(x) if maximization is needed)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))

    # Individuals in the generation
    creator.create("Individual", array.array, typecode='d',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    BOUND_LOW, BOUND_UP = 0.0, 1.0  # Lower and upper variable bounds

    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_mo, propvec=propvec, Kinp=KK)

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
    hof = tools.HallOfFame(50)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pf.update(pop)

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
        pf.update(offspring)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    pop.sort(key=lambda x: x.fitness.values)
    front = numpy.array([ind.fitness.values for ind in pop])
    print("NSGA done; hof: {}".format(pf[0]))
    print("K = {}; Score: {}".format(KK, -eval_mo(pf[0],propvec,KK)[0]))
    print("pv RON = {}".format(propvec['NAME']))
    return front[:, 1], -front[:, 0]

#    return pop, logbook, hof, pf
