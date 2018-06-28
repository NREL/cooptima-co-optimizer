#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

import numpy

from math import sqrt

import matplotlib.pyplot as plt

from fuelsdb_interface import *
import cooptima_plotting_tools as cpt
from merit_functions import mmf_single, mft_mmf
from blend_functions import blend_linear as blend

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools

def eval_prop_linear(propDB, comp, prop='COST'):
    pval = 0.0
    for k, v in comp.iteritems():
        pval += propDB[k][prop]*v
    return pval


# Load data from Bob & Gina's database of fuel properties
import csv
import copy
def load_fuelsdb(dbfile, cas=None):
    propDB = {}
    print("Reading fuel properties database")
    with open(dbfile) as fueldbfile:
        fuelsdb = csv.reader(fueldbfile, delimiter=',', quotechar='\"')
        firstrow = True
        for row in fuelsdb:
            if firstrow:
                hdrs = row
                firstrow = False
                # print hdrs
            else:
                propDB_entry = {}
                for h, r in zip(hdrs,row):
                    propDB_entry[h] = r
                if(cas):
                    if(propDB_entry['Pure_CAS'] in cas):
                       propDB[propDB_entry['Pure_CAS']] = copy.deepcopy(propDB_entry)
                else:
                   propDB[propDB_entry['Pure_CAS']] = copy.deepcopy(propDB_entry)
    return propDB


import property_mapping as pm
from fuelsdb_interface import load_propDB


def fraction_constraint(individual): #compute the constraint function value (sum over all parameters = 1)
    return sum(individual)

def eval_mo(individual, Roninp,Sinp, Oninp, HoVinp, SLinp, LFinp, PMinp, Costinp, Kinp):
    # Do some hard computing on the individual

    this_ron = 0
    this_s = 0
    this_on = 0
    this_HoV = 0
    this_SL = 0
    this_LFV150 = 0
    this_PMI = 0
    cost_f=0
    for ii in range(1,18):
        this_ron = this_ron + individual[ii-1] * Roninp[ii]
        this_s = this_s + individual[ii-1] * Sinp[ii]
        this_on = this_on + individual[ii-1] * Oninp[ii]
        this_HoV = this_HoV + individual[ii-1] * HoVinp[ii]
        this_SL = this_SL + individual[ii-1] * SLinp[ii]
        this_LFV150 = this_LFV150 + individual[ii-1] * LFinp[ii]
        this_PMI = this_PMI + individual[ii-1] * PMinp[ii]
        cost_f = cost_f + individual[ii-1] * Costinp[ii]

    #print("individual", individual, "cost:", cost_f)

    merit_f = mmf_single(RON=this_ron, S=this_s, ON=this_on, HoV=this_HoV,SL=this_SL, K=Kinp) #note change in mmf_single (dynamic def of K)
    g_val = fraction_constraint(individual)

    if (numpy.abs(g_val-1) <=1.0e-3):
        penalty_mmf =0
        penalty_cost=0
    else: #theoretically, this part of the condition will never be entered
        raise 'Your parameters dont add up to 1 '

    c_p_p = cost_f+penalty_cost #penalty_cost = 0 if nothing is changed about creation of individuals
    mmf_p_p = -(merit_f+penalty_mmf) #penalty_mmf = 0 if nothing is changed about creation of individuals

    return mmf_p_p, c_p_p

    
def uniform(low, up, size=None): #generate uniform random numbrs in [0,1] and scale so that sum = 1
    X=[random.uniform(a, b) for a, b in zip([low] * size, [up] * size)] #uniformly generate new points within parameter bounds
    sumX=sum(X) #sum up over all parameters in each individual
    Y=[X[ii]/sumX for ii in range(len(X))] #divide each individual's parameters by the sum to satisfy sum x = 1 constraint;
    return Y 

def scale():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                sumChild = sum(child)

                for i in xrange(len(child)):
                    child[i]=child[i]/sumChild
            
            return offspring
        return wrapper
    return decorator

def main(KK,seed=None):

    NDIM = 17 #number of parameters

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) #minimize both objectives (min -f(x) if maximization is needed)
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin) #individuals in the generation

    toolbox = base.Toolbox()
   
    BOUND_LOW, BOUND_UP = 0.0, 1.0 #lower and upper variable bounds 

    propDB = load_propDB("propDB_fiction.xls")

    # Assemble property vectors for each composition
    ncomp = 0
    SPNM = []
    for k in propDB.keys():
        for kk in propDB[k].keys():
            print ("key: {}".format(kk))
        SPNM.append(k)
    
    RONVEC = {}
    SVEC = {}
    ONVEC = {}
    HoVVEC = {}
    SLVEC = {}
    LFV150VEC = {}
    PMIVEC = {}
    COSTVEC = {}
    XVEC = {}

    ncomp = len(SPNM)
    for i in range(1, ncomp+1):
        RONVEC[i] =  ( propDB[SPNM[i-1]]['RON'] )
        SVEC[i] =  ( propDB[SPNM[i-1]]['S'] )
        ONVEC[i] =  ( propDB[SPNM[i-1]]['ON'] )
        HoVVEC[i] =  ( propDB[SPNM[i-1]]['HoV'] )
        SLVEC[i] =  ( propDB[SPNM[i-1]]['SL'] )
        LFV150VEC[i] =  ( propDB[SPNM[i-1]]['LFV150'] )
        PMIVEC[i] =  ( propDB[SPNM[i-1]]['PMI'] )
        COSTVEC[i] =  ( propDB[SPNM[i-1]]['COST'] )
        XVEC[i] =  0.05

    #print COSTVEC


    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_mo, Roninp=RONVEC,Sinp=SVEC, Oninp=ONVEC, HoVinp=HoVVEC, SLinp=SLVEC, LFinp=LFV150VEC, PMinp=PMIVEC, Costinp=COSTVEC,Kinp=KK)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)

    toolbox.decorate("mate", scale())
    toolbox.decorate("mutate",scale())

    toolbox.register("select", tools.selNSGA2)


    #random.seed(KK)

    #these are parameters that can be adjusted and may change the algorithm's performance 
    NGEN = 200 #number of generations 
    MU = 100 #number of individuals
    CXPB = 0.75 #cross-over probability, [0,1]

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

    return pop, logbook, hof, pf
        
if __name__ == "__main__":
    
    KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0] #-vector
    clr =  ['fuchsia','b' ,  'g', 'r', 'y',  'm', 'c','k','g','r','y','m']
    mrk =  ['o',  'o', 'o',  'o', 'o', 'o',  'o', 'o','x','x','x','x','x']
    plt.close()
    for KK,col,mk in zip(KVEC,clr,mrk):

        pop, stats, hof, pf = main(KK)
        pop.sort(key=lambda x: x.fitness.values)    
        front = numpy.array([ind.fitness.values for ind in pop])
    
        numpy.savetxt('test.out', front)
        plt.scatter(front[:,1], -front[:,0], label="K={}".format(KK),marker=mk, color=col)
        
    plt.axis("tight")
    plt.xlabel('Cost')
    plt.ylabel('MMF')
    plt.legend(loc=8, ncol= 3,fontsize=10)
    plt.savefig("nsga2_pareto_K={}.pdf".format(KK),form='pdf')
    plt.close()
    