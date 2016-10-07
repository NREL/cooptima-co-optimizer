import numpy as np
from fuelsdb_interface import *
import cooptima_plotting_tools as cpt
from merit_functions import mmf_single, mft_mmf
from blend_functions import blend_linear as blend
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyomo.environ
import matplotlib.pyplot as plt


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
if __name__ == '__main__':

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

    print RONVEC

    def run_optimize(cstar, KK):

        def obj_fun(model):
            this_ron = summation( model.X, model.RON,index=model.I)
            this_s = summation( model.X, model.S,index=model.I)
            this_on = summation( model.X, model.ON,index=model.I)
            this_HoV = summation( model.X, model.HoV,index=model.I)
            this_SL = summation( model.X, model.SL,index=model.I)
            this_LFV150 = summation( model.X, model.LFV150,index=model.I)
            this_PMI = summation( model.X, model.PMI,index=model.I)
            return mmf_single(RON=this_ron, S=this_s, ON=this_on, HoV=this_HoV,
                              SL=this_SL, K=KK)

        def fraction_constraint(model):
            return ( np.abs( summation(model.X) - 1.0) <= 1.0e-3) # Need fractions to sum to unity


        def cost_constraint(model):
            return ( summation(model.X, model.COST, index=model.I)  <= cstar )


   #     pyomo formulation

        
        model = ConcreteModel()
        #model = AbstractModel()

        # Properties as parameters
        model.n = Param(within=NonNegativeIntegers, initialize = ncomp)
        model.I = RangeSet(1, model.n)

        model.RON = Param(model.I, initialize=RONVEC)
        model.S = Param(model.I, initialize=SVEC)
        model.ON = Param(model.I, initialize=ONVEC)
        model.HoV = Param(model.I, initialize=HoVVEC)
        model.SL = Param(model.I, initialize=SLVEC)
        model.LFV150 = Param(model.I, initialize=LFV150VEC)
        model.PMI = Param(model.I, initialize=PMIVEC)
        model.COST = Param(model.I, initialize=COSTVEC)
        model.X = Var(model.I, domain=NonNegativeReals, initialize=XVEC)

        # Objective function and constraints
        model.obj = Objective(rule=obj_fun, sense=maximize)
        model.unity = Constraint(rule=fraction_constraint)
        model.cost = Constraint(rule=cost_constraint)


        # Solve
        #opt = SolverFactory('ipopt',solver_io='nl')
        opt = SolverFactory('ipopt', solver_io='nl')
        #opt = SolverFactory('bnb',solver_io='nl')#, solver_io='asl')

        for i in range(1,ncomp+1):
            model.X[i].value = 0.0
        model.X[3] = 0.7
        model.X[7] = 0.3

        #inst = model.create_instance()
        #result = opt.solve(model, tee=True,
        #                   options={'bound_push':1.0e-20,'warm_start_init_point':'yes','acceptable_tol':1.0e-15,'max_iter':10000})
        result = opt.solve(model)

        print result
        #model.load(result)
        print("\nDisplaying Soluiton\n" + '-'*60)
        model.pprint()

        # Extract composition and plot it
        comp = {}
        for i in range(1,ncomp+1):
            comp[SPNM[i-1]] = model.X[i].value

        # Evaluate resulting mmf value
        print comp
        prop_list = ['RON', 'S', 'ON', 'HoV', 'SL', 'LFV150', 'PMI']
        props = {}
        for p in prop_list:
            props[p] = blend(p, propDB, comp)
        mmf = mft_mmf(props)
        cost = eval_prop_linear(propDB, comp, 'COST')
        print "Cost: {}, cstar = {} ".format(cost,cstar)

        # This to make a radar plot of the composition for each solution
        # cpt.plot_comp_radar(propDB, comp, savefile="{}_mfK{}".format(cstar,KK),
        #                     title="Composition, C*={}, mmf={:.2f}".format(cstar,mmf))
        return cost, mmf

    KVEC = [-2.0,     -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0]
    clr =  ['fuchsia','b' ,  'g', 'r', 'y',  'm', 'c','k','g','r','y','m']
    mrk =  ['o',  'o', 'o',  'o', 'o', 'o',  'o', 'o','x','x','x','x','x']
    plt.close()
    for KK,col,mk in zip(KVEC,clr,mrk):
        C = []
        M = []
        for cs in np.linspace(1.5,15.0,30):
            c, m = run_optimize(cs, KK)
            C.append(c)
            M.append(m)

        plt.scatter(C, M,label="K={}".format(KK),marker=mk, color=col)
        plt.xlabel('Cost')
        plt.ylabel('MMF')
        #plt.savefig("mmf_pareto_K={}.pdf".format(KK),form='pdf')
        #plt.close()
    plt.legend(loc=8, ncol= 3,fontsize=10)
    plt.savefig("mmf_pareto_Ksweep.pdf",form='pdf')
    
    #c, m = run_optimize(12, 4.0)
