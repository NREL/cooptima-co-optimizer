import numpy as np
#from fuelsdb_interface import *
import cooptima_plotting_tools as cpt
from merit_functions import mmf_single, mft_mmf
from blend_functions import blend_linear_propDB as blend_linear_propDB
from blend_functions import blend_linear_pyomo as blend_linear_pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyomo.environ
import matplotlib.pyplot as plt

from fuelsdb_interface import make_property_dict

def run_optimize(cstar, KK, propDB):
        print("Running optimizer based on IPOPT for c^* = {}, KK={}".format(cstar, KK))
        ncomp, spc_names, propvec = make_property_dict(propDB)

        def obj_fun(model):
            this_ron = blend_linear_pyomo(model,'RON')
            this_s = blend_linear_pyomo(model,'S')
            this_on = blend_linear_pyomo(model,'ON')
            this_HoV = blend_linear_pyomo(model,'HoV')
            this_SL = blend_linear_pyomo(model,'SL')
            this_LFV150 = blend_linear_pyomo(model,'LFV150')
            this_PMI = blend_linear_pyomo(model,'PMI')
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

        model.RON = Param(model.I, initialize=propvec['RON'])
        model.S = Param(model.I, initialize=propvec['S'])
        model.ON = Param(model.I, initialize=propvec['ON'])
        model.HoV = Param(model.I, initialize=propvec['HoV'])
        model.SL = Param(model.I, initialize=propvec['SL'])
        model.LFV150 = Param(model.I, initialize=propvec['LFV150'])
        model.PMI = Param(model.I, initialize=propvec['PMI'])
        model.COST = Param(model.I, initialize=propvec['COST'])

        #XVEC = np.zeros(ncomp)
        XVEC = 0.05
        model.X = Var(model.I, domain=NonNegativeReals, initialize=XVEC)

        # Objective function and constraints
        model.obj = Objective(rule=obj_fun, sense=maximize)
        model.unity = Constraint(rule=fraction_constraint)
        model.cost = Constraint(rule=cost_constraint)


        # Solve
        opt = SolverFactory('ipopt', solver_io='nl')

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
            comp[spc_names[i-1]] = model.X[i].value

        return comp


def comp_to_cost_mmf(comp, propDB):
        # Evaluate resulting mmf value
        print comp
        prop_list = ['RON', 'S', 'ON', 'HoV', 'SL', 'LFV150', 'PMI']
        props = {}
        for p in prop_list:
            props[p] = blend_linear_propDB(p, propDB, comp)
        mmf = mft_mmf(props)
        cost = blend_linear_propDB('COST', propDB, comp)
#        print "Cost: {}, cstar = {} ".format(cost,cstar)

        # This to make a radar plot of the composition for each solution
        # cpt.plot_comp_radar(propDB, comp, savefile="{}_mfK{}".format(cstar,KK),
        #                     title="Composition, C*={}, mmf={:.2f}".format(cstar,mmf))
        return cost, mmf

import property_mapping as pm
from fuelsdb_interface import load_propDB
if __name__ == '__main__':

    propDB = load_propDB("propDB_fiction.xls")

    p = {}
    p['RON'] = 103.0
    p['S'] = 16.9
    p['ON'] = 86.1
    p['HoV'] = 392.2
    p['SL'] = 26.0
    p['LFV150'] = 0.0
    p['PMI'] = 1.4
    print "cheapest mmf: ", mft_mmf(p)

    # KVEC = [-2.0,     -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0]
    KVEC = [-2.0, 0.5,  1.0]
    clr =  ['fuchsia','b' ,  'g', 'r', 'y',  'm', 'c','k','g','r','y','m']
    mrk =  ['o',  'o', 'o',  'o', 'o', 'o',  'o', 'o','x','x','x','x','x']
    plt.close()
    for KK,col,mk in zip(KVEC,clr[0:2],mrk[0:2]):
        print("hi")
        C = []
        M = []
        for cs in np.linspace(1.5,15.0, 10):
            comp = run_optimize(cs, KK, propDB)
            c, m = comp_to_cost_mmf(comp, propDB)
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
