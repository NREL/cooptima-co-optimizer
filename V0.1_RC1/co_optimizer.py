# This file is part of the 'co-optimizer' for the Co-optimization of fuels
# and engine project

# Authors: Ray Grout (ray.grout@nrel.gov) and Juliane Muller (julianemueller@lbl.gov)

# Created with the support of the DOE/EERE/VTO fuels and lubricants program 
# Program Manager Kevin Stork

from __future__ import print_function
import sys
from fuelsdb_interface import load_propDB, make_property_vector
from optimizer import run_optimize as run_optimize_pyomo, comp_to_cost_mmf
from nsga2_k import nsga2_pareto_K as run_optmize_nsga2
import matplotlib.pyplot as plt
import numpy as np
clr =  ['fuchsia','b' ,  'g', 'r', 'y',  'm', 'c','k','g','r','y','m']
mrk =  ['o',  'o', 'o',  'o', 'o', 'o',  'o', 'o','x','x','x','x','x']

if __name__ == '__main__':


    import cooptimizer_input
    print ("====================================")
    print ("Welcome to the Co-optimizer")
    print ("====================================")

    print ('--------------------------------------')
    print ("Setting up:")
    print ("Reading fuel component properties from: ", cooptimizer_input.component_properties_database)
    propDB = load_propDB(cooptimizer_input.component_properties_database)
    print ("<skipped> Reading fuel component costs from: ", cooptimizer_input.component_cost_database)
    print ('--------------------------------------')
    
    output_files = []

    for t, v in cooptimizer_input.task_list.iteritems():
        if v:
    	    ans = 'Yes'
        else:
            ans = 'No'
        print ('Planning to perform task: ', t, '\t\t', ans)

    if cooptimizer_input.task_list['cost_vs_merit_Pareto']:
        plt.close()
        print ("Running cost vs merit function Pareto front analysis")
        n = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(n, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector(propDB)
    

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method (not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK,col,mk in zip(cooptimizer_input.KVEC,clr[0:n+1],mrk[0:n+1]):
            if cooptimizer_input.use_pyomo:
                C = []
                M = []
                for cs in np.linspace(1.5,15.0, 10):
                    comp = run_optimize_pyomo(cs, KK, propDB)
                    c, m = comp_to_cost_mmf(comp, propDB)
                    C.append(c)
                    M.append(m)
            elif cooptimizer_input.use_deap_NSGAII:
                C, M = run_optmize_nsga2(KK, propvec)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
            plt.scatter(C, M,label="K={}".format(KK),marker=mk, color=col)
            plt.xlabel('Cost')
            plt.ylabel('Merit')
            #plt.savefig("mmf_pareto_K={}.pdf".format(KK),form='pdf')
        #plt.close()
        plt.legend(loc=8, ncol= 3,fontsize=10)
        plt.savefig("mmf_pareto_Ksweep.pdf",form='pdf')

    print("====================================")
    print("Analysis completed; new output files")
    for f in output_files:
    	print(f)
    print("====================================")
    sys.exit(-1)