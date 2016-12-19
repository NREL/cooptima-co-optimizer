# This file is part of the 'co-optimizer' for the Co-optimization of fuels
# and engine project

# Authors:
# Ray Grout (ray.grout@nrel.gov) and Juliane Muller (julianemueller@lbl.gov)

# Created with the support of the DOE/EERE/VTO fuels and lubricants program
# Program Manager Kevin Stork

from __future__ import print_function
import sys
from fuelsdb_interface import load_propDB, make_property_vector
from optimizer import run_optimize_vs_C as run_optimize_pyomo_C,\
                      comp_to_cost_mmf, comp_to_mmf,\
                      run_optimize_vs_K as run_optimize_pyomo_K
from nsga2_k import nsga2_pareto_K as run_optmize_nsga2
import matplotlib.pyplot as plt
import numpy as np
import cooptimizer_input
clr = ['fuchsia', 'b', 'g', 'r', 'y', 'm', 'c', 'k', 'g', 'r', 'y', 'm']
mrk = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x', 'x']


def write_composition(f, c, hdr_in=None, prefix=None):
    vals = ""
    hdr = []
    if hdr_in is None:
        for k in c:
            hdr.append(k)
        if prefix is not None:
            for i in range(len(prefix)):
                f.write("\t")
        for h in hdr:
            f.write("{}\t".format(h))
        f.write("\n")
    else:
        hdr = hdr_in.copy()

    for h in hdr:
        f.write("{}\t".format(c[h]))
    f.write("\n")


if __name__ == '__main__':

    print ("=================================================================")
    print ("Welcome to the Co-optimizer")
    print ("=================================================================")

    print ('-----------------------------------------------------------------')
    print ("Setting up:")
    print ("Reading fuel component properties from: ",
           cooptimizer_input.component_properties_database)
    propDB = load_propDB(cooptimizer_input.component_properties_database,
                         maxrows=18, maxcols=14)
    print ("Reading fuel component costs from: ",
           cooptimizer_input.component_cost_database)
    propDB = load_propDB(cooptimizer_input.component_cost_database,
                         propDB_initial=propDB, maxrows=18, maxcols=2)
    print ('-----------------------------------------------------------------')

    output_files = []

    for t, v in cooptimizer_input.task_list.iteritems():
        if v:
            ans = 'Yes'
        else:
            ans = 'No'
        print ('Planning to perform task: ', t, '\t\t', ans)

    if cooptimizer_input.task_list['cost_vs_merit_Pareto']:
        plt.close()
        compfile = open(cooptimizer_input.cost_vs_merit_datafilename, 'w')
        print ("Running cost vs merit function Pareto front analysis")
        n = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(n, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:n+1], mrk[0:n+1]):
            if cooptimizer_input.use_pyomo:
                C = []
                M = []
                compfile.write("K = {}-------------------------\n".format(KK))
                for cs in np.linspace(1.5, 15.0, 10):
                    comp, isok = run_optimize_pyomo_C(cs, KK, propDB)
                    if (isok):
                        c, m = comp_to_cost_mmf(comp, propDB, KK)
                        C.append(c)
                        M.append(m)
                        write_composition(compfile, comp)
                compfile.write("\n")
            elif cooptimizer_input.use_deap_NSGAII:
                C, M = run_optmize_nsga2(KK, propvec)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
            plt.scatter(C, M, label="K={}".format(KK), marker=mk, color=col)
            plt.xlabel('Cost')
            plt.ylabel('Merit')
        plt.legend(loc=8, ncol=3, fontsize=10)
        plt.savefig(cooptimizer_input.cost_vs_merit_plotfilename, form='pdf')
        output_files.append(cooptimizer_input.cost_vs_merit_plotfilename)
        output_files.append(cooptimizer_input.cost_vs_merit_datafilename)

    if cooptimizer_input.task_list['K_vs_merit_sweep']:
        plt.close()
        compfile = open(cooptimizer_input.k_sweep_datafilename, 'w')
        print ("Running K vs merit function sweep")
        n = len(cooptimizer_input.KVEC)
        # print ("Running {} K values: {}".format(n, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)

        if cooptimizer_input.use_deap_NSGAII:
            print("Not yet implemented using NSGAII")
            sys.exit(-1)

        M = []
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:n+1], mrk[0:n+1]):
            compfile.write("K = {}-------------------------\n".format(KK))
            if cooptimizer_input.use_pyomo:
                comp, isok = run_optimize_pyomo_K(KK, propDB)
                if (isok):
                    write_composition(compfile, comp)
                    compfile.write("\n")
                    m = comp_to_mmf(comp, propDB, KK)
                    M.append(m)
            elif cooptimizer_input.use_deap_NSGAII:
                C, M = run_optmize_nsga2(KK, propvec)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
        print ("{}".format(cooptimizer_input.KVEC))
        print ("{}".format(M))
        plt.scatter(cooptimizer_input.KVEC, M)
        plt.xlabel('K')
        plt.ylabel('Merit')
        plt.savefig(cooptimizer_input.k_sweep_plotfilename, form='pdf')
        output_files.append(cooptimizer_input.k_sweep_plotfilename)
        output_files.append(cooptimizer_input.k_sweep_datafilename)

    print("==================================================================")
    print("Analysis completed; new output files")
    for f in output_files:
        print(f)
    print("==================================================================")
    sys.exit(0)
