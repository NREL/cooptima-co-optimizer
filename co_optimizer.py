# -*- coding: utf-8; -*-
"""co_optimizer.py: Driver code for the co_optimizer
--------------------------------------------------------------------------------
Developed by the NREL Computational Science Center
and LBNL Center for Computational Science and Engineering
Contact: Ray Grout <ray.grout@nrel.gov>

Authors: Ray Grout and Juliane Mueller
--------------------------------------------------------------------------------


This file is part of the Co-optimizer, developed as part of the Co-Optimization
of Fuels & Engines (Co-Optima) project sponsored by the U.S. Department of 
Energy (DOE) Office of Energy Efficiency and Renewable Energy (EERE), Bioenergy 
Technologies and Vehicle Technologies Offices. (Optional): Co-Optima is a 
collaborative project of multiple national laboratories initiated to 
simultaneously accelerate the introduction of affordable, scalable, and 
sustainable biofuels and high-efficiency, low-emission vehicle engines.

"""

from __future__ import print_function
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from fuelsdb_interface import load_propDB, make_property_vector
from optimizer import run_optimize_vs_C as run_optimize_pyomo_C,\
                      comp_to_cost_mmf, comp_to_mmf,\
                      run_optimize_vs_K as run_optimize_pyomo_K
from nsga2_k import nsga2_pareto_K as run_optmize_nsga2
import numpy as np
import cooptimizer_input
from matplotlib.backends.backend_pdf import PdfPages
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
    if cooptimizer_input.task_list['K_sampling']:
        plt.close()
        compfile = open(cooptimizer_input.k_sampling_datafilename, 'w')
        print ("Running K vs merit function sweep")
        n = cooptimizer_input.nsamples
        
        ncomp, spc_names, propvec = make_property_vector(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)

        if cooptimizer_input.use_deap_NSGAII:
            print("Not yet implemented using NSGAII")
            sys.exit(-1)

        M = []
        KVEC = []
        KVEC = np.random.normal(cooptimizer_input.kmean,cooptimizer_input.kvar,n)
        for ii in range(n):
            KK = KVEC[ii]
            if cooptimizer_input.use_pyomo:
                comp, isok = run_optimize_pyomo_K(KK, propDB)
                if (isok):
                    write_composition(compfile, comp)
                    compfile.write("\n")
                    m = comp_to_mmf(comp, propDB, KK)
                    M.append(m)
                    print ("sample m = {}".format(m))
            elif cooptimizer_input.use_deap_NSGAII:
                C, M = run_optmize_nsga2(KK, propvec)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
        print ("{}".format(cooptimizer_input.KVEC))
        print ("{}".format(M))  
        f, axs = plt.subplots(1,2)
        axs[0].hist(M,bins=max(20,n/100))
        axs[0].set_xlabel('Maximum Merit distribuiton')

        axs[1].hist(KVEC,bins=max(20,n/100))
        axs[1].set_xlabel('K input distribution')
      
        plt.savefig(cooptimizer_input.k_sampling_plotfilename, form='pdf')
        output_files.append(cooptimizer_input.k_sampling_plotfilename)
        output_files.append(cooptimizer_input.k_sampling_datafilename)
    
    if cooptimizer_input.task_list['UP']:
        plt.close()
        compfile = open(cooptimizer_input.UP_datafilename, 'w')
        print ("Running uncertainty propagation")
        n = cooptimizer_input.nsamples
        
        ncomp, spc_names, propvec = make_property_vector(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)

        if cooptimizer_input.use_deap_NSGAII:
            print("Not yet implemented using NSGAII")
            sys.exit(-1)

        KK = cooptimizer_input.KVEC[0]
        M = []

        sen_samples = {}
        ref_samples = {}
        for kk in cooptimizer_input.sen_mean.keys():
            sen_samples[kk] = []
        for kk in cooptimizer_input.ref_mean.keys():
            ref_samples[kk] = []

        nn = 0
        sen = {}
        ref = {}
        while nn < n:
            # Draw a sample candidate
            for kk in cooptimizer_input.sen_mean.keys():
                sen[kk] = np.random.normal(cooptimizer_input.sen_mean[kk],cooptimizer_input.sen_var[kk])
            for kk in cooptimizer_input.ref_mean.keys():
                ref[kk] = np.random.normal(cooptimizer_input.ref_mean[kk],cooptimizer_input.ref_var[kk])

            # Reject samples outside bounds
            if ref['PMI'] < 0.0:
                continue 
            if ref['S'] < 0.0:
                continue  
            nn += 1

            # If we're good, sotre it and then go on to evaluation
            for kk in cooptimizer_input.sen_mean.keys():
                sen_samples[kk].append(sen[kk])
            for kk in cooptimizer_input.ref_mean.keys():
                ref_samples[kk].append(ref[kk])


            if cooptimizer_input.use_pyomo:
                comp, isok = run_optimize_pyomo_K(KK, propDB, ref=ref, sen=sen)
                if (isok):
                    write_composition(compfile, comp)
                    compfile.write("\n")
                    m = comp_to_mmf(comp, propDB, KK,ref=ref,sen=sen)
                    M.append(m)
                    print ("sample m = {}".format(m))
            elif cooptimizer_input.use_deap_NSGAII:
                C, M = run_optmize_nsga2(KK, propvec)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
        print ("{}".format(cooptimizer_input.KVEC))
        print ("{}".format(M))  
        with PdfPages(cooptimizer_input.UP_plotfilename) as pdf:
            f, ax = plt.subplots(1,1)
            ax.hist(M,bins=max(20,n/100))
            ax.set_xlabel('Maximum Merit distribuiton')
            pdf.savefig()
            plt.close()
            f, axs = plt.subplots(2,4)
            axs[0,0].hist(sen_samples['ON'],alpha=0.4)
            axs[0,0].set_xlabel('ON')
            axs[0,0].locator_params(nbins=4, axis='x')
            
            axs[0,1].hist(sen_samples['ONHoV'],alpha=0.4)
            axs[0,1].set_xlabel('ONHoV')
            axs[0,1].locator_params(nbins=2, axis='x')
            
            axs[0,2].hist(sen_samples['HoV'],alpha=0.4)
            axs[0,2].set_xlabel('HoV')
            axs[0,2].locator_params(nbins=2, axis='x')

            axs[0,3].hist(sen_samples['SL'],alpha=0.4)
            axs[0,3].set_xlabel('SL')
            axs[0,3].locator_params(nbins=4, axis='x')

            axs[1,0].hist(sen_samples['LFV150'],alpha=0.4)
            axs[1,0].set_xlabel('LFV150')
            axs[1,0].locator_params(nbins=4, axis='x')

            axs[1,1].hist(sen_samples['PMIFIX'],alpha=0.4)
            axs[1,1].set_xlabel('PMIFIX')
            axs[1,1].locator_params(nbins=4, axis='x')
            
            axs[1,2].hist(sen_samples['PMIVAR'],alpha=0.4)
            axs[1,2].set_xlabel('PMIVAR')
            axs[1,2].locator_params(nbins=4, axis='x')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            f, axs = plt.subplots(2,3)
            axs[0,0].hist(ref_samples['RON'])
            axs[0,0].set_xlabel('Research Octane')
            axs[0,0].locator_params(nbins=4, axis='x')

            axs[0,1].hist(ref_samples['S'])
            axs[0,1].set_xlabel('Sensitivity')
            axs[0,1].locator_params(nbins=4, axis='x')

            axs[0,2].hist(ref_samples['HoV'])
            axs[0,2].set_xlabel('Heat of Vaporization')
            axs[0,2].locator_params(nbins=4, axis='x')

            axs[1,0].hist(ref_samples['SL'])
            axs[1,0].set_xlabel('Laminar Flame Speed')
            axs[1,0].locator_params(nbins=4, axis='x')

            axs[1,1].hist(ref_samples['PMI'])
            axs[1,1].set_xlabel('Particulate Matter Index')
            axs[1,1].locator_params(nbins=4, axis='x')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        output_files.append(cooptimizer_input.UP_plotfilename)
        output_files.append(cooptimizer_input.UP_datafilename)
    print("==================================================================")
    print("Analysis completed; new output files")
    for f in output_files:
        print(f)
    print("==================================================================")
    sys.exit(0)
