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
import copy
import time

from fuelsdb_interface import load_propDB,\
                              make_property_vector_all_sample_cost,\
                              make_property_vector_all_sample_cost_UP_single
from fuelsdb_interface import make_property_vector_all, make_property_vector 
from optimizer import run_optimize_vs_C as run_optimize_pyomo_C,\
                      comp_to_cost_mmf, comp_to_mmf,\
                      run_optimize_vs_K as run_optimize_pyomo_K
from nsga2_k import nsga2_pareto_K as run_optmize_nsga2
import numpy as np
import cooptimizer_input
from matplotlib.backends.backend_pdf import PdfPages
from maximize_merit import maximize_merit
from surrogate_optimization import surrogate_optimization
from surrogateMO_optimization import surrogateMO_optimization
#parallel stuff
from multiprocessing import Pool
import random
#-----------------------------------
clr = ['fuchsia', 'b', 'g', 'r', 'y', 'm', 'c', 'k', 'g', 'r', 'y', 'm']
mrk = ['o', 'x', 's', '^', '<', '*', 'o', 'o', 'x', 'x', 'x', 'x', 'x']

#-----------------------------------
def wrapper_func(n,varID,propDB, KK):
    np.random.seed(n)
    ncomp, spc_names, propvec = make_property_vector_all_sample_cost_UP_single(propDB, change_name) #only randomly sample cost for one (change_name) fuel component
    #print(propvec['COST'], ncomp)
    
    Pfront = run_optmize_nsga2(KK, propvec, propDB) #Pfront = [-merit, cost, obj3]
    return Pfront, propvec, spc_names


def wrapper_func_all(n,propDB, KK): #randomly samples cost for all fuel components
    np.random.seed(n)
    ncomp, spc_names, propvec = make_property_vector_all_sample_cost(propDB)
    Pfront = run_optmize_nsga2(KK, propvec, propDB) #Pfront = [-merit, cost, obj3]

    

    return Pfront, propvec, spc_names


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
    t0=time.time()
    np.random.seed(0)
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
                         propDB_initial=propDB, maxrows=18, maxcols=3)
    print ('-----------------------------------------------------------------')

    output_files = []

    for t, v in cooptimizer_input.task_list.items():
        if v:
            ans = 'Yes'
        else:
            ans = 'No'
        print ('Planning to perform task: ', t, '\t\t', ans)

    if cooptimizer_input.task_list['cost_vs_merit_Pareto']:
        plt.close()
        compfile = open(cooptimizer_input.cost_vs_merit_datafilename, 'w')
        print ("Running cost vs merit function Pareto front analysis")
        nK = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(nK, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector_all(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:nK+1], mrk[0:nK+1]):
            if cooptimizer_input.use_pyomo:
                sys.exit(-2)
                C = []
                M = []
                compfile.write("K = {}-------------------------\n".format(KK))
                for cs in np.linspace(1.5, 15.0, 15):
                    comp, isok = run_optimize_pyomo_C(cs, KK, propDB)
                    if (isok):
                        c, m = comp_to_cost_mmf(comp, propDB, KK)
                        C.append(c)
                        M.append(m)
                        write_composition(compfile, comp)
                compfile.write("\n")
            elif cooptimizer_input.use_deap_NSGAII:
                
                front = run_optmize_nsga2(KK, propvec, propDB)
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)

            st1 = '{}'
            for stc in range(front.shape[1]-1):
                st1=st1+',{}'
            st1 = st1+'\n'
            pareto_txt = 'poster_pareto_MMF_GPrejectoutsider_K_'+str(KK)+'.txt'
            output_files.append(pareto_txt)
            spdfile = open(pareto_txt,'w') 
            for lin in range(front.shape[0]):
                f_write = list()
                for addf in range(front.shape[1]):
                    f_write.append(front[lin, addf])
                spdfile.write(st1.format(*tuple(f_write)))
            spdfile.close()
                
            if front.shape[1]==2: #make scatter plot if we have only 2 objectives    
                fig = plt.figure(figsize=(10,8))
                #plt.scatter(front[:,0], front[:,1], label="K={}".format(KK), marker=mk, color=col)
                plt.scatter(front[:,0], front[:,1],  marker=mk, color=col)
                #plt.ylabel('Min Std')
                plt.xlabel('Maximize MMF')
                plt.ylabel('Maximize NMEP')
                #plt.xlabel('Maximize mean NMEP')
                #plt.ylabel('Minimize variance NMEP')

        if front.shape[1]==2:        
            plt.legend(loc=1, ncol=1, fontsize=20)
            #plt.title("Co-Optimizer NSGA2 MMF vs cost")
            ax = plt.gca()
            ax.ticklabel_format(style = 'plain', axis ='both')
            ax.ticklabel_format(useOffset=False, style='plain')
            plt.savefig(cooptimizer_input.cost_vs_merit_plotfilename, form='pdf')
            output_files.append(cooptimizer_input.cost_vs_merit_plotfilename)
        #output_files.append(cooptimizer_input.cost_vs_merit_datafilename)

    if cooptimizer_input.task_list['mean_vs_var_Pareto']:
        plt.close()
        compfile = open(cooptimizer_input.mean_vs_var_merit_datafilename, 'w')
        print ("Running merit mean vs merit variance multiobjective analysis")
        nK = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(nK, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector_all(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:nK+1], mrk[0:nK+1]):
            if cooptimizer_input.use_pyomo:
                sys.exit(-2)
                C = []
                M = []
                compfile.write("K = {}-------------------------\n".format(KK))
                for cs in np.linspace(1.5, 15.0, 15):
                    comp, isok = run_optimize_pyomo_C(cs, KK, propDB)
                    if (isok):
                        c, m = comp_to_cost_mmf(comp, propDB, KK)
                        C.append(c)
                        M.append(m)
                        write_composition(compfile, comp)
                compfile.write("\n")
            elif cooptimizer_input.use_deap_NSGAII:
                front = run_optmize_nsga2(KK, propvec, propDB)

            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)

            st1 = '{}'
            for stc in range(front.shape[1]-1):
                st1=st1+',{}'
            st1 = st1+'\n'
            pareto_txt = 'sampling_pareto_mean_var_K_'+str(KK)+'.txt'
            output_files.append(pareto_txt)
            spdfile = open(pareto_txt,'w') 
            for lin in range(front.shape[0]):
                f_write = list()
                for addf in range(front.shape[1]):
                    f_write.append(front[lin, addf])
                spdfile.write(st1.format(*tuple(f_write)))
            spdfile.close()
                
            if front.shape[1]==2: #make scatter plot if we have only 2 objectives    
                plt.scatter(front[:,0], front[:,1], label="K={}".format(KK), marker=mk, color=col)
                plt.xlabel('Merit mean (maximize)')
                plt.ylabel('Merit variance (minimize)')

        if front.shape[1]==2:        
            plt.legend(loc=8, ncol=3, fontsize=10)
            plt.title("Co-Optimizer GA Pareto Front")
            plt.savefig(cooptimizer_input.mean_vs_var_merit_plotfilename, form='pdf')
            output_files.append(cooptimizer_input.mean_vs_var_merit_plotfilename)  

    # UQ for individual cost components (assuming uncertainty in only a single component while all others are deterministic)
    if cooptimizer_input.task_list['cost_vs_merit_Pareto_UP_single']:
        plt.close()

        compfile = open(cooptimizer_input.cost_vs_merit_Pareto_UP_single_datafilename, 'w')
        print ("Running cost vs merit function Pareto front analysis with sampling of individual cost components")
        nK = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(nK, cooptimizer_input.KVEC))
        ncomp, init_names, propvec = make_property_vector_all(propDB)
        
        printed = False
        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:nK+1], mrk[0:nK+1]):
            if cooptimizer_input.use_pyomo:
                print("Not yet implemented using PyOmo")
                sys.exit(-1)
                C = []
                M = []
                compfile.write("K = {}-------------------------\n".format(KK))
                for cs in np.linspace(1.5, 15.0, 15):
                    comp, isok = run_optimize_pyomo_C(cs, KK, propDB)
                    if (isok):
                        c, m = comp_to_cost_mmf(comp, propDB, KK)
                        C.append(c)
                        M.append(m)
                        write_composition(compfile, comp)
                compfile.write("\n")

            elif cooptimizer_input.use_deap_NSGAII:
                for varID in range(ncomp):
                    costlist = []
                    Front = []
                    change_name = init_names[varID]
                    
                    print('sampling ', change_name)

                    pool = Pool()
                    pool_res = pool.map_async(wrapper_func, ((ns, change_name, propDB, KK) for ns in range(cooptimizer_input.nsamples)))
                    result = pool_res.get()
                    pool.close()
                    for ns in range(cooptimizer_input.nsamples):
                        Front.append(result[ns][0])
                        costlist.append(result[ns][1]['COST'])
                        
                        xnames=result[ns][2] 
                    spc_names =xnames

                    pareto_txt = 'merit_cost_pareto_data_'+change_name+'_K_'+str(KK)+'.txt'
                    output_files.append(pareto_txt)
                    spdfile = open(pareto_txt,'w')

                    madeplot = False
                    st1 = '{}'
                    for stc in range(Front[0].shape[1]-1):
                        st1=st1+',{}'
                    st1 = st1+'\n'

                    for f in zip(Front):
                        for lin in range(f[0].shape[0]):
                            f_write = list()
                            for addf in range(f[0].shape[1]):
                                f_write.append(f[0][lin, addf])

                            spdfile.write(st1.format(*tuple(f_write))) #f[0][lin,0], f[0][lin,1],f[0][lin,2]))
                            #spdfile.write("{},{}\n".format(f[0][lin,0], f[0][lin,1]))
                        if f[0].shape[1] == 2:
                            plt.scatter(f[0][:,0], f[0][:,1],  marker='.')
                            plt.xlabel('Merit (maximize)')
                            plt.ylabel('Cost (minimize)')
                            madeplot = True

                    spdfile.close()

                    if madeplot:
                        plt.legend(loc=8, ncol=3, fontsize=10)
                        plt.title("Co-Optimizer GA Pareto Front")
                        pltname = cooptimizer_input.cost_vs_merit_Pareto_UP_single_plotfilename+'_'+change_name+'_K_'+str(KK)+'.pdf'
                        output_files.append(pltname)
                        plt.savefig(pltname, form='pdf')
                        plt.close()
                    
                    costarray = np.array(costlist)
                    filename = 'cost_samples_'+change_name+'_K_'+str(KK)+'ln.pdf'
                    with PdfPages(filename) as pdf:
                        for i in range(costarray.shape[1]):
                            plt.hist(costarray[:,i])
                            plt.xlabel("{}".format(spc_names[i]))
                            pdf.savefig()
                            plt.close()
                    output_files.append(filename)
                    printed = True

            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)
         
        if not(printed):   
            #plt.legend(loc=8, ncol=3, fontsize=10)
            plt.savefig(cooptimizer_input.cost_vs_merit_plotfilename, form='pdf')
            costarray = np.array(costlist)
            plt.close()
            with PdfPages('cost_samples.pdf') as pdf:
                for i in range(costarray.shape[1]):
                    plt.hist(costarray[:,i])
                    plt.xlabel("{}".format(spc_names[i]))
                    pdf.savefig()
                    plt.close()
        
          
            output_files.append(cooptimizer_input.cost_vs_merit_plotfilename)
            output_files.append(cooptimizer_input.cost_vs_merit_datafilename)


    #Ray's stuff
    if cooptimizer_input.task_list['cost_vs_merit_Pareto_UP']:
        plt.close()
        compfile = open(cooptimizer_input.cost_vs_merit_Pareto_UP_datafilename, 'w')
        print ("Running cost vs merit function Pareto front analysis with sampling of all cost components")
        nK = len(cooptimizer_input.KVEC)
        print ("Running {} K values: {}".format(nK, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector_all(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:nK+1], mrk[0:nK+1]):
            if cooptimizer_input.use_pyomo:
                print("Not yet implemented using PyOmo")
                sys.exit(-1)
                C = []
                M = []
                comppfile.write("K = {}-------------------------\n".format(KK))
                for cs in np.linspace(1.5, 15.0, 15):
                    comp, isok = run_optimize_pyomo_C(cs, KK, propDB)
                    if (isok):
                        c, m = comp_to_cost_mmf(comp, propDB, KK)
                        C.append(c)
                        M.append(m)
                        write_composition(compfile, comp)
                compfile.write("\n")

            elif cooptimizer_input.use_deap_NSGAII:
                costlist = []
                Front = []
                pool = Pool()
                pool_res = pool.map_async(wrapper_func_all, ((ns, propDB, KK) for ns in range(cooptimizer_input.nsamples)))
                result = pool_res.get()
                pool.close()
                for ns in range(cooptimizer_input.nsamples):
                    Front.append(result[ns][0])
                    costlist.append(result[ns][1]['COST'])
                    xnames=result[ns][2]

                    pareto_txt = 'merit_cost_pareto_data_all_K'+str(KK)+'_sample_'+str(ns)+'.txt'
                    spdfile = open(pareto_txt,'w')
                    #spdfile = open('sampling_pareto_data_UP_all.txt','w')
                    madeplot = False
                    st1 = '{}'
                    for stc in range(Front[0].shape[1]-1):
                        st1=st1+',{}'
                    st1 = st1+'\n'
                            
                    for f in zip(Front):
                        for lin in range(f[0].shape[0]):
                            f_write = list()
                            for addf in range(f[0].shape[1]):
                                f_write.append(f[0][lin, addf])
                            spdfile.write(st1.format(*tuple(f_write))) #f[0][lin,0], f[0][lin,1],f[0][lin,2]))
                            #spdfile.write("{},{}\n".format(f[0][lin,0], f[0][lin,1]))
                        #if f[0].shape[1] == 2:
                        #    plt.scatter(f[0][:,0], f[0][:,1], marker='.')
                        #    plt.xlabel('Merit (maximize)')
                        #    plt.ylabel('Cost (minimize)')
                        #    madeplot = True
                    spdfile.close()

                    
                spc_names =xnames
            else:
                print("No valid optimization algorithm specified")
                sys.exit(-1)

            pareto_txt = 'merit_cost_pareto_data_all_K'+str(KK)+'.txt'
            spdfile = open(pareto_txt,'w')
            #spdfile = open('sampling_pareto_data_UP_all.txt','w')
            madeplot = False
            st1 = '{}'
            for stc in range(Front[0].shape[1]-1):
                st1=st1+',{}'
            st1 = st1+'\n'
            
            for f in zip(Front):
                for lin in range(f[0].shape[0]):
                    f_write = list()
                    for addf in range(f[0].shape[1]):
                        f_write.append(f[0][lin, addf])
                    spdfile.write(st1.format(*tuple(f_write))) #f[0][lin,0], f[0][lin,1],f[0][lin,2]))
                    #spdfile.write("{},{}\n".format(f[0][lin,0], f[0][lin,1]))
                if f[0].shape[1] == 2:
                    plt.scatter(f[0][:,0], f[0][:,1], marker='.')
                    plt.xlabel('Merit (maximize)')
                    plt.ylabel('Cost (minimize)')
                    madeplot = True
            spdfile.close()
            output_files.append(pareto_txt)

            if madeplot:
                plt.legend(loc=8, ncol=3, fontsize=12)
                plt.title("Co-Optimizer GA Pareto Front")
                pltname = cooptimizer_input.cost_vs_merit_Pareto_UP_plotfilename+'_K_'+str(KK)+'.pdf'
                output_files.append(pltname)
                plt.savefig(pltname, form='pdf')
                plt.close()

            filename = 'cost_samples_K_'+str(KK)+'.pdf'
            costarray = np.array(costlist)
            with PdfPages(filename) as pdf:
                for i in range(costarray.shape[1]):
                    plt.hist(costarray[:,i])
                    plt.xlabel("{}".format(spc_names[i]))
                    pdf.savefig()
                    plt.close()
            output_files.append(filename)        
            printed = True
      
        
        if not(printed):#this needs to be updated (only active when pyomo is used)
            #plt.legend(loc=8, ncol=3, fontsize=10)
            plt.savefig(cooptimizer_input.cost_vs_merit_Pareto_UP_plotfilename+'_K_'+str(KK)+'.pdf')
            costarray = np.array(costlist)
            plt.close()
            with PdfPages('cost_samples.pdf') as pdf:
                for i in range(costarray.shape[1]):
                    plt.hist(costarray[:,i])
                    plt.xlabel("{}".format(spc_names[i]))
                    pdf.savefig()
                    plt.close()
            output_files.append(cooptimizer_input.cost_vs_merit_plotfilename)
            output_files.append(cooptimizer_input.cost_vs_merit_datafilename)


    if cooptimizer_input.task_list['K_vs_merit_sweep']:
        plt.close()
        compfile = open(cooptimizer_input.k_sweep_datafilename, 'w')
        print ("Running K vs merit function sweep")
        n = len(cooptimizer_input.KVEC)
        # print ("Running {} K values: {}".format(n, cooptimizer_input.KVEC))
        ncomp, spc_names, propvec = make_property_vector_all(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)

        if cooptimizer_input.use_deap_NSGAII:
            M, F = [],[]
            
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
                    #merit = maximize_merit(KK, propvec, propDB)
                    #M.append(merit)
                    
                    data = surrogate_optimization(KK, propvec,  propDB)
                    F.append(-data.Fbest)
                    
                    #data = surrogateMO_optimization(KK, propvec,  propDB)
                    #F.append(-data.Fbest)
                else:
                    print("No valid optimization algorithm specified")
                    sys.exit(-1)
            print ("{}".format(cooptimizer_input.KVEC))
            print ("{}".format(M))
            #plt.scatter(cooptimizer_input.KVEC, M, marker = 'o', c= 'r',s=40,label='python optimizer')
            plt.scatter(cooptimizer_input.KVEC, F, marker = '^', c = 'b',s=40,label='surrogate optimizer')
            plt.xlabel('K')
            plt.ylabel('Merit')
            plt.legend(loc=0, fontsize=10)
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
        KVEC = np.random.normal(cooptimizer_input.kmean, cooptimizer_input.kvar,n)
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
    
    if cooptimizer_input.task_list['UPMO']:
        plt.close()
        compfile = open(cooptimizer_input.mean_vs_var_merit_datafilename, 'w')
        print ("Running uncertainty propagation")
        n = cooptimizer_input.nsamples
        nK = len(cooptimizer_input.KVEC)
        for KK, col, mk in zip(cooptimizer_input.KVEC, clr[0:nK+1], mrk[0:nK+1]):    
            ncomp, spc_names, propvec = make_property_vector_all(propDB)#make_property_vector

            if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
                print("Choose only 1 optimizer method")
                print("(not use_pyomo and use_deap_NSGAII)!")
                sys.exit(-2)

            if cooptimizer_input.use_deap_NSGAII:
                print("Using NSGAII for optimization")
                front = run_optmize_nsga2(KK, propvec, propDB)#,  sen=sen_samples, ref=ref_samples) #first output is merit, second is variance
                print(front)
                print(front.shape, type(front))
                np.savetxt('Paretofront_UPMO.txt', front)
                plt.scatter(front[:,0], front[:,1], label="K={}".format(KK), marker=mk, color=col)
            
            st1 = '{}'
            for stc in range(front.shape[1]-1):
                st1=st1+',{}'
            st1 = st1+'\n'
            pareto_txt = 'sampling_pareto_uncertfuelprop_mean_var_K_'+str(KK)+'.txt'
            spdfile = open(pareto_txt,'w') 
            for lin in range(front.shape[0]):
                f_write = list()
                for addf in range(front.shape[1]):
                    f_write.append(front[lin, addf])
                spdfile.write(st1.format(*tuple(f_write)))
            spdfile.close()

        plt.xlabel('Maximize expected MMF value')
        plt.ylabel('Minimize MMF variance')
        plt.legend(loc=2, ncol=3, fontsize=10)
        #plt.title("Co-Optimizer NSGAII Pareto Front")
        plt.savefig(cooptimizer_input.mean_vs_var_merit_plotfilename+'_ln.pdf')
        output_files.append(cooptimizer_input.mean_vs_var_merit_plotfilename)
        output_files.append(cooptimizer_input.mean_vs_var_merit_datafilename)


    if cooptimizer_input.task_list['UP']:
        plt.close()
        compfile = open(cooptimizer_input.UP_datafilename, 'w')
        print ("Running uncertainty propagation")
        n = cooptimizer_input.nsamples
        
        ncomp, spc_names, propvec = make_property_vector_all(propDB)

        if cooptimizer_input.use_pyomo and cooptimizer_input.use_deap_NSGAII:
            print("Choose only 1 optimizer method")
            print("(not use_pyomo and use_deap_NSGAII)!")
            sys.exit(-2)

        if cooptimizer_input.use_deap_NSGAII:
            print("Not yet implemented using NSGAII")
            #sys.exit(-1)

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
                front = run_optmize_nsga2(KK, propvec,propDB)
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
    t1 = time.time()-t0
    print('time needed:',t1)
    sys.exit(0)
