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
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from fuelsdb_interface import load_propDB, make_property_vector,\
                              make_property_vector_sample_cost,\
                              make_property_vector_all
#from optimizer import run_optimize_vs_C as run_optimize_pyomo_C,\
#                      comp_to_cost_mmf, comp_to_mmf,\
#                      run_optimize_vs_K as run_optimize_pyomo_K
#from nsga2_k import nsga2_pareto_K as run_optmize_nsga2
import numpy as np
#import cooptimizer_input
from matplotlib.backends.backend_pdf import PdfPages
from merit_functions import mmf_single_param, mmf_single

from blend_functions import blend_linear_vec, blend_fancy_vec, volume_to_other
from merit_functions import jim_bob_mf, revised_mf
import xlrd
def get_xvec(comp, spids):

    x = np.zeros([len(spids)])
    for i in range(len(spids)):
        if spids[i] in comp.keys():
            x[i] = comp[spids[i]]

    print("{}".format(x))

    return x



if __name__ == '__main__':
    print ("=================================================================")
    print ("Testing non-linear blending")
    print ("=================================================================")

    propDB = load_propDB('testDB.xls')
    ncomp, spids, propvec = make_property_vector_all(propDB)
    print("{}; {}".format( ncomp, spids))
#    print("{}".format(propvec))

    # Set up some compositions to test:
    #comp1 = {'BOB-1':1.0}
    #comp1 = {'64-17-5':1.0}

    # Volume fractions:

    wb = xlrd.open_workbook('test_list.xlsx')
    ws = wb.sheet_by_index(0)
    cas_list = []
    jbm_list = []
    cas_col = ws.col(0)
    for c,jb in zip(cas_col[1:], ws.col(2)[1:]):
        cas_list.append(c.value)
        jbm_list.append(jb.value)



    mf_vals = []
    mf_idx = []
    mf_knom = []
    mf_knom_idx = []
    isamp = 0
    for c in cas_list:
        if len(c) > 0:
            print(" c = {}".format(c))
            comp = {c:0.1, 'BOB-1':0.9}
            x = get_xvec(comp, spids)
            x = volume_to_other(x, propvec, 'MOLE')
            print ("mole fracs: {}".format(x))
             
            ron = blend_linear_vec(x,propvec,'RON')
            print ("Mixture RON: {}".format(ron))
    
            ron = blend_fancy_vec(x,propvec,'RON')
            print ("Fancy Mixture RON: {}".format(ron))
    
            sen = blend_fancy_vec(x,propvec,'S')
            print ("{} Fancy Mixture S: {}".format(isamp, sen))
    
            HoV = blend_fancy_vec(x,propvec,'HoV')
            print ("Fancy Mixture HoV: {}".format(HoV))
    
            AFR = blend_fancy_vec(x,propvec,'AFR_STOICH')
            print ("Fancy Mixture AFR: {}".format(AFR))
    
            LFV150 = blend_fancy_vec(x,propvec,'LFV150')
            print ("Fancy Mixture LFV150: {}".format(LFV150))
    
            PMI = blend_fancy_vec(x,propvec,'PMI')
            print ("Fancy Mixture PMI: {}".format(PMI))
    
            #jbm = jim_bob_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI)
            #print ("jim_bob_merit = {}".format(jbm))
    
            # revm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI)
            # print ("rev_merit = {}".format(revm))
    # 
            # Sample mf
            ksamp = np.random.normal(-1.25,0.5,5000)
            for k in ksamp:
                jbm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI, K=k)
            #print ("jim_bob_merit = {}".format(jbm))
                mf_vals.append(jbm)
                mf_idx.append(isamp)

            jbm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI, K=-1.25)
            mf_knom.append(jbm)
            mf_knom_idx.append(isamp)
            isamp += 1


    plt.plot(mf_idx, mf_vals,'g.',ms=4)
    plt.plot(mf_knom_idx, mf_knom,'k+',ms=6)
    plt.savefig('mf_scatter.png')
#    plt.plot(jbm_list,'r+',ms=14)
#    plt.show()

    #for i in range(len(spids)):
    #    fld = 'BOB'
    #    print("{}  \t\t\t\t\t {} \t {} \t {}".format(spids[i],fld,propvec[fld][i],x[i]))
#
    #ron = blend_linear_vec(x,propvec,'RON')
    #print ("Mixture RON: {}".format(ron))
#
    #ron = blend_fancy_vec(x,propvec,'RON')
    #print ("Fancy Mixture RON: {}".format(ron))
#
    #sen = blend_fancy_vec(x,propvec,'S')
    #print ("Fancy Mixture S: {}".format(sen))
#
    #HoV = blend_fancy_vec(x,propvec,'HoV')
    #print ("Fancy Mixture HoV: {}".format(HoV))
#
    #AFR = blend_fancy_vec(x,propvec,'AFR_STOICH')
    #print ("Fancy Mixture AFR: {}".format(AFR))
#
    #LFV150 = blend_fancy_vec(x,propvec,'LFV150')
    #print ("Fancy Mixture LFV150: {}".format(LFV150))
#
    #PMI = blend_fancy_vec(x,propvec,'PMI')
    #print ("Fancy Mixture PMI: {}".format(PMI))
#
    #jbm = jim_bob_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI)
    #print ("jim_bob_merit = {}".format(jbm))
#
    #revm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI)
#    print ("rev_merit = {}".format(revm))
        #mmf_single(RON=bp['RON'], S=bp['S'],
        #                            HoV=bp['HoV'], SL=bp['SL'],
        #                            LFV150=bp['LFV150'], PMI=bp['PMI'],
        #                            K=ksamp[ns])


