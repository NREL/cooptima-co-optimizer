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
from scipy.optimize import minimize, fmin

def get_xvec(comp, spids):

    x = np.zeros([len(spids)])
    for i in range(len(spids)):
        if spids[i] in comp.keys():
            x[i] = comp[spids[i]]

    #print("{}".format(x))

    return x

def eval_merit(x, propvec, K=-1.25):
    ron = blend_fancy_vec(x,propvec,'RON')
    sen = blend_fancy_vec(x,propvec,'S')
    HoV = blend_fancy_vec(x,propvec,'HoV')
    AFR = blend_fancy_vec(x,propvec,'AFR_STOICH')       
    LFV150 = blend_fancy_vec(x,propvec,'LFV150')
    PMI = blend_fancy_vec(x,propvec,'PMI')
    return revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI, K=K)

def comp_mf(cas, BOB, spids, propvec, vf):
        thiscomp = {cas:vf, BOB:(1-vf)}
        x = get_xvec(thiscomp, spids)
        x = volume_to_other(x, propvec, 'MOLE')
        M = eval_merit(x, propvec)
        return M


def find_e10_blend(cas,BOB, spids, propvec):
    # E10 merit in this BOB

    e10comp = {'64-17-5':0.1, BOB:0.9}
#    e10comp = {'64-17-5':0.3, BOB:0.7}
    x = get_xvec(e10comp, spids)
    x = volume_to_other(x, propvec, 'MOLE')
    ME10 = eval_merit(x, propvec )

    fsq = lambda vf: (comp_mf(cas, BOB, spids, propvec, vf) - ME10)**2

    vf = 0.3
    res = minimize(fsq, vf, method='Nelder-Mead')
    print("Residual: {}".format(res.fun))
    vfrac_E10_equiv = res.x[0]
    #    print("{}".format(res))
    if res.success and res.fun < 1.0e-5:
        return ME10, vfrac_E10_equiv
    else:
        return ME10, None



if __name__ == '__main__':
    print ("=================================================================")
    print ("Testing non-linear blending")
    print ("=================================================================")

    propDB = load_propDB('testDB.xls')
    ncomp, spids, propvec = make_property_vector_all(propDB)
    #    print("{}; {}".format( ncomp, spids))
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

    if True:
        plt.close()
        e10lvl = {}
        for BOB in ['BOB-1', 'BOB-2', 'BOB-3']:
#            BOB = 'BOB-1'
            e10lvl[BOB] = []
            for cas in cas_list:
                ME10, vfrac_E10_equiv = find_e10_blend(cas, BOB=BOB, spids=spids, propvec=propvec)
                print ("Merit function, E10, {} = {}; equivalent VF = {} ".format(BOB, ME10,vfrac_E10_equiv))
                if vfrac_E10_equiv is not None and vfrac_E10_equiv > 0.0 and vfrac_E10_equiv < 1.0:
                    e10lvl[BOB].append(vfrac_E10_equiv)
                else:
                    e10lvl[BOB].append(None)
            #mf_results = []
            #for cas, vf in zip(cas_list, bob1_e10lvl):
            #    if vf is not None:
            #       mf_results.append(comp_mf(cas, BOB=BOB, spids=spids, propvec=propvec, vf=vf))
            #    else:
            #        mf_results.append(None)

         
            print ("Done {}".format(BOB))
        f, ax = plt.subplots()
        for k, v in e10lvl.iteritems():
            ax.plot(v,'o',ms=10,label=k)
        ax.plot([0,18],[0.1,0.1],'--k')
        ax.legend()
        plt.savefig('e10level.pdf')
        plt.show()
#            ax.plot(bob1_e10lvl,label=BOB)
#        ax.legend()
#        axs[0].get_yaxis().get_major_formatter().set_useOffset(False)
#        plt.show()

        #plt.plot(mf_results,'gx')
        #plt.show()


    if False:
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



