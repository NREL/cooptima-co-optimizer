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
from nsga2_k import eval_mo
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

    propDB = load_propDB('prop_db_AMR.xls')
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

    if False:
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
#                comp = {c:0.1, 'BOB-1':0.9}
                comp = {c:1.0, 'BOB-1':0.0}
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
  #              ksamp = np.random.normal(-1.25,0.5,5000)
  #              for k in ksamp:
  #                  jbm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI, K=k)
  #              #print ("jim_bob_merit = {}".format(jbm))
  #                  mf_vals.append(jbm)
  #                  mf_idx.append(isamp)
    
                jbm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, PMI=PMI, K=-1.0)
                mf_knom.append(jbm)
                mf_knom_idx.append(isamp)
                isamp += 1
                print ("mf: {}".format(jbm))
    

        plt.plot(mf_idx, mf_vals,'g.',ms=4)
        plt.plot(mf_knom_idx, mf_knom,'k+',ms=6)
        plt.savefig('mf_scatter.png')

    KK = -0.5
    x_shuff = [0.001324310350291002, 0.004999871054905616, 0.009597839797339565, 0.020249992558423184, 0.10512280120845688, 0.030188172710182325, 0.05934060289277107, 0.007108223208273045, 0.022569635194655057, 0.07539302507083424, 0.03808834477204709, 0.007216109025727148, 0.14587147969492825, 0.018599742824815118, 0.0007668699773392273, 0.0008101533174170688, 0.008910494766329137, 0.011742609648713554, 0.0746615887576285, 0.07292404878028737, 0.18110486660397077, 0.10340921778466486]

    KK = -2.0
    x_shuff = [5.429193184724211e-06, 5.4700121342959545e-06, 0.00023574675787848193, 0.0006634005877650475, 0.33420041342149914, 2.253088358924713e-08, 2.929083304649458e-07, 7.84952012879686e-05, 2.539318697385498e-07, 6.472335781271855e-07, 0.0002801601866101847, 5.78982495877495e-06, 0.25715789806490985, 0.000724824623564603, 3.3710529613876674e-05, 0.00013076272054106164, 2.4256566440012568e-06, 3.7455938815765416e-05, 0.13974210154549355, 3.6569331184504476e-06, 0.2665392129798876, 0.0001518292174307078]
    # Score: 64.4243251413

    KK = -1.25
    x_shuff = [2.334882429827181e-05, 2.3073828291791606e-06, 1.1373883308286529e-06, 8.460933750952283e-07, 0.352349270601257, 1.9769097040567398e-05, 1.2381570105591349e-06, 0.0008524641601872692, 1.5924796560953116e-06, 5.838187371641677e-05, 1.2989269851950561e-06, 5.051312220823206e-06, 0.24365125365611764, 5.9298089320557183e-05, 1.3779517723124228e-06, 0.00012597507923429462, 5.74394624825527e-07, 7.046856723913778e-06, 0.12045975848472151, 3.0348674144966435e-05, 0.2822741257687364, 7.35347476964213e-05]

    my_names = [u'Ethyl butanoate', u'Diisobutylene', u'2-butanone (MEK)', u'Ethanol', u'2-pentanol', u'Methanol', u'Butyl acetate', u'2-butanol', u'2.4 dimethyl-3- pentanone', u'2-Me-1-butanol', u'sBOB', u'CARBOB', u'wCBOB', u'Isobutanol', u'2-pentanone', u'1-butanol', u'Ethyl acetate', u'Triptane', u'Methyl acetate', u'Anisole', u'Methyl furan', u'Cyclopentanone']
    nsga_names = [u'2.4 dimethyl-3- pentanone', u'2-Me-1-butanol', u'sBOB', u'CARBOB', u'Methanol', u'Ethyl butanoate', u'Isobutanol', u'2-butanol', u'2-pentanone', u'1-butanol', u'2-butanone (MEK)', u'Ethyl acetate', u'Ethanol', u'2-pentanol', u'Triptane', u'wCBOB', u'Butyl acetate', u'Methyl acetate', u'Diisobutylene', u'Anisole', u'Methyl furan', u'Cyclopentanone']

    x = np.zeros_like(x_shuff)
    for nm, xs  in zip(nsga_names, x_shuff):
        spot = my_names.index(nm)
        print ("{} goes in spot {}, which has name {}".format(nm, spot, my_names[spot]))
        x[spot] = xs

    isamp = 1
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
    

    jbm = revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, PMI=PMI, K=KK)
    print("revised merit fcn: {}".format(jbm))
  #  print("nsga2 fcn: {}".format(eval_mo(x,propvec,KK)))
    print("sum: {}".format(np.sum(x)))
    print("pv RON = {}".format(propvec['NAME']))

    for x, nm in zip(x_shuff, nsga_names):
        print("{}: {}".format(nm, x))




