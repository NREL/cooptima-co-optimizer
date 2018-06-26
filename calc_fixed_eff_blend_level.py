# -*- coding: utf-8; -*-
"""calc_fixed_eff_blend_level.py
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
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
from fuelsdb_interface import load_propDB, make_property_vector,\
                              make_property_vector_sample_cost,\
                              make_property_vector_all
import numpy as np
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
    return x


def eval_merit(x, propvec, K=-1.25):
    ron = blend_fancy_vec(x, propvec, 'RON')
    sen = blend_fancy_vec(x, propvec, 'S')
    HoV = blend_fancy_vec(x, propvec, 'HoV')
    AFR = blend_fancy_vec(x, propvec, 'AFR_STOICH')
    LFV150 = blend_fancy_vec(x, propvec, 'LFV150')
    PMI = blend_fancy_vec(x, propvec, 'PMI')
    return revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR,
                      LFV150=LFV150, PMI=PMI, K=K)


def comp_mf(cas, BOB, spids, propvec, vf):
        thiscomp = {cas: vf, BOB: (1-vf)}
        x = get_xvec(thiscomp, spids)
        x = volume_to_other(x, propvec, 'MOLE')
        M = eval_merit(x, propvec)
        return M


def find_e10_blend(cas, BOB, spids, propvec):
    # E10 merit in this BOB

    e10comp = {'64-17-5': 0.1, BOB: 0.9}
#    e10comp = {'64-17-5':0.3, BOB:0.7}
    x = get_xvec(e10comp, spids)
    x = volume_to_other(x, propvec, 'MOLE')
    ME10 = eval_merit(x, propvec)

    fsq = lambda vf: (comp_mf(cas, BOB, spids, propvec, vf) - ME10)**2

    vf = 0.3
    res = minimize(fsq, vf, method='Nelder-Mead')
    print("Residual: {}".format(res.fun))
    vfrac_E10_equiv = res.x[0]
    if res.success and res.fun < 1.0e-5:
        return ME10, vfrac_E10_equiv
    else:
        return ME10, None


def get_bl_eff(targeteff, cas, BOB, spids, propvec):
    fsq = lambda vf: (comp_mf(cas, BOB, spids, propvec, vf) - targeteff)**2
    vf = 0.5
    res = minimize(fsq, vf, method='Nelder-Mead')
    print("Residual: {}".format(res.fun))
    volfrac = res.x[0]
    #    print("{}".format(res))
    if res.success and res.fun < 1.0e-5 and volfrac <= 1.0:
        return comp_mf(cas, BOB, spids, propvec, volfrac), volfrac
    else:
        eff = comp_mf(cas, BOB, spids, propvec, 1.0)
        return eff, 1.0


if __name__ == '__main__':
    print("=================================================================")
    print("Finding blend levels for fixed efficiency")
    print("=================================================================")
    matplotlib.rc('xtick', labelsize=24)
    matplotlib.rc('ytick', labelsize=24)
    matplotlib.rc('axes', titlesize=24)
    matplotlib.rc('font', size=24)
    propDB = load_propDB('testDB.xls')
    ncomp, spids, propvec = make_property_vector_all(propDB)

    # Volume fractions:
    wb = xlrd.open_workbook('test_list.xlsx')
    ws = wb.sheet_by_index(0)
    cas_list = []
    nm_list = []
    jbm_list = []
    cas_col = ws.col(0)
    for c, jb, nm in zip(cas_col[1:], ws.col(2)[1:], ws.col(1)[1:]):
        cas_list.append(c.value)
        jbm_list.append(jb.value)
        nm_list.append(nm.value)

    if True:
        plt.close()
        blend_level = {}
        mf = {}
        for BOB in ['BOB-1', 'BOB-2', 'BOB-3']:
            blend_level[BOB] = []
            mf[BOB] = []
            for cas in cas_list:
                # merit, bl = get_bl_eff(10.0, cas, BOB=BOB,
                #                        spids=spids, propvec=propvec)
                merit = comp_mf(cas, BOB, spids, propvec, 0.3)
                bl = 0.2
                blend_level[BOB].append(bl)
                mf[BOB].append(merit)

        for BOB in ['BOB-1', 'BOB-2', 'BOB-3']:
            print("{}-MF\t {}-BL\t".format(BOB, BOB), end="")
        print("")

        for i in range(len(cas_list)):
            print("{}\t".format(cas_list[i]), end="")
            for BOB in ['BOB-1', 'BOB-2', 'BOB-3']:
                print("{}\t {}\t".format(blend_level[BOB][i],
                                         mf[BOB][i]), end="")
            print("")
