# -*- coding: utf-8; -*-
"""cooptimizer_input.py: Input / configuration file for the co-optimizer
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

from merit_functions import mmf_single_param, mmf_single
import numpy as np
import matplotlib.pyplot as plt
import sys


def sample_mmf_k(bplist, kmean, kvar, nsamples):

    # Now let some parameters be random variables
    # First look at K
    mlist = []
    klist = []
    for bp in bplist:
        ksamp = np.random.normal(kmean, kvar, nsamples)
        msamp = np.zeros([nsamples])
        for ns in range(nsamples):
            msamp[ns] = mmf_single(RON=bp['RON'], S=bp['S'],
                                   HoV=bp['HoV'], SL=bp['SL'],
                                   LFV150=bp['LFV150'], PMI=bp['PMI'],
                                   K=ksamp[ns])
        klist.append(ksamp)
        mlist.append(msamp)

    return mlist, klist


def samples_mmf_refsen(bplist, ref_mean, ref_var, sen_mean,
                       sen_var, nsamples, k):
    Mlist = []
    slist = []
    for bp in bplist:
        samples = {}
        for kk in sen_mean.keys():
            samples["sen"+kk] = []
        for kk in ref_mean.keys():
            samples["ref"+kk] = []
        ns = 0
        msamp = np.zeros([nsamples])
        while (ns < nsamples):
            sen = {}
            ref = {}

            for kk in sen_mean.keys():
                if sen_var is not None:
                    sen[kk] = np.random.normal(sen_mean[kk], sen_var[kk])
                else:
                    sen[kk] = sen_mean[kk]

            for kk in ref_mean.keys():
                if sen_var is not None:
                    ref[kk] = np.random.normal(ref_mean[kk], ref_var[kk])
                else:
                    ref[kk] = ref_mean[kk]

            if ref['PMI'] < 0.0:
                continue
            if ref['S'] < 0.0:
                continue

            if sen['HoV'] == 0:
                print("sen: {}, ref:{}".format(sen, ref))
            for kk in sen_mean.keys():
                samples["sen"+kk].append(sen[kk])
            for kk in ref_mean.keys():
                samples["ref"+kk].append(ref[kk])

            msamp[ns] = mmf_single_param(ref, sen, RON=bp['RON'], S=bp['S'],
                                         HoV=bp['HoV'], SL=bp['SL'],
                                         LFV150=bp['LFV150'], PMI=bp['PMI'],
                                         K=k)
            ns += 1
        Mlist.append(msamp)
        slist.append(samples)

    return Mlist, slist


if __name__ == '__main__':

    nsamples = 200000

    bplist = []
    bp = {'RON': 102, 'S': 14, 'HoV': 4786, 'SL': 56, 'LFV150': 0, 'PMI': 0}
    bplist.append(bp)
    bp = {'RON': 101.1, 'S': 0.9, 'HoV': 4983, 'SL': 57, 'LFV150': 0, 'PMI': 0}
    bplist.append(bp)

    bpnames = ['Best for k=-2', 'Best for k=0.5, 1']

    # To do uncertainty propagation from K to merit function
    Mlist, klist = sample_mmf_k(bplist, -0.5, 1, nsamples)

    f, axs = plt.subplots(1, 2)
    for bp, M, bpname in zip(bplist, Mlist, bpnames):
        axs[1].hist(M, bins=int(max(20, len(Mlist[0])/1000)),
                    alpha=0.3, edgecolor='None', normed='True', label=bpname)

    for bp, k, bpname in zip(bplist, klist, bpnames):
        axs[0].hist(k, bins=int(max(20, len(klist[0])/1000)),
                    alpha=0.3, edgecolor='None', normed='True', label=bpname)

    axs[1].legend()
    axs[0].legend()
    axs[0].set_xlabel('K')
    axs[1].set_xlabel('M')

    plt.savefig('mf_k_sampling.png')

    # To do uncertainty propagation from K, ref, sen to merit function
    ref_mean = {}
    ref_mean['RON'] = 92.0
    ref_mean['S'] = 10.0
    ref_mean['HoV'] = 415.0
    ref_mean['SL'] = 46.0
    ref_mean['PMI'] = 2.0

    ref_var = {}
    ref_var['RON'] = 8.0
    ref_var['S'] = 10.0
    ref_var['HoV'] = 20.0
    ref_var['SL'] = 2.0
    ref_var['PMI'] = 2.0

    sen_mean = {}
    sen_mean['ON'] = 1.0/1.6
    sen_mean['ONHoV'] = 0.01
    sen_mean['HoV'] = 1.0/130.0
    sen_mean['SL'] = 1.0/3.0
    sen_mean['LFV150'] = 1.0
    sen_mean['PMIFIX'] = 0.67
    sen_mean['PMIVAR'] = 0.5

    sen_var = {}
    sen_var['ON'] = 1.0/1.6*0.1
    sen_var['ONHoV'] = 0.01*.1
    sen_var['HoV'] = 1.0/130.0*.1
    sen_var['SL'] = 1.0/3.0*.1
    sen_var['LFV150'] = 0.1
    sen_var['PMIFIX'] = 0.67*.1
    sen_var['PMIVAR'] = 0.5*.1

    Mlist, slist = samples_mmf_refsen(bplist, ref_mean, ref_var,
                                      sen_mean, sen_var, nsamples, -0.5)

    plt.close()

    def mk_subplot(axs, f, i, nr, nc, l):
        ii = int(i % nr)
        jj = int((i-ii)/nr)
        axs[jj, ii].hist(f, bins=int(max(20, len(klist[0])/1000)),
                         alpha=0.4, edgecolor='None', normed='True')
        axs[jj, ii].set_xlabel(l)
        axs[jj, ii].locator_params(nbins=2, axis='x')

        if i == 3:
            axs[jj, ii].locator_params(nbins=2, axis='x')

    nc = 6
    nr = 2
    f, axs = plt.subplots(nr, nc)
    f.set_size_inches([16, 7])

    p = 0
    for bp, M, bpname, samples in zip(bplist, Mlist, bpnames, slist):
        mk_subplot(axs, M, 0, 4, 2, 'Merit')
        mk_subplot(axs, samples['senON'], 1, nc, nr, 'ON Sensitivity')
        mk_subplot(axs, samples['senONHoV'], 2, nc, nr, 'ON/HoV Sensitivity')
        mk_subplot(axs, samples['senHoV'], 3, nc, nr, 'HoV Sensitivity')
        mk_subplot(axs, samples['senSL'], 4, nc, nr, 'SL Sensitivity')
        mk_subplot(axs, samples['senPMIFIX'], 5, nc, nr,
                   'PMI Fixed Sensitivity')
        mk_subplot(axs, samples['senPMIVAR'], 6, nc, nr,
                   'PMI Variable Sensitivity')

        mk_subplot(axs, samples['refRON'], 7, nc, nr, 'Reference RON')
        mk_subplot(axs, samples['refS'], 8, nc, nr, 'Reference S')
        mk_subplot(axs, samples['refHoV'], 9, nc, nr, 'Reference HoV')
        mk_subplot(axs, samples['refSL'], 10, nc, nr, 'Reference SL')
        mk_subplot(axs, samples['refPMI'], 11, nc, nr, 'Reference PMI')

    plt.tight_layout()
    plt.savefig('coefficient_sensitivity.png')

    plt.close()
    plt.figure(1, figsize=(16, 7))
    for bp, M, bpname, samples in zip(bplist, Mlist, bpnames, slist):
        plt.hist(M, bins=int(max(20, len(klist[0])/1000)),
                 alpha=0.4, edgecolor='None', normed='True')
    plt.xlabel('Merit function')
    plt.xlim([50, 100])
    plt.savefig('coefficient_sensitivity_merit.png')
