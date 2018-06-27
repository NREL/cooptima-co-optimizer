# -*- coding: utf-8; -*-
"""merit_function.py: Miles merit function based on finished fuel properties
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


def mmf_single(RON=100.0, S=10.0, HoV=415.0, SL=46.0,
               LFV150=0.0, PMI=1.0, K=0.5):
    if PMI > 2.0:
        return ((RON-92.0)/1.6 -
                (K * (S-10.0)/1.6) +
                (0.01*(HoV-415.0))/1.6 +
                (HoV - 415.0)/130.0 +
                (SL - 46.0)/3.0 -
                LFV150 - (0.67+0.5*(PMI-2.0)))
    else:
        return ((RON-92.0)/1.6 -
                (K * (S-10.0)/1.6) +
                (0.01*(HoV-415.0))/1.6 +
                (HoV - 415.0)/130.0 +
                (SL - 46.0)/3.0 -
                LFV150)


def mmf_single_param(ref, sen, RON=100.0, S=10.0, HoV=415.0, SL=46.0,
                     LFV150=0.0, PMI=1.0, K=0.5):

    if PMI > ref['PMI']:
        return ((RON-ref['RON'])*sen['ON'] -
                (K * (S-ref['S'])*sen['ON']) +
                (sen['ONHoV']*(HoV-ref['HoV']))*sen['ON'] +
                (HoV - ref['HoV'])*sen['HoV'] +
                (SL - ref['SL'])*sen['SL'] -
                LFV150*sen['LFV150'] -
                (sen['PMIFIX']+sen['PMIVAR']*(PMI-ref['PMI'])))
    else:
        return ((RON-ref['RON'])*sen['ON'] -
                (K * (S-ref['S'])*sen['ON']) +
                (sen['ONHoV']*(HoV-ref['HoV']))*sen['ON'] +
                (HoV - ref['HoV'])*sen['HoV'] +
                (SL - ref['SL'])*sen['SL'] -
                LFV150*sen['LFV150'])


def jim_bob_mf(RON=91, S=8, HoV=414, SL=46, AFR=14.3,
               LFV150=0.0, PMI=1.6, K=-1.25):
    m = ((RON-91.0)/1.6 -
         (K * (S-8.0)/1.6) +
         0.085 * (HoV/(AFR+1.0) - 415.0/(14.3+1.0))/1.6 +
         (HoV/(AFR+1.0) - 415.0/(14.3+1.0))/15.38 +
         (SL - 46.0)/1.6 -
         LFV150)

    if PMI > 1.6:
        m -= 0.7 + 0.5*(PMI-1.4)

    return m


def revised_mf(RON=91, S=8, HoV=414, SL=46, AFR=14.0,
               LFV150=0.0, PMI=1.6, K=-1.25, TC90_conv=0.0, TC90_mix=0.0):
    m = ((RON-91.0)/1.6 -
         (K * (S-8.0)/1.6) +
         0.085 * (HoV/(AFR+1.0) - 415.0/(14.0+1.0))/1.6 +
         (HoV/(AFR+1.0) - 415.0/(14.0+1.0))/15.2 +
         (SL - 46.0)/5.4 +
         0.008*(TC90_conv - TC90_mix))

    if PMI > 1.6:
        m -= 0.7 + 0.5*(PMI-1.4)

    return m


def rand_composition(propDB):
    comp = {}
    tot = 0.0
    for k in propDB.keys():
        comp[k] = np.random.rand(1)
        tot += comp[k]

    for k in propDB.keys():
        comp[k] = comp[k]/tot

    return comp


def mft_assert(fuel_component):
    pass


def mft_mkt_trans(fuel_component):
    pass
