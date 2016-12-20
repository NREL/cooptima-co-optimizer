# -*- coding: utf-8; -*-
"""property_mapping.py: Functions to map the properties necessary for merit 
function evaluation to the properties in the database
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

# Functions to map the properties necessary for merit function evaluation to
# the properties in the database

import sys


def check_data(dbprops, prop):
    for p in prop:
        if not dbprops[p]:
            sys.stderr.write("Data missing: {} for {}".format(p,
                             dbprops['concat_name_cas2']))
            return False
    return True


def RON(dbprops):
    if dbprops['___Identifier'] == 'Pure Compounds':
        if not check_data(dbprops, ['Pure_RON']):
            return 0.0
        return float(dbprops['Pure_RON'])
    elif dbprops['___Identifier'] == 'Fuel Blends':
        if not check_data(dbprops, ['Blend_RON']):
            return 0.0
        return float(dbprops['Blend_RON'])


def LHV(dbprops):
    if dbprops['___Identifier'] == 'Pure Compounds':
        if not check_data(dbprops, ['Pure_LHV']):
            return 0.0
        return float(dbprops['Pure_LHV'])
    elif dbprops['___Identifier'] == 'Fuel Blends':
        if not check_data(dbprops, ['Blend_LHV']):
            return 0.0
        return float(dbprops['Blend_LHV'])


def K(dbprops):
    return 0.0


def sensitivity(dbprops):
    if dbprops['___Identifier'] == 'Pure Compounds':
        if not check_data(dbprops, ['Pure_RON', 'Pure_MON']):
            return 0.0
        return float(dbprops['Pure_RON']) - float(dbprops['Pure_MON'])
    elif dbprops['___Identifier'] == 'Fuel Blends':
        if not check_data(dbprops, ['Blend_RON', 'Blend_MON']):
            return 0.0
        return float(dbprops['Blend_RON']) - float(dbprops['Blend_MON'])


def HoV(dbprops):
    if dbprops['___Identifier'] == 'Pure Compounds':
        if not check_data(dbprops, ['Pure_Heat_of_Vaporization',
                                    'Pure_Molecular_Weight']):
            return 0.0
        return \
            float(dbprops['Pure_Heat_of_Vaporization'])\
          / float(dbprops['Pure_Molecular_Weight'])*1000.0

    elif dbprops['___Identifier'] == 'Fuel Blends':
        if not check_data(dbprops, ['Blend_Heat_of_Vaporization',
                          'Blend_Molecular_Weight']):
            return 0.0
        return float(dbprops['Blend_HoV_temp']) \
             / float(dbprops['Blend_Molecular_Weight'])*1000.0


def laminar_flame_speed(dbprops):
    return 0.0


def lfv150(dbprops):
    return 0.0


def AKI(dbprops):
    return 0.0


def PMI(dbprops):
    return 0.0


def getname(dbprops):
    return dbprops['concat_name_cas2']


def formula(dbprops):
    return dbprops['Pure_Molecular_Formula']


def molwt(dbprops):
    return float(dbprops['Pure_Molecular_Weight'])*1000.0


def bp(dbprops):
    return float(dbprops['Pure_Boiling_Point'])


def hof_liq(dbprops):
    return 0.0


def lhv(dbprops):
    return dbprops['Pure_LHV']


def prop_mapping(prop, propDB):
    prop_conv = {
        'RON': RON,
        'LHV': LHV,
        'K': K,
        'S': sensitivity,
        'HoV': HoV,
        'SL': laminar_flame_speed,
        'LFV150': lfv150,
        'ON': AKI,
        'PMI': PMI,
        'NAME': getname,
        'FORMULA': formula,
        'MOLWT': molwt,
        'BP': bp,
        'HoF_liq': hof_liq,
        'LHV': lhv,
    }
    fconv = prop_conv.get(prop)
    return fconv(propDB)
