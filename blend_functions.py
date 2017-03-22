# -*- coding: utf-8; -*-
"""blend_functions.py: Implementation of blending model for fuel properties
    based on component mixture properties
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

import numpy as np
from pyomo.environ import *
import scipy.interpolate as interp

import matplotlib.pyplot as plt

# General blend function - not currently used
def blend_fcn(prop, propDB, comp):
    prop_blend = {
        'RON': blend_linear,
        'LHV': blend_linear,
        'S': blend_linear,
        'HoV': blend_linear,
        'SL': blend_linear,
        'LFV150': blend_linear,
        'PMI': blend_linear,
        'MOLWT': blend_linear,
        'BP': blend_linear,
        'HoF_liq': blend_linear,
        'LHV': blend_linear,
    }
    fblend = prop_blend.get(prop)
    return fblend(prop, propDB, comp)


# Variations of linear blending for different arguments
# This is the general one
def blend_linear_propDB(prop, propDB, comp):
    prop_vec = np.zeros(len(comp))
    x_vec = np.zeros(len(comp))
    i = 0
    for k in comp.keys():
        x_vec[i] = comp[k]
        prop_vec[i] = propDB[k][prop]
        i += 1
    prop = np.sum(prop_vec*x_vec)
    return prop


# This is the one for pyomo
def blend_linear_pyomo(model, whichprop):
    if whichprop == 'RON':
        return summation(model.X, model.RON, index=model.I)
    elif whichprop == 'S':
        return summation(model.X, model.S, index=model.I)
    elif whichprop == 'HoV':
        return summation(model.X, model.HoV, index=model.I)
    elif whichprop == 'SL':
        return summation(model.X, model.SL, index=model.I)
    elif whichprop == 'LFV150':
        return summation(model.X, model.LFV150, index=model.I)
    elif whichprop == 'PMI':
        return summation(model.X, model.PMI, index=model.I)
    else:
        print("Unknown property {}".format(whichprop))
        sys.exit(-1)


# This is the one for general (numpy based) use, including nsga2
def blend_linear_vec(x_vec, prop_vec, whichprop):
    return np.sum(x_vec*prop_vec[whichprop])

def blend_fancy_vec(x_vec, prop_vec, whichprop):
    if whichprop == 'RON':
        # Need volume fractions
        x_volume = get_other_fractions(x_vec, prop_vec, 'VOLUME')
        print x_volume
        x_mass = get_other_fractions(x_vec, prop_vec, 'MASS')
        print "x_mass", x_mass
        bRON = np.zeros(len(x_vec))
        for i in range(len(x_vec)):
            if prop_vec['BOB'][i] == 1:
                bRON[i] = prop_vec['RON'][i]
            else:
                if prop_vec['bRON_datapts'][i] > 0:
                    bRON_y = []
                    bRON_x = []

                    ib = 0
                    for b in range(int(prop_vec['bRON_datapts'][i])):
                        x0 = (0.0)
                        r0 = (prop_vec["base_RON_bRON_{}".format(ib)][i])
                        x1 = (prop_vec["vol_frac_bRON_{}".format(ib)][i])
                        r1 = (prop_vec["blend_RON_bRON_{}".format(ib)][i])

                        bRON_y.append( (r1 - (1.0-x1)*r0)/x1)
                        bRON_x.append( (x1) )
                        ib += 1

                    bRON_y.append(prop_vec['RON'][i])
                    bRON_x.append( (1.0) )

                    plt.show()
                    bRON_fcn = interp.interp1d(bRON_x, bRON_y, fill_value='extrapolate')     
                    bRON[i] = bRON_fcn(x_volume[i])
                    #print '-----------------------'
                    #if x_volume[i] > 0:
                    #    print "i = ", i
                    #    print bRON_x
                    #    print bRON_y
                    #    print bRON[i]
                    #print '-----------------------'
                else:
                    bRON[i] = prop_vec['RON'][i]
        return np.sum(x_volume*bRON)

    if whichprop == 'S':
        # Need volume fractions
        x_volume = get_other_fractions(x_vec, prop_vec, 'VOLUME')
        #print x_volume

        #print "x_mass", x_mass
        bS = np.zeros(len(x_vec))
        for i in range(len(x_vec)):
            if prop_vec['BOB'][i] == 1:
                bS[i] = prop_vec['S'][i]
            else:
                if prop_vec['bS_datapts'][i] > 0:
                    bS_y = []
                    bS_x = []

                    ib = 0
                    for b in range(int(prop_vec['bS_datapts'][i])):
                        x0 = (0.0)
                        r0 = (prop_vec["base_S_bS_{}".format(ib)][i])
                        x1 = (prop_vec["vol_frac_bS_{}".format(ib)][i])
                        r1 = (prop_vec["blend_S_bS_{}".format(ib)][i])

                        bS_y.append( (r1 - (1.0-x1)*r0)/x1)
                        bS_x.append( (x1) )
                        ib += 1

                    bS_y.append(prop_vec['S'][i])
                    bS_x.append( (1.0) )

                    plt.show()
                    bS_fcn = interp.interp1d(bS_x, bS_y, fill_value='extrapolate')     
                    bS[i] = bS_fcn(x_volume[i])
                    #print '-----------------------'
                    #if x_volume[i] > 0:
                    #    print "i = ", i
                    #    print bRON_x
                    #    print bRON_y
                    #    print bRON[i]
                    #print '-----------------------'
                else:
                    bS[i] = prop_vec['S'][i]
        return np.sum(x_volume*bS)

    if whichprop == 'HoV':
        x_mass = get_other_fractions(x_vec, prop_vec, 'MASS')
        return np.sum(x_mass*prop_vec['HoV'])

    if whichprop == 'AFR_STOICH':
        x_mass = get_other_fractions(x_vec, prop_vec, 'MASS')
        return np.sum(x_mass*prop_vec['AFR_STOICH'])

    if whichprop == 'LFV150':
        x_volume = get_other_fractions(x_vec, prop_vec, 'VOLUME')
        return np.sum(x_volume*prop_vec['LFV150'])

    if whichprop == 'PMI':
        x_mass = get_other_fractions(x_vec, prop_vec, 'MASS')
        return np.sum(x_mass*prop_vec['PMI'])

    else:
        return blend_linear_vec(x_vec, prop_vec, whichprop)


    # Build function for bRON, bMON on the fly
    # If fuel is pure component, RON is the bRON at that fraction
    # If fuel is blendstock, RON is the measured RON
    
    # Mass, mole, energy, volume weighted as appropriate

    # AFR, HoV, PMI - mass basis
    # RON, MON, S - bRON, bMON, bS, volume basis
    # LFV150 - volume basis
    # 

def get_other_fractions(x_vec, prop_vec, whichbasis):
    if whichbasis is 'VOLUME':
        V = x_vec * prop_vec['MOLWT']/prop_vec['DENSITY']
        Vfrac = V/np.sum(V)
        return Vfrac

        # Need density on mole basis
        # V_i  in g/cm^3 from database
        # V_i/w_i should be mol/cm^3
        # V = sum(x_i/(V_i/w_i))
        # vfrac_i = x_i/(V_i/w_i)/V

    if whichbasis is 'MASS':
        M = x_vec * prop_vec['MOLWT']
        Mfrac = M/np.sum(M)
        return Mfrac
        # Base on fraction of avg. molecular weight
        # w_avg = sum(w_i x_i)
        # y_i = w_i*x_i/w_avg


    if whichbasis is 'ENERGY':
        pass
        # Need LHV on mole basis; assume db is in kJ/kg
        # Skip this for now, only need for S_L


def volume_to_other(v_vec, prop_vec, whichbasis='MOLE'):
    if whichbasis is 'MOLE':
        print "molwt: ", prop_vec['MOLWT']
        x = v_vec*prop_vec['DENSITY']/prop_vec['MOLWT']
        x_frac = x/np.sum(x)
        return x_frac

        # Need density on mole basis
        # V_i  in g/cm^3 from database
        # V_i/w_i should be mol/cm^3
        # V = sum(x_i/(V_i/w_i))
        # vfrac_i = x_i/(V_i/w_i)/V

    if whichbasis is 'MASS':
        pass



def get_bRON_fcns(fp):

    print fp

    bRON_dict = {}
    bMON_dict = {}
    bS_dict = {}

    for fname, fuel in fp.iteritems():
        x = []
        r = []
        bRON_y = []
        bRON_x = []

        ib = 0
        for b in range(fuel['bRON_datapts']):
            x0 = (0.0)
            r0 = (fuel["base_RON_bRON_{}".format(ib)])

            x1 = (fuel["vol_frac_bRON_{}".format(ib)])
            r1 = (fuel["blend_RON_bRON_{}".format(ib)])

            bRON_y.append( (r1 - (1.0-x1)*r0)/x1)
            bRON_x.append( (x1) )

            # Maybe put in pure component RON at x=1.0?
            
            ib += 1
        
        bRON = interp.interp1d(bRON_x, bRON_y, fill_value='extrapolate')       
        bRON_dict[fname] = bRON        

        ib = 0
        bMON_y = []
        bMON_x = []
        for b in range(fuel['bMON_datapts']):

            x0 = (0.0)
            r0 = (fuel["base_MON_bMON_{}".format(ib)])

            x1 = (fuel["vol_frac_bMON_{}".format(ib)])
            r1 = (fuel["blend_MON_bMON_{}".format(ib)])

            bMON_y.append( (r1 - (1.0-x1)*r0)/x1)
            bMON_x.append( (x1) )

            print " computing bMON for :", ib, fname    
            print "    base MON : ", r0
            print "    vol_frac : ", x1
            print "    blend MON : ", r1
            # Maybe put in pure component RON at x=1.0?
            
            ib += 1
        
        bMON = interp.interp1d(bMON_x, bMON_y, fill_value='extrapolate')       
        bMON_dict[fname] = bMON  

        ib = 0
        bS_y = []
        bS_x = []
        for b in range(fuel['bS_datapts']):
            x0 = (0.0)
            r0 = (fuel["base_S_bS_{}".format(ib)])

            x1 = (fuel["vol_frac_bS_{}".format(ib)])
            r1 = (fuel["blend_S_bS_{}".format(ib)])

            bS_y.append( (r1 - (1.0-x1)*r0)/x1)
            bS_x.append( (x1) )

            # Maybe put in pure component RON at x=1.0?
            
            ib += 1
        
        bS = interp.interp1d(bS_x, bS_y, fill_value='extrapolate')       
        bS_dict[fname] = bS  


    return bRON_dict, bMON_dict, bS_dict


