import numpy as np
from pyomo.environ import *


# General blend function - not currently used
def blend_fcn(prop, propDB, comp):
    prop_blend = {
        'RON': blend_linear,
        'LHV': blend_linear,
        'S': blend_linear,
        'HoV': blend_linear,
        'SL': blend_linear,
        'LFV150': blend_linear,
        'ON': blend_linear,
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
    elif whichprop == 'ON':
        return summation(model.X, model.ON, index=model.I)
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
