import numpy as np
def blend_linear(prop, propDB, comp):
    prop_vec = np.zeros(len(comp))
    x_vec = np.zeros(len(comp))
    i = 0
    for k in comp.keys():
        x_vec[i] = comp[k]
        prop_vec[i] = propDB[k][prop]
        i += 1
    prop = np.sum(prop_vec*x_vec)
    return prop

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
        'HoF_liq':blend_linear,
        'LHV':blend_linear,
    }
    fblend = prop_blend.get(prop)
    return fblend(prop, propDB, comp)


