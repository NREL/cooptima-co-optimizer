def mft_mmf(properties):
    print ("properties to mmf: {}", properties)
    #K = 0.5
    K = 20.0
    if properties['PMI']>2.0:
        return ((properties['RON']-92.0)/1.6
               -(K * (properties['S']-10.0)/1.6)
               +(0.01*properties['ON']*(properties['HoV']-415.0))/1.6
               +(properties['HoV'] - 415.0)/130.0
               +(properties['SL'] - 46.0)/3.0
               -properties['LFV150']
               -(0.67+0.5*(properties['PMI']-2.0)))
    else:
        return ((properties['RON']-92.0)/1.6
               -(K * (properties['S']-10.0)/1.6)
               +(0.01*properties['ON']*(properties['HoV']-415.0))/1.6
               +(properties['HoV'] - 415.0)/130.0
               +(properties['SL'] - 46.0)/3.0
               -properties['LFV150'])

def mmf_single(RON=100.0, S=10.0, ON=90.0, HoV=415.0, SL=46.0,
               LFV150=0.0, PMI=1.0, K=0.5):
    if PMI>2.0:
        return ((RON-92.0)/1.6
               -(K * (S-10.0)/1.6)
               +(0.01*ON*(HoV-415.0))/1.6
               +(HoV - 415.0)/130.0
               +(SL - 46.0)/3.0
               -LFV150
               -(0.67+0.5*(PMI-2.0)))
    else:
        return ((RON-92.0)/1.6
               -(K * (S-10.0)/1.6)
               +(0.01*ON*(HoV-415.0))/1.6
               +(HoV - 415.0)/130.0
               +(SL - 46.0)/3.0
               -LFV150)

def rand_composition(propDB):
    comp = {}
    tot = 0.0
    for k in propDB.keys():
        comp[k] = np.random.rand(1)
        tot += comp[k]

    for k in propDB.keys():
        comp[k] = comp[k]/tot
    
    return comp
import property_mapping as pm
from fuelsdb_interface import load_propDB
import numpy as np
from blend_functions import blend_linear as blend
if __name__ == '__main__':
    propDB = load_propDB("propDB_fiction.xls")
    print("{}".format(len(propDB)))


    prop_list = ['RON', 'S', 'ON', 'HoV', 'SL', 'LFV150', 'PMI']

    comp_samples = []
    prop_samples = []

    for i in range(10):
        comp =  rand_composition(propDB)
        comp_samples.append(comp)
        props = {}
        for p in prop_list:
            props[p] = blend(p, propDB, comp)
        print("props:{}".format(props))
        mmf = mft_mmf(props)
        print("MMF = {}".format(mmf))
        prop_samples.append(props)

    #cpt.plot_prop_parallel(prop_samples)
    mmf = mft_mmf(prop_samples[0])
    print ("MMF = {}".format(mmf) )


def mft_assert(fuel_component):
    pass

def mft_mkt_trans(fuel_component):
    pass

def get_prop(fuel_components):
    pass

#def goodness_fcn_terms(fuel_components):
#    properties_components = get_prop(fuel_components)
#    properties = blend(properties_components, fuel_components)
#
#    mft = {}
#    mft['assert'] = mft_assert(fuel_components)
#    mft['mkt_trans'] = mft_mkt_trans(fuel_components)
#    mft['engine_merit'] = mft_mmf(properties)
#
#
def goodness_fcn(fuel_components, weights):
    mft = merit_fcn_terms(fuel_components)
    mf = 0.0
    for w,v in weights.iteritems():
        mf += v*mft[w]
    return mf

def eval_merit(fuel_component,weights):
    pass

