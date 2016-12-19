# def mft_mmf(properties):
#     print ("properties to mmf: {}", properties)
#     K = 20.0
#     if properties['PMI']>2.0:
#         return ((properties['RON']-92.0)/1.6
#                -(K * (properties['S']-10.0)/1.6)
#                +(0.01*properties['ON']*(properties['HoV']-415.0))/1.6
#                +(properties['HoV'] - 415.0)/130.0
#                +(properties['SL'] - 46.0)/3.0
#                -properties['LFV150']
#                -(0.67+0.5*(properties['PMI']-2.0)))
#     else:
#         return ((properties['RON']-92.0)/1.6
#                -(K * (properties['S']-10.0)/1.6)
#                +(0.01*properties['ON']*(properties['HoV']-415.0))/1.6
#                +(properties['HoV'] - 415.0)/130.0
#                +(properties['SL'] - 46.0)/3.0
#                -properties['LFV150'])
#


def mmf_single(RON=100.0, S=10.0, ON=90.0, HoV=415.0, SL=46.0,
               LFV150=0.0, PMI=1.0, K=0.5):
    if PMI > 2.0:
        return ((RON-92.0)/1.6
               -(K * (S-10.0)/1.6)
               +(0.01*ON*(HoV-415.0))/1.6
               +(HoV - 415.0)/130.0
               +(SL - 46.0)/3.0
               -LFV150
               -(0.67+0.5*(PMI-2.0)))
    else:
        return ((RON-92.0)/1.6
                 - (K * (S-10.0)/1.6)
                 + (0.01*ON*(HoV-415.0))/1.6
                 + (HoV - 415.0)/130.0
                 + (SL - 46.0)/3.0
                 - LFV150)


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
