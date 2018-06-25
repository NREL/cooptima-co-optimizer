# -*- coding: utf-8; -*-
"""fuelsdb_interface.py: Tools to read an export from the co-optima fuel
properties database and also simplified spreadsheets based on the list of 20
or similar downselects
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

# Load data from Bob & Gina's database of fuel properties
import csv
import copy
import numpy as np
import xlrd
import xlwt
import sys
import math
import scipy.stats as stats

def ResizeList(l, size, fill_with=None):
    l += [fill_with]*(size-len(l))

def make_property_vector_all(propDB):
    ncomp = 0
    spids = []

    for k in propDB.keys():
        spids.append(k)

    ncomp = len(spids)

    props = list(propDB[list(propDB.keys())[0]].keys())

    propvec = {}
    for prop in props:
        propvec[prop] = np.zeros(ncomp)
        #ResizeList(propvec[prop], ncomp)

        if prop == 'NAME':
            propvec[prop] = []
            for i in range(0, ncomp):
                propvec[prop].append(propDB[spids[i]][prop])

        else:
            for i in range(0, ncomp):
                try:
                    propvec[prop][i] = propDB[spids[i]][prop]
                except:
                    propvec[prop][i] = 0

    return ncomp, spids, propvec

def make_property_vector_all_sample_cost(propDB):
    ncomp = 0
    spids = []

    for k in propDB.keys():
        spids.append(k)

    ncomp = len(spids)

    props = propDB[propDB.keys()[0]].keys()

    propvec = {}
    for prop in props:
        propvec[prop] = np.zeros(ncomp)
        #ResizeList(propvec[prop], ncomp)

        if prop == 'NAME':
            propvec[prop] = []
            for i in range(0, ncomp):
                propvec[prop].append(propDB[spids[i]][prop])

        if prop == 'COST':
            for i in range(0, ncomp):
                zz = 0
                while True:
                    if propDB[spids[i]]['COSTVAR'] >0:
                        #print(spids[i])
                        
                        #normal distribution
                        #propvec['COST'][i] = np.random.normal(propDB[spids[i]]['COST'], np.sqrt(propDB[spids[i]]['COSTVAR']))
                        
                        #print(spids[i], propvec['COST'][i])

                        #uniform distribution
                        #propvec['COST'][i] = np.random.uniform(low = (min(0,propDB[spids[i]]['COST']-3*np.sqrt(propDB[spids[i]]['COSTVAR']))),\
                        #high =(propDB[spids[i]]['COST']+3*np.sqrt(propDB[spids[i]]['COSTVAR'])))
                        
                        #lognormal
                        mu = propDB[spids[i]]['COST']
                        v= propDB[spids[i]]['COSTVAR']
                        sigma = np.sqrt(v)
                        propvec['COST'][i] = np.random.lognormal(mu, sigma)

                        '''
                        #truncated normal distribution
                        mu = propDB[spids[i]]['COST']
                        v= propDB[spids[i]]['COSTVAR']
                        myclip_b = mu+3*np.sqrt(v)
                        lower, upper = 0, myclip_b
                        sigma = np.sqrt(v)
                        X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        #N = stats.norm(loc=mu, scale=sigma)
                        propvec['COST'][i] = X.rvs(1)
                        '''




                    else:
                        propvec['COST'][i] = propDB[spids[i]]['COST']


                    '''
                    propvec['COST'][i] = \
                    np.random.normal(propDB[spids[i]]['COST'], \
                        propDB[spids[i]]['COSTVAR'])
                    '''
                    if (propvec['COST'][i] > 0.0):
                        break
                    zz += 1
                    if( zz > 1000):
                        print ("Should never have gotten here")
                        sys.exit(-1)
                       
        else:
            for i in range(0, ncomp):
                try:
                    propvec[prop][i] = propDB[spids[i]][prop]
                except:
                    propvec[prop][i] = 0

    return ncomp, spids, propvec




def make_property_vector_all_sample_cost_UP_single(propDB, change_name):
    ncomp = 0
    spids = []

    for k in propDB.keys():
        spids.append(k)

    ncomp = len(spids)

    props = propDB[propDB.keys()[0]].keys()

    propvec = {}
    for prop in props:
        propvec[prop] = np.zeros(ncomp)
        #ResizeList(propvec[prop], ncomp)

        if prop == 'NAME':
            propvec[prop] = []
            for i in range(0, ncomp):
                propvec[prop].append(propDB[spids[i]][prop])

        if prop == 'COST':
            for i in range(0, ncomp):
                zz = 0
                while True:
                    if spids[i] == change_name:
                        print("changing component number: ", i, change_name)
                        #propDB[spids[i]]['COSTVAR'] >0:
                        #print(spids[i])
                        #NORMAL DISTRIBUTION
                        #propvec['COST'][i] = np.random.normal(propDB[spids[i]]['COST'], np.sqrt(propDB[spids[i]]['COSTVAR'])) #normal distribution
                        #print(propvec['COST'][i])
                        
                        #uniform distribution
                        #propvec['COST'][i] = np.random.uniform(low = (min(0,propDB[spids[i]]['COST']-3*np.sqrt(propDB[spids[i]]['COSTVAR']))),\
                        #high =(propDB[spids[i]]['COST']+3*np.sqrt(propDB[spids[i]]['COSTVAR'])))
                        
                        #lognormal
                        mu = propDB[spids[i]]['COST']
                        v= propDB[spids[i]]['COSTVAR']
                        sigma = np.sqrt(v)
                        propvec['COST'][i] = np.random.lognormal(mu, sigma)
                        
                        #truncated normal distribution
                        #mu = propDB[spids[i]]['COST']
                        #v= propDB[spids[i]]['COSTVAR']
                        #myclip_b = mu+3*np.sqrt(v)
                        #lower, upper = 0, myclip_b
                        #sigma = np.sqrt(v)
                        #X = stats.truncnorm( (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        #N = stats.norm(loc=mu, scale=sigma)
                        #propvec['COST'][i] = X.rvs(1)
                        

                        #print(spids[i], propvec['COST'][i])
                    else:
                        propvec['COST'][i] = propDB[spids[i]]['COST']


                    '''
                    propvec['COST'][i] = \
                    np.random.normal(propDB[spids[i]]['COST'], \
                        propDB[spids[i]]['COSTVAR'])
                    '''
                    if (propvec['COST'][i] > 0.0): #truncated normal distribution instead
                        break
                    zz += 1
                    if( zz > 1000):
                        print ("Should never have gotten here")
                        sys.exit(-1)
                       
        else:
            for i in range(0, ncomp):
                try:
                    propvec[prop][i] = propDB[spids[i]][prop]
                except:
                    propvec[prop][i] = 0

    return ncomp, spids, propvec



def make_property_vector(propDB):

    # Assemble property vectors for each composition
    # Basically just transpose the information in propDB
    ncomp = 0
    SPNM = []
    for k in propDB.keys():
        # for kk in propDB[k].keys():
            # print ("key: {}".format(kk))
        SPNM.append(k)
    ncomp = len(SPNM)

    RONVEC = np.zeros(ncomp)
    SVEC = np.zeros(ncomp)
    ONVEC = np.zeros(ncomp)
    HoVVEC = np.zeros(ncomp)
    SLVEC = np.zeros(ncomp)
    LFV150VEC = np.zeros(ncomp)
    PMIVEC = np.zeros(ncomp)
    COSTVEC = np.zeros(ncomp)
    XVEC = np.zeros(ncomp)

    for i in range(0, ncomp):
        RONVEC[i] = (propDB[SPNM[i-1]]['RON'])
        SVEC[i] = (propDB[SPNM[i-1]]['S'])
        HoVVEC[i] = (propDB[SPNM[i-1]]['HoV'])
        #SLVEC[i] = (propDB[SPNM[i-1]]['SL'])
        LFV150VEC[i] = (propDB[SPNM[i-1]]['LFV150'])
        PMIVEC[i] = (propDB[SPNM[i-1]]['PMI'])
        COSTVEC[i] = (propDB[SPNM[i-1]]['COST'])
        XVEC[i] = 0.05

    propvec = {}
    propvec['RON'] = RONVEC.copy()
    propvec['S'] = SVEC.copy()
    propvec['HoV'] = HoVVEC.copy()
    propvec['SL'] = SLVEC.copy()
    propvec['LFV150'] = LFV150VEC.copy()
    propvec['PMI'] = PMIVEC.copy()
    propvec['COST'] = COSTVEC.copy()

    return ncomp, SPNM, propvec

def make_property_vector_sample_cost(propDB):

    # Assemble property vectors for each composition
    # Basically just transpose the information in propDB
    ncomp = 0
    SPNM = []
    for k in propDB.keys():
        # for kk in propDB[k].keys():
            # print ("key: {}".format(kk))
        SPNM.append(k)
    ncomp = len(SPNM)

    RONVEC = np.zeros(ncomp)
    SVEC = np.zeros(ncomp)
    ONVEC = np.zeros(ncomp)
    HoVVEC = np.zeros(ncomp)
    SLVEC = np.zeros(ncomp)
    LFV150VEC = np.zeros(ncomp)
    PMIVEC = np.zeros(ncomp)
    COSTVEC = np.zeros(ncomp)
    XVEC = np.zeros(ncomp)

    for i in range(0, ncomp):
        RONVEC[i] = (propDB[SPNM[i-1]]['RON'])
        SVEC[i] = (propDB[SPNM[i-1]]['S'])
        HoVVEC[i] = (propDB[SPNM[i-1]]['HoV'])
        SLVEC[i] = (propDB[SPNM[i-1]]['SL'])
        LFV150VEC[i] = (propDB[SPNM[i-1]]['LFV150'])
        PMIVEC[i] = (propDB[SPNM[i-1]]['PMI'])
        zz = 0
        while True:
            COSTVEC[i] = np.random.normal(propDB[SPNM[i-1]]['COST'], propDB[SPNM[i-1]]['COSTVAR'])
            if COSTVEC[i] > 0:
                break
            zz += 1
            if( zz > 1000):
                print ("Should never have gotten here")
                sys.exit(-1)

        XVEC[i] = 0.05

    propvec = {}
    propvec['RON'] = RONVEC.copy()
    propvec['S'] = SVEC.copy()
    propvec['HoV'] = HoVVEC.copy()
    propvec['SL'] = SLVEC.copy()
    propvec['LFV150'] = LFV150VEC.copy()
    propvec['PMI'] = PMIVEC.copy()
    propvec['COST'] = COSTVEC.copy()

    return ncomp, SPNM, propvec


def make_property_dict(propDB):

    # Assemble property vectors for each composition
    # Basically just transpose the information in propDB
    ncomp = 0
    SPNM = []
    for k in propDB.keys():
        # for kk in propDB[k].keys():
        #    print ("key: {}".format(kk))
        SPNM.append(k)
    ncomp = len(SPNM)

    RONVEC = {}
    SVEC = {}
    ONVEC = {}
    HoVVEC = {}
    SLVEC = {}
    LFV150VEC = {}
    PMIVEC = {}
    COSTVEC = {}
    XVEC = {}

    for i in range(1, ncomp+1):
        RONVEC[i] = (propDB[SPNM[i-1]]['RON'])
        SVEC[i] = (propDB[SPNM[i-1]]['S'])
        HoVVEC[i] = (propDB[SPNM[i-1]]['HoV'])
        SLVEC[i] = (propDB[SPNM[i-1]]['SL'])
        LFV150VEC[i] = (propDB[SPNM[i-1]]['LFV150'])
        PMIVEC[i] = (propDB[SPNM[i-1]]['PMI'])
        COSTVEC[i] = (propDB[SPNM[i-1]]['COST'])
        XVEC[i] = 0.05

    propvec = {}
    propvec['RON'] = RONVEC.copy()
    propvec['S'] = SVEC.copy()
    propvec['HoV'] = HoVVEC.copy()
    propvec['SL'] = SLVEC.copy()
    propvec['LFV150'] = LFV150VEC.copy()
    propvec['PMI'] = PMIVEC.copy()
    propvec['COST'] = COSTVEC.copy()

    return ncomp, SPNM, propvec


def load_fuelsdb(dbfile, cas=None, propDB_initial=None):
    propDB = {}
    if propDB is not None:
        propDB = propDB_initial.copy()

    print("Reading fuel properties database")
    with open(dbfile) as fueldbfile:
        fuelsdb = csv.reader(fueldbfile, delimiter=',', quotechar='\"')
        firstrow = True
        for row in fuelsdb:
            if firstrow:
                hdrs = row
                firstrow = False
                # print hdrs
            else:
                propDB_entry = {}
                for h, r in zip(hdrs, row):
                    propDB_entry[h] = r
                if(cas):
                    if(propDB_entry['Pure_CAS'] in cas):
                        propDB[propDB_entry['Pure_CAS']]\
                             = copy.deepcopy(propDB_entry)
                else:
                    propDB[propDB_entry['Pure_CAS']]\
                            = copy.deepcopy(propDB_entry)
    return propDB


def load_listof20(listof20file):
    xl_wb = xlrd.open_workbook(listof20file)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(3)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    for c, i in zip(cas_col[1:19], id_col[1:19]):
        if c.value:
            cas.append(c.value)
            ids.append(int(i.value))

    return cas, ids


def write_xl_db(propDB, outfile, order=None, keyorder=None):
    wb = xlwt.Workbook()
    ws = wb.add_sheet("propDB Summary")

    # Write header row
    i = 1
    if not keyorder:
        for key in propDB.values()[0].keys():
            ws.write(0, i, key)
            i += 1
    else:
        for k in keyorder:
            ws.write(0, i, k)
            i += 1

    if not order:
        i = 1
        for f, p in propDB.iteritems():
            ws.write(i, 0, f)
            j = 1
            for k, v in p.iteritems():
                ws.write(i, j, p[k])
                j += 1
            i += 1
    else:
        (cs, ivs) = order
        for c, i in zip(cs, ivs):
            # print("{}, {}".format(c, i))
            # print("{}".format(propDB[c]))
            ws.write(i, 0, c)
            if not keyorder:
                j = 1
                for k, v in propDB[c].iteritems():
                    ws.write(ti, j, propDB[c][k])
                    j += 1
            else:
                j = 1
                for k in keyorder:
                    # print(" writing {} into column {}".format(k, j))
                    ws.write(i, j, propDB[c][k])
                    j += 1
    wb.save(outfile)


def load_propDB(fname, propDB_initial=None, maxrows=18, maxcols=14):
    xl_wb = xlrd.open_workbook(fname)
    xl_s = xl_wb.sheet_by_index(0)
    ncol = xl_s.ncols;
    nrow = xl_s.nrows;
    cas_col = xl_s.col(0)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    hdr = xl_s.row(0)

    print("maximum columns:{}".format(ncol))
    print("rows: {}".format(nrow))
    i = 0
    propDB = {}
    if propDB_initial is not None:
        propDB = propDB_initial.copy()
    for i in range(1, xl_s.nrows):
        vals = xl_s.row(i)[0:xl_s.ncols]
        newcomponent = {}
        for h, v in zip(hdr, vals):
            newcomponent[h.value] = v.value
        # If no initial database was supplied,
        # just copy the entire dictionary over

        # key for item we're looking at:
        # Key preference is CAS, pk_UUID, co_ID
        if('CAS' in newcomponent.keys() and newcomponent['CAS']):
            kk = newcomponent['CAS']
        elif('pk_UUID' in newcomponent.keys() and newcomponent['pk_UUID']):
            kk = newcomponent['pk_UUID']
        elif('co_ID' in newcomponent.keys() and newcomponent['co_ID']):
            kk = newcomponent['co_ID']
        else:
            continue

        if propDB_initial is None:
                propDB[kk] = (copy.deepcopy(newcomponent))
        else:
            print ("Updating record with key = {}".format(kk))
            if kk in propDB.keys():
                print ("key already exists:{}".format(kk))
                print ("   Old name:{}".format(propDB[kk]['NAME']))
                print ("   New name:{}".format(newcomponent['NAME']))
                for k, v in newcomponent.items():
                    # print "Old record", propDB[newcomponent['CAS']]
                    if k is 'NAME':
                        if propDB[kk][k] is not v:
                            if propDB[kk]['XTRA_NAMES']:
                                propDB[kk]['XTRA_NAMES'].append(v)
                            else:
                                propDB[kk]['XTRA_NAMES'] = []
                                propDB[kk]['XTRA_NAMES'].append(v)
    
                    else:
                        propDB[kk][k] = v
            else:
                propDB[kk] = {}
                for k, v in newcomponent.iteritems():
                    propDB[kk][k] = v
    # for cas, rec in propDB.iteritems():
    #     propDB[cas]['NAME'] = "RON {},Sl {}".format(propDB[cas]['RON'],
    #                                                 propDB[cas]['SL'])

    # print "PROPDB:", propDB
    return propDB


def load_listof20(listof20file):
    xl_wb = xlrd.open_workbook(listof20file)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(3)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    for c, i in zip(cas_col[1:19], id_col[1:19]):
        if c.value:
            cas.append(c.value)
            ids.append(int(i.value))

    return cas, ids


def write_xl_db(propDB, outfile, order=None, keyorder=None):
    wb = xlwt.Workbook()
    ws = wb.add_sheet("propDB Summary")

    # Write header row
    i = 1
    if not keyorder:
        for key in propDB.values()[0].keys():
            ws.write(0, i, key)
            i += 1
    else:
        for k in keyorder:
            ws.write(0, i, k)
            i += 1

    if not order:
        i = 1
        for f, p in propDB.iteritems():
            ws.write(i, 0, f)
            j = 1
            for k, v in p.iteritems():
                ws.write(i, j, p[k])
                j += 1
            i += 1
    else:
        (cs, ivs) = order
        for c, i in zip(cs, ivs):
            # print("{}, {}".format(c, i))
            # print("{}".format(propDB[c]))
            ws.write(i, 0, c)
            if not keyorder:
                j = 1
                for k, v in propDB[c].iteritems():
                    ws.write(i, j, propDB[c][k])
                    j += 1
            else:
                j = 1
                for k in keyorder:
                    # print(" writing {} into column {}".format(k, j))
                    ws.write(i, j, propDB[c][k])
                    j += 1
    wb.save(outfile)
