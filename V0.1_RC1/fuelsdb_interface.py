# Load data from Bob & Gina's database of fuel properties
import csv
import copy
import numpy as np

def make_property_vector(propDB):
       
    # Assemble property vectors for each composition
    # Basically just transpose the information in propDB
    ncomp = 0
    SPNM = []
    for k in propDB.keys():
        for kk in propDB[k].keys():
            print ("key: {}".format(kk))
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
        RONVEC[i] =  ( propDB[SPNM[i-1]]['RON'] )
        SVEC[i] =  ( propDB[SPNM[i-1]]['S'] )
        ONVEC[i] =  ( propDB[SPNM[i-1]]['ON'] )
        HoVVEC[i] =  ( propDB[SPNM[i-1]]['HoV'] )
        SLVEC[i] =  ( propDB[SPNM[i-1]]['SL'] )
        LFV150VEC[i] =  ( propDB[SPNM[i-1]]['LFV150'] )
        PMIVEC[i] =  ( propDB[SPNM[i-1]]['PMI'] )
        COSTVEC[i] =  ( propDB[SPNM[i-1]]['COST'] )
        XVEC[i] =  0.05

    propvec = {}
    propvec['RON'] = RONVEC.copy()
    propvec['S'] = SVEC.copy()
    propvec['ON'] = ONVEC.copy()
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
        for kk in propDB[k].keys():
            print ("key: {}".format(kk))
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
        RONVEC[i] =  ( propDB[SPNM[i-1]]['RON'] )
        SVEC[i] =  ( propDB[SPNM[i-1]]['S'] )
        ONVEC[i] =  ( propDB[SPNM[i-1]]['ON'] )
        HoVVEC[i] =  ( propDB[SPNM[i-1]]['HoV'] )
        SLVEC[i] =  ( propDB[SPNM[i-1]]['SL'] )
        LFV150VEC[i] =  ( propDB[SPNM[i-1]]['LFV150'] )
        PMIVEC[i] =  ( propDB[SPNM[i-1]]['PMI'] )
        COSTVEC[i] =  ( propDB[SPNM[i-1]]['COST'] )
        XVEC[i] =  0.05

    propvec = {}
    propvec['RON'] = RONVEC.copy()
    propvec['S'] = SVEC.copy()
    propvec['ON'] = ONVEC.copy()
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
                for h, r in zip(hdrs,row):
                    propDB_entry[h] = r
                if(cas):
                    if(propDB_entry['Pure_CAS'] in cas):
                       propDB[propDB_entry['Pure_CAS']] = copy.deepcopy(propDB_entry)
                else:
                   propDB[propDB_entry['Pure_CAS']] = copy.deepcopy(propDB_entry)
    return propDB

import xlrd
import xlwt
def load_listof20(listof20file):
    xl_wb = xlrd.open_workbook(listof20file)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(3)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    for c,i in zip(cas_col[1:19],id_col[1:19]):
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
        i=1
        for f,p in propDB.iteritems():
            ws.write(i,0,f)
            j=1
            for k, v in p.iteritems():
                ws.write(i,j,p[k])
                j += 1
            i+=1
    else:
        (cs, ivs) = order
        for c, i in zip(cs,ivs):
            print("{}, {}".format( c, i))
            print("{}".format(propDB[c]))
            ws.write(i,0,c)
            if not keyorder:
                j=1
                for k, v in propDB[c].iteritems():
                    ws.write(ti,j,propDB[c][k])
                    j += 1
            else:
                j=1
                for k in keyorder:
                    print(" writing {} into column {}".format(k,j))
                    ws.write(i,j,propDB[c][k])
                    j += 1



    wb.save(outfile)

    
def load_propDB(fname, propDB_initial=None, maxrows=18, maxcols=14):
    xl_wb = xlrd.open_workbook(fname)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(0)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    hdr = xl_s.row(0)

    i = 0
    propDB = {}
    if propDB_initial is not None:
        propDB = propDB_initial.copy()
    for i in range(1,maxrows):
        vals = xl_s.row(i)[0:maxcols]
        newcomponent = {}
        for h,v in zip(hdr,vals):
            newcomponent[h.value] = v.value
        # If no initial database was supplied, just copy the entire dictionary over
        if propDB_initial is None:
            propDB[newcomponent['CAS']] = (copy.deepcopy(newcomponent))
        else:
            for k,v in newcomponent.iteritems():
                print "Old record", propDB[newcomponent['CAS']]
                if k is not 'CAS':
                    print "adding key: ", k, ' value:', v
                    propDB[newcomponent['CAS']][k] = v

    # for cas, rec in propDB.iteritems():
    #     propDB[cas]['NAME'] = "RON {},Sl {}".format(propDB[cas]['RON'],
    #                                                 propDB[cas]['SL'])
        
    print "PROPDB:", propDB
    return propDB




import xlrd
import xlwt
def load_listof20(listof20file):
    xl_wb = xlrd.open_workbook(listof20file)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(3)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    for c,i in zip(cas_col[1:19],id_col[1:19]):
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
        i=1
        for f,p in propDB.iteritems():
            ws.write(i,0,f)
            j=1
            for k, v in p.iteritems():
                ws.write(i,j,p[k])
                j += 1
            i+=1
    else:
        (cs, ivs) = order
        for c, i in zip(cs,ivs):
            print("{}, {}".format( c, i))
            print("{}".format(propDB[c]))
            ws.write(i,0,c)
            if not keyorder:
                j=1
                for k, v in propDB[c].iteritems():
                    ws.write(i,j,propDB[c][k])
                    j += 1
            else:
                j=1
                for k in keyorder:
                    print(" writing {} into column {}".format(k,j))
                    ws.write(i,j,propDB[c][k])
                    j += 1



    wb.save(outfile)
