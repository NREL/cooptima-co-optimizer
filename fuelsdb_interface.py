#from PyFileMaker import FMServer
#
#fm = FMServer('rgrout_nrel:n4zhZ#>!@https://fuelsdb.nrel.gov/fmi/webd/fuels%20Engine%20CoOptimization')
#
#fm.getDbNames()

# Load data from Bob & Gina's database of fuel properties
import csv
import copy
def load_fuelsdb(dbfile, cas=None):
    propDB = {}
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

    
def load_propDB(fname):
    xl_wb = xlrd.open_workbook(fname)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(3)
    id_col = xl_s.col(0)
    cas = []
    ids = []
    hdr = xl_s.row(0)

    i = 0
    propDB = {}
    for i in range(1,18):
        vals = xl_s.row(i)[0:15]
        newcomponent = {}
        for h,v in zip(hdr,vals):
            newcomponent[h.value] = v.value
        propDB[newcomponent['CAS']] = (copy.deepcopy(newcomponent))

    for cas, rec in propDB.iteritems():
        propDB[cas]['NAME'] = "RON {},Sl {}".format(propDB[cas]['RON'],
                                                    propDB[cas]['SL'])
        
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
