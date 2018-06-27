# -*- coding: utf-8; -*-
"""co_optimizer.py: Driver code for the co_optimizer
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

import xlrd
import xlwt
from fuelsdb_interface import load_propDB
import csv
import re


def read_input_list(input_file):
    xl_wb = xlrd.open_workbook(input_file)
    xl_s = xl_wb.sheet_by_index(0)
    cas_col = xl_s.col(0)
    uuid_col = xl_s.col(1)
    co_id_col = xl_s.col(2)
    ids = []

    for c in cas_col[1:]:
        if c.value:
            ids.append(c.value)
    for u in uuid_col[1:]:
        if u.value:
            ids.append(u.value)
    for c in co_id_col[1:]:
        if c.value:
            ids.append(c.value)
    return ids


def load_raw_fpdatabase(input_file, cas_list, keys):
    with open(input_file, 'r') as fpdb:
        fpdbreader = csv.DictReader(fpdb, delimiter=',')
        data_extract = []
        for row in fpdbreader:
            if row['Pure_CAS'] in cas_list:
                a = {}
                for k in keys:
                    a[k] = row[k]
                data_extract.append(a)
            elif row['pk_UUID'] in cas_list:
                a = {}
                for k in keys:
                    a[k] = row[k]
                data_extract.append(a)

    return data_extract


def mf_pc_inputs(input_name, comp):
    # Do mapping for pure components
    if input_name is 'CAS':
        return comp['Pure_CAS']

    if input_name is 'pk_UUID':
        return comp['pk_UUID']

    if comp['Pure_IUPAC_name']:
        if input_name is 'BOB':
            return 0

        if input_name is 'NAME':
            return comp['Pure_IUPAC_name']

        if input_name is 'FORMULA':
            return comp['Pure_Molecular_Formula']

        if input_name is 'MOLWT':
            try:
                return float(comp['Pure_Molecular_Weight'])
            except ValueError:
                print("Could not convert:{}".
                      format(comp['Pure_Molecular_Weight']))
                return None

        if input_name is 'BP':
            try:
                return float(comp['Pure_Boiling_Point'])
            except ValueError:
                return None

        if input_name is 'LHV':
            try:
                return float(comp['Pure_LHV'])
            except ValueError:
                return None

        if input_name is 'HoV':
            try:
                return float(comp['Pure_Heat_of_Vaporization']) / \
                       mf_pc_inputs('MOLWT', comp)*1000
            except ValueError:
                return None

        if input_name is 'RON':
            try:
                return float(re.sub("[^0-9^.]", "", comp['Pure_RON']))
            except ValueError:
                return None

        if input_name is 'PMI':
            try:
                return float(comp['Pure_PMI'])
            except ValueError:
                return None

        if input_name is 'S':
            try:
                return float(re.sub("[^0-9^.]", "", comp['Pure_RON'])) - \
                       float(re.sub("[^0-9^.]", "", comp['Pure_MON']))
            except ValueError:
                return None

        if input_name is 'C' or input_name is 'H' or input_name is 'O':
            formula = comp['Pure_Molecular_Formula']
            m = re.search('{}([0-9]+)'.format(input_name), formula)
            if m:
                return int(m.group(1))
            else:
                m = re.search('{}'.format(input_name), formula)
                if m:
                    return 1
                else:
                    return 0

        if input_name is 'LFV150':
            if mf_pc_inputs('BP', comp) < 150:
                return 0.0
            else:
                return 1.0

        if input_name is 'AFR_STOICH':
            nC = mf_pc_inputs('C', comp)
            nH = mf_pc_inputs('H', comp)
            nO = mf_pc_inputs('O', comp)
            mw = mf_pc_inputs('MOLWT', comp)

            if None in (nC, nH, nO, mw):
                return None
            try:
                return ((nC*2+nH/2-nO)/2.0*(32.0+28.0*3.76))/mw
            except ValueError:
                return None

        if input_name is 'DENSITY':
            try:
                return float(comp['Pure_Density'])
            except ValueError:
                return None

        if input_name is 'bRON':
            npts = int(comp['bRON_npts'])
            for i in range(npts):
                bRON_pts['value'] = comp['bRON_value_{}'.format(i)]
                bRON_pts['level'] = comp['bRON_level_{}'.format(i)]

        # this is going to be about getting a function:
        # bRON(vol_fract) that takes the bRON_pts and finds bRON
        # for that level
        # then blend is sum(b_RON*vol_fracts)

        # to debug need to be able to generate bRON(volfract)
        # and also calc RON(volfract, BOB)

        # Going to also need a function to convert molefract to
        # volfract to energyfract to massfract
        # in context of blending model
        # Going to need to know about the rest of the mixture...

        # Need something like this that can read in blends -
        # can I find the blends
        # in the database?
    elif comp['Blend_Name']:
        if input_name is 'NAME':
            return comp['Blend_Name']

        if input_name is 'BOB':
            return 1

        if input_name is 'FORMULA':
            return None

        if input_name is 'MOLWT':
            return None

        if input_name is 'BP':
            return None

        if input_name is 'LHV':
            try:
                return float(comp['Blend_LHV'])
            except ValueError:
                return None

        if input_name is 'HoV':
            return None

        if input_name is 'RON':
            try:
                return float(re.sub("[^0-9^.]", "", comp['Blend_RON']))
            except ValueError:
                return None

        if input_name is 'PMI':
            try:
                return float(comp['Blend_PMI'])
            except ValueError:
                return None

        if input_name is 'S':
            try:
                return float(re.sub("[^0-9^.]", "", comp['Blend_RON'])) -\
                       float(re.sub("[^0-9^.]", "", comp['Blend_MON']))
            except ValueError:
                return None

        # if input_name is 'C' or input_name is 'H' or input_name is 'O':
        #     formula = comp['Pure_Molecular_Formula']
        #     m = re.search('{}([0-9]+)'.format(input_name),formula)
        #     if m:
        #         return int(m.group(1))
        #     else:
        #         m = re.search('{}'.format(input_name),formula)
        #         if m:
        #             return 1
        #         else:
        #             return 0

        # if input_name is 'LFV150':
        #     if mf_pc_inputs('BP',comp) < 150:
        #         return 0.0
        #     else:
        #         return 1.0

        # if input_name is 'AFR_STOICH':
        #     nC = mf_pc_inputs('C',comp)
        #     nH = mf_pc_inputs('H',comp)
        #     nO = mf_pc_inputs('O',comp)
        #     mw = mf_pc_inputs('MOLWT',comp)

        #     if None in (nC,nH,nO,mw):
        #         return None
        #     try:
        #         return ((nC*2+nH/2-nO)/2.0*(32.0+28.0*3.76))/mw
        #     except ValueError:
        #         return None

        if input_name is 'DENSITY':
            try:
                return float(comp['Blend_Density'])
            except ValueError:
                return None

        if input_name is 'bRON':
            npts = int(comp['bRON_npts'])
            for i in range(npts):
                bRON_pts['value'] = comp['bRON_value_{}'.format(i)]
                bRON_pts['level'] = comp['bRON_level_{}'.format(i)]

    else:
        raise ValueError


def write_propDB_remap(output_file, fpdb_extract):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Components')
    i = 0
    keys = ['CAS', 'pk_UUID', 'co_ID', 'NAME', 'BOB', 'FORMULA', 'MOLWT',
            'BP', 'LHV', 'RON', 'S', 'HoV', 'SL', 'LFV150', 'PMI',
            'C', 'H', 'O', 'AFR_STOICH', 'DENSITY']

    for k in keys:
        ws.write(0, i, k)
        i += 1
    j = 1
    for comp in fpdb_extract:
        i = 0
        for k in keys:
            ws.write(j, i, mf_pc_inputs(k, comp))
            i += 1
        j += 1

    wb.save(output_file)


def write_propDB(output_file, propDB):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Components')
    i = 0

    keys = ['CAS', 'pk_UUID', 'co_ID', 'NAME', 'BOB', 'FORMULA',
            'MOLWT', 'BP', 'LHV', 'RON', 'S', 'HoV', 'SL',
            'LFV150', 'PMI', 'C', 'H', 'O', 'AFR_STOICH', 'DENSITY']
    max_blend_pts = 5
    keys.append('bRON_datapts')
    for b in range(max_blend_pts):
        keys.append('base_RON_bRON_{}'.format(b))
        keys.append('vol_frac_bRON_{}'.format(b))
        keys.append('blend_RON_bRON_{}'.format(b))
    keys.append('bMON_datapts')
    for b in range(max_blend_pts):
        keys.append('base_MON_bMON_{}'.format(b))
        keys.append('vol_frac_bMON_{}'.format(b))
        keys.append('blend_MON_bMON_{}'.format(b))

    keys.append('bS_datapts')
    for b in range(max_blend_pts):
        keys.append('base_S_bS_{}'.format(b))
        keys.append('vol_frac_bS_{}'.format(b))
        keys.append('blend_S_bS_{}'.format(b))

    for k in keys:
        ws.write(0, i, k)
        i += 1
    j = 1
    for key, comp in propDB.items():
        i = 0
        print("{}".format(comp))
        for k in keys:
            if k in comp.keys():
                ws.write(j, i, comp[k])
            i += 1
        j += 1

    wb.save(output_file)


if __name__ == '__main__':
    cas = read_input_list('assert_20.xls')
    # Add blendstocks to this as well as flag for is it a blend so that
    # know how to map it to fpdatabase fields
    print("{}".format(cas))

    keys = ['Pure_CAS', 'pk_UUID', 'Pure_Molecular_Formula', 'Pure_IUPAC_name',
            'Pure_Molecular_Weight', 'Pure_Boiling_Point',
            'Pure_LHV', 'Pure_Heat_of_Vaporization',
            'Pure_RON', 'Pure_MON', 'Pure_PMI', 'Pure_Density',
            'Blend_Name', 'Blend_Density', 'Blend_HoV', 'Blend_LHV',
            'Blend_MON', 'Blend_RON', 'Blend_PMI']

    data = load_raw_fpdatabase('Fuel Engine Co Optimization-2.csv', cas, keys)

    write_propDB_remap('testpropDB.xls', data)

    # Now re-read that file, combine with the bRON data and write
    # out the result:
    propDB = load_propDB('testpropDB.xls')
    propDB = load_propDB('bRON_data_with_CAS.xls', propDB_initial=propDB)
    write_propDB('testDB.xls', propDB)
