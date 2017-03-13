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
	cas_col = xl_s.col(2)
	id_col = xl_s.col(0)
	cas = []
	ids = []
	for c, i in zip(cas_col[1:19], id_col[1:19]):
		if c.value:
			cas.append(c.value)
			ids.append(int(i.value))
	return cas, ids

def load_raw_fpdatabase(input_file,cas_list,keys):
	with open(input_file, 'r') as fpdb:
		fpdbreader = csv.DictReader(fpdb, delimiter=',')
		data_extract = []
		for row in fpdbreader:
			if row['Pure_CAS'] in cas_list:
				a = {}
				for k in keys:
					a[k] = row[k]
				data_extract.append(a)
	

	return data_extract


def write_propDB(output_file,fpdb_extract):
	wb = xlwt.Workbook()
	ws = wb.add_sheet('Components')
	i = 0
	keys = ['CAS','NAME','FORMULA','MOLWT','BP','HoF_liq','LHV','RON','S','HoV','SL','LFV150','PMI','ON']
	dbmap = {}
	dbmap['CAS'] = 'Pure_CAS'
	dbmap['NAME'] = 'Pure_IUPAC_name'
	dbmap['FORMULA'] = 'Pure_Molecular_Formula'
	dbmap['MOLWT'] = 'Pure_Molecular_Weight'
	dbmap['BP'] = 'Pure_Boiling_Point'
	dbmap['LHV'] = 'Pure_LHV'
	dbmap['HoV'] = 'Pure_Heat_of_Vaporization'
	dbmap['RON'] = 'Pure_RON'

	for k in keys:
		ws.write(0,i,k)
		i += 1
	j = 1
	for comp in fpdb_extract:
		i = 0
		for k in keys:
			if k in dbmap:
				ws.write(j,i,comp[dbmap[k]])
			if k is 'S':
				s =  float(re.sub("[^0-9^.]","",comp['Pure_RON'])) - float(re.sub("[^0-9^.]","",comp['Pure_MON']))
				ws.write(j,i,s)
			i += 1
		j += 1
	
	wb.save(output_file)

if __name__ == '__main__':
	cas, id = read_input_list('assert_20.xls')
	print cas
 
 	keys = ['Pure_CAS','Pure_Molecular_Formula','Pure_IUPAC_name',
 	'Pure_Molecular_Weight','Pure_Boiling_Point',
 	'Pure_LHV','Pure_Heat_of_Vaporization',
 	'Pure_RON','Pure_MON']
	data = load_raw_fpdatabase('Fuel Engine Co Optimization-2.csv',cas,keys)
	write_propDB('testpropDB.xls',data)

