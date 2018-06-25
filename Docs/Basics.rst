Basics
======

Input files
-----------

#. Input scripts
	The behaviour of the co-optimizer is controlled by the entires in the cooptimizer_input.py file.

#.  Fuel properties database
	The fuel properties that are used to estimate blend properties and ultimately evaluate the merit function are read in from a spreadsheet formatted like the example provided and specified in the cooptimizer_input.py file:
	::

		component_properties_database = 'prop_db_AMR.xls'

#. Component cost
	Component costs are read from a separate database, with an example formate provided and also specified in the cooptimizer_input.py fiel:
	::

		component_cost_database = 'cost_db_AMR_OG.xlsx'





Using individual co-optimizer features
--------------------------------------

Apart from optimizing a multi-component blend, the co-optimizer also provides some basic tools to manipulate a database of fuel pure component and blend properties. Loading a spreadsheet of data is done using, for example:

::
	
	from fuelsdb_interface import load_propDB, make_property_vector,\
	                          make_property_vector_sample_cost,\
	                          make_property_vector_all
	propDB = load_propDB('testDB.xls')
	ncomp, spids, propvec = make_property_vector_all(propDB)

The resulting property data structure can then be fed to the blending routines in a function similar to the below:

::

	from blend_functions import blend_fancy_vec
	from merit_functions import revised_mf
	
	def eval_merit(x, propvec, K=-1.25):
		ron = blend_fancy_vec(x,propvec,'RON')
		sen = blend_fancy_vec(x,propvec,'S')
		HoV = blend_fancy_vec(x,propvec,'HoV')
		AFR = blend_fancy_vec(x,propvec,'AFR_STOICH')       
		LFV150 = blend_fancy_vec(x,propvec,'LFV150')
		PMI = blend_fancy_vec(x,propvec,'PMI')
		return revised_mf(RON=ron, S=sen, HoV=HoV, AFR=AFR, LFV150=LFV150, PMI=PMI, K=K)
