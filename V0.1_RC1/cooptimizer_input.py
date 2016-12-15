task_list = {}
displacement_setup = {}
property_min = {}
property_max = {}
component_min = {}
component_max = {}
# DO NOT EDIT ABOVE THIS LINE -------------------------------------------------


# This is the input file for the optimizer - edit the functions and
# declarations below to configure the behavior

# -----------------------------------------------------------------------------
# What the co-optimizer should do

# Run a tradeoff analysis between cost and merit function
task_list['cost_vs_merit_Pareto'] = False

# Find the composition that maximizes the merit function subject to constraints
# below - NOT YET IMPLEMENTED,  PLACEHOLDER FOR FUTURE DEV
# task_list['maximize_merit'] = False

# Find the compositition that produces at least the minimum merit function
# value for the least cost - NOT YET IMPLEMENTED, PLACEHOLDER FOR FUTURE DEV
# task_list['find_min_cost_sol'] = False

# Find a list of (n) components that could displace a given component in a 
# reference fuel blend. Inputs for 'displacement_setup' below must be 
# filled in,  NOT YET IMPLEMENTED, PLACEHOLDER FOR FUTURE DEV
# task_list['displace_components'] = False
# -----------------------------------------------------------------------------
# How the optimizer does its work
use_pyomo = False
use_deap_NSGAII = True

# -----------------------------------------------------------------------------
# Other input / data files
component_properties_database = 'propDB_fiction.xls'
component_cost_database = 'costDB_fiction.xls' 

# -----------------------------------------------------------------------------
# Displace components setup - PLACEHOLDER FOR FUTURE DEVELOPMENT
# displacement_setup['Displaced_component'] = 'A'
# displacement_setup['Number_max_new_components'] = 2
# displacement_setup['Min_displaced_volume_fraction'] = 0.2 # Ttl for new comps.
# displacement_setup['Max_displaced_volume_fraction'] = 0.4 # Ttl for new comps.
# displacement_setup['Ref_composition'] = {}
# displacement_setup['Ref_composition']['3'] = 0.2
# displacement_setup['Ref_composition']['2'] = 0.1
# displacement_setup['Ref_composition']['7'] = 0.7

# -----------------------------------------------------------------------------
# Running multiple "K" values in merit function
#KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0] #-vector
KVEC = [-2.0, 0.5,  1.0] #-vector


# -----------------------------------------------------------------------------
# Property bounds - these can be omitted to have no bound
# NOT YET IMPLEMENTED 
# property_min['RON'] = 87.0
# property_min['ON'] = 87.0
# property_min['S'] = 36.0
# #property_min['HoV'] = 0.0
# property_min['SL'] = 36.0
# property_min['LFV150'] = 0.0
# property_min['PMI'] = 0.0
# 
# property_max['RON'] = 87.0
# property_max['ON'] = 87.0
# property_max['S'] = 0.0
# #property_max['HoV'] = 100.0
# property_max['SL'] = 36.0
# property_max['LFV150'] = 0.0
# property_max['PMI'] = 0.0

# -----------------------------------------------------------------------------
# Constraints on minimum/maximum volume fraction of a given component
# NOT YET IMPLEMENTED
# component_min['3'] = 0.7
# component_max['2'] = 0.1
# -----------------------------------------------------------------------------

