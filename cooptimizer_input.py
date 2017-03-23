# -*- coding: utf-8; -*-
"""cooptimizer_input.py: Input / configuration file for the co-optimizer
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

# Run a tradeoff analysis between cost and obtainable merit function
task_list['cost_vs_merit_Pareto'] = False

task_list['cost_vs_merit_Pareto_UP'] = True
#TODO: work out how to get distribution of merit possible for a given target cost
#      based on uncertainty in cost properties. Do by sampling. Then we can put in bin+-delta.


# Run a tradeoff analysis between engine design and obtainable merit function
task_list['K_vs_merit_sweep'] = False

# Do sampling for various uncertain parameters
task_list['K_sampling'] = False
k_sampling_datafilename = 'k_sampling.txt'
k_sampling_plotfilename = 'k_sampling.pdf'
nsamples = 100
kmean = 0.5
kvar = 1.0

# Do uncertainity propagation for uncertainty in merit function
task_list['UP'] = False
UP_datafilename = 'UP.txt'
UP_plotfilename = 'UP.pdf'

# Coefficients in merit function indicating potential improvement
sen_mean = {}
sen_mean['ON']= 1.0/1.6
sen_mean['ONHoV'] = 0.01
sen_mean['HoV'] = 1.0/130.0
sen_mean['SL'] = 1.0/3.0
sen_mean['LFV150'] = 1.0
sen_mean['PMIFIX'] = 0.67
sen_mean['PMIVAR'] = 0.5

sen_var = {}
sen_var['ON']= 1.0/1.6*.1
sen_var['ONHoV'] = 0.01*.1
sen_var['HoV'] = 1.0/130.0*.1
sen_var['SL'] = 1.0/3.0*.1
sen_var['LFV150'] = 0.1
sen_var['PMIFIX'] = 0.67*.1
sen_var['PMIVAR'] = 0.5*.1

# Coefficients in merit function indicating reference fuel properties
ref_mean = {}
ref_mean['RON'] = 92.0
ref_mean['S'] = 10.0
ref_mean['HoV'] = 415.0
ref_mean['SL'] = 46.0
ref_mean['PMI'] = 2.0

ref_var = {}
ref_var['RON'] = 8.0
ref_var['S'] = 10.0
ref_var['HoV'] = 20.0
ref_var['SL'] = 2.0
ref_var['PMI'] = 2.0

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
print("CAUTION --- USING PLACEHOLDER PROPERTIES")
component_properties_database = 'prop_db_AMR.xls'
component_cost_database = 'cost_db_AMR.xlsx'

# -----------------------------------------------------------------------------
# Output file names
cost_vs_merit_datafilename = "cost_merit_pareto.txt"
cost_vs_merit_plotfilename = "cost_merit_pareto.pdf"

k_sweep_datafilename = "ksweep.txt"
k_sweep_plotfilename = "ksweep.pdf"
# -----------------------------------------------------------------------------
# Displace components setup - PLACEHOLDER FOR FUTURE DEVELOPMENT
# displacement_setup['Displaced_component'] = 'A'
# displacement_setup['Number_max_new_components'] = 2
# Ttls for new components
# displacement_setup['Min_displaced_volume_fraction'] = 0.2
# displacement_setup['Max_displaced_volume_fraction'] = 0.4
# displacement_setup['Ref_composition'] = {}
# displacement_setup['Ref_composition']['3'] = 0.2
# displacement_setup['Ref_composition']['2'] = 0.1
# displacement_setup['Ref_composition']['7'] = 0.7

# -----------------------------------------------------------------------------
# Running multiple "K" values in merit function
#KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0] #-vector
#KVEC = [-2.0, 0.5,  1.0]  # -vector
#KVEC = [-2.0, -1.25, -0.5]  # -vector
KVEC = [-1.25]  # -vector


# -----------------------------------------------------------------------------
# Constraints on minimum/maximum mole fraction of a given component
# NOT YET IMPLEMENTED
# component_min['3'] = 0.7
# component_max['2'] = 0.1
# -----------------------------------------------------------------------------
