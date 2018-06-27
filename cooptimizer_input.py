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

# the optimizations to be done -- several options should
# be possible iteratively or in parallel
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
task_list['cost_vs_merit_Pareto'] = True

task_list['mean_vs_var_Pareto'] = False

task_list['cost_vs_merit_Pareto_UP_single'] = False

task_list['cost_vs_merit_Pareto_UP'] = False


# TODO: work out how to get distribution of merit possible for a
#       given target cost
#       based on uncertainty in cost properties.
#       Do by sampling. Then we can put in bin+-delta.

# Run a tradeoff analysis between engine design and obtainable merit function
task_list['K_vs_merit_sweep'] = False

# Do sampling for various uncertain parameters
task_list['K_sampling'] = False
k_sampling_datafilename = 'k_sampling.txt'
k_sampling_plotfilename = 'k_sampling.pdf'
# number of random numbers used for each cost
# (basically nsamples runs of the optimizer with different random numbers)
nsamples = 5  # 30
kmean = 0.5
kvar = 1.0

# Do uncertainity propagation for uncertainty in merit function
task_list['UPMO'] = False
task_list['UP'] = False
UP_datafilename = 'UP.txt'
UP_plotfilename = 'UP.pdf'


if task_list['cost_vs_merit_Pareto'] or task_list['mean_vs_var_Pareto'] or task_list['UPMO'] or task_list['K_vs_merit_sweep']:
    parallel_nsgaruns = False
else:
    parallel_nsgaruns = True

# Coefficients in merit function indicating potential improvement
sen_mean = {}
sen_mean['ON'] = 1.0/1.6
sen_mean['ONHoV'] = 0.01
sen_mean['HoV'] = 1.0/130.0
sen_mean['SL'] = 1./5.4  # old:1.0/3.0
sen_mean['LFV150'] = 1.0
sen_mean['PMIFIX'] = 0.7  # old:0.67
sen_mean['PMIVAR'] = 0.5

sen_var = {}
mul = 0.1  # orig: .1
sen_var['ON'] = 1.0/1.6*mul
sen_var['ONHoV'] = 0.01*mul
sen_var['HoV'] = 1.0/130.0*mul
sen_var['SL'] = 1./5.4*mul  # old:1.0/3.0*mul
sen_var['LFV150'] = mul
sen_var['PMIFIX'] = 0.7*mul  # old:0.67*mul
sen_var['PMIVAR'] = 0.5*mul

# Coefficients in merit function indicating reference fuel properties
ref_mean = {}
ref_mean['RON'] = 91.  # old:92.0
ref_mean['S'] = 8.  # old:10.0
ref_mean['HoV'] = 415.0
ref_mean['SL'] = 46.0
ref_mean['PMI'] = 1.6  # old:2.0

ref_var = {}
tul = 1
ref_var['RON'] = 8.0*tul
ref_var['S'] = 8.*tul  # 10.0*tul
ref_var['HoV'] = 20.0*tul
ref_var['SL'] = 2.0*tul
ref_var['PMI'] = 1.6*tul  # old:2.0*tul

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
component_cost_database = 'cost_db_AMR_OG.xlsx'

# -----------------------------------------------------------------------------
# Output file names
cost_vs_merit_datafilename = "cost_revmerit_pareto.txt"  # for pyomo
cost_vs_merit_plotfilename = "cost_revmerit_pareto"


mean_vs_var_merit_plotfilename = "mean_var_merit_pareto"
mean_vs_var_merit_datafilename = "mean_var_merit_pareto.txt"  # for pyomo

cost_vs_merit_Pareto_UP_single_plotfilename = "UPsingle_cost_vs_merit_pareto"
cost_vs_merit_Pareto_UP_single_datafilename = "UPsingle_cost_vs_merit_pareto.txt"

cost_vs_merit_Pareto_UP_plotfilename = "UP_cost_vs_merit_pareto"
cost_vs_merit_Pareto_UP_datafilename = "UP_cost_vs_merit_pareto.txt"

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
# KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0] #-vector
KVEC = [-1.25]  # -vector


# -----------------------------------------------------------------------------
# Constraints on minimum/maximum mole fraction of a given component
# NOT YET IMPLEMENTED
# component_min['3'] = 0.7
# component_max['2'] = 0.1
# -----------------------------------------------------------------------------
