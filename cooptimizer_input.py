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
task_list['cost_vs_merit_Pareto'] = True


# Run a tradeoff analysis between engine design and obtainable merit function
task_list['K_vs_merit_sweep'] = False

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
use_pyomo = True
use_deap_NSGAII = False

# -----------------------------------------------------------------------------
# Other input / data files
print("CAUTION --- USING PLACEHOLDER PROPERTIES")
component_properties_database = 'propDB_fiction.xls'
component_cost_database = 'costDB_fiction.xls'

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
# KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0] #-vector
KVEC = [-2.0, 0.5,  1.0]  # -vector


# -----------------------------------------------------------------------------
# Constraints on minimum/maximum mole fraction of a given component
# NOT YET IMPLEMENTED
# component_min['3'] = 0.7
# component_max['2'] = 0.1
# -----------------------------------------------------------------------------
