# -*- coding: utf-8; -*-
"""optimizer.py: PYOMO based approach to cost-merit Pareto front identification 
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


import numpy as np
import cooptima_plotting_tools as cpt
from merit_functions import mmf_single, mmf_single_param
from blend_functions import blend_linear_propDB as blend_linear_propDB
from blend_functions import blend_linear_pyomo as blend_linear_pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ
import matplotlib.pyplot as plt
import sys

from fuelsdb_interface import make_property_dict


def run_optimize_vs_C(cstar, KK, propDB, initial_X=None):
        print("Running optimizer based on"
              " IPOPT for c^* = {}, KK={}".format(cstar, KK))
        ncomp, spc_names, propvec = make_property_dict(propDB)

        def obj_fun(model):
            this_ron = blend_linear_pyomo(model, 'RON')
            this_s = blend_linear_pyomo(model, 'S')
            this_on = blend_linear_pyomo(model, 'ON')
            this_HoV = blend_linear_pyomo(model, 'HoV')
            this_SL = blend_linear_pyomo(model, 'SL')
            this_LFV150 = blend_linear_pyomo(model, 'LFV150')
            this_PMI = blend_linear_pyomo(model, 'PMI')
            return mmf_single(RON=this_ron, S=this_s, ON=this_on, HoV=this_HoV,
                              SL=this_SL, K=KK)

        def fraction_constraint(model):
            # Need fractions to sum to unity
            return (np.abs(summation(model.X) - 1.0) <= 1.0e-3)

        def cost_constraint(model):
            return (summation(model.X, model.COST, index=model.I) <= cstar)

    #    pyomo formulation
        model = ConcreteModel()

        # Properties as parameters
        model.n = Param(within=NonNegativeIntegers, initialize=ncomp)
        model.I = RangeSet(1, model.n)

        model.RON = Param(model.I, initialize=propvec['RON'])
        model.S = Param(model.I, initialize=propvec['S'])
        model.ON = Param(model.I, initialize=propvec['ON'])
        model.HoV = Param(model.I, initialize=propvec['HoV'])
        model.SL = Param(model.I, initialize=propvec['SL'])
        model.LFV150 = Param(model.I, initialize=propvec['LFV150'])
        model.PMI = Param(model.I, initialize=propvec['PMI'])
        model.COST = Param(model.I, initialize=propvec['COST'])

        XVEC = 0.05
        model.X = Var(model.I, domain=NonNegativeReals, initialize=XVEC)

        # Objective function and constraints
        model.obj = Objective(rule=obj_fun, sense=maximize)
        model.unity = Constraint(rule=fraction_constraint)
        model.cost = Constraint(rule=cost_constraint)

        # Solve
        opt = SolverFactory('ipopt', solver_io='nl')

        initial_value = 1.0/(ncomp)
        # print 'Initializing uniform composition', initial_value
        initial_sum = 0.0
        for i in range(1, ncomp+1):
            model.X[i].value = initial_value
            initial_sum += model.X[i].value

        # print "Initial sum = ", initial_sum
        # model.X[7] = 0.3
        # model.X[3] = 0.7

        # inst = model.create_instance()
        # result = opt.solve(model, tee=True,
        #                   options={'bound_push':1.0e-20,
        #                            'warm_start_init_point':'yes',
        #                            'acceptable_tol':1.0e-15,
        #                             'max_iter':10000})
        result = opt.solve(model, options={'max_iter': 10000})
        newcomp = {}
        isok = True
        if (result.solver.status == SolverStatus.ok):
            isok = True
        else:
            print "... Something wrong, maybe try restaring "\
                  " trying to find a new starting place from a"\
                  " perturbed problem"
            newcomp, perturbok = run_optimize_vs_C(cstar*1.1, KK,
                                    propDB, initial_X=None)
            if (perturbok):
                print "... Perturbed solution succeeded, "\
                      "setting up to restart from it"
                for i in range(1, ncomp+1):
                    model.X[i].value = newcomp[spc_names[i-1]]

                perturbokresult = opt.solve(model,
                                            options={'max_iter': 10000})
                if (perturbokresult.solver.status == SolverStatus.ok):
                    print "... Found solution to original problem"\
                           " from perturbed solution, success"
                    isok = True
                else:
                    print "... Failed to find solution to original"\
                          " problem from perturbed solution"
                    isok = False

            else:
                print "... Perturbed solution failed"
                isok = False
        # print result
        # print("\nDisplaying Soluiton\n" + '-'*60)
        # model.pprint()

        # Extract composition and plot it
        comp = {}
        for i in range(1, ncomp+1):
            comp[spc_names[i-1]] = model.X[i].value

        return comp, isok


def run_optimize_vs_K(KK, propDB, initial_X=None,ref=None,sen=None):
        print("Running optimizer based on IPOPT for KK={}".format(KK))
        ncomp, spc_names, propvec = make_property_dict(propDB)

        if ref is not None and sen is None:
            print("Error; must spec both ref and sen, or neither")
            sys.exit(-3)
        if sen is not None and ref is None:
            print("Error; must spec both ref and sen, or neither")
            sys.exit(-4)

        def obj_fun(model):
            this_ron = blend_linear_pyomo(model, 'RON')
            this_s = blend_linear_pyomo(model, 'S')
            this_HoV = blend_linear_pyomo(model, 'HoV')
            this_SL = blend_linear_pyomo(model, 'SL')
            this_LFV150 = blend_linear_pyomo(model, 'LFV150')
            this_PMI = blend_linear_pyomo(model, 'PMI')
            if( ref is None):
                return mmf_single(RON=this_ron, S=this_s, HoV=this_HoV,
                                  SL=this_SL, K=KK)
            else:
                return mmf_single_param(ref,sen, RON=this_ron, S=this_s, HoV=this_HoV,
                                        SL=this_SL, K=KK)

        def fraction_constraint(model):
            # Need fractions to sum to unity
            return (np.abs(summation(model.X) - 1.0) <= 1.0e-3)

    #    pyomo formulation
        model = ConcreteModel()

        # Properties as parameters
        model.n = Param(within=NonNegativeIntegers, initialize=ncomp)
        model.I = RangeSet(1, model.n)

        model.RON = Param(model.I, initialize=propvec['RON'])
        model.S = Param(model.I, initialize=propvec['S'])
        model.HoV = Param(model.I, initialize=propvec['HoV'])
        model.SL = Param(model.I, initialize=propvec['SL'])
        model.LFV150 = Param(model.I, initialize=propvec['LFV150'])
        model.PMI = Param(model.I, initialize=propvec['PMI'])

        XVEC = 0.05
        model.X = Var(model.I, domain=NonNegativeReals, initialize=XVEC)

        # Objective function and constraints
        model.obj = Objective(rule=obj_fun, sense=maximize)
        model.unity = Constraint(rule=fraction_constraint)

        # Solve
        opt = SolverFactory('ipopt', solver_io='nl')

        initial_value = 1.0/(ncomp)
        # print 'Initializing uniform composition', initial_value
        initial_sum = 0.0
        for i in range(1, ncomp+1):
            model.X[i].value = initial_value
            initial_sum += model.X[i].value

        # print "Initial sum = ", initial_sum
        # model.X[7] = 0.3
        # model.X[3] = 0.7

        # inst = model.create_instance()
        # result = opt.solve(model, tee=True,
        #                   options={'bound_push':1.0e-20,'warm_start_init_point':'yes',
        # 'acceptable_tol':1.0e-15,'max_iter':10000})
        result = opt.solve(model, options={'max_iter': 10000})
        newcomp = {}
        isok = True
        if (result.solver.status == SolverStatus.ok):
            isok = True
        else:
            print "... Something wrong, maybe try restaring "\
                  " trying to find a new starting place from a"\
                  " perturbed problem"
            newcomp, perturbok = run_optimize_vs_K(KK*1.1, propDB,
                                                   initial_X=None)
            if (perturbok):
                print "... Perturbed solution succeeded,"\
                      " setting up to restart from it"
                for i in range(1, ncomp+1):
                    model.X[i].value = newcomp[spc_names[i-1]]

                perturbokresult = opt.solve(model, options={'max_iter': 10000})
                if (perturbokresult.solver.status == SolverStatus.ok):
                    print "... Found solution to original problem \
                    from perturbed solution, success"
                    isok = True
                else:
                    print "... Failed to find solution to original"\
                          " problem from perturbed solution"
                    isok = False

            else:
                print "... Perturbed solution failed"
                isok = False
        # print result
        # print("\nDisplaying Soluiton\n" + '-'*60)
        # model.pprint()

        # Extract composition and plot it
        comp = {}
        for i in range(1, ncomp+1):
            comp[spc_names[i-1]] = model.X[i].value

        return comp, isok


def comp_to_cost_mmf(comp, propDB, k):
        # Evaluate resulting mmf value
        # print comp
        prop_list = ['RON', 'S', 'ON', 'HoV', 'SL', 'LFV150', 'PMI']
        props = {}
        for p in prop_list:
            props[p] = blend_linear_propDB(p, propDB, comp)
        mmf = mmf_single(RON=props['RON'], S=props['S'], ON=props['ON'],
                         HoV=props['HoV'], SL=props['SL'],
                         LFV150=props['LFV150'], PMI=props['PMI'], K=k)
        cost = blend_linear_propDB('COST', propDB, comp)
#        print "Cost: {}, cstar = {} ".format(cost,cstar)

        # This to make a radar plot of the composition for each solution
        # cpt.plot_comp_radar(propDB,comp,savefile="{}_mfK{}".format(cstar,KK),
        #                     title="Composition, "\
        #                     "C*={}, mmf={:.2f}".format(cstar,mmf))
        return cost, mmf


def comp_to_mmf(comp, propDB, k,ref=None,sen=None):
        # Evaluate resulting mmf value
        # print comp
        prop_list = ['RON', 'S', 'HoV', 'SL', 'LFV150', 'PMI']
        props = {}
        for p in prop_list:
            props[p] = blend_linear_propDB(p, propDB, comp)
        if ref is None and sen is None:
            mmf = mmf_single(RON=props['RON'], S=props['S'], 
                             HoV=props['HoV'], SL=props['SL'],
                             LFV150=props['LFV150'], PMI=props['PMI'], K=k)
        else:   
            mmf = mmf_single_param(ref,sen,RON=props['RON'], S=props['S'], 
                                   HoV=props['HoV'], SL=props['SL'],
                                   LFV150=props['LFV150'], PMI=props['PMI'], K=k)
        cost = blend_linear_propDB('COST', propDB, comp)

        return mmf
