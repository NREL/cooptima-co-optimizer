Optimization
============

Finding blends that maximize merit function
-------------------------------------------

* Unconstrained optimization
* Single components into blendstock

Multi-objective Optimization
----------------------------

A user interface for this capability is a work in progress; the following notes provide an overview of the available capabilities and how to access them. Selection of the correct form of the objective function for the otpimizer is currently done by uncommenting the correct definition from lines 566--571 in nsga2_k.py. 


A-1
~~~
option 'cost_vs_merit_Pareto'
in nsga2_k: objective function is eval_mo
outfiles: cost_vs_merit_plotfilename (if 2 objectives only); txt file with Pareto front points: sampling_pareto_merit_cost_K_str(KK).txt
parto front [Merit, Cost]
adjust before running: 
in eval_mo: add objective functions (computation and as additional output arguments)
in nsga2_pareto_K: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0)) -- add -1 or 1 for additional objectives
parallel_nsgaruns =False (nothing else is parallelized)

A-2
~~~~
option 'cost_vs_merit_Pareto' - also does maximization of Miles function and Maximization of NMEP
in nsga2_k: objective function is eval_mmf_gp
outfiles: cost_vs_merit_plotfilename (if 2 objectives only); txt file with Pareto front points: sampling_pareto_merit_cost_K_str(KK).txt
parto front [Merit, Cost]
adjust before running: 
in eval_mmf_gp: scaling of engine-related parameters that do not show up in the Miles function: currently uses smallest and largest observed values based on which the GP is computed
in nsga2_pareto_K: creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0)) -- add -1 or 1 for additional objectives
parallel_nsgaruns =False (nothing else is parallelized)
must adjust in main nsga2_k the number of parameters 22+6

A-3
~~~~
option 'cost_vs_merit_Pareto' - also does maximization of Miles function and Maximization of optimized NMEP
in nsga2_k: objective function is eval_MMF_gp_opt
consider only the fuel properties that were measured in experiments for fitting the GP -- fuel compositions that lead to llfuel properties outside of the learning range get bad fitness values and thus they wont be used to create the next generation.
With set fuel properties (in the learning range of the GP), we optimize the engine parameters over the range over which the GP was trained.
outfiles: cost_vs_merit_plotfilename (if 2 objectives only); txt file with Pareto front points: sampling_pareto_merit_cost_K_str(KK).txt
parto front [Merit, Cost]
adjust before running: 
in eval_MMF_gp_opt: scaling of engine-related parameters that do not show up in the Miles function: currently uses smallest and largest observed values based on which the GP is computed; fuel property ranges that are outside the GP learning range must be adjusted if new data comes in
in nsga2_pareto_K: creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0)) -- add -1 or 1 for additional objectives
parallel_nsgaruns =False (nothing else is parallelized)

A-4
~~~~
option 'cost_vs_merit_Pareto' - also does maximization of NMEP and minimization of the variance
in nsga2_k: objective function is eval_gp
outfiles: cost_vs_merit_plotfilename (if 2 objectives only); txt file with Pareto front points: sampling_pareto_merit_cost_K_str(KK).txt
parto front [Merit, Cost]
adjust before running: 
in eval_gp: scaling of fuel property and engine-related parameters: currently uses smallest and largest observed values based on which the GP is computed
in nsga2_pareto_K: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0)) -- add -1 or 1 for additional objectives
parallel_nsgaruns =False (nothing else is parallelized)
must adjust in main nsga2_k the number of parameters

B
~~~
option 'mean_vs_var_Pareto'
in nsga2_k: objective function is eval_mo2
outfiles: mean_vs_var_merit_plotfilename (if 2 objectives only); txt file with Pareto front points: sampling_pareto_mean_var_K_str(KK).txt
in eval_mo: add objective functions (computation and as additional output arguments)
in nsga2_pareto_K: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0)) -- add -1 or 1 for additional objectives
parallel_nsgaruns =False (nothing else is parallelized)

A) and B) can be modularized into one (only differ wrt file names and the objectives that are being optimized






general settings:
in nsga2_k: NGEN (number of generations); MU (number of individuals)
parto front [Merit mean, Merit variance]
nsamples for uncertainty propagation



Uncertainty
-----------

C
~~
option 'cost_vs_merit_Pareto_UP_single'
here we do several NSGA2 runs in parallel and therefore parallel_nsgaruns =True (cannot do nested paralellism with multiprocessing)
wrapper function (wrapper_func) draws random samples for one cost component at a time
n nsga2_k: objective function is eval_mo
decorators!!

D
~~
option 'cost_vs_merit_Pareto_UP'
here we do several NSGA2 runs in parallel and therefore parallel_nsgaruns =True (cannot do nested paralellism with multiprocessing)
n nsga2_k: objective function is eval_mo
wrapper function (wrapper_func_all) draws random samples for all cost components

E
~~
option 'UPMO'
uncertainty in fuel properties
parallelization happens within NSGA2, for a given individual, we do nsamples function evaluations to obtain expected merit and variance
in nsga2: objective function is eval_mo2
outfiles: cooptimizer_input.mean_vs_var_merit_plotfilename, sampling_pareto_data_mean_var_K_+str(KK).txt
pareto front default [max mean, min var] + possibly cost
adjust before running: select eval_mo2, number of objectives
