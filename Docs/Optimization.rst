Optimization
============

The co-optimizer comes with a variety of optimization options that are aimed at solving single and multi-objective optimization problems. There are pre-defined options the user can choose from (see below) and there is the possibility to query self-defined objective functions. In the following, we describe step-by-step how and what files to change in order to run the given options. At the end, we outline how the user can define their own objective functions.  A user interface for this capability is a work in progress; the following notes provide an overview of the available capabilities and how to access them. Configuration currently requires the user to make option selections and changes in two files, namely in cooptimizer_input.py and nsga2_k.py. 


Single-objective Optimization
-----------------------------


This returns one solution that maximizes the SI merit function (known colloquially within co-optima as the 'original Miles merit function') for different values of K, given by:

.. math::
	M = \frac{RON-92}{1.6} - K\frac{S-10}{1.6} + 0.01\frac{HoV-415}{1.6} + \frac{HoV-415}{130} + \frac{S_L-46}{3} \\ 
	- LFV_{150} - \delta (PMI-2.0)\left[0.67 - 0.5(PMI-2.0)\right]

Over time, the merit function used for SI efficiency within co-optima has evolved, and the newer optimization approach (based on genetic algorithms) has been updated to use the revised merit function. To reduce confusion the pyomo interface will eventually be removed from the co-optimizer but for the time being it remains for historical reasons.

To use it, in cooptimizer_input.py set:
::

	task_list['K_vs_merit_sweep'] = True

and also select a range of K values in cooptimizer_input.py, for example:
::
	l. 165: KVEC = [-2.0, -1.5, -1.0,-0.5, 0.5,  1.0, 1.5,2.0,2.5,3.0,3.5,4.0]

A separate single objective optimizations will be done for each K values. 

In addition to the pyomo/IPOPT approach two more robust optimization options are available:	

* The first solves the problem directly and is well suited for a cheap-to-compute objective function. This option is selected in co_optimizer_par.py by setting:
	:: 

		l. 515: merit = maximize_merit(KK, propvec, propDB)
		l. 516: M.append(merit)
		l. 527: plt.scatter(cooptimizer_input.KVEC, M, marker = 'o', c= 'r',s=40,label='python optimizer')

* The second constructs a surrogate function to aid the optimization and is well suited for an expensive-to-compute objective function. To use this option, set in co_optimizer_par.py

	::

		 l. 518: data = surrogate_optimization(KK, propvec,  propDB)
		 l. 519: F.append(-data.Fbest); 
		 l. 528: plt.scatter(cooptimizer_input.KVEC, F, marker = '^', c = 'b', s=40, label='surrogate optimizer')


Multi-objective Optimization
----------------------------

Multi-objective optimization results in (a family) of Pareto fronts rather than a particular vale. Selection of the correct form of the objective function for the optimizer is currently done by uncommenting the correct definition from lines 566--571 in nsga2_k.py.  

Deterministic optimization: return one Pareto front
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For these analysis, in cooptimizer_input.py set:
::

	l.39: task_list['cost_vs_merit_Pareto'] = True

Several combinations of objective functions are possible:

A. Maximize the Miles merit function, minimize the associated costs
	In nsga2_k.py, set:
	:: 
	  	
		l. 569: toolbox.register("evaluate", eval_mo, propvec=propvec, Kinp=KK)
		l. 543: NDIM = len(propvec['COST'])
		l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))

	l. 578 & 579 – must be active

B. Maximize the Miles merit function, maximize the net mean effective pressure (NMEP)
	In nsga2_k.py set:
	::

		l. 567: toolbox.register("evaluate", eval_MMF_gp, propvec=propvec, Kinp=KK, GP = GP, scal = scal)
		l. 543: NDIM = len(propvec['COST'])+3
		l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
		l. 538: use GP, scal  = run_GP()

	l. 578 & 579 – must be active

C.	Maximize the Miles merit function and also maximize NMEP by solving an optimization subproblem on the NMEP-related variables
	In nsga2_k.py set:
	::

		l. 566: toolbox.register("evaluate", eval_MMF_gp_opt, 	vec=propvec, Kinp=KK, GP = GP, scal = scal)
		l. 543: NDIM = len(propvec['COST'])
		l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
		l. 538: scal = run_GP()

	l. 578 & 579 – must be active

D.	Maximize the expected NMEP, minimize the variance of NMEP
	In nsga2_k.py use:
	:: 

		l. 568: toolbox.register("evaluate", eval_gp, GP = GP, scal = scal)
		l. 543: NDIM= 6
		l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
		l. 538: use GP, scal  = run_GP()

	Comment-out L. 578 & 579

Optimization under uncertainty in MMF coefficients: return one Pareto front
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This analysis is to maximize the mean of MMF and minimize the MMF variance (assuming uncertainty in coefficients of the MMF, as defined in lines 75-108 in cooptimizer_input.py). 

In cooptimizer_input.py l.41 set:
::

	task_list['mean_vs_var_Pareto'] = True

In nsga2_k.py set:

::

	l. 570: toolbox.register("evaluate", eval_mo2, propvec=propvec, Kinp=KK)
	l. 543: NDIM= = len(propvec['COST'])
	l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
	l. 578 & 579 – must be active
	l. 338 and following: choose distribution from which to sample coefficients

Optimization under uncertainty in one cost coefficient: returns several Pareto fronts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximizes the MMF, and minimizes the cost, where the cost is a random variable for a single fuel component (all other fuel components are kept with deterministic costs). We draw nsamples (see cooptimizer_input.py l. 58) values for the i-th cost component and thus do nsamples optimizations, yielding nsamples Pareto fronts

In cooptimizer_input.py set:
::
	task_list['cost_vs_merit_Pareto_UP_single'] = True

In nsga2_k.py:
::
	l. 569: toolbox.register("evaluate", eval_mo, propvec=propvec, Kinp=KK)
	l. 543: NDIM= = len(propvec['COST'])
	l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))

l. 578 & 579 – must be active


Optimization under uncertainty in all cost coefficients: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This returns  several Pareto fronts that maximizes the MMF and minimizes the cost, where all component costs are randomly drawn from a distribution. We draw nsamples (see cooptimizer_input.py l. 58) values for each cost component and thus do nsamples optimizations, yielding nsamples Pareto fronts

In cooptimizer_input.py set:
::
	task_list['cost_vs_merit_Pareto_UP'] = True

In nsga2_k.py:
::
	l. 569, toolbox.register("evaluate", eval_mo, propvec=propvec, K	=KK)
	l. 543: NDIM= = len(propvec['COST'])
	l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))

l. 578 & 579 – must be active

