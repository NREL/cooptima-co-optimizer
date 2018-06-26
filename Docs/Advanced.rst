User defined objective functions
================================

In order to solve your own multi-objective problems, the user has to implement their functions such that it is a simple function call that returns m scalars, m being the number of objective functions. In the following is information on the necessary modifications. It is probably easiest to use one of the given objectives and modify them:

* Use for example the objective eval_mo starting on l. 162 of nsga2_k.py. The important part is that the return values have to contain the objective function values (in the example mmf_p_p, c_p_p). If you have more than 2 objectives, then correspondingly more values have to be returned. The user is responsible for implementing the computation of the return values. 

* Also modify nsga2_pareto_K starting at l. 535 in nsga2_k.py:

	* l. 543: NDIM (the number of optimization parameters) must be adjusted
	* l. 547: creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0)): the weights represent whether we maximize (1.0) or minimize (-1.0). These weights must be adjusted depending on the sequence of objective function values that are returned. For example, if the order of returned objectives is f1, f2, f3, f4, and we want to maximize f1 and f2, and minimize f3 and f4, the weight vector would be weights=(1.0, 1.0, -1.0, -1.0).
	* l. 556: BOUND_LOW, BOUND_UP = 0,1:  the lower and upper bounds for the optimization must be adjusted. However, we recommend doing the optimization over the scaled hyper unit cube and rescaling the parameters to their true ranges for evaluating the user-defined objective functions. 
	* l. 578, 579: toolbox.decorate("mate", scale_2()), toolbox.decorate("mutate", scale_2()): this is the place where additional optimization constraints are defined. 
