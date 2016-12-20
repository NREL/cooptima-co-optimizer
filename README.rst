Co-optimization of fuels and engines Co-optimizer
=================================================

This collection of python scripts is the first (internal) release of the co-optimizer,
or the co-optima scenario analysis tool.

This release (V0.1) is not intended for broad distribution, and should not be 
shared outside of co-optima.

The intention of this release is to add concreteness to the co-optimization concepts, 
and facilitate discussion between the teams as the tool evolves.

Questions, comments, and rants can be directed to Ray.Grout@nrel.gov

Availability:
-------------
Currently, the code is hosted on github in a private repository.

For access, follow the following steps:

- Create an account on github.com if you don't have one already

- Email Ray.Grout@nrel.gov and request access. It would be useful if you include a note about any potential usage you have in mind (so that I can save you time if it isn't currently possible, and so it can be added to the wishlist or we can discuss what would be necessary to support it.)

- Clone the git repository at `git@github.com:rgrout/cooptimizer.git`


Prerequisites
-------------

- python
- `Numpy <http://www.numpy.org>`_
- `Matplotlib <http://matplotlib.org>`_
- `PYOMO <http://www.pyomo.org>`_
- `IPOPT <https://projects.coin-or.org/Ipopt>`_
- deap / nsga2 (`Installation guide` <http://deap.readthedocs.io/en/master/installation.html>`_) 
- `xlrd <http://www.python-excel.org>`_

Input files
------------

- cooptimizer_input.py
- Properties database input file (example: propDB_fiction.xls)
- Cost input file (example: costDB_fiction.xls)

There are tools included in this distribution to read a file output from the *fuel properties database*; however, since at the moment the properties necessary for the merit function largely need to be filled in by the user, they are not used. Instead, it is left to the user to fill in components and properties in the template files "propDB.xls" and "costDB.xls". The names are arbitrary and are specified in the cooptimizer_input.py file.



Usage and capabilities:
-----------------------

This release is capable of 2 primary capabilities.

1. Firstly, sweeping out a Pareto front to study the cost-merit trade off; that is, finding the composition that maximizes
merit for a given cost (or, equivalently, finding the composition that minimizes cost for a given merit).

2. Sweeping across a range of values of "K" in the merit function and finding the composition that maximizes the merit function 
irrespective of cost (or for no data available)

The mode is selected by setting either::

task_list['cost_vs_merit_Pareto'] = True

or::

task_list['K_vs_merit_sweep'] = True

in the input file `cooptimizer_input.py`. Note that as of this release, only the former is implemented in the GA formulation.

The co-optimizer is run by::

python co_optimizer.py


Methodology
-----------
The co-optimizer uses two alternative algorithms to solve the merit function optimization problem, specified in the input file.  The 'pyomo' implementation 
currently uses IPOPT for solving the non-linear interior point optimization problem. However, this method sometimes lacks robustness. On a single failure, the 
co-optimizer will attempt some heuristics to obtain a successful solution; if this is not possible, it will display an error message. There is also a 'deap_NSGAII'
option, which will use the DEAP toolbox implementation of the NSGA2 (Non Sorting Genetic Algorithm II) to find the Pareto front. This method is more robust, but can take longer. 




