Co-optimization of fuels and engines Co-optimizer
=================================================

This collection of python scripts is the co-optimizer,
or the co-optima scenario analysis tool.

This release (V0.2) is intended primarily to analysis such as done to support co-optima project planning and identification of potentially fruitful research directions rather than specify particular fuel blends. 

The intention of this release is to add concreteness to the co-optimization concepts, 
and facilitate discussion between the teams and stakeholders as the tool evolves.

The user interface is currently non-existant from a productions standpoint.

Questions, comments, and rants can be directed to Ray.Grout@nrel.gov

Availability:
-------------
The code is hosted on github in a private repository.

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
- deap / nsga2 (`Installation guide <http://deap.readthedocs.io/en/master/installation.html>`_) 
- `xlrd and xlwt <http://www.python-excel.org>`_


Setting up to run on Peregrine
------------------------------

The co-optimizer is set up and installed on Peregrine 
in /projects/optima/applications/co-optimizer. To set up your environment to run 
the co-optimizer on Peregine, a few modules are necessary::

module load python
module load mkl

Then you can use pip to install the prerequisite python tools in your 
home directory::

pip install --user pyomo
pip install --user xlrd
pip install --user xlwt
pip install --user deap

Finally, to use the pre-build ipopt executable you need to add to your path::

export PATH=$PATH:/projects/optima/applications/co-optimizer/utils/ipopt/bin


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

python co_optimizer_par.py


Methodology
-----------
The co-optimizer uses two alternative algorithms to solve the merit function optimization problem, specified in the input file.  The 'pyomo' implementation 
currently uses IPOPT for solving the non-linear interior point optimization problem. However, this method sometimes lacks robustness. On a single failure, the 
co-optimizer will attempt some heuristics to obtain a successful solution; if this is not possible, it will display an error message. There is also a 'deap_NSGAII'
option, which will use the DEAP toolbox implementation of the NSGA2 (Non Sorting Genetic Algorithm II) to find the Pareto front. This method is more robust, but can take longer. 

Acknowledgement
---------------
The co-optimizer was developed as part of the Co-Optimization of Fuels & Engines (Co-Optima) project sponsored by the U.S. Department of Energy (DOE) Office of Energy Efficiency and Renewable Energy (EERE), Bioenergy Technologies and Vehicle Technologies Offices. (Optional): Co-Optima is a collaborative project of multiple national laboratories initiated to simultaneously accelerate the introduction of affordable, scalable, and sustainable biofuels and high-efficiency, low-emission vehicle engines.


