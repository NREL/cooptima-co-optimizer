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
- numpy
- matplotlib
- pyomo 
- ipopt
- deap / nsga2
- xlrd

Input files
------------

- cooptimizer_input.py
- Properties database input file (example: propDB_fiction.xls)
- Cost input file (example: costDB_fiction.xls)

Usage and capabilities:
-----------------------

This release is capable of 3 primary capabilities.




