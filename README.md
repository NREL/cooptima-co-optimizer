# cooptimizer
Scenario analysis toolkit for Co-optima

Run the "optimizer.py" script; currently set up to sweep out a pareto front for cost vs. merit function by altering constraint value.

This is the verision that *does not* function properly, as the "optima" found for c<=c\* is sometimes less than the optima found for a lower value of c\*

## prereqs

At the moment, you need at least

* PYOMO
http://www.pyomo.org

* IPOPT
http://www.coin-or.org/projects/Ipopt.xml
