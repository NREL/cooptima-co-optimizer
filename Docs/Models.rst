Models
======

Merit Functions
---------------

Spark ignition merit functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

#. Miles merit function (MMF): this merit function is defined in the report [insert name] and measures the efficiency one can obtain from a given fuel blend. In this implementation, we assume that we have 22 fuel components, whose properties and costs are assumed to be given (synthetic data for experiments). The fuel components are mixed with a linear blending model to obtain values for RON, S, HOV, etc., which enter the MMF. The costs are derived in the same linear fashion. See the files merit_functions.py (revised_mf) for computing MMF and blend_functions.py (blend_fancy_vec) for computation of RON, S, HOV, COST.
#. Net mean effective pressure (NMEP): this merit function is based on data from Ratcliff [insert paper]. We used their data to train a Gaussian process (GP) model (GPmerit.py). We use the Gaussian process model as “truth model” objective function in the optimization. The GP gives for an unsampled parameter vector an estimate of the mean response and its variance. Thus, we can maximize the expected (mean) NMEP while minimizing the associated predicted variance as done in 1d above. 
#. Optimization under uncertainty: this is generally done by assuming a distribution on either the coefficients in the MMF (2 above), or assuming a distribution on the costs of the fuel components (3 and 4 above). For the fuel component uncertainty, we run several multi-objective optimizations from a realization of the random cost variables and we obtain several Pareto fronts. We can look at the mean and median Pareto fronts to obtain an idea of the spread and sensitivity of the tradeoff solutions. When considering uncertainty in the coefficients of the MMF, for a given parameter vector, we draw a large number N of random values form the distribution, compute N MMF values, and then maximize the mean of these N values while minimizing their variance (2 above).
#. Singe objective optimization (see 5 above): there is an option for computationally cheap and computationally expensive single objective optimization. The current implementation allows the user to select a range of K values that go into the Miles function (MMF). Given K, the MMF is maximized with a local optimizer and a single best solution is returned. For expensive-to-evaluate objective functions, there is an option that uses surrogate model approximations during the optimization in order to minimize the number of expensive evaluations that is required. In the current implementation, we use the MMF as placeholder for an expensive evaluation. 


Blend Models
------------

Computing the properties of a blend of compoents remains an open problem. Several models are implmented in blend_functions.py; the most comprehensive model available is the in the blend_fancy_vec function which blends the properties as:

+-----------+-------------------+
|Property   | Model             |
+===========+===================+
|RON        | Non-linear        |
+-----------+-------------------+     
|S          | Non-linear        |
+-----------+-------------------+     
|HoV        | Linear by mass    |
+-----------+-------------------+         
|Stoic. AFR | Linear by mass    |
+-----------+-------------------+         
|LFV150     | Linear by volume  |
+-----------+-------------------+           
|PMI        | Linear by mass    |
+-----------+-------------------+         
|COST       | Linear by volume  |      
+-----------+-------------------+     

The routines used to compute these are flexible to allow linear by an arbitrary basis, for example if laminar flame speed was included it could be blended linearly on an energy basis. 

The two non-linear options are utilize tabulated data for blending the component into a surrogate blendstock assuming that the apparent property effect is independent of the composition of that blendstock. For example, if a given component blends into a surrogate blendstock such that it has an apparent ('blending RON') of 105 at 10\% the model assumes that it will blend into an arbitrary combination of components and blends with RON 105 at the 10\% level. 

.. warning:: The non-linear blend data provided is for single components into a four-component surrogate up to 30\% levels, but the blend model will use this data to extrapolate to higher levels with possibly misleading results. The estimates must be confirmed before relying on them.


