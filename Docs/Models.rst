Models
======

Merit Functions
---------------

Spark ignition merit functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

#. Miles merit function (MMF): this merit function reflects the potential efficiency improvements that could be realized in a spark-ignition engine by altering fuel properties. Several variations of this are defined in the merit_functions.py file; the most recent (revised_mf) is:

	.. math::
	
		M = \frac{RON-91}{1.6} - K\frac{S-8}{1.6} 
		+ 0.085\left[ \frac{HoV}{AFR+1} - \frac{415}{15}\right]\frac{1}{1.6}
		+ \left[ \frac{HoV}{AFR+1} - \frac{415}{15}\right] \frac{1}{15.2} \\
		+ \frac{S_L - 46}{5.4} + 0.008\left( Tc_{90, conv} - Tc_{90,mix} 	\right) - \left[0.7 + 0.5(PMI-1.4)\right]\delta(PMI-1.6)
	
	In this implementation, we assume that we have 22 fuel components, 	whose properties and costs are assumed to be given (synthetic data for 	experiments). The fuel components with the blending models discussed 	below to obtain values for RON, S, HOV, etc., which enter the merit function. The costs are derived by linear blending. 

#. Net mean effective pressure (NMEP): this merit function is based on data provided by the M. Ratcliff withing the engines group at NREL. We used their data to train a Gaussian process (GP) model (GPmerit.py). We use the Gaussian process model as “truth model” objective function in the optimization. The GP gives for an unsampled parameter vector an estimate of the mean response and its variance. Thus, we can maximize the expected (mean) NMEP while minimizing the associated predicted variance. 

Uncertainty
~~~~~~~~~~~

Optimization under uncertainty is generally done by assuming a distribution on either the coefficients in the merit function, or assuming a distribution on the costs of the fuel components. For the fuel component uncertainty, we run several multi-objective optimizations from a realization of the random cost variables and we obtain several Pareto fronts. We can look at the mean and median Pareto fronts to obtain an idea of the spread and sensitivity of the tradeoff solutions. When considering uncertainty in the coefficients of the merit function, for a given parameter vector, we draw a large number N of random values form the distribution, compute N merit function values, and then maximize the mean of these N values while minimizing their variance.



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


