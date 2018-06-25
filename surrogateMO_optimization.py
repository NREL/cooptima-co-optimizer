import numpy as np
from pyDOE import *

from utility import *
from merit_functions import revised_mf as mmf_single
from blend_functions import blend_fancy_vec as blend




import matplotlib as mpl
#mpl.use('Agg')

mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as scp
import math
import copy
import time
import random
from utility import *
from sklearn.cluster import KMeans
from multiprocessing import Pool
#this is supposed to do the outer optimization
#optimize over all weights (N weights)
#weights live [0,1]
#weights add up to 1



def my_objective(individual, propvec, Kinp, data):

	f_out = np.zeros(data.nobj_exp+data.nobj_chp)
	f_chp = data.objfunction_chp(individual, propvec, Kinp)
	
	f_exp = []#RBF predictions
	for ii in range(data.nobj_exp):
		f = predict_rbf(individual, data, ii)
		f_exp.append(f[0,0])
		f_out[ii] = f
	for ii in range(data.nobj_chp):
		if data.nobj_chp>1:
			print('fix the assignmentof f_chp to f_out')
			kkk
		f_out[data.nobj_exp+ii] = f_chp#[ii]
	#print(f_exp, f_chp, f_out)
	#kkk
	return f_out#f_exp, f_chp


	
def predict_rbf(CandPoint, data, obj_id):
	CandPoint = np.asmatrix(CandPoint)
	CandPoint = CandPoint[0,0:data.dim]
	numpoints = CandPoint.shape[0] # determine number of candidate points
	# compute pairwise distances between candidates and already sampled points
	dist_val = np.transpose(scp.distance.cdist(CandPoint, data.S[0:data.m, :]))
	# compute radial basis function value for distances
	U_Y = phi(dist_val, data.phifunction)
	# determine the polynomial tail (depending on rbf model)
	if data.polynomial == 'none':
		PolyPart = np.zeros((numpoints, 1))
	elif data.polynomial == 'constant':
		PolyPart = data.ctail * np.ones((numpoints, 1))
	elif data.polynomial == 'linear':
		PolyPart = np.concatenate((np.ones((numpoints, 1)), CandPoint), axis = 1) * data.ctail[:,obj_id]
	else:
		raise myException('Error: Invalid polynomial tail.')
	RBFvalue = np.asmatrix(U_Y).T * np.asmatrix(data.llambda[:,obj_id]) + PolyPart

	return RBFvalue#, np.asmatrix(dist_val)

def clustering(data, CandPoint, Pfront):

	#delete all points that are too close to already sampled points
	dist_val = np.transpose(scp.distance.cdist(CandPoint[:,0:data.dim], data.S[0:data.m, :]))
	min_dist_val = np.asmatrix(np.amin(dist_val, axis = 0)).T
	del_id =np.where(min_dist_val<=data.tolerance)[0]
	if len(del_id)>0:
		CandPoint = np.delete(CandPoint,del_id, axis=0)
		Pfront = np.delete(Pfront, del_id, axis = 0)
	
	#use clustering of the remaining Pfront points (cluster on Pareto front)
	kmeans = KMeans(n_clusters=data.n_samp,random_state=0).fit(Pfront)#(elemeffects.reshape(-1, 1))
	#KMeans(n_clusters=2, random_state=0).fit(X)
	#print(kmeans.labels_, kmeans.cluster_centers_)


	plt.scatter(Pfront[:,0], Pfront[:,1], marker = 'o', c= 'r',s=40, label ='Pareto points')
	plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker = '^', c = 'b',s=40,label='kmeans centers')
	plt.xlabel('Merit')
	plt.ylabel('Cost')
	plt.legend(loc=0, fontsize=10)
	plt.show()
	plt.savefig('MO_surr.png')
	plt.close('all')

	group_cent= np.zeros((data.n_samp,data.dim+1))
	for ii in range(data.n_samp):
		group_id = np.where(kmeans.labels_==ii)[0]
		if len(group_id)>0:
			ingroup = CandPoint[group_id,:]
			#print(ingroup)
			group_cent[ii,:] = np.mean(ingroup, axis = 0) #group centroids in parameter space will add up to 1 because of mean values and sum over the matrix

	#print(group_cent, np.sum(group_cent,axis = 1))
	#check that group centroids are not too close to already evaluated point
	dist_val = np.transpose(scp.distance.cdist(group_cent[:,0:data.dim], data.S[0:data.m, :]))
	min_dist_val = np.asmatrix(np.amin(dist_val, axis = 0)).T
	del_id =np.where(min_dist_val<=data.tolerance)[0]
	if len(del_id)>0:
		group_cent = np.delete(group_cent,del_id, axis=0)
	

	return group_cent


def sample_selection(data, CandPoint):

	CandValue, dist_val = predict_rbf(CandPoint, data)
	MinCandValue = np.amin(CandValue)
	MaxCandValue = np.amax(CandValue)

	if MinCandValue == MaxCandValue:
		ScaledCandValue = np.ones((CandValue.shape[0], 1))
	else:
		ScaledCandValue = (CandValue - MinCandValue) / (MaxCandValue - MinCandValue)

	normval = {}
	if data.n_samp == 1:
		#valueweight = 0.95
		CandMinDist = np.asmatrix(np.amin(dist_val, axis = 0)).T
		MaxCandMinDist = np.amax(CandMinDist)
		MinCandMinDist = np.amin(CandMinDist)
		if MaxCandMinDist == MinCandMinDist:
			ScaledCandMinDist = np.ones((CandMinDist.shape[0], 1))
		else:
			ScaledCandMinDist = (MaxCandMinDist - CandMinDist) / (MaxCandMinDist - MinCandMinDist)

		# compute weighted score for all candidates
		CandTotalValue = data.rbf_weight * ScaledCandValue + (1 - data.rbf_weight) * ScaledCandMinDist

		# assign bad scores to candidate points that are too close to already sampled
		# points
		CandTotalValue[CandMinDist < data.tolerance] = np.inf

		MinCandTotalValue = np.amin(CandTotalValue)
		selindex = np.argmin(CandTotalValue)
		xselected = np.array(CandPoint[selindex, :])
		normval[0] = np.asmatrix((dist_val[:, selindex])).T
	else:
			raise myException('Error: Selection of several sample points not yet implemented.')
	return xselected, normval

def uniform(low, up, size=None):
	"""generate uniform random numbrs in [0,1] and scale so that sum = 1"""

	# Uniformly generate new points within parameter bounds
	X = [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
	sumX = sum(X[0:22])  # sum up over all parameters in each individual

	# Divide each individual's parameters by the sum to satisfy
	# sum x = 1 constraint;
	Y = [X[ii] / sumX for ii in range(len(X))]

	return Y#X

def scale_2():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                sumChild = sum(child[0:22])

                for i in xrange(22):#len(child)):
                    child[i] = child[i]/sumChild

            return offspring
        return wrapper
    return decorator

def surrogate_mo(data, KK, propvec, pDB):
	"""
	This creates ncands*dim points that are randomly perturbed from the best point.
	#todo: candidate points plus another dimension must add to 1
	(easy: we do know what the full weight vector of xbest is
	-- so we perturb all N values, make them sum 1 and delete last column)
	"""

	import array
	
	import cooptimizer_input
	from deap import algorithms
	from deap import base
	from deap import benchmarks
	from deap.benchmarks.tools import diversity, convergence
	from deap import creator
	from deap import tools
	NDIM = data.dim+1 # Number of parameters = 21 (surrogate model)
    
	# Minimize both objectives (min -f(x) if maximization is needed)
	creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))#everything must be minimized

	# Individuals in the generation
	creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	#parameter sequence: RON, S, HOV, SL, LFV150, PMI, CA50, IAT, KI

	BOUND_LOW, BOUND_UP = 0,1 #0 , 1#[0.0]*6 + [6.7] + [35.]+2., [1.0]*6 + [23.8] + [90.]+[10.5]  # Lower and upper variable bounds
	if not(cooptimizer_input.parallel_nsgaruns):
		pool =Pool()
		toolbox.register("map",pool.map)
	#toolbox.register("map",futures.map)
	toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
	toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.attr_float)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	#toolbox.register("evaluate",eval_MMF_gp_opt, propvec=propvec, Kinp=KK, GP = GP, scal = scal)
	#toolbox.register("evaluate", eval_MMF_gp, propvec=propvec, Kinp=KK, GP = GP, scal = scal)#, propvec=propvec, Kinp=KK)
	#toolbox.register("evaluate", eval_gp, GP = GP, scal = scal)#, propvec=propvec, Kinp=KK)
	#toolbox.register("evaluate", eval_mo, propvec=propvec, Kinp=KK)#, ref_in = ref, sen_in = sen )
	#toolbox.register("evaluate", eval_mo2, propvec=propvec, Kinp=KK)#, ref_in = ref, sen_in = sen )
	#toolbox.register("evaluate", eval_mean_var, propDB=pDB, Kinp=KK)

	toolbox.register("evaluate", my_objective, propvec=propvec, Kinp=KK, data = data)#, ref_in = ref, sen_in = sen )

	toolbox.register("mate", tools.cxSimulatedBinaryBounded,low=BOUND_LOW, up=BOUND_UP, eta=20.0)
	toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW,up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)

	toolbox.decorate("mate", scale_2())
	toolbox.decorate("mutate", scale_2())

	toolbox.register("select", tools.selNSGA2)

	#  These are parameters that can be adjusted and may change
	#  the algorithm's performance
    
	NGEN=150
	MU=40
	CXPB = 0.75  # Cross-over probability, [0,1]

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	pf = tools.ParetoFront()
	hof = tools.HallOfFame(100)
	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	pop = toolbox.population(n=MU)
    
	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit
	pf.update(pop)
	hof.update(pop)
	# This is just to assign the crowding distance to the individuals
	# no actual selection is done
	#print(pop)

	pop = toolbox.select(pop, len(pop))

	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(invalid_ind), **record)
	#print(logbook.stream)

	# Begin the generational process
	for gen in range(1, NGEN):
		# Vary the population
		offspring = tools.selNSGA2(pop, len(pop))
		offspring = tools.selTournamentDCD(pop, len(pop))
		offspring = [toolbox.clone(ind) for ind in offspring]

		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			if random.random() <= CXPB:
				toolbox.mate(ind1, ind2)
			toolbox.mutate(ind1)
			toolbox.mutate(ind2)
			del ind1.fitness.values, ind2.fitness.values

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		try:
			pf.update(offspring)
		except:
			print([ind.fitness.values for ind in offspring])
			lll

		hof.update(offspring)
		# Select the next generation population
		pop = toolbox.select(pop + offspring, MU)
		record = stats.compile(pop)
		logbook.record(gen=gen, evals=len(invalid_ind), **record)
		print(logbook.stream)

	pop.sort(key=lambda x: x.fitness.values)

	front = np.array([ind.fitness.values for ind in pf])
	print("NSGA done; hof: {}".format(pf[0]))
	#print("K = {}; Score: {}".format(KK, -eval_mo(pf[0],propvec,KK)[0]))
	#print("pv RON = {}".format(propvec['NAME']))
	print('paretofront:',front)
	#plt.scatter(front[:,0], front[:,1], marker = 'o', c= 'r',s=40)
	#plt.show()
	#plt.savefig('MO_surr.png')
	#kkk

	pf_pop = np.array([ind for ind in pf])
	print(pf_pop)
	print(np.sum(pf_pop,axis = 1))
	print(pf_pop.shape)
	print(front.shape)

	#for ii in range(pf_pop.shape[0]):
	#	o = objective_function(pf_pop[ii,:], propvec, KK)
	#	print(o)
	#	print(front[ii,:])
	if not(cooptimizer_input.parallel_nsgaruns):
		pool.close()
	return pf_pop, front

def phi(r, phitype):

    if phitype == 'linear':
        output = r
    elif phitype == 'cubic':
        output = np.power(r, 3)

    else:
        raise myException('Error: Unkonwn type.')

    return output


def initialize_matrices(data, PairwiseDistance):

    PHI = np.zeros((data.maxeval, data.maxeval))
    if data.phifunction == 'linear':
        PairwiseDistance = PairwiseDistance
    elif data.phifunction == 'cubic':
        PairwiseDistance = PairwiseDistance ** 3


    PHI[0:data.m, 0:data.m] = PairwiseDistance
    phi0 = phi(0, data.phifunction) # phi-value where distance of 2 points =0 (diagonal entries)

    if data.polynomial == 'None':
        pdim = 0
        P = np.array([])
    elif data.polynomial == 'constant':
        pdim = 1
        P = np.ones((data.maxeval, 1)), data.S
    elif data.polynomial == 'linear':
        pdim = data.dim + 1
        P = np.concatenate((np.ones((data.maxeval, 1)), data.S), axis = 1)
    else:
        raise myException('Error: Invalid polynomial tail.')
    return np.asmatrix(PHI), np.asmatrix(phi0), np.asmatrix(P), pdim





#def constraint_function(w): #currently not used
#	return w/np.sum(w)


def objective_function_exp(x, propvec, Kinp):
	if len(x)==21: #needs to be generalized later -- add a component to make the sum = 1
		w = np.zeros(len(x)+1)
		w[0:len(x)] = x
		w[-1] = 1-np.sum(x)
	else:
		w = x
	this_ron = blend(w, propvec, 'RON')
	this_s = blend(w, propvec, 'S')
	this_HoV = blend(w, propvec, 'HoV')
	this_SL = blend(w, propvec, 'SL')
	this_AFR = blend(w, propvec, 'AFR_STOICH')
	this_LFV150 = blend(w, propvec, 'LFV150')
	this_PMI = blend(w, propvec, 'PMI')
	cost_f = blend(w, propvec, 'COST')
	f0 = mmf_single(RON=this_ron, S=this_s,HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)
	#f1 = cost_f
	return -f0#, f1 #minimize negative merit

def objective_function_chp(x, propvec, Kinp):
	if len(x)==21: #needs to be generalized later -- add a component to make the sum = 1
		w = np.zeros(len(x)+1)
		w[0:len(x)] = x
		w[-1] = 1-np.sum(x)
	else:
		w = x
	this_ron = blend(w, propvec, 'RON')
	this_s = blend(w, propvec, 'S')
	this_HoV = blend(w, propvec, 'HoV')
	this_SL = blend(w, propvec, 'SL')
	this_AFR = blend(w, propvec, 'AFR_STOICH')
	this_LFV150 = blend(w, propvec, 'LFV150')
	this_PMI = blend(w, propvec, 'PMI')
	cost_f = blend(w, propvec, 'COST')
	#f0 = mmf_single(RON=this_ron, S=this_s,HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)
	f1 = cost_f
	return f1 #minimize negative merit




def surrogateMO_optimization(KK, propvec,  pDB):
	data = Data()

	data.dim = len(propvec['COST'])-1 # for equality constrained case, need dimension d-1 because of sum=1 constraint

	#caution when there are other objectives that have different (additional) parameters
	data.maxeval = 200 #number of local optimizations
	n0 = 2*(data.dim+1+1)
	data.xlow=0
	data.xup = 1
	data.objfunction_chp = objective_function_chp
	data.objfunction_exp = objective_function_exp
	data.nobj_chp=1
	data.nobj_exp=1
	#data.constraint = constraint_function #may integrate this in generation of candidate points
	
	data.phifunction = 'cubic'
	data.polynomial = 'linear'
	data.n_samp = 4 # number of samples to generate per iteration -- for now 1, serial
	data.Ncand = 500*data.dim #some big number
	data.pertP = 1 #probability to perturb each variable of xbest
	# algorithm parameters
	data.tolerance = 0.001 * np.linalg.norm(np.ones((1, data.dim)))
	sigma_stdev_default = 0.2
	data.sigma_stdev = sigma_stdev_default # current mutation rate 
	maxshrinkparam = 5 # maximal number of shrikage of standard deviation for normal distribution when generating the candidate points
	failtolerance = max(5,data.dim)
	succtolerance =3

	# initializations
	iterctr = 0 # number of iterations
	shrinkctr = 0 # number of times sigma_stdev was shrunk
	failctr = 0 # number of consecutive unsuccessful iterations
	localminflag = 0  # indicates whether or not xbest is at a local minimum
	succctr=0 # num

	data.rbf_weight = 1.0
	#recast the problem as one with N-1 parameters and optimize in (N-1)-dimensional space
	
	startpoints = lhs(data.dim+1, samples = n0, criterion="maximin")  #initial design #dimension+1
	#sum=1 constraint
	p_init=np.divide(startpoints,np.asmatrix(np.sum(startpoints, axis = 1)).T) 
	rank_s = np.linalg.matrix_rank(np.concatenate((p_init[:,0:data.dim], np.ones((n0,1))), axis = 1))
	while rank_s < data.dim+1: #regenerate initial design until rank condition satisfied
		startpoints = lhs(data.dim+1, samples = n0, criterion="maximin")  #initial design #dimension+1
		#sum=1 constraint
		p_init=np.divide(startpoints,np.asmatrix(np.sum(startpoints, axis = 1)).T) 
		rank_s = np.linalg.matrix_rank(np.concatenate((p_init[:,0:data.dim], np.ones((n0,1))), axis = 1))

	
	#evaluate "expensive function" and identify current Pareto points in data.Y
	f0f1 = np.zeros((n0,data.nobj_exp+data.nobj_chp))
	P=[]
	Px=[]
	for ii in range(n0):	
		if data.nobj_chp>0: #at least 1 cheap objective
			f0f1[ii,0:data.nobj_exp] = data.objfunction_exp(np.ravel(p_init[ii,0:data.dim]), propvec, KK)
			f0f1[ii,data.nobj_exp:data.nobj_exp+data.nobj_chp] = data.objfunction_chp(np.ravel(p_init[ii,0:data.dim]), propvec, KK)
		else: #all objectives expensive
			f0f1[ii,:] = data.objfunction_exp(np.ravel(p_init[ii,0:data.dim]), propvec, KK)

		do_not_include = False
		if len(P) == 0:
			P = np.asmatrix(f0f1[ii,:])
			Px = np.asmatrix(p_init[ii,0:data.dim])
		else:
			del_index=[] #delete dominated points from P
			for jj in range(P.shape[0]):
				if (f0f1[ii,:]<P[jj,:]).all(): #new point dominates what's in P
					del_index.append(jj) 
				elif (f0f1[ii,:]>P[jj,:]).all(): #new point dominates what's in P
					do_not_include = True
			if len(del_index)>0:
				P = np.delete(P, del_index,axis = 0)
				Px = np.delete(Px, del_index, axis = 0)
			if not(do_not_include):
				P =np.concatenate((P,np.asmatrix(f0f1[ii,:])), axis = 0)
				Px =np.concatenate((Px,np.asmatrix(p_init[ii,0:data.dim])), axis = 0)

	plt.scatter(np.ravel(P[:,0]), np.ravel(P[:,1]), marker = 'o', c= 'r',s=40, label ='Pareto set')
	plt.scatter(np.ravel(f0f1[:,0]), np.ravel(f0f1[:,1]), marker = '^', c = 'b',s=40,label='all points')
	plt.xlabel('Merit')
	plt.ylabel('Cost')
	plt.legend(loc=0, fontsize=10)
	plt.show()
	plt.savefig('checkP.png')
	plt.close('all')

	
	#fit surrogate model
	data.s = p_init[:,0:data.dim] #this is the sample site matrix we fit the RBF to (dimension -1)
	data.y = f0f1 #initial vector of function values
	data.m = n0
	data.S = np.concatenate((data.s, np.zeros((data.maxeval - data.m, data.dim))), axis = 0)
	data.Y = np.concatenate((data.y, np.zeros((data.maxeval - data.m, data.nobj_chp+data.nobj_exp))), axis = 0)
	data.Pfront = P
	data.Ppoints = Px


	#fit surrogate surface 
	PairwiseDistance = scp.distance.cdist(data.S[0:data.m, :], data.S[0:data.m, :], 'euclidean')
	# initial RBF matrices
	PHI, phi0, P, pdim = initialize_matrices(data, PairwiseDistance)

	#optimization loop
	while data.m < data.maxeval and localminflag == 0:
		iterctr = iterctr + 1 # increment iteration counter
		print ('\n Iteration: {} \n'.format(iterctr))
		print ('\n fEvals: {} \n'.format(data.m))
		#print '\n Best value in this restart: %2.4f \n' %data.Fbest
	
		# number of new samples in an iteration
		NumberNewSamples = min(data.n_samp,data.maxeval - data.m)

		# replace large function values by the median of all available function values--skip for now
		Ftransform=np.zeros((data.m,data.nobj_exp))
		for ii in range(data.nobj_exp): #first columns are expensive, rest is cheap
			Ftransform[:,ii] = np.copy(np.asarray(data.Y[:,ii])[0:data.m])
		#medianF = np.median(np.asarray(data.Y)[0:data.m])
		#Ftransform[Ftransform > medianF] = medianF

		# fit the response surface
		# Compute RBF parameters
		a_part1 = np.concatenate((PHI[0:data.m, 0:data.m], P[0:data.m, :]), axis = 1)
		a_part2 = np.concatenate((np.transpose(P[0:data.m, :]), np.zeros((pdim, pdim))), axis = 1)
		a = np.concatenate((a_part1, a_part2), axis = 0)

		eta = math.sqrt((1e-16) * np.linalg.norm(a, 1) * np.linalg.norm(a, np.inf))
		
		C=np.zeros((data.m+pdim,data.nobj_exp))
		#print(C.shape)
		for ii in range(data.nobj_exp):
			coeff = np.linalg.solve((a + eta * np.eye(data.m + pdim)),\
				np.concatenate((np.asmatrix(Ftransform[:,ii]).T, np.zeros((pdim, 1))), axis = 0))
			#print (coeff.shape)
			C[:,ii] = np.ravel(coeff)
		# rbf parameters
		data.llambda = coeff[0:data.m, :]
		data.ctail = coeff[data.m: data.m + pdim,:]
		
		#-------------------------------------------------------------------------------------
		# select the next function evaluation point:
		# use NSGA 2 to solve the multi-objective surrogate problem
		CandPoint, Pfront =surrogate_mo(data, KK, propvec, pDB) #retruns matrix with Pareto points and Pareto function values
		xselected=clustering(data, CandPoint, Pfront)
		
		# more than one new point, do parallel evaluation
		# instead of parfor in MATLAB, multiprocessing pool is used here
		Fnew = np.zeros((xselected.shape[0],data.nobj_chp+data.nobj_exp)) #just for scatterplot
		for ii in range(xselected.shape[0]):	
			if data.nobj_chp>0: #at least 1 cheap objective
				Fselected = np.zeros((1,data.nobj_chp+data.nobj_exp))
				Fselected[0,0:data.nobj_exp] = data.objfunction_exp(np.ravel(xselected[ii,:]), propvec, KK)
				Fselected[0,data.nobj_exp:data.nobj_exp+data.nobj_chp] = data.objfunction_chp(np.ravel(xselected[ii,:]), propvec, KK)
			else: #all objectives expensive
				Fselected = np.asmatrix(data.objfunction_exp(np.ravel(xselected[ii,:]), propvec, KK))

			#print(Fselected)
			Fnew[ii,:] =Fselected
			#check if new point dominates any previous points
			do_not_include = False
			del_index=[] #delete dominated points from P
			for jj in range(data.Pfront.shape[0]):
				if (Fselected<data.Pfront[jj,:]).all(): #new point dominates what's in P
					del_index.append(jj) 
				elif (Fselected>data.Pfront[jj,:]).all(): #new point dominates what's in P
					do_not_include = True
					break
			if len(del_index)>0:
				data.Pfront = np.delete(data.Pfront, del_index,axis = 0)
				data.Ppoints = np.delete(data.Ppoints, del_index, axis = 0)
			if not(do_not_include):
				data.Pfront =np.concatenate((data.Pfront,np.asmatrix(Fselected)), axis = 0)
				data.Ppoints =np.concatenate((data.Ppoints,np.asmatrix(xselected[ii,0:data.dim])), axis = 0)

			data.S[data.m, :] = xselected[ii,0:data.dim]
			data.Y[data.m, :] = Fselected
			data.m = data.m + 1
			#Fselected = np.zeros((xselected.shape[0], 1))
			#Time = np.zeros((xselected.shape[0], 1))

			#pool = Pool()
			#pool_res = pool.map_async(wrapper_func, ((i, data.objfunction) for i in xselected.tolist()))
			#pool.close()
			#pool.join()
			#result = pool_res.get()
			#for ii in range(len(result)):
			#	Fselected[ii, 0] = result[ii][0]
			#	Time[ii, 0] = result[ii][1]

			#data.fevaltime[data.m:data.m+xselected.shape[0], 0] = Time
			#data.S[data.m:data.m+xselected.shape[0], :] = xselected
			#data.Y[data.m:data.m+xselected.shape[0], 0] = Fselected
			#data.m = data.m + xselected.shape[0]
			#raise myException('Error: Not yet implemented.')
		#else:
		#	#time1 = time.time()
		#	Fselected = data.objfunction(np.ravel(xselected), propvec, KK )
		#	#data.fevaltime[data.m, 0] = time.time() - time1
		#	data.S[data.m, :] = xselected
		#	data.Y[data.m, 0] = Fselected
		#	data.m = data.m + 1
		#	print(Fselected, data.Fbest)

		
		#plt.scatter(np.ravel(-data.Y[0:data.m,0]), np.ravel(data.Y[0:data.m,1]), marker =  '^', color = 'b',s=70,label='All already evaluated points')
		plt.scatter(np.ravel(-data.Pfront[:,0]), np.ravel(data.Pfront[:,1]), marker = 'o', color='r',s=70, label ='Pareto front')
		#plt.scatter(np.ravel(-Fnew[:,0]), np.ravel(Fnew[:,1]), marker ='*', color = 'y', s = 70, label = 'Points selected in this iteration')
		plt.xlabel('Maximize merit')
		plt.ylabel('Minimize cost')
		plt.legend(loc=0, fontsize=10)
		#plt.show()
		plt.savefig('Pfront'+str(iterctr)+'.png')
		plt.close('all')
	
		
		'''
		# determine best one of newly sampled points
		minSelected = np.amin(Fselected)
		IDminSelected = np.argmin(Fselected)
		xMinSelected = xselected[IDminSelected, :]
		if minSelected < data.Fbest:
			if data.Fbest - minSelected > (1e-3)*math.fabs(data.Fbest):
				# "significant" improvement
				failctr = 0
				succctr = succctr + 1
			else:
				failctr = failctr + 1
				succctr = 0
			data.xbest = xMinSelected
			data.Fbest = minSelected
		else:
			failctr = failctr + 1
			succctr = 0

		# check if algorithm is in a local minimum
		shrinkflag = 1
		if failctr >= failtolerance:
			if shrinkctr >= maxshrinkparam:
				shrinkflag = 0
				print 'Stopped reducing sigma because the maximum reduction has been reached.'
			failctr = 0

			if shrinkflag == 1:
				shrinkctr = shrinkctr + 1
				sigma_stdev = data.sigma_stdev / 2
				print 'Reducing sigma by a half!'
			else:
				localminflag = 1
				print 'Algorithm is probably in a local minimum! Restarting the algorithm from scratch.' 

		if succctr >= succtolerance:
			data.sigma_stdev = min(2 * data.sigma_stdev, sigma_stdev_default)
			succctr = 0
		'''

		# update PHI matrix only if planning to do another iteration
		if data.m < data.maxeval and localminflag == 0:
			'''
			n_old = data.m - xselected.shape[0]
			
			for kk in range(xselected.shape[0]):
				dist_val = np.transpose(scp.distance.cdist(xselected[kk, :], data.S[0:data.m, :]))

				new_phi = phi(dist_val[kk], data.phifunction)
				PHI[n_old + kk, 0: n_old + kk] = new_phi
				PHI[0:n_old+kk, n_old+kk] = np.asmatrix(new_phi).T
				PHI[n_old+kk, n_old+kk] = phi0
				P[n_old+kk, 1:data.dim+1] = xselected[kk, :]
			'''
			#fit surrogate surface 
			PairwiseDistance = scp.distance.cdist(data.S[0:data.m, :], data.S[0:data.m, :], 'euclidean')
			# initial RBF matrices
			PHI, phi0, P, pdim = initialize_matrices(data, PairwiseDistance)	
	

	data.S = data.S[0:data.m, :]
	data.Y = data.Y[0:data.m, :]
	#data.fevaltime = data.fevaltime[0:data.m, :]
	data.NumberFevals = data.m
	print(-data.Fbest)
	print(np.sum(data.xbest), len(data.xbest))
	return data
