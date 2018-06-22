import numpy as np
from pyDOE import *

from utility import *
from merit_functions import revised_mf as mmf_single
from blend_functions import blend_fancy_vec as blend


import numpy as np
from pyDOE import *
import matplotlib as mpl
#mpl.use('Agg')
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as scp
import math
import copy
import time

from utility import *


#this is supposed to do the outer optimization
#optimize over all weights (N weights)
#weights live [0,1]
#weights add up to 1


def predict_rbf(CandPoint, data):
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
		PolyPart = np.concatenate((np.ones((numpoints, 1)), CandPoint), axis = 1) * data.ctail
	else:
		raise myException('Error: Invalid polynomial tail.')
	RBFvalue = np.asmatrix(U_Y).T * np.asmatrix(data.llambda) + PolyPart

	return RBFvalue, np.asmatrix(dist_val)



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


def create_cands(data):
    """
    This creates ncands*dim points that are randomly perturbed from the best point.
    #todo: candidate points plus another dimension must add to 1
    (easy: we do know what the full weight vector of xbest is
    -- so we perturb all N values, make them sum 1 and delete last column)
    """
    best = np.zeros(data.dim+1)
    best[0:data.dim] = data.xbest
    best[data.dim]=1-np.sum(data.xbest)

    # Ncand times the best value
    cp_e = np.kron(np.ones((data.Ncand,1)), np.asmatrix(best))
    # This generates random perturbations
    r=np.random.rand(data.Ncand,data.dim+1) #need dim+1 to account for the "missing" value
    a = r<data.pertP
    idx= np.where(np.sum(a,axis=1)==0)
    for ii in range(len(idx[0])):
        f = np.random.permutation(data.dim+1)
        a[idx[0][ii],f[0]] = True
    randnums = np.random.randn(data.Ncand, data.dim+1)
    randnums[a==False]=0
    pv = randnums*data.sigma_stdev
    # Create new points by adding random fluctucations to best point
    new_pts = cp_e+pv

    # Iterative, column wise procedure to force the randomly sampled point to be in [0,1]
    for ii in range(data.dim+1):
        vec_ii = new_pts[:,ii]
        adj_l = np.where(vec_ii < data.xlow)
        vec_ii[adj_l[0]] = data.xlow + (data.xlow - vec_ii[adj_l[0]])
        adj_u = np.where(vec_ii > data.xup)
        vec_ii[adj_u[0]] = data.xup - (vec_ii[adj_u[0]]-data.xup)
        stillout_u = np.where(vec_ii > data.xup)
        vec_ii[stillout_u[0]] = data.xlow
        stillout_l = np.where(vec_ii < data.xlow)
        vec_ii[stillout_l[0]] = data.xup
        new_pts[:,ii] = copy.copy(vec_ii)

    new_pts = new_pts/np.sum(new_pts, axis =1)

    cp_e = copy.copy(new_pts)
    rand_pts = np.asmatrix(np.random.uniform(0,1, [data.Ncand, data.dim+1]))
    cp_r = rand_pts/np.sum(rand_pts, axis = 1)

    CandPoint= np.concatenate((cp_e, cp_r), axis =0)
    CandPoint_out = CandPoint[:,0:data.dim] #return only data.dim candidate points

    return CandPoint_out

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


def objective_function(x, propvec, Kinp):
	
	w = np.zeros(len(x)+1)
	w[0:len(x)] = x
	w[-1] = 1-np.sum(x)
	this_ron = blend(w, propvec, 'RON')
	this_s = blend(w, propvec, 'S')
	this_HoV = blend(w, propvec, 'HoV')
	this_SL = blend(w, propvec, 'SL')
	this_AFR = blend(w, propvec, 'AFR_STOICH')
	this_LFV150 = blend(w, propvec, 'LFV150')
	this_PMI = blend(w, propvec, 'PMI')
	cost_f = blend(w, propvec, 'COST')
	f0 = -mmf_single(RON=this_ron, S=this_s,HoV=this_HoV, AFR=this_AFR, PMI=this_PMI, K=Kinp)

	return f0



def surrogate_optimization(KK, propvec,  pDB):
	data = Data()

	data.dim = len(propvec['COST'])-1 # for equality constrained case, need dimension d-1 because of sum=1 constraint

	#caution when there are other objectives that have different (additional) parameters
	data.maxeval = 300 #number of local optimizations
	n0 = 2*(data.dim+1+1)
	data.xlow=0
	data.xup = 1
	data.objfunction = objective_function
	#data.constraint = constraint_function #may integrate this in generation of candidate points
	
	data.phifunction = 'cubic'
	data.polynomial = 'linear'
	data.n_samp = 1 # number of samples to generate per iteration -- for now 1, serial
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

	
	#evaluate "expensive function"
	f0 = np.zeros((n0,1))
	for ii in range(n0):
		f0[ii,0] = data.objfunction(np.ravel(p_init[ii,0:data.dim]), propvec, KK)

	#fit surrogate model
	data.s = p_init[:,0:data.dim] #this is the sample site matrix we fit the RBF to (dimension -1)
	data.y = f0 #initial vector of function values

	data.Fbest = np.amin(data.y)
	data.xbest = np.ravel(data.s[np.argmin(data.y),:])
	data.m = n0
	data.S = np.concatenate((data.s, np.zeros((data.maxeval - data.m, data.dim))), axis = 0)
	data.Y = np.concatenate((data.y, np.zeros((data.maxeval - data.m,        1))), axis = 0)

    
	#fit surrogate surface 
	PairwiseDistance = scp.distance.cdist(data.S[0:data.m, :], data.S[0:data.m, :], 'euclidean')
	# initial RBF matrices
	PHI, phi0, P, pdim = initialize_matrices(data, PairwiseDistance)

	#optimization loop
	while data.m < data.maxeval and localminflag == 0:
		iterctr = iterctr + 1 # increment iteration counter
		print '\n Iteration: %d \n' % iterctr
		print '\n fEvals: %d \n' % data.m
		print '\n Best value in this restart: %2.4f \n' %data.Fbest
	
		# number of new samples in an iteration
		NumberNewSamples = min(data.n_samp,data.maxeval - data.m)

		# replace large function values by the median of all available function values--skip for now
		Ftransform = np.copy(np.asarray(data.Y)[0:data.m])
		#medianF = np.median(np.asarray(data.Y)[0:data.m])
		#Ftransform[Ftransform > medianF] = medianF

		# fit the response surface
		# Compute RBF parameters
		a_part1 = np.concatenate((PHI[0:data.m, 0:data.m], P[0:data.m, :]), axis = 1)
		a_part2 = np.concatenate((np.transpose(P[0:data.m, :]), np.zeros((pdim, pdim))), axis = 1)
		a = np.concatenate((a_part1, a_part2), axis = 0)

		eta = math.sqrt((1e-16) * np.linalg.norm(a, 1) * np.linalg.norm(a, np.inf))
		
		coeff = np.linalg.solve((a + eta * np.eye(data.m + pdim)),\
			np.concatenate((Ftransform, np.zeros((pdim, 1))), axis = 0))

		# rbf parameters
		data.llambda = coeff[0:data.m]
		data.ctail = coeff[data.m: data.m + pdim]
		#-------------------------------------------------------------------------------------
		# select the next function evaluation point:
		# introduce candidate points  
		CandPoint =create_cands(data)
		xselected, dist_val=sample_selection(data, CandPoint)
		
		# more than one new point, do parallel evaluation
		# instead of parfor in MATLAB, multiprocessing pool is used here
		if xselected.shape[0] > 1:
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
			raise myException('Error: Not yet implemented.')
		else:
			#time1 = time.time()
			Fselected = data.objfunction(np.ravel(xselected), propvec, KK )
			#data.fevaltime[data.m, 0] = time.time() - time1
			data.S[data.m, :] = xselected
			data.Y[data.m, 0] = Fselected
			data.m = data.m + 1
			print(Fselected, data.Fbest)

	
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
		
		# update PHI matrix only if planning to do another iteration
		if data.m < data.maxeval and localminflag == 0:
			n_old = data.m - xselected.shape[0]
			for kk in range(xselected.shape[0]):
				new_phi = phi(dist_val[kk], data.phifunction)
				PHI[n_old + kk, 0: n_old + kk] = new_phi
				PHI[0:n_old+kk, n_old+kk] = np.asmatrix(new_phi).T
				PHI[n_old+kk, n_old+kk] = phi0
				P[n_old+kk, 1:data.dim+1] = xselected[kk, :]
	data.S = data.S[0:data.m, :]

	data.Y = data.Y[0:data.m, :]
	#data.fevaltime = data.fevaltime[0:data.m, :]
	data.NumberFevals = data.m
	print(-data.Fbest)
	print(np.sum(data.xbest), len(data.xbest))
	return data
