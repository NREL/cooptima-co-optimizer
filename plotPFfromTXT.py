import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as P


def load_data(KK,comp):
	PFpoints=np.zeros([35586,2])
	ii = 0

	if comp==None:
		filename = "merit_cost_pareto_data_all_K"+str(KK) + ".txt"
		#filename = "pareto_cost_merit_all/sampling_pareto_data_all_K_"+str(KK) + ".txt"
	else:
		kk
		filename = "pareto_cost_merit_K_" +str(KK) +"/sampling_pareto_data_"+comp+"_K_"+str(KK) + ".txt"

	with open(filename, "r") as filestream:
		for line in filestream:
			L = line.strip().split(',')
			#print L
			PFpoints[ii,1] = float(L[0])
			PFpoints[ii,0] = float(L[1])
			ii = ii+1     
	
	merit = PFpoints[:,0]
	cost = PFpoints[:,1]
	#print(merit)
	
	#100 Pareto points per run are stored

	return PFpoints, merit, cost



'''
def extract_merit_distribution_for_cost(KK,comp):
	merit, cost = load_data(KK, comp)
	median = np.median(data)
upper_quartile = np.percentile(data, 75)
lower_quartile = np.percentile(data, 25)

iqr = upper_quartile - lower_quartile
upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
'''

def extract_data(KK, comp, merit_bracket):
	
	PFpoints, merit, cost = load_data(KK,comp)  #load data from files for individual cost components
	#merit value analysis
	max_merit = np.max(np.ravel(merit)) # maximum achievable merit
	#print('maxmerit: ', max_merit)
	low_bound_merit = max_merit-merit_bracket*max_merit
	#sort according to descending merit value
	s_ind = np.argsort(np.ravel(-merit), axis = -1)
	sorted_PFpoints = PFpoints[s_ind,:]
	#find all PF points with merit larger than low_bound_merit
	best_percent_merit_ind = np.where(np.ravel(merit)>low_bound_merit)[0]	
	best_percent = PFpoints[best_percent_merit_ind,:] #[cost, merit]
	best_percent_costmean = np.mean(best_percent[:,0]) #expected value of cost to pay for getting to x% max merit
	best_percent_costmedian = np.median(best_percent[:,0]) #median price to pay for x% max merit
	best_percent_coststd = np.std(best_percent[:,0])
	best_percent_costmin = np.min(best_percent[:,0])
	best_percent_costmax = np.max(best_percent[:,0])

	#can we define likelihood for prices or merit?

	return PFpoints

def extract_data_all(KK, merit_bracket):
	comp = None
	PFpoints, merit, cost = load_data(KK,comp)  #load data from files for individual cost components
	#merit value analysis
	max_merit = np.max(np.ravel(merit)) # maximum achievable merit
	#print('maxmerit: ', max_merit)
	low_bound_merit = max_merit-merit_bracket*max_merit
	#sort according to descending merit value
	s_ind = np.argsort(np.ravel(-merit), axis = -1)
	sorted_PFpoints = PFpoints[s_ind,:]
	#find all PF points with merit larger than low_bound_merit
	best_percent_merit_ind = np.where(np.ravel(merit)>low_bound_merit)[0]	
	best_percent = PFpoints[best_percent_merit_ind,:] #[cost, merit]
	best_percent_costmean = np.mean(best_percent[:,0]) #expected value of cost to pay for getting to x% max merit
	best_percent_costmedian = np.median(best_percent[:,0]) #median price to pay for x% max merit
	best_percent_coststd = np.std(best_percent[:,0])
	best_percent_costmin = np.min(best_percent[:,0])
	best_percent_costmax = np.max(best_percent[:,0])

	#can we define likelihood for prices or merit?

	return PFpoints



def analysis4(cost_matrix, merit_matrix, cost, merit, KK,component):
	median_cost = np.median(cost_matrix, axis =1)
	median_merit = np.median(merit_matrix, axis= 1)
	mean_cost = np.mean(cost_matrix, axis =1)
	mean_merit = np.mean(merit_matrix, axis= 1)
	p1 = plt.scatter(cost, merit, label="K={}".format(KK), marker='.')
	p2 = plt.scatter(mean_cost, mean_merit, label="K={}".format(KK), marker='*', color='r')
	p3 =plt.scatter(median_cost, median_merit, label="K={}".format(KK), marker='o', color='g')
	plt.xlabel('Cost')
	plt.ylabel('Merit')
	plt.title('Component_'+component)
	plt.legend([p1,p2,p3,],['all data', 'mean','median'], loc=4)
	font = {'size': 16}
	plt.rc('font', **font)
	plt.tight_layout()
	#plt.title("{}".format(change_name))
	figname = "MeanMedian_Pareto_"+component+"_K_"+str(KK)+".png"
	plt.savefig(figname)
	plt.close('all')
	return mean_cost, mean_merit, median_cost, median_merit

def analysis5_plots(Kvals, mean_K_matrix_c, mean_K_matrix_m, median_K_matrix_c, median_K_matrix_m, mrk, clr, component):
	fig,ax = plt.subplots()
	fig.set_size_inches(12.,8.)
	plt.subplots_adjust(left = 0.075, right = 0.95, top = 0.9, bottom = 0.25)

	for jj in range(len(Kvals)):
		plt.scatter(mean_K_matrix_c[:,jj], mean_K_matrix_m[:,jj], marker=mrk[jj], s=40, color=clr[jj], label = 'K='+str(Kvals[jj]))
		
	plt.legend(loc=2)	
	plt.xlabel('Cost')
	plt.ylabel('Merit')
	plt.title('Mean Pareto fronts for component '+component)
	font = {'size': 20}
	plt.rc('font', **font)
	plt.tight_layout()
	figname = "MeanParetoFront_all_K_"+component+'.png'
	plt.savefig(figname)
	plt.close('all')
	
	#plot median pareto fronts for all K values for each component
	fig,ax = plt.subplots()
	fig.set_size_inches(12.,8.)
	plt.subplots_adjust(left = 0.075, right = 0.95, top = 0.9, bottom = 0.25)
	for jj in range(len(Kvals)):
		plt.scatter(median_K_matrix_c[:,jj], median_K_matrix_m[:,jj], marker=mrk[jj],s=40, color=clr[jj], label = 'K='+str(Kvals[jj]))
	plt.xlabel('Cost')
	plt.ylabel('Merit')
	plt.legend(loc=2)	
	plt.title('Median Pareto fronts for component '+component)
	font = {'size': 20}
	plt.rc('font', **font)
	plt.tight_layout()
	figname = "MedianParetoFront_all_K_"+component+'.png'
	plt.savefig(figname)
	plt.close('all')
	
	return 0

def analysis2(merit, merit_bracket, bpldata, KK, boxlabel, alldata):
	max_merit = np.max(np.ravel(merit)) # maximum achievable merit
	print(max_merit)
	low_bound_merit = max_merit-merit_bracket*max_merit
	#find all PF points with merit larger than low_bound_merit
	best_percent_merit_ind = np.where(np.ravel(merit)>low_bound_merit)[0]	
	best_percent = alldata[best_percent_merit_ind,:] #[cost, merit]
	best5=list(best_percent[:,0]) #costs associated with getting to best x% of max merit
	bpldata.append(best5) #cost data for boxplots that show costs needed to get within x% of max merit
	newstr = 'K='+str(KK)+'\n merit in [{0:.2f}'.format(low_bound_merit)+', {0:.2f}'.format(max_merit)+']' 
	boxlabel.append(newstr)
	#max_comp_merit.append(max_merit)
	return bpldata, boxlabel

def analysis1(KK,C_statistic,C_eps, cost, merit, boxlabel2, bpldata2):
	boxlabel2.append('K='+str(KK))
	ind_interest = np.where(cost <= C_statistic+C_eps*C_statistic)[0]
	C_fewer = cost[ind_interest]
	M_fewer = merit[ind_interest]
	ind_interest2 = np.where(C_fewer >= C_statistic-C_eps*C_statistic)[0]
	merit_data = M_fewer[ind_interest2]
	cost_data  = C_fewer[ind_interest2]
	bpldata2.append(list(merit_data)) 

	return bpldata2, boxlabel2


def plotPFfromTXT():
	
	componentwise_analysis = False

	#cost components
	if componentwise_analysis:	
		comps = ['6032-29-7','137-32-6', '67-56-1', '105-54-4', '78-83-1','78-92-2','107-87-9',\
		'71-36-3', '78-93-3', '141-78-6', '64-17-5', '565-80-0', '464-06-2', '123-86-4',\
		'79-20-9', '107-39-1','534-22-5', '120-92-3', '100-66-3', 'BOB-1','BOB-2','BOB-3']
	else:
		comps = ['all']

	clr = ['fuchsia', 'b', 'g', 'r', 'y', 'm', 'c', 'k', 'g', 'r', 'y', 'm']
	mrk = ['.', 'o', 'x', '*', 'd', '>', 'o', 'o', 'x', 'x', 'x', 'x', 'x']
	scenario1 = True
	scenario2 = True
	scenario3 = False
	scenario4 = True#for mean/median: subdivide x-axis into samll intervals, then for each interval compute mean/median value and plot that 
	scenario5 = True

	
	C_statistic = 3#the cost we are willing to pay plus minus C_eps%
	C_eps = 0.25 #percent for wiggle room around C-statistic

	merit_bracket = 0.02 #find all solutions within x% of max merit
	for kk in range(len(comps)):
		bpldata = []#cost data for boxplots that show costs needed to get within x% of max merit
		bpldata2 =[] #list of merit values attainable by incesting [C-eps, C+eps] cost
		boxlabel=[]
		boxlabel2 = []
		max_comp_merit=[]
		Kvals = [-1.25]#[-2.0, -1.0, 1.0, 2.0, 3.0, 4.0]#[-2, -1, 1, 2]
		max_merit_poss = np.zeros(len(Kvals))
		cost_matrix = np.zeros([100,100])
		merit_matrix = np.zeros([100,100])
		mean_K_matrix_c = np.zeros((100, len(Kvals)))
		mean_K_matrix_m = np.zeros((100, len(Kvals)))
		median_K_matrix_c = np.zeros((100, len(Kvals)))
		median_K_matrix_m = np.zeros((100, len(Kvals)))

		for jj in range(len(Kvals)):
			KK = Kvals[jj]
			if componentwise_analysis:
				alldata = extract_data(KK, comps[kk], merit_bracket)
			else:
				alldata = extract_data_all(KK, merit_bracket)
			merit = alldata[:,1]
			cost = alldata[:,0]	
			if scenario4 or scenario5:
				for ijk in range(100):
					cost_s = cost[ijk*100:(ijk+1)*100]
					merit_s = merit[ijk*100:(ijk+1)*100]
					s_ind = np.argsort(np.ravel(cost_s), axis = -1)
					sort_c = cost_s[s_ind]
					sort_m = merit_s[s_ind]
					cost_matrix[:,ijk] = sort_c
					merit_matrix[:,ijk] = sort_m

				#SCENARIO 4: compute average over all pareto fronts
				mean_cost, mean_merit, median_cost, median_merit = analysis4(cost_matrix, merit_matrix, cost, merit, KK,comps[kk])
				#SCENARIO 5: plot mean pareto fronts and median pareto fronts for all K values into one frame
				if scenario5:
					mean_K_matrix_c[:,jj] = mean_cost
					mean_K_matrix_m[:,jj] = mean_merit
					median_K_matrix_c[:,jj] = median_cost
					median_K_matrix_m[:,jj] = median_merit



			if scenario3:
					#SCENARIO 3: collect maximum possible merit for each K componentwise
					max_merit_poss[jj] = np.max(np.ravel(merit))
					print(max_merit_poss)

			if scenario2:
					#SCENARIO 2: find all pareto points for which merit is within merit_bracket percent of max merit
					print(merit_bracket)
					bpldata, boxlabel = analysis2(merit, merit_bracket, bpldata, KK, boxlabel, alldata)

			if scenario1:
					###SCENARIO 1: find all merits I can get for cost in [C_statistic - C_eps, C_statistic + C_eps]
					bpldata2, boxlabel2 = analysis1(KK,C_statistic,C_eps, cost, merit,boxlabel2, bpldata2)

			'''
			##print(merit_data)
			#print(min(cost_data), max(cost_data))
			#hhh
			#weights = np.ones_like(merit_data)/float(len(merit_data))
			#plot histograms that show the distribution of attainable merits with cost at least "statistic"
			#fig,ax = plt.subplots()
			#fig.set_size_inches(12.,8.)
			#n, bins, patches = P.hist(merit_data, 30,  histtype='stepfilled', weights = weights)
			#P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
			# add a line showing the expected distribution
				#y = P.normpdf(bins, merit_data.mean(), merit_data.std())
				#l = P.plot(bins, y, 'k--', linewidth=1.5)
				#P.xlabel('Merit achievable with cost of at most {0:.2f} '.format(statistic))
				#P.ylabel('Normed frequency')
				#P.title(comps[kk] + ' for K='+str(KK))
				#save_name='merit_dist_by_cost_'+comps[kk]+'_K_'+str(KK)+'.png'
				#plt.savefig(save_name)
				#plt.close('all')

				#merit analysis: select a merit we want to achieve and plot the corresponding distribution of costs that get us this merit value
				#max_merit = max(merit)
				#perc = 0.1 #we want solutions that are within perc% of overall best merit
				#statistic = max_merit-perc*max_merit

				#ind_interest = np.where(merit >= statistic)[0]
				#cost_data = cost[ind_interest]
				#weights = np.ones_like(cost_data)/float(len(cost_data))
				#plot histograms that show the distribution of attainable merits with cost at least "statistic"
				#fig,ax = plt.subplots()
				#fig.set_size_inches(12.,8.)
				#n, bins, patches = P.hist(cost_data, 20,  histtype='stepfilled', cumulative = True, weights = weights)
				#P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
				# add a line showing the expected distribution
				#P.xlabel('Cost necessary to achieve merit of at least {0:.2f} '.format(statistic))
				#P.ylabel('Normed frequency')
				#P.title(comps[kk] + ' for K='+str(KK))
				#save_name='cost_dist_by_merit_'+comps[kk]+'_K_'+str(KK)+'.png'
				#plt.savefig(save_name)
				#plt.close('all')
			'''
		#plot mean pareto fronts for all K values for each component
		if scenario5: #dot eh plots
			analysis5_plots(Kvals, mean_K_matrix_c, mean_K_matrix_m, median_K_matrix_c, median_K_matrix_m, mrk, clr, comps[kk])

	
		if scenario3:
			fig, ax = plt.subplots()
			fig.set_size_inches(12.,8.)
			plt.subplots_adjust(left = 0.075, right = 0.95, top = 0.9, bottom = 0.25)
			plt.plot(Kvals, max_merit_poss, marker ='*', markersize = 20)
			plt.ylabel('Maximum merit possible')
			plt.xlabel('Kvalues')
			plt.title('Component '+comps[kk])
			font = {'size': 20}
			plt.rc('font', **font)
			plt.tight_layout()
			figname = "Maxmerit_by_K_"+comps[kk]+".png"
			plt.savefig(figname)
			plt.close('all')
		
		if scenario1:
			fig,ax = plt.subplots()
			fig.set_size_inches(12.,8.)
			plt.subplots_adjust(left = 0.075, right = 0.95, top = 0.9, bottom = 0.25)
			ax.boxplot(bpldata2)
			xTickNames = plt.setp(ax,xticklabels= boxlabel2)
			plt.setp(xTickNames, rotation = 75, fontsize =16)
			plt.ylabel('Merit')
			plt.title('Merit attainable for cost in [{0:.2f},'.format(C_statistic-C_eps*C_statistic)+\
				'{0:.2f}]'.format(C_statistic+C_eps*C_statistic)+'\n Component '+comps[kk])
			font = {'size': 20}
			plt.rc('font', **font)
			plt.tight_layout()
			figname = "K_boxplots_merit_"+comps[kk]+".png"
			plt.savefig(figname)
			plt.close('all')
		
		if scenario2:
			fig,ax = plt.subplots()
			fig.set_size_inches(12.,8.)
			plt.subplots_adjust(left = 0.075, right = 0.95, top = 0.9, bottom = 0.25)
			ax.boxplot(bpldata)
			xTickNames = plt.setp(ax,xticklabels= boxlabel)
			plt.setp(xTickNames, rotation = 75, fontsize =16)
			plt.ylabel('Cost')
			plt.title('Costs required to get within {0:.0f}% of max merit'.format(100*merit_bracket))
			font = {'size': 20}
			plt.rc('font', **font)
			plt.tight_layout()
			figname = "K_boxplots_comp_"+comps[kk]+".png"
			plt.savefig(figname)
			plt.close('all')
		

if __name__ == "__main__":
	plotPFfromTXT()