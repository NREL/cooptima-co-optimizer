from merit_functions import mmf_single_param, mmf_single
import numpy as np
import matplotlib.pyplot as plt
bp = {}
bp['RON'] = 100.0
bp['S'] = 8.0
bp['ON'] = 90.0
bp['HoV'] = 415.0
bp['SL'] = 46.0
bp['LFV150'] = 0.0
bp['PMI'] = 1.0

ref = {}
ref['RON'] = 92.0
ref['S'] = 10.0
ref['HoV'] = 415.0
ref['SL'] = 46.0
ref['PMI'] = 2.0

sen = {}
sen['ON']= 1.0/1.6
sen['ONHoV'] = 0.01
sen['HoV'] = 1.0/130.0
sen['SL'] = 1.0/3.0
sen['LFV150'] = 1.0
sen['PMIFIX'] = 0.67
sen['PMIVAR'] = 0.5

# Check single evaluatio works
ms = mmf_single(RON=bp['RON'], S=bp['S'],
    ON=bp['ON'], HoV=bp['HoV'],SL=bp['SL'],
    LFV150=bp['LFV150'],PMI=bp['PMI'],K=0.5)

msp = mmf_single_param(ref,sen,RON=bp['RON'], S=bp['S'],
    ON=bp['ON'], HoV=bp['HoV'],SL=bp['SL'],
    LFV150=bp['LFV150'],PMI=bp['PMI'],K=0.5)

print 'ms=', ms
print 'msp=', msp

# Now let some parameters be random variables
# First look at K
nsamples = 200000 
#ksamp = np.random.normal(0.5,2.0,nsamples)
#msamp = np.zeros([nsamples])
#for ns in range(nsamples):
#    msamp[ns]  = mmf_single_param(ref,sen,RON=bp['RON'], S=bp['S'],
#    ON=bp['ON'], HoV=bp['HoV'],SL=bp['SL'],
#    LFV150=bp['LFV150'],PMI=bp['PMI'],K=ksamp[ns])
#print ksamp, msamp
#plt.hist(ksamp)
#plt.title('K')
#plt.show()
#plt.hist(msamp)
#plt.title('M')
#plt.show()

# Next look at impact of reference composition
# These in Paul's report
ref_mean = {}
ref_mean['RON'] = 92.0
ref_mean['S'] = 10.0
ref_mean['HoV'] = 415.0
ref_mean['SL'] = 46.0
ref_mean['PMI'] = 2.0

# This such that it looks more like certification fuel
ref_var = {}
ref_var['RON'] = 8.0
ref_var['S'] = 10.0
ref_var['HoV'] = 20.0
ref_var['SL'] = 2.0
ref_var['PMI'] = 2.0

msamp = np.zeros([nsamples])
ronsamp = np.zeros([nsamples])
ssamp = np.zeros([nsamples])
hovsamp = np.zeros([nsamples])
slsamp = np.zeros([nsamples])
pmisamp = np.zeros([nsamples])

ns = 0
while (ns < nsamples):
    for kk in ref_mean.keys():
        ref[kk] = np.random.normal(ref_mean[kk],ref_var[kk])
    if ref['PMI'] < 0.0:
        continue 
    if ref['S'] < 0.0:
        continue  
    ronsamp[ns] = ref['RON']
    ssamp[ns] = ref['S']
    hovsamp[ns] = ref['HoV']
    slsamp[ns] = ref['SL']
    pmisamp[ns] = ref['PMI']
    msamp[ns] =  mmf_single_param(ref,sen,RON=bp['RON'], S=bp['S'],
    ON=bp['ON'], HoV=bp['HoV'],SL=bp['SL'],
    LFV150=bp['LFV150'],PMI=bp['PMI'],K=0.5) 
    ns += 1

f, axs = plt.subplots(2,3)
axs[1,2].hist(msamp)
axs[1,2].set_xlabel('Merit')

axs[0,0].hist(ronsamp)
axs[0,0].set_xlabel('Research Octane')

axs[0,1].hist(ssamp)
axs[0,1].set_xlabel('Sensitivity')

axs[0,2].hist(hovsamp)
axs[0,2].set_xlabel('Heat of Vaporization')

axs[1,0].hist(slsamp)
axs[1,0].set_xlabel('Laminar Flame Speed')

axs[1,1].hist(pmisamp)
axs[1,1].set_xlabel('Particulate Matter Index')
plt.show()

ref = {}
ref['RON'] = 92.0
ref['S'] = 10.0
ref['HoV'] = 415.0
ref['SL'] = 46.0
ref['PMI'] = 2.0

sen_mean = {}
sen_mean['ON']= 1.0/1.6
sen_mean['ONHoV'] = 0.01
sen_mean['HoV'] = 1.0/130.0
sen_mean['SL'] = 1.0/3.0
sen_mean['LFV150'] = 1.0
sen_mean['PMIFIX'] = 0.67
sen_mean['PMIVAR'] = 0.5

sen_var = {}
sen_var['ON']= 1.0/1.6*.1
sen_var['ONHoV'] = 0.01*.1
sen_var['HoV'] = 1.0/130.0*.1
sen_var['SL'] = 1.0/3.0*.1
sen_var['LFV150'] = 0.1
sen_var['PMIFIX'] = 0.67*.1
sen_var['PMIVAR'] = 0.5*.1

samples = {}
for kk in sen_mean.keys():
    samples[kk] = []

ns = 0
while (ns < nsamples):
    for kk in sen_mean.keys():
        sen[kk] = np.random.normal(sen_mean[kk],sen_var[kk])
        samples[kk].append(sen[kk])
   
    msamp[ns] =  mmf_single_param(ref,sen,RON=bp['RON'], S=bp['S'],
    ON=bp['ON'], HoV=bp['HoV'],SL=bp['SL'],
    LFV150=bp['LFV150'],PMI=bp['PMI'],K=0.5) 
    ns += 1

f, axs = plt.subplots(2,4)
axs[1,3].hist(msamp,bins=200,alpha=0.4)
axs[1,3].set_xlabel('Merit')

axs[0,0].hist(samples['ON'],bins=200,alpha=0.4)
axs[0,0].set_xlabel('ON')

axs[0,1].hist(samples['ONHoV'],bins=200,alpha=0.4)
axs[0,1].set_xlabel('ONHoV')

axs[0,2].hist(samples['HoV'],bins=200,alpha=0.4)
axs[0,2].set_xlabel('HoV')

axs[0,3].hist(samples['SL'],bins=200,alpha=0.4)
axs[0,3].set_xlabel('SL')

axs[1,0].hist(samples['LFV150'],bins=200,alpha=0.4)
axs[1,0].set_xlabel('LFV150')

axs[1,1].hist(samples['PMIFIX'],bins=200,alpha=0.4)
axs[1,1].set_xlabel('PMIFIX')

axs[1,2].hist(samples['PMIVAR'],bins=200,alpha=0.4)
axs[1,2].set_xlabel('PMIVAR')

plt.show()

plt.show()
