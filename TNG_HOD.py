import numpy as np
import matplotlib.pyplot as plt
import pdb
from astropy.table import Table
import illustris_python  as il
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
import math as mt
from scipy import stats 

basePath = "/research/astro/gama/loveday/Data_local/TNG/TNG300-1/output" # Edit as required.

# Number of histogram bins to be used.
b = 100 # 100

# Mass range to be used.
u = np.log10(5e13)
l = np.log10(2e11)

# Limit on stellar mass of a central.
g_mass = np.log10(1e10)

# Limit on mass of a satelite (No difference to centrals).
s_mass = np.log10(1e10)

#===============================
# Importing and formatting data.
#===============================

# Table containing list of galaxies with a link to their host halo.
#------------------------------------------------------------------

# Importing required subhalos data. 
fields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloMass','SubhaloMassType']
subhalos = il.groupcat.loadSubhalos(basePath, 87, fields = fields)

# Stellar mass of the subhalo.
S_Mass = []
for i in subhalos['SubhaloMassType']:
    S_Mass.append(i[4])
subhalos.pop('SubhaloMassType')

subhalos = pd.DataFrame.from_dict(subhalos) # Converting to dataframe.
subhalos['S_Mass'] = S_Mass

# Adding index column to join on halos table.
subhalos['GroupFirstSub'] = range(0,len(subhalos))

# Converting units to Msun/h.
subhalos['S_Mass'] = subhalos['S_Mass'].apply(lambda x: np.log10(x*(1e10)*1.4))

# Applying galaxy stellar mass limit.
sat_subhalos = subhalos.loc[subhalos['S_Mass'] > s_mass]
subhalos = subhalos.loc[subhalos['S_Mass'] > g_mass]

# Selecting only galaxies.
subhalos = subhalos.loc[subhalos['SubhaloFlag'] == 1]
sat_subhalos = sat_subhalos.loc[sat_subhalos['SubhaloFlag'] == 1]

print('Number of galaxies in the sample:',len(subhalos))

# Table containing list of halos and their mass.
#-----------------------------------------------

# Importing halo data.
fields = ['GroupMassType', 'GroupFirstSub']
halos = il.groupcat.loadHalos(basePath, 87, fields = fields)

# Mass of DM particles in halos.
DMmass = []
for i in halos['GroupMassType']:
    DMmass.append(i[1])
halos.pop('GroupMassType')

halos = pd.DataFrame.from_dict(halos) # Converting to dataframe.
halos['DMMass'] = DMmass

# Removing halos that do not have a subhalo.
halos = halos.loc[halos['GroupFirstSub'] != -1]

# Converting units to Msun/h.
halos['DMMass'] = halos['DMMass'].apply(lambda x: np.log10(x*(1e10)))

halos['SubhaloGrNr'] = range(0,len(halos)) # Adding index to match to subhalos

# Selecting halos within desired mass range.
halos = halos.loc[halos['DMMass'] < u]
halos = halos.loc[halos['DMMass'] > l]

print('Halos in the sample:',len(halos))
#----------

# Merging subhalo and halo tables.

c_hosts = subhalos.merge(halos, on='GroupFirstSub', how = 'inner', suffixes = ('_1', '_2'))

#==========
# Centrals only HOD.
#==========

print('Centrals:')

# Binning the distribution of halos and central hosts.
plt.figure(1)
(nh, binsh, patchesh) = plt.hist(halos['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'All halos')
(nc, binsc, patchesc) = plt.hist(c_hosts['DMMass'], bins = b, range = (l,u), color = 'orange', label = 'Central hosts')

# Finding probability of central within each bin and uncertainty.
DMMasses = []
P = []
P_err = []
i = 0
while i<len(binsh)-1:
    
    # Center of each bin.
    center = (binsh[i+1] + binsh[i])/2

    # Number of entries in each bin.
    n_halos = nh[i]
    n_hosts = nc[i]
    
    # Probability of a halo hosting a central in each bin.
    P_central = n_hosts/n_halos
    
    # Error calculations.
    if n_hosts > 0:
        P_e = np.sqrt((nc[i]/(nh[i]**2)) + ((nc[i]**2)/(nh[i]**3)))
        DMMasses.append(center)
        P.append(P_central)
        P_err.append(P_e)
        i = i + 1
    if n_hosts == 0:
        i = i + 1

# Plotting the distribution of halos and central hosts.
plt.figure(2)
(nh, binsh, patchesh) = plt.hist(halos['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'All halos')
plt.errorbar(DMMasses, nh, yerr = np.sqrt(nh), fmt = 'none')
(nc, binsc, patchesc) = plt.hist(c_hosts['DMMass'], bins = b, range = (l,u), color = 'orange', label = 'Central hosts')
plt.errorbar(DMMasses, nc, yerr = np.sqrt(nc), fmt = 'none')
#plt.title(u'Mass distribution of $M_{*} > 10^{10}$ $M_\u2609 h^{-1}$ central hosts at z = 0.15')
plt.xlabel(u'$\log_{10}\\frac{M_{halo}h}{M_\u2609}$', fontsize = 9)
plt.ylabel('Frequency')
plt.legend(frameon = False)
plt.savefig('Central_Host_Distribution.PDF')


# Defining central HOD function and fitting to data.

def fit(x ,a, z):
    return (1+erf((x - a)/z))*0.5

popt, pcov = curve_fit(fit, DMMasses, P, sigma = P_err, p0 = [11.8, 0.5])

Mmin = popt[0]
Sigma = popt[1]

print('logMmin =',popt[0])
print('sigmalogMmin =',popt[1])
print('Covariance matrix',pcov)

# Plotting fitted HOD.
plt.figure(3)
plt.errorbar(DMMasses, P, P_err, fmt ='k.', color = 'black', ms=2,label='TNG central occupation')
plt.plot(DMMasses, fit(DMMasses, popt[0], popt[1]), label = 'HOD central occupation', color = 'blue')
plt.ylim(0.01,1.1)
plt.xlim(11.25,13)
plt.yscale('log')
#plt.title(u'Probability of a halo at z = 0.15 hosting a galaxy of $M_{*} > 10^{10}$ $M_\u2609 h^{-1}$')
plt.xlabel(u'$\\log_{10}\\frac{M_{halo}h}{M_\u2609}$', fontsize = 9)
plt.ylabel(u'$ \\langle N_{cen} \\rangle$')
plt.legend(frameon = False)
plt.savefig('Central_HOD.PDF')

# Chi-squared calculation.

Chi2 = 0
i = 0
while i < len(P):
    res = ((P[i] - fit(DMMasses[i], popt[0], popt[1]))**2)/(P_err[i]**2)
    Chi2 = Chi2 + res
    i = i + 1
print('Chi2 =', str(Chi2))
print('Reduced Chi2 =', str(Chi2/(b-2)))


a = stats.chi2.cdf(Chi2, b-2)
print('Probability of Chi2 =', str(1-a))

#=====================
# Number of satelites.
#=====================

print('Satellites:')

# Table containing satelites and centrals.
sat_subhalos['SubhaloGrNr_1'] = sat_subhalos['SubhaloGrNr']
s_hosts = c_hosts.merge(sat_subhalos, on='SubhaloGrNr_1', how='inner')

# Binning distribution of all galaxies and central hosts.
plt.figure(4)
(ns, binsh, patchesh) = plt.hist(s_hosts['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'Number of satellites and centrals')
(nc, binsc, patchesc) = plt.hist(c_hosts['DMMass'], bins = b, range = (l,u), color = 'orange', label = 'Central hosts')

# Calculating mean number of satelites per halo for each bin.
DMMasses2 = []
sats = []
s_err = []
nsl = []
ncl = []
i = 0
while i<len(binsc)-1:
    
    # Bin centres (same as centrals).
    center = (binsc[i+1] + binsc[i])/2
    
    # Average number of satellites.
    n_sat = (ns[i]-nc[i])/nh[i]
    
    # Error calculations.
    if n_sat >0:
        sat_err = np.sqrt(((ns[i]+nc[i])/(nh[i]**2)) + (((ns[i]**2)+(nc[i]**2))/(nh[i]**3)))
        s_err.append(sat_err)
        sats.append(n_sat)
        DMMasses2.append(center)
        nsl.append(ns[i])
        ncl.append(nc[i])
        i = i + 1
    if n_sat == 0:
        i = i + 1


# Plotting distribution of central hosts and satellites.
plt.figure(5)
(ns, binsh, patchesh) = plt.hist(s_hosts['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'Number of satellites and centrals')
plt.errorbar(DMMasses2, nsl, yerr = np.sqrt(nsl), fmt = 'none')
(nc, binsc, patchesc) = plt.hist(c_hosts['DMMass'], bins = b, range = (l,u), color = 'orange', label = 'Central hosts')
plt.errorbar(DMMasses2, ncl, yerr = np.sqrt(ncl), fmt = 'none')
#plt.title(u'Galaxy and mass distribution of central hosting halos at z = 0.15')
plt.xlabel(u'$\\log_{10}\\frac{M_{halo}h}{M_\u2609}$', fontsize = 9)
plt.ylabel('Frequency')
plt.legend(frameon = False)
plt.savefig('Satelite_Distribution.PDF')



# Fitting satellite HOD to data.
def SatFit(x,c,z,u):
    return (((x)-(c))/(z))**u

popt, pcov = curve_fit(SatFit, DMMasses2, sats, sigma = s_err, p0 = [12,13,1])

print('M0 = '+str(popt[0]))
print('M1 = '+str(popt[1]))
print('a = '+str(popt[2]))

M0 = popt[0]
M1 = popt[1]
alpha = popt[2]

print('Covariance matrix:',pcov)

# Chi squared calculations.
Chi2 = 0
i = 0
while i < len(sats):
    res = ((sats[i] - SatFit(DMMasses2[i], popt[0], popt[1],popt[2]))**2)/(s_err[i]**2)
    Chi2 = Chi2 + res
    i = i + 1
print('Chi2 =', str(Chi2))
print('Reduced Chi2 =', str(Chi2/(b-2)))

a = stats.chi2.cdf(Chi2, b-2)
print('Probability Chi2 =', str(1-a))

# Plotting satelite HOD.
plt.figure(6)
plt.plot(DMMasses2, SatFit(DMMasses2, popt[0], popt[1], popt[2]), label='HOD satellite occupation', color = 'orange', zorder = 1)
plt.errorbar(DMMasses2, sats, s_err, fmt ='.',label='TNG satellite occupation', color = '#009ACD',zorder=0)
plt.xlabel(u'$\\log_{10}\\frac{M_{halo}h}{M_\u2609}$', fontsize = 9)
plt.ylabel(u'$ \\langle N_{sat} \\rangle$')
plt.yscale('log')
#plt.title(u'Expected number of $M_{*} > 10^{10}$ $M_\u2609 h^{-1}$ satellites in halos at z = 0.15')
plt.ylim(1e-2,11) 
plt.legend(frameon = False)
plt.savefig('Satellite_HOD.PDF')

#================================
# Combined central and satelites.
#================================

print('Complete HOD:')

# Binning all halos and then 'centrals and stellites'.
plt.figure(7)
(nh, binsh, patchesh) = plt.hist(halos['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'All Halos')
(ns, binss, patchesh) = plt.hist(s_hosts['DMMass'], bins = b, range = (l, u), color = 'blue', label = 'Number of Satelites and Centrals')


# Calculating mean number of galaxies per halo in a bin.
DMMasses3 = []
ntot = []
tot_err = []
i = 0
while i<len(binsc)-1:
    
    # Bin centres (same as previous). 
    center = (binss[i+1] + binsh[i])/2
    
    # Mean number of galaxies in each bin.
    tot = (ns[i])/nh[i]
    
    # Error calculations.
    if tot  >0:
        comb_err = np.sqrt((ns[i]/(nh[i]**2)) + ((ns[i]**2)/(nh[i]**3)))
        ntot.append(tot)
        tot_err.append(comb_err)
        DMMasses3.append(center)
        i = i + 1
    if tot==0:
        i = i + 1

# Fitting full HOD.

def fullfit(x,a,b,c,d,e):
    return (((1+erf((x - a)/b)))*0.5)+(((x-c)/d)**e)

popt, pcov = curve_fit(fullfit, DMMasses3, ntot, sigma = tot_err, p0=[11.7,0.2,11,1,4])

print('Mmin =',popt[0])
print('sigmalogMmin =',popt[1])
print('M_0 = '+str(popt[2]))
print('M_1 = '+str(popt[3]))
print('a = '+str(popt[4]))

print('Covariance matrix:',pcov)

# Chi squared calculations.
Chi2 = 0
i = 0
while i < len(ntot):
    res = ((ntot[i] - fullfit(DMMasses3[i], popt[0], popt[1],popt[2],popt[3],popt[4]))**2)/(tot_err[i]**2)
    Chi2 = Chi2 + res
    i = i + 1
print('Chi2 =', str(Chi2))
print('Reduced Chi2 =', str(Chi2/(b-5)))

a = stats.chi2.cdf(Chi2, b-5)
print('Probability of Chi2 =', str(1-a))

#----------

#Plotting full HOD and individual centrals and satellites.
plt.figure(8)

plt.errorbar(DMMasses, P, P_err, fmt ='none', color = 'black', label = 'TNG centrals', zorder = 0)
plt.plot(DMMasses, fit(DMMasses, Mmin, Sigma), label = 'HOD centrals', color = 'blue', zorder = 1)

plt.errorbar(DMMasses2, sats, s_err, fmt ='none', label = 'TNG satellites', color = '#009ACD',zorder =2)
plt.plot(DMMasses2, SatFit(DMMasses2, M0, M1,alpha), label = 'HOD satellites', color = 'orange',zorder = 3)

plt.errorbar(DMMasses3, ntot, tot_err, fmt = 'none',label = 'TNG total galaxies', color = '#008B00', zorder = 4)
plt.plot(DMMasses3, fullfit(DMMasses3, popt[0], popt[1],popt[2],popt[3],popt[4]), label = 'HOD total galaxies', color = 'red', zorder = 5)

plt.ylim(1e-2,15)
plt.legend(frameon = False)
plt.yscale('log')
#plt.title(u'Expected number of $M_{*} > 10^{10}$ $M_\u2609 h^{-1}$ galaxies in halos at z = 0.15')
plt.xlabel(u'$\\log_{10}\\frac{M_{halo}h}{M_\u2609}$', fontsize = 9)
plt.ylabel(u'$ \\langle N \\rangle$')
plt.savefig('Complete_HOD.PDF')

