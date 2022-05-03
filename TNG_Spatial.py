import numpy as np
import matplotlib.pyplot as plt
import pdb
from astropy.table import Table
import illustris_python  as il
import pandas as pd

basePath = "/research/astro/gama/loveday/Data_local/TNG/TNG300-1/output" # Edit as required.

# Upper and lower mass range to be considered.
u = [14]
l = [13]

# Limit on stellar mass of a galaxy.
g_mass = np.log10(1e10)

#===============================
# Importing and formatting data.
#=============================== 

# Table containing list of galaxies with a link to their host halo.
#------------------------------------------------------------------

# Importing subhalos data.
fields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloMassType','SubhaloPos']
subhalos = il.groupcat.loadSubhalos(basePath, 87, fields = fields)

# Stellar  mass of the subhalo.
S_Mass = []
for i in subhalos['SubhaloMassType']:
    S_Mass.append(i[4])
subhalos.pop('SubhaloMassType')

# x and y positions of each subhalo.
Gx_Pos = []
Gy_Pos = []
for i in subhalos['SubhaloPos']:
    Gx_Pos.append(i[0])
    Gy_Pos.append(i[1])
subhalos.pop('SubhaloPos')

subhalos = pd.DataFrame.from_dict(subhalos)# Converting to dataframe.
subhalos['S_Mass'] = S_Mass
subhalos['Gx_Pos'] = Gx_Pos
subhalos['Gy_Pos'] = Gy_Pos

# Adding index column to join on halos table.
subhalos['GroupFirstSub'] = range(0,len(subhalos))

# Converting units to Msun/h and applying stellar mass correction.
subhalos['S_Mass'] = subhalos['S_Mass'].apply(lambda x: np.log10(x*(1e10)*1.4))

# Must be a galaxy.
subhalos = subhalos.loc[subhalos['SubhaloFlag'] == 1]

subhalos2 = subhalos

# Table containing list of halos and their mass.
#-----------------------------------------------

# Importing halos data.
fields = ['GroupMassType', 'GroupFirstSub','GroupMass','GroupPos','Group_R_TopHat200']
halos = il.groupcat.loadHalos(basePath, 87, fields = fields)

# Mass of DM particles in halo.
DM_Mass = []
for i in halos['GroupMassType']:
    DM_Mass.append(i[1])
halos.pop('GroupMassType')

# x and y position of halos.
Hx_Pos = []
Hy_Pos = []
for i in halos['GroupPos']:
    Hx_Pos.append(i[0])
    Hy_Pos.append(i[1])
halos.pop('GroupPos')

halos = pd.DataFrame.from_dict(halos) # Converting to dataframe.
halos['DM_Mass'] = DM_Mass
halos['Hx_Pos'] = Hx_Pos
halos['Hy_Pos'] = Hy_Pos

# Converting units to Msun/h.
halos['DM_Mass'] = halos['DM_Mass'].apply(lambda x: np.log10(x*(1e10)))

halos['SubhaloGrNr'] = range(0,len(halos)) #Adding index to match to subhalos

# Removing halos that do not have a subhalo.
halos = halos.loc[halos['GroupFirstSub'] != -1]

# Merging subhalo and halo tables.
c_halos3 = c_halos  = subhalos.merge(halos, on='GroupFirstSub', how = 'inner', suffixes = ('_1', '_2'))

print('Halo and subhalo data read.')


# Calculating separation distributions. This is in an iterative form so different values can be added to 'u' and 'l' above to allow for investigation of different mass ranges.

z = 0
while z < len(u):
    # Selecting halos within desired mass range.
    c_halos = c_halos.loc[c_halos['DM_Mass'] < u[z]]
    c_halos2 = c_halos = c_halos.loc[c_halos['DM_Mass'] > l[z]]


    # Calculating the normalised separations within each halo.
    R = []
    for i in c_halos['SubhaloGrNr_1']:
        # Selecting halo and corresponding subhalos.
        subhalos = subhalos[subhalos['SubhaloGrNr'] == i]
        c_halos = c_halos[c_halos['SubhaloGrNr_1'] == i]
        # Removing central galaxy.
        subhalos = subhalos[subhalos['GroupFirstSub'] != float(c_halos['GroupFirstSub'])]
        
        # Calculating separations in x and y dimmensions and then full 2D separation.
        sepx = []
        for j in subhalos['Gx_Pos']:
            sepx.append(abs(c_halos['Hx_Pos']-j))
        sepy = []
        for j in subhalos['Gy_Pos']:
            sepy.append(abs(c_halos['Hy_Pos']-j))
        sep = []
        y = 0
        while y < len(sepy):
            sep.append(np.sqrt((sepx[y]**2) + (sepy[y]**2)))
            y = y + 1

        subhalos['Sep'] = sep

        # Calculating R_50.
        subhalos['NormSep'] = subhalos['Sep']/np.median(list(subhalos['Sep']))
        
        # Selecting galaxies satisfying the stellar mass limit.
        subhalos = subhalos[subhalos['S_Mass']> g_mass]
        for j in subhalos['NormSep']:
            R.append(float(j))
    
    
        subhalos = subhalos2
        c_halos = c_halos2
    
    print('Number of galaxy separations:',len(R))
    

    plt.figure(1)

    # Binning separations.
    x,bins,patches=plt.hist(R, bins = 40, range = (0,4))

    Prob  = []
    Prob_err = []
    Den = []
    Den_err = []
    k = 0
    while k < len(x):
        # Calculating density of galaxies in each bin.
        Den.append(x[k]/(((np.pi*(bins[k+1]**2))-(np.pi*(bins[k]**2)))*len(R)))
        Den_err.append(np.sqrt(x[k])/(((np.pi*(bins[k+1]**2))-(np.pi*(bins[k]**2)))*len(R)))
        # Probability of finding a galaxy in each bin.
        Prob.append(x[k]/len(R))
        Prob_err.append(np.sqrt(x[k])/len(R))
        k = k + 1
    
    k = 0
    # Calculating bin centres.
    centres = []
    while k <len(bins)-1:
        centre = (bins[k+1]+bins[k])/2
        centres.append(centre)
        k = k + 1
        
    # Probability distribution plot.
    plt.figure(2)
    plt.bar(centres,Prob, width = 0.1)
    plt.errorbar(centres, Prob,yerr = Prob_err, fmt = 'none', capsize = 2.5, color = 'black')
    plt.ylabel('$P$')
    plt.xlabel('$\\frac{r}{R_{50}}$', fontsize = 12)
    #plt.title('Spatial distribution of $M_{*} > 10^{10}$ $M_{\u2609}h^{-1}$ TNG satellites')
    plt.savefig('SpatialPlot.PDF')
    
    # Density plot (normalised by maximum density).
    plt.figure(3)
    plt.bar(centres,Den/Den[0], width = 0.1)
    plt.errorbar(centres, Den/Den[0],yerr = Den_err/Den[0], fmt = 'none', capsize = 2.5, color = 'black')
    plt.ylabel('$\\rho$')
    plt.xlabel('$\\frac{r}{R_{50}}$', fontsize = 12)
    #plt.title('Spatial distribution of $M_{*} > 10^{10}$ $M_{\u2609}h^{-1}$ TNG satellites')
    plt.savefig('SpatialPlot2.PDF')
    
    
    print('Probability of r < R_50',sum(Prob[0:10]))

    c_halos = c_halos3
    z = z + 1
