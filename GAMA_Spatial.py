#Importing required modules.
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#===============
# Importing data
#===============

# Reading stellar mass data and converting to pandas df.
hdul = fits.open('StellarMassesv19.fits')
sm = hdul[1].data
hdul.close()
sm = Table(sm)
sm = sm.to_pandas()

# Reading galaxy data and converting to pandas df.
hdul = fits.open('G3CGalv10.fits')
gal = hdul[1].data
hdul.close()
gal = Table(gal)
gal = gal.to_pandas()

# Joining the two tables.
gals = gal.merge(sm, on='CATAID', suffixes = ('_gal','_sm'))

# Aperture correction.

# Selecting realistic fluxscale values.
gals = gals[gals['fluxscale']>0.5]
gals = gals[gals['fluxscale']<2]

# Applying correction.
gals['absmag_r'] = gals['absmag_r'] - (2.5*np.log10(gals['fluxscale']))
gals['logmstar'] = (gals['logmstar'] + np.log10(gals['fluxscale']))-(2*np.log10(1/0.7))

# Applying volume limiting cuts.
gals = gals[gals['absmag_r']<-20]
gals = gals[gals['absmag_r']>-23]

gals = gals[gals['Z_sm']<0.2]
gals = gals[gals['Z_sm']>0.003]

# Selecting galaxies that belong to a group.
gals = gals[gals['GroupID'] != 0]

gals.sort_values('GroupID') # Sorting data by group IDs

# Moving group IDs to a list.
ID = []
for i in gals['GroupID']:
    ID.append(i)

# Indentifying the groups for which the 5th brightest galaxy is in the sample.
Groups = []
for i in ID:
    a = ID.count(i)
    if a > 4:
        Groups.append(i)
Groups = set(Groups)

# Reading FoF group data.
hdul = fits.open('G3CFoFGroupv10.fits')
FoF = hdul[1].data
hdul.close()
FoF = Table(FoF)
FoF = FoF.to_pandas()

# Selecting FoF groups with 5 galaxies in the volume limited sample.
df = pd.DataFrame(Groups)
df = df.rename(columns = {0:'GroupID'})
FoF = df.merge(FoF, on='GroupID')

# Selecting only groups that are fully within the sample.
FoF = FoF[FoF['GroupEdge'] == 1]

# Converting to log10 mass.
FoF['MassA'] = np.log10(FoF['MassA'])

FoF.index = np.arange(0,len(FoF))

#====================
#Spatial Distribution
#====================

# Selecting galaxies satisfying the minimum stellar mass condition.
gals = gals[gals['logmstar']>10]

# Removing central galaxies.
gals = gals[gals['RankIterCen'] != 1]

FoFGals = gals.merge(FoF, on='GroupID', suffixes = ('_gal','_sm'))

# Calculating normalised separation.
FoFGals['NormSep'] = FoFGals['AngSepIterCen']/FoFGals['Rad50']

# Imposing halo mass limits.
FoFGals = FoFGals[FoFGals['MassA']>13]
FoFGals = FoFGals[FoFGals['MassA']<14]

FoF  = FoF[FoF['MassA']>13]
FoF  = FoF[FoF['MassA']<14]

print('Number of galaxies considered =',len(FoFGals))
print('Number of groups considered = ',len(FoF))

# Binning distribution of separations.
plt.figure(1)
x,bins,patches = plt.hist(FoFGals['NormSep'], bins = 40, range = (0,4))

# Calculating probability and density in each bin.
Probs = []
Probs_err = []
Dens = []
Dens_err = []
i = 0
while i <len(x):
    Probs.append(x[i]/len(FoFGals['NormSep']))
    Probs_err.append(np.sqrt(x[i])/len(FoFGals['NormSep']))
    Dens.append(x[i]/(((np.pi*(bins[i+1]**2))-(np.pi*(bins[i]**2)))*len(FoFGals['NormSep'])))
    Dens_err.append(np.sqrt(x[i])/(((np.pi*(bins[i+1]**2))-(np.pi*(bins[i]**2)))*len(FoFGals['NormSep'])))
    i = i + 1

print('Probability r < R_50 =',sum(Probs[0:10]))

i = 0
centres = []
while i <len(bins)-1:
    centre = (bins[i+1]+bins[i])/2
    centres.append(centre)
    i = i + 1
# Probability distribution.
plt.figure(2)
plt.bar(centres,Probs,width = 0.1)
plt.errorbar(centres,Probs,yerr = Probs_err, fmt = 'none', color = 'black', capsize = 2.5)
plt.ylabel('$P$')
plt.xlabel('$\\frac{r}{R_{50}}$', fontsize = 12)
#plt.title(u'Spatial distribution of $M_{*} > 10^{10}$ $M_{\u2609} h^{-1}$ GAMA satellites')
plt.savefig('Spatial_probability.PDF')

# Density distribution.
plt.figure(3)
plt.bar(centres,Dens/Dens[0],width = 0.1)
plt.errorbar(centres,Dens/Dens[0],yerr = Dens_err/Dens[0], fmt = 'none', color = 'black', capsize = 2.5)
plt.ylabel('$\\rho$')
plt.xlabel('$\\frac{r}{R_{50}}$', fontsize = 12)
#plt.title(u'Spatial distribution of $M_{*} > 10^{10}$ $M_{\u2609} h^{-1}$ GAMA satellites')
plt.savefig('Spatial_density.PDF')




