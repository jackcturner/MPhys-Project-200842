#Importing required modules.
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy import stats
from seaborn import kdeplot

import warnings # Fitting throws up unimportant warnings related to logs. Remvoing these. 
warnings.filterwarnings("ignore")

#====================================================
# Importing data and creating volume limitied sample.
#====================================================

# Reading stellar mass data and converting to pandas dataframe.
hdul = fits.open('StellarMassesv19.fits')
sm = hdul[1].data
hdul.close()
sm = Table(sm)
sm = sm.to_pandas()

# Reading galaxy data and converting to pandas dataframe.
hdul = fits.open('G3CGalv10.fits')
gal = hdul[1].data
hdul.close()
gal = Table(gal)
gal = gal.to_pandas()

# Joining the two tables.
gals2 = gals = gal.merge(sm, on='CATAID', suffixes = ('_gal','_sm'))

# Aperture correction.

# Selecting realistic fluxscale values.
gals = gals[gals['fluxscale']>0.5]
gals = gals[gals['fluxscale']<2]

# Applying correction.
gals['absmag_r'] = gals['absmag_r'] - (2.5*np.log10(gals['fluxscale']))
gals['logmstar'] = (gals['logmstar'] + np.log10(gals['fluxscale']))-(2*np.log10(1/0.7))

gals4 = gals3 = gals

# Plotting distribution of galaxies.
plt.figure(1)
plt.scatter(gals['Z_sm'], gals['absmag_r'],s=2)
#kdeplot(x=gals['Z_sm'],y = gals['absmag_r'], color = 'black',levels =5)
plt.ylim(-10,-25)
plt.xlim(0,0.6)
plt.xlabel('z')
plt.ylabel('$M_{r}$')
#plt.title('GAMAII DR4 Galaxies')
plt.savefig('GAMA_galaxies.PDF')
plt.show()

# Applying volume limiting cuts.
gals = gals[gals['absmag_r']<-20]
gals = gals[gals['absmag_r']>-23]

gals = gals[gals['Z_sm']<0.2]
gals = gals[gals['Z_sm']>0.003]

# Selecting galaxies that belong to a group.
gals = gals[gals['GroupID'] != 0]

# Plotting volume limited sample.
plt.figure(2)
plt.scatter(gals['Z_sm'],gals['absmag_r'],s=2)
#kdeplot(x = gals['Z_sm'], y = gals['absmag_r'], color = 'black', levels =5)
plt.ylim(-20,-23)
plt.xlim(0,0.2)
plt.xlabel('z')
plt.ylabel('$M_{r}$')
#plt.title('GAMAII DR4 volume limited sample')
plt.savefig('GAMA_volume_limited.PNG')
plt.show()

plt.figure(3)
plt.hist(gals['logmstar'], bins = 'fd')
plt.axvline(10, color = 'black',linestyle = '--')
plt.xlabel('$\log_{10}\\frac{M^{*}h}{M_{\u2609}}$', fontsize = 9)
plt.ylabel('Frequency')
plt.savefig('Mass_Distribution.PDF')
plt.show()

# Sorting galaxies by their FoF group ID.
gals.sort_values('GroupID')

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

# The number of M* > 10^10 galaxies in each group. 
NFoF = []
for i in Groups:
        j = gals2[gals2['GroupID'] == i]
        j = j[j['logmstar'] > 10]
        NFoF.append(len(j))

# Reading FoF group data.
hdul = fits.open('G3CFoFGroupv10.fits')
FoF = hdul[1].data
hdul.close()
FoF = Table(FoF)
FoF = FoF.to_pandas()

# Selecting those FoF groups with 5 galaxies in the volume limited sample.
df = pd.DataFrame(Groups)
df = df.rename(columns = {0:'GroupID'})
FoF = df.merge(FoF, on='GroupID')

# Adding column for number of galaxies in a group with M* > 10^10.
FoF['NFoF10'] = NFoF

# Adding a column for the number of satellites.
FoF['Nsat'] = FoF['NFoF10'] - 1

#=====================
# HOD Parameterisation
#=====================

# Adding estimate for error on halo mass.
FoF['MassA_err'] = 1 - 0.43*np.log10(FoF['Nfof'])
FoF.loc[FoF['Nfof'] > 50, 'MassA_err'] = 0.27

# Selecting only groups that are fully within the sample.
FoF = FoF[FoF['GroupEdge'] == 1]

# Number of groups that satisfy all our conditions.
print('Number of groups for consideration:',len(FoF))

# Converting to log10 mass.
FoF['MassA'] = np.log10(FoF['MassA'])
FoF['MassAfunc'] = np.log10(FoF['MassAfunc'])

# Calculation scaling relation mass.
FoF['ScaledMass'] = (0.81)*((FoF['LumB']/(10**11.5))**(1.01))
FoF['ScaledMass'] = FoF['ScaledMass']*(10**14)
FoF['ScaledMass'] = np.log10(FoF['ScaledMass'])
FoF['ScaledMass_err'] = 0.1

FoF2 = FoF

# Redshift cuts to be considered.
l = [0.03,0.14]
u = [0.2,0.16]

# Mass estimates to be considered.
Mdefs = ['MassA','MassAfunc','ScaledMass']
Merrs = ['MassA_err','MassA_err','ScaledMass_err'] 


# Iterating over each mass estimate and redshift distribution.
y = 0
while y<len(Mdefs):
    
    print('Mass estimate:',Mdefs[y])
 
    Rchi = []
    parameters = []
    j = 0
    while j<len(l):
        # Redshift cuts.
        FoF = FoF[FoF['IterCenZ']<u[j]]
        FoF = FoF[FoF['IterCenZ']>l[j]]

        FoF.index = np.arange(0,len(FoF))
    
        # Mass estimates.
        M = FoF[Mdefs[y]]; M_err = FoF[Merrs[y]]

        #Function determined using HOD.
        def TNGHOD(x):
            return (0.5*(1+erf((x-11.701)/0.194)))+(((x - 11.3)/1.4)**4.4)

        m = np.linspace(12.5,15.3,500)

        # Determining a Chi2 for the TNG function based on the GAMA groups.
        #Requires numerical solution due to error on M not N.

        def TNGHOD2(x,N):
            return (0.5*(1+erf((x-11.701)/0.194)))+(((x - 11.3)/1.4)**4.4)-N
            
        # TNG HOD-GAMA data Chi2 calculation.
        Chi2 = 0
        i = 0
        while i < len(FoF):
            C = ((M[i] - fsolve(TNGHOD2, x0 = M[i], args = FoF['NFoF10'][i]))**2)/(M_err[i]**2)
            Chi2 = Chi2 + C
            i = i + 1

        print('Reduced Chi2 of TNG fit to GAMA data (z'+str(j)+') = '+str(float(Chi2)/(len(FoF))))
    
        # Fitting the satellite part of the function to GAMA data for comparision with TNG
        def TNGSat(x,b,c):
            return (1.4*np.exp(np.log(x)/b)) + c

        popt,pcov = curve_fit(TNGSat, FoF['Nsat'], M, sigma = M_err, p0 = [4.3,11.32])
        
        # Chi2 calculation.
        Chi2 = 0
        i = 0
        while i < len(FoF):
            C = ((M[i] - TNGSat(FoF['Nsat'][i],popt[0],popt[1]))**2)/(M_err[i]**2)
            Chi2 = Chi2 + C
            i = i + 1

        print('Reduced Chi2 of fit (z'+str(j)+') = '+str(Chi2/(len(FoF)-2)))
    
        Rchi.append(round(float(Chi2)/len(FoF),2))
    
        parameters.append(popt[0])
        parameters.append(popt[1])
        FoF = FoF2
        j = j + 1

    M = FoF[Mdefs[y]]; M_err = FoF[Merrs[y]]

    def TNGSat2(x,a,b,c):
        return ((x-a)/b)**c

    n = np.linspace(1,120,500)

    plt.figure(4+y)
    plt.scatter(x = M, y = FoF['Nsat'], c = FoF['IterCenZ'], label = 'GAMA satellites', s=10)
    plt.plot(TNGSat(n,parameters[0],parameters[1]),n, color = 'blue',label = 'GAMA HOD satellites (0.03 < z < 0.2)')
    plt.plot(TNGSat(n,parameters[2],parameters[3]),n, color = 'orange',label = 'GAMA HOD satellites (0.14 < z < 0.16)')

    plt.plot(m, TNGSat2(m,11.3,1.4,4.4), color = 'red',label = 'TNG HOD satellites')

    plt.yscale('log')
    plt.xlabel('$\log_{10}\\frac{M_{halo}h}{M_\u2609}$')
    plt.ylabel('$N_{sat}$')
    #plt.title(u'Number of $M_{*} > 10^{10}$ $M_{\u2609}h^{-1}$ satellites in GAMA groups')
    plt.xlim(11.5,15.5)
    plt.legend(loc=2,frameon = False)
    plt.colorbar(label="z", orientation="vertical")
    plt.savefig('GAMA_Fits'+str(y)+'.PDF')

    y = y + 1
