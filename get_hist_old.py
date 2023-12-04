import numpy as np
import matplotlib.pyplot as plt
import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM




#setting the global cosmology parameters
omg_m = 0.3
params = {'flat': True, 'H0': 100, 'Om0': omg_m, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', **params)


# we have fixed for now but we can later on use a SMHM relation
log_mstel = 10 # in units of h-2 Msun
log_mhalo = 12 # in units of h-1 Msun default M200m
siglogM = 0.2 # in dex
zlens = 0.1


# define a square(can be anullus) around the lense center
nsrcs = 30 # number per arcmin^2   Euclid expectations
Rmax = 1 # comoving radius Mpch-1
Rmin = 0.01 # Mpch-1

# area of square and estimaing the number of sourcer
dcom = cosmo.comovingDistance(z_min=0.0, z_max=zlens, transverse=True)
area = 1.0/dcom**2 * (180*60/np.pi)**2  # str-->arcmin
Nsrc = 10#round(area/nsrcs)
print("number of source galaxies = ", Nsrc)
# randomly sampling around the lens placed at the origin
np.random.seed(123)
pos = np.arctan((1.0 + 0.0*np.random.uniform(high=Rmax, size =int(2*Nsrc)))*1/dcom) * (180/np.pi) # in degrees
#pos = (1.0 + 0.0*np.random.uniform(high=Rmax, size =int(2*Nsrc)))*1/dcom * (180/np.pi) # in degrees
pos = pos.reshape((-1,2))

zsrc = 0.8

#galsim takes 200c
def get_etarr(logmh):
    nfw = galsim.NFWHalo(mass=10**logmh, conc=3, redshift=zlens, omega_m=omg_m, omega_lam=1-omg_m)
    print('scale radius', nfw.M, nfw.c, nfw.rs)
    e1_arr,e2_arr = nfw.getShear((60, 0 ), zsrc, reduced=True) #convert degree angled to arcsec for the galsim computations
    #e1_arr,e2_arr = nfw.getShear((0.0,pos[:,1]*60**2 ), zsrc, reduced=True) #convert degree angled to arcsec for the galsim computations
    #et_arr, ex_arr  = get_et(lra=0.0, ldec=0.0, sra=0.0, sdec=pos[:,1], se1=e1_arr, se2=e2_arr)
    print(e1_arr,e2_arr)
    #print(et_arr, ex_arr)
    return e1_arr, e2_arr

et_arr, ex_arr = get_etarr(logmh)
#plt.subplot(2,2,1)
#for logmh in [12]:
#    et_arr, ex_arr = get_etarr(logmh)
##    plt.hist(np.log10(et_arr), histtype='step', bins=20, density=1, label='logMh = %2.2f'%logmh)
#
#plt.yscale('log')
##plt.xscale('log')
#plt.legend()
#plt.savefig('test.png', dpi=300)



