import numpy as np
from astropy.cosmology import FlatLambdaCDM
from halopy import halo
cc = FlatLambdaCDM(H0=100, Om0=0.3)

def get_sigma_crit_inv(lzred, szred, cc=cc):
    # some important constants for the sigma crit computations
    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    cee = 3e5 #km s^-1
    # sigma_crit_calculations for a given lense-source pair
    sigm_crit_inv = cc.angular_diameter_distance(lzred).value * cc.angular_diameter_distance_z1z2(lzred, szred).value * (1.0 + lzred)**2 * 1.0/cc.angular_diameter_distance(szred).value
    sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
    print(4*np.pi*gee*1.0/cee**2)
    #sigm_crit_inv = 1e12*sigm_crit_inv #esd's are in pc not in Mpc

    return sigm_crit_inv

M200c = 1e11
c200c = 3


hp = halo(M200c, c200c)
print(hp.r_200/hp.c)
yy = hp.esd(np.ones(5))*get_sigma_crit_inv(0.1,0.8)*1.0/(1 - hp.sigma_nfw_scalar(1.0)*get_sigma_crit_inv(0.1,0.8))
print(yy)



