# updated the assign sizes given the cosmological parameter.
import numpy as np
from astropy.cosmology import FlatLambdaCDM
H0=67.7
cc_molwa_2019 = FlatLambdaCDM(H0=H0, Om0=0.308)
hval = H0/100
def get_re(logMstel, lzred, cosmo):
    """assigns the half-light radius given the stellar mass and the cosmology instance"""
    print("log10 of the half-light radius(kpc) assignment takes stellar masses in units of log(Mstel/ Msun)")
    # Mstel in units of h-1 Msun
    Mstel   =   10**logMstel/hval
    rp      =   3.8
    Mp      =   10**10.3
    alpha   =   0.09
    beta    =   0.37
    # we use equation 2 from arxiv:1901.05014
    # we convert Kpc to the Mpc
    re = rp*(Mstel/Mp)**alpha * (0.5 * (1 + (Mstel/Mp)**6))**((beta-alpha)/6)/1e3
    theta = re/cc_molwa_2019.angular_diameter_distance(lzred).value
    logre = np.log10(theta*cosmo.angular_diameter_distance(lzred).value)
    idx = Mstel<10**9
    logre[idx] = -999
    return logre

