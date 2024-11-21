import numpy as np

def get_re(logMstel):
    print("log10 of the half-light radius(kpc) assignment takes stellar masses in units of log(Mstel/ Msun)")
    # Mstel in units of Msun
    Mstel   =   10**logMstel
    rp      =   3.8
    Mp      =   10**10.3
    alpha   =   0.09
    beta    =   0.37
    #we use equation 2 from arxiv:1901.05014
    re = rp*(Mstel/Mp)**alpha * (0.5 * (1 + (Mstel/Mp)**6))**((beta-alpha)/6    )
    logre = np.log10(re)
    idx = Mstel<10**9
    logre[idx] = -999
    return logre

