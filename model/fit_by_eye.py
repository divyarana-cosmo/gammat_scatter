import sys
sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from halopy import halo
from stellarpy import stellar
import numpy as np
import argparse
import yaml
import emcee
import pandas as pd
from schwimmbad import MPIPool
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import concentration
import matplotlib.pyplot as plt

Om0 =   0.25
H0  =   100
params = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', **params)


def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return ans

def model(x, rbins):
    logmstel, log_re, logmh, cfac = x
    # we are evaluating at redshift of 0.3
    lconc   = concentration.concentration(10**logmh, '200m', 0.3, model = 'diemer19')
    conc    =   cfac * lconc
    hp          = halo(logmh, conc, omg_m=Om0)
    stel        = stellar(logmstel, log_re=log_re)
    esd_s       = stel.esd_deVaucouleurs(rbins)
    esd_dm      = hp.esd_nfw(rbins)
    return esd_s/1e12, esd_dm/1e12

if __name__ == "__main__":
    import sys
    logMmin =   float(sys.argv[1])
    logMmax =   float(sys.argv[2])

    njacks = 100
    rbins, data, err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/cov_dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax))                  

#    _nanfix = sum(~np.isfinite(data))
#    rbins = rbins[_nanfix:]
#    data  = data[_nanfix:]
#    
#    cov   = cov[_nanfix:,_nanfix:]

    plt.subplot(2,2,1)
    plt.errorbar(rbins, data, yerr=np.diag(cov)**0.5, fmt='.', capsize=3)

    logmstel    = 13
    log_re      = 3e-3
    logmh       = 11.5
    cfac        =  4

    x = [logmstel, log_re, logmh, cfac]
    esd_s, esd_dm = model(x, rbins)
    plt.plot(rbins, esd_s) 
    plt.plot(rbins, esd_dm) 
    plt.plot(rbins, esd_s + esd_dm) 

    plt.xlabel("R")
    plt.ylabel("$\Delta \Sigma$")

    plt.xscale('log')
    plt.yscale('log')

    plt.savefig("test.png", dpi=300)

