import sys
import os
import numpy as np
import argparse
import yaml
import emcee
import pandas as pd
from model import model
from colossus.cosmology import cosmology
from colossus.halo import concentration
params = {'flat': True, 'H0': 100, 'Om0': 0.319, 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
cosmology.addCosmology('myCosmo', **params)
cosmo = cosmology.setCosmology('myCosmo')
import matplotlib.pyplot as plt 


def lnprob(x, rbins, data, icov, mm, median_lzred, invsigc):
    import time
    begin = time.time()
    
    # model prediction
    ## setting up the splines
    #delta_sigma, sigma = mm.set_esd_spl(x, rbins, lzred=median_lzred)
    ## change Mpc to pc units as the invsigc is in pc units
    #sigma       =   sigma/1e12
    #delta_sigma =   delta_sigma/1e12

    ##sigma       = (x[0] * mm.sigma_s + mm.sigma_dm)/1e12
    ##delta_sigma = (x[0] * mm.esd_s + mm.esd_dm)/1e12
    #
    ## correction for the reduced shear

    #esd = delta_sigma / (1 - sigma*invsigc )

    _, esd = mm.set_esd_spl(x, rbins, lzred=median_lzred, reduced=False)
    print('time_elaspsed', time.time() - begin)
    Delta = data - esd
    return Delta



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--logMmin", help="minimum stellar mass", default=9.5, type=float)
    parser.add_argument("--logMmax", help="maximum stellar mass", default=11.0, type=float)
    parser.add_argument("--seed", help="seed", default=123, type=int)



    # for the test case 
    H0          =  100      
    Om0         =  0.319    
    lenstype    =  'desi'     
    logMmin     =  11.4
    logMmax     =  11.5 
    seed        =   19
    
    zlmin       =  0.0
    zlmax       =  0.4
    Njacks      =  100
    zdiff       =  0.2




    #creating modelling class instance
    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

    rbins, data, err, xdata, err, stelesd, invsigc, invsigcsq    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(logMmin, logMmax, seed, logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%( logMmin, logMmax, seed,logMmin, logMmax))                  
    cov     =   np.diag(np.diag(cov))

    # removing the first bin
    idx         = (rbins<0.1) & (rbins>0.01)
    rbins       = rbins[idx]
    data        = data[idx]
    stelesd     = stelesd[idx]
    invsigc     = invsigc[idx]
    invsigcsq   = invsigcsq[idx]
    cov         = np.delete(cov, ~idx, axis=0)
    cov         = np.delete(cov, ~idx, axis=1)


    fpath = './precompute/'+'pzl_%s_%s.dat'%(logMmin, logMmax)
    if not os.path.exists(fpath):
        print('please run the precompute first')
        exit()

    zlbins, pzl, mean_redshift = np.loadtxt(fpath, unpack=1)
    median_lzred = mean_redshift[0]/2
    print(median_lzred)
    zlbins, pzl, mean_redshift = np.loadtxt(fpath, unpack=1)
    from scipy.interpolate import interp1d
    func = interp1d(np.cumsum(pzl), zlbins)
    median_lzred = func(0.5)
 
    #running with the first ten rbins
    outputdir       = 'output_mcmc_desi_runs' 
    os.system('mkdir -p %s'%outputdir)
    icov            = np.linalg.inv(cov)
    hartlap_factor  = (Njacks - len(data) - 2) * 1.0/(Njacks - 1)
    icov            = hartlap_factor*icov

    x         = np.array([ 1.0,  13.71779508, 6.69845559]).T

    delta = lnprob(x, rbins, data, icov, mm, median_lzred, invsigc)
    
    plt.subplot(2,2,1)
    plt.plot(rbins, delta/np.diag(cov)**0.5)
    plt.xscale('log')
    plt.savefig('fit_by_eye.pdf')


