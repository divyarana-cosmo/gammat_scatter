#MCMC Code to produce logmstel, logmh
import numpy as np
import pandas
import matplotlib.pyplot as pl
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import concentration
import mpmath
import emcee
import corner
import argparse
import yaml
from schwimmbad import MPIPool
from scipy.integrate import simps
import sys
sys.path.append('../')
from distort import simshear
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM

#@np.vectorize
def get_sigma_crit_inv(lzred, szred, cc):
    "evaluates the lensing efficency geometrical factor"
    sigm_crit_inv = 0.0*szred
    idx =  szred>lzred   # if sources are in foreground then lensing is zero
    if sum(idx)==0:
        return sigm_crit_inv
    else:
        # some important constants for the sigma crit computations
        gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
        cee = 3e5 #km s^-1
        # sigma_crit_calculations for a given lense-source pair
        sigm_crit_inv[idx] = cc.angular_diameter_distance(lzred[idx]).value * cc.angular_diameter_distance_z1z2(lzred[idx], szred[idx]).value * (1.0 + lzred[idx])**2 * 1.0/cc.angular_diameter_distance(szred[idx]).value
        sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
    return sigm_crit_inv




def model(x, lzred, szred, rbin, cosmo_param):
    logmstel, logmh, conc = x
    H0 , Om0 , Ob0 , Tcmb0 , Neff, sigma8, ns = cosmo_param
    params = dict(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff)
    cc = FlatLambdaCDM(**params)
    hp      = halo(logmh, conc, omg_m=Om0)
    stel    = stellar(logmstel)
    #print(hp.esd_nfw(rbin))
    etan = (hp.esd_nfw(rbin) + stel.esd_pointmass(rbin))*get_sigma_crit_inv(lzred, szred, cc)
    return etan

def lnprior(param):
    logmstel, logmh, conc = param
    if 8<logmstel<16 and 8<logmh<16 and 1<conc<10:
        return 0.0
    else:
        return -np.inf

def lnprob(x, yd, icov, rbin, lzred, szred, cosmo_param):
    lp = lnprior(x)
    if not np.isfinite(lp):
       return -np.inf

    mod = model(x, lzred, szred, rbin, cosmo_param)
    Delta = mod - yd
    chisq = np.sum(Delta**2)*icov
    #chisq = np.dot(Delta, np.dot(icov, Delta))

    print( 'log_Mstel, log_Mh, conc, chisq')
    print( x,chisq)
    res = lp-chisq*0.5

    return res




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    yd    = np.array([])
    lzred = np.array([])
    szred = np.array([])
    rbins = np.array([])
    logmstel = np.array([])
    logmh = np.array([])
    c = np.array([])

    import pandas as pd
    from glob import glob
    flist = glob('../debug/simed_sources.dat_no_shape_noise_proc_*')
    for fil in flist[:-1]:
        df = pd.read_csv(fil, delim_whitespace=1)
        yd    =   np.append(yd, df['etan_obs'])
        rbins =  np.append(rbins, df['proj_sep'])
        lzred = np.append(lzred, df['lzred'])
        szred = np.append(szred, df['szred'])
        logmstel = np.append(logmstel, df['llogmstel'])
        logmh = np.append(logmh, df['llogmh'])
        c = np.append(c, df['lconc'])
    #added hack to cleanup numerical issues
    idx = (yd>0) & (yd<1.0) & (np.random.uniform(size=len(yd)) < 1.0) & (rbins<0.01)
    yd    =  yd[idx]
    rbin  = rbins[idx]
    lzred = lzred[idx]
    szred = szred[idx]
    logmstel = logmstel[idx]
    logmh = logmh[idx]
    c = c[idx]

    print(np.median(logmstel), np.median(logmh), np.median(c))
    exit()

    yerr = 0.3 # putting from maryam thesis
    icov = 1/yerr**2#np.linalg.inv(cov)
    from multiprocessing import Pool
    with Pool() as  pool:
        #if not pool.is_master():
        #    print("ballo")
        #    pool.wait()
        #    sys.exit(0)
        ndim = 3
        nwalkers = 8
        niter = 2000
        np.random.seed(123)
        p_log_Mstel = np.random.uniform(8, 13, nwalkers)
        p_log_Mh = np.random.uniform(12, 16, nwalkers)
        p_conc = np.random.uniform(1, 10, nwalkers)
        p0 = np.transpose([p_log_Mstel, p_log_Mh, p_conc])

        cosmo_param = np.array([config['H0'], config['Om0'], config['Ob0'], config['Tcmb0'], config['Neff'], config['sigma8'], config['ns']])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [yd, icov, rbin, lzred, szred, cosmo_param], pool=pool)

        print("Running burn-in...")
        pos = sampler.run_mcmc(p0, 2000)

        sampler.reset()

        print("Running production...")

        sampler.run_mcmc(pos, niter)

        print("Shape of sampler.chain", np.shape(sampler.chain))
        print("Shape of sampler.flatchain", np.shape(sampler.flatchain))

        print("Execution completed")
        samples = sampler.get_chain()


        np.savetxt("./Chainfile.dat", sampler.flatchain)
        print("Execution completed")


    print(np.median(logmstel), np.median(logmh), np.median(c))

