import sys
sys.path.append('../utils/')
from lensutils import get_re

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

Om0 =   0.25
H0  =   100
params = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', **params)

def get_sigma_crit_inv(lzred, szred):
    "evaluates the lensing efficency geometrical factor"
    sigma_crit_inv = 0.0*szred + 0.0*lzred
    idx =  szred>lzred   # if sources are in foreground then lensing is zero
    if np.isscalar(idx):
        lzred = np.array([lzred])
        szred = np.array([szred])
        idx = np.array([idx])
        sigma_crit_inv = np.array([sigma_crit_inv])
    # some important constants for the sigma crit computations
    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    cee = 3e5 #km s^-1
    # sigma_crit_calculations for a given lense-source pair
    #in physical units
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100, Om0=0.25)
    sigma_crit_inv = cosmo.comoving_distance(lzred).value*(cosmo.comoving_distance(szred).value - cosmo.comoving_distance(lzred).value)
    sigma_crit_inv /=cosmo.comoving_distance(szred).value
    sigma_crit_inv /=(1+lzred)
    sigma_crit_inv[~idx]=0.0 
    sigma_crit_inv = sigma_crit_inv * 4*np.pi*gee*1.0/cee**2
    return sigma_crit_inv




def get_avg_sigma_crit_inv(lzred, zmax, zdiff):
    "assigns redshifts respecting the distribution"
    z0 = 0.9/(2)**0.5
    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
    return 0
    


 

def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return ans

def model(x, zred, rbins):
    logmstel, log_re, logmh, cfac = x
    rbins = rbins*(1+zred)
    # we are evaluating at redshift of 0.3
    lconc   = 1.0#concentration.concentration(10**logmh, '200m', 0.2, model = 'diemer19')
    conc    =   cfac * lconc
    hp          = halo(logmh, conc, omg_m=Om0)
    stel        = stellar(logmstel, log_re=log_re*(1+zred))
    sigma_s    = stel.sigma_deVaucouleurs(rbins) 
    sigma_dm   = hp.sigma_nfw(rbins)         
 
    esd_s       = (1+zred)**2 * stel.esd_deVaucouleurs(rbins)
    esd_dm      = (1+zred)**2 * hp.esd_nfw(rbins)
    return esd_s/1e12, esd_dm/1e12, sigma_s*get_sigma_crit_inv(0.2,0.8), sigma_dm*get_sigma_crit_inv(0.2,0.8)

def lnprior(x):
    logmstel, log_re, logmh, c = x
    if 7.0<=logmstel<=16  and np.log10(0.001)<log_re<np.log10(0.05) and 9<=logmh<=16 and  0<c<20:
        return 0.0 #+ np.log(gauss(c,mean=1.0, sigma=0.16))
    return -np.inf

def lnprob(x, zred, rbins, data, icov):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(2*len(data) + 1)
       return -np.inf,dirt
    mod = model(x, zred, rbins)
    #Delta = (mod[0]*(1+lzred)**2 + mod[1]*(1+lzred)**2)/(1-(mod[2]+mod[3])*(1+lzred)**2) - data
    Delta = mod[0] + mod[1] - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(mod[0],mod[1])
    blob = np.append(blob, chisq)

    print( 'log_Mstel, log_re, log_Mh, cfac, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq):
        return -np.inf, 5*np.ones(2*len(data) + 1)
    res = lp-1.7*chisq*0.5 #added 1.7 to scale micecat area to the whole desi-DR1 area 
    #res = lp-chisq*0.5 #added 3 to scale micecat area to the whole euclid area 

    return res,blob


def runchain(Ntotal,sampler,chainf,blobf,pos, nwalkers):
    print(np.shape(pos))
    blnk=[""];
    fchain=open(chainf,"w");
    fblob=open(blobf,"w");
    iterno=1;
    # Store chainfile and prednfile in the same format as before
    for result in sampler.sample(pos, iterations=Ntotal, store=1):
        posn,probn,staten,blobsn = result;
        for i in range(nwalkers):
            np.savetxt(fchain,posn[i],newline=' ');
            np.savetxt(fchain,[sampler.acceptance_fraction[i],-2.*probn[i]],newline=' ');
            np.savetxt(fblob,blobsn[i],newline=' ');
            np.savetxt(fchain,blnk,fmt='%s');
            np.savetxt(fblob,blnk,fmt='%s');
        print("Iteration number: %d of %d done"%(iterno,Ntotal));
        iterno=iterno+1;
        posnew=result[0];

    fchain.close();
    fblob.close();
    return posnew;


def run_mcmc(logMmin, logMmax, zred, pool):
    njacks = 100
    _rbins, _data, err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/test_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax), unpack=1)
    _cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/cov_test_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax))                  
    
    #running with the first ten rbins
    for Rmin in _rbins[:10]:
        idx     =   (_rbins>Rmin) #& (rbins<0.3)
        rbins   =   _rbins[idx]
        data    =   _data[idx]
        cov     =   np.delete(_cov, ~idx, axis=0)
        cov     =   np.delete(cov, ~idx, axis=1)

        outputdir       = 'output_mcmc' 
        icov            =   np.linalg.inv(cov)
        hartlap_factor  = (njacks - len(data) - 2) * 1.0/(njacks - 1)
        icov            = hartlap_factor*icov

        ndim = 4
        nwalkers = 256
        
        np.random.seed(123)
        p_logmstel  = np.random.uniform(9, 10.5, nwalkers) 
        p_log_re    = np.random.uniform(np.log10(0.001), np.log10(0.05), nwalkers)    
        p_logmh     = np.random.uniform(9, 16, nwalkers)
        p_c         = np.random.uniform(0.1, 20, nwalkers)

        p_0         = np.transpose([p_logmstel, p_log_re, p_logmh, p_c])
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[zred,rbins,data,icov])

        print("Running burn-in...")
        Ntotal = 3000

        burnfile        =   './%s/burnfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_Rmin_%2.2f'%(outputdir, logMmin, logMmax, Rmin*1e3)
        burnpredfile    =   './%s/burnpredfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_Rmin_%2.2f'%(outputdir, logMmin, logMmax, Rmin*1e3)

        pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0, nwalkers)
        sampler.reset()

        print("Running production...")
        Ntotal = 4000
        chainfile = './%s/chainfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_Rmin_%2.2f'%(outputdir, logMmin, logMmax, Rmin*1e3)
        predfile  = './%s/predfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_Rmin_%2.2f'%(outputdir, logMmin, logMmax, Rmin*1e3)

        pos = runchain(Ntotal,sampler, chainfile, predfile, pos, nwalkers)
        print("Execution completed for", logMmin, logMmax, Rmin)
    return 0


if __name__ == "__main__":
    import sys
    Nlens,_logMmin,_logMmax,_lzred,avglogMstel,avglogMh = np.loadtxt('logMstel_bins_seln.dat', unpack=1)
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)                 
 
    for logMmin, logMmax, zred  in zip(_logMmin,_logMmax,_lzred):
        run_mcmc(logMmin, logMmax, zred, pool=pool)
    pool.close()

