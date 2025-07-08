import sys
sys.path.append('../utils/')
from lensutils import get_re
import os
import numpy as np
import argparse
import yaml
import emcee
import pandas as pd
from schwimmbad import MPIPool
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import concentration


sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from halopy import halo
from stellarpy import stellar
from distort_com import simshear

#integration
from scipy.integrate import quad

Om0 =   0.25
H0  =   100
params = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', **params)
 
sys.path.append('/home/rana/github_0/gammat_scatter/')
from get_data import lens_select

def get_zlarr(lenstype, logMmin, logMmax, zmin, zmax):
    lensargs = {}
    lensargs['type'] == lenstype
    lensargs['logmstelmin'] =   logMmin
    lensargs['logmstelmax'] =   logMmax
    lensargs['zmin']        =   zmin
    lensargs['zmax']        =   zmax

    lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs)
    return lzred

def nzsrc(z):
    "assigns redshifts respecting the distribution"
    z0 = 0.9/(2)**0.5
    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
    return f(z)
#zmin = 0.0
#zmax = 3


ss = simshear(H0=100, Om0=0.25)

def model(x, rbin, lzredarr, lzredmax, zdiff):
    logmstel, log_re, logmh, cfac = x
    #assigning the concentration from diemer-joyce relation 
    lconc   = concentration.concentration(10**logmh, '200m', np.median(lzredarr), model = 'diemer19')
    esd_s, esd_dm, sigma_s, sigma_dm  = ss._get_esd(logmstel=logmstel, logre=log_re, logmh=logmh, lconc=lconc*cfac, proj_sep=rbin)

    ans = 0.0*rbin
    norm = quad(nsrc, 0,3)[0] # could have done this analytically via feynman technique 
    for ii, rr in enumerate(rbin):
        sigma       =   sigma_s[ii] + sigma_dm[ii]
        integrand   =   lambda xx: np.mean(1/(1-(sigma*ss._get_sigma_crit_inv(lzred=lzredarr, szred=np.array([xx])))))
        ans[ii]     =  quad(integrand, lzredmax+zdiff, 3.0)[0]/norm

    esd = (esd_s + esd_dm)*ans
    return esd/1e12




def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return ans


def lnprior(x):
    logmstel, log_re, logmh, c = x
    if 8<logmstel<14.5  and -4<log_re<0 and 9<=logmh<=16 and  0<c<20:
        return 0.0 + np.log(gauss(c,mean=1.0, sigma=0.16))
    return -np.inf

def lnprob(x, rbins, data, icov, lzredarr, lzredmax, zdiff):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(len(data) + 1)
       return -np.inf,dirt
    esd =   model(x, rbin, lzredarr, lzredmax, zdiff):
    Delta = esd - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(esd)
    blob = np.append(blob, chisq)

    print( 'log_Mstel, log_re, log_Mh, cfac, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq):
        return -np.inf, 5*np.ones(len(data) + 1)
    res = lp- chisq*0.5  
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

if __name__ == "__main__":
    import sys
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)                 
    # for the test case 
    logMmin = 9.5
    logMmax = 11.0
    zmin    = 0.1
    zmax    = 0.4

    njacks = 50
    rbins, data, err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/test_desi_z_0.1_0.4/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/test_desi_z_0.1_0.4/cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax))                  
    
    #running with the first ten rbins
    outputdir       = 'output_mcmc_test_desi_runs' 
    os.system('mkdir -p %s'%outputdir)
    icov            = np.linalg.inv(cov)
    hartlap_factor  = (njacks - len(data) - 2) * 1.0/(njacks - 1)
    icov            = hartlap_factor*icov

    ndim = 4
    nwalkers = 256
    
    np.random.seed(123)
    p_logmstel  = np.random.uniform(9.5, 11.0, nwalkers) 
    p_log_re    = np.random.uniform(np.log10(0.001), np.log10(0.05), nwalkers)    
    p_logmh     = np.random.uniform(9, 16, nwalkers)
    p_c         = np.random.uniform(0.1, 20, nwalkers)

    p_0         = np.transpose([p_logmstel, p_log_re, p_logmh, p_c])

    #getting the lense redshift array
    lenstype    = 'test_desi_runs'    
    lzredarr    = get_zlarr(lenstype, logMmin, logMmax, zmin, zmax):
    lzredmax    = zmax
    zdiff       = 0.0  
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins,data,icov, lzredarr, lzredmax, zdiff])

    print("Running burn-in...")
    Ntotal = 3000
    burnfile        =   './%s/burnfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    burnpredfile    =   './%s/burnpredfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0, nwalkers)
    sampler.reset()

    print("Running production...")
    Ntotal = 4000
    chainfile = './%s/chainfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    predfile  = './%s/predfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, chainfile, predfile, pos, nwalkers)
    print("Execution completed for", logMmin, logMmax, Rmin)
 
    pool.close()


