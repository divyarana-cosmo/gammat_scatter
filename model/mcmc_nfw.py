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

def lnprior(x):
    logmstel, log_re, logmh, c = x
    if 7.0<=logmstel<=16  and np.log10(0.001)<log_re<np.log10(0.05) and 9<=logmh<=16 and  c>0:
        return 0.0 + np.log(gauss(c,mean=1.0, sigma=0.16))
    return -np.inf

def lnprob(x, rbins, data, icov):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(2*len(data) + 1)
       return -np.inf,dirt
    mod = model(x, rbins)
    Delta = (mod[0] + mod[1]) - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(mod[0],mod[1])
    blob = np.append(blob, chisq)

    print( 'log_Mstel, log_re, log_Mh, cfac, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq):
        return -np.inf, 5*np.ones(2*len(data) + 1)
    res = lp-1.8*chisq*0.5 #added 3 to scale micecat area to the whole euclid area 
    #res = lp-chisq*0.5 #added 3 to scale micecat area to the whole euclid area 

    return res,blob


def runchain(Ntotal,sampler,chainf,blobf,pos):
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
    logMmin =   float(sys.argv[1])
    logMmax =   float(sys.argv[2])

    njacks = 100
    rbins, data, err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/cov_dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax))                  

    outputdir = 'output_mcmc' 

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)                 
    
    icov    =   np.linalg.inv(cov)
    hartlap_factor = (njacks - len(data) - 2) * 1.0/(njacks - 1)
    icov = hartlap_factor*icov

    ndim = 4
    nwalkers = 256
    
    np.random.seed(123)
    p_logmstel  = (logMmin + logMmax)*0.5 + 0.01*np.random.uniform(-1, 1, nwalkers) 
    p_log_re    = np.random.uniform(np.log10(0.001), np.log10(0.05), nwalkers)    
    p_logmh     = np.random.uniform(9, 16, nwalkers)
    p_c         = np.random.uniform(0.8, 1.2, nwalkers)

    p_0         = np.transpose([p_logmstel, p_log_re, p_logmh, p_c])
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins,data,icov])

    print("Running burn-in...")
    Ntotal = 4000

    burnfile        =   './%s/burnfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    burnpredfile    =   './%s/burnpredfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0)
    sampler.reset()

    print("Running production...")
    Ntotal = 4000
    chainfile = './%s/chainfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    predfile  = './%s/predfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, chainfile, predfile, pos)

    print("Execution completed")
    pool.close()

