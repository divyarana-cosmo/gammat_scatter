import sys
import os
import numpy as np
import argparse
import yaml
import emcee
import pandas as pd
from schwimmbad import MPIPool
from model import model

def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return ans

def lnprior(x):
    logalpha, logmh, cfac = x
    if -2<logalpha<2 and 9<logmh<16 and  cfac>0:
        return 0.0 + np.log(gauss(cfac,mean=1.0, sigma=0.16))
    return -np.inf

def lnprob(x, rbins, data, icov, mm):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(len(data) + 1)
       return -np.inf,dirt

    import time
    begin = time.time()
    # model prediction
    esd =  mm.esd( x, rbins) 
    print('time_elaspsed', time.time() - begin)
    Delta = esd - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(esd, chisq)
    print( 'logalpha, log_Mh, cfac, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq) :
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
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)                 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # for the test case 
    H0          =  config['H0'] 
    Om0         =  config['Om0'] 
    lenstype    =  config['lenstype']     
    logMmin     =  config['logMmin'] 
    logMmax     =  config['logMmax'] 
    zlmin       =  config['zlmin'] 
    zlmax       =  config['zlmax'] 
    Njacks      =  config['Njacks'] 
    zdiff       =  config['zdiff'] 


    #creating modelling class instance
    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)


    rbins, data, err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/test_desi_z_0.1_0.4/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/test_desi_z_0.1_0.4/cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax))                  
    
    #running with the first ten rbins
    outputdir       = 'output_mcmc_test_desi_runs' 
    os.system('mkdir -p %s'%outputdir)
    icov            = np.linalg.inv(cov)
    hartlap_factor  = (Njacks - len(data) - 2) * 1.0/(Njacks - 1)
    icov            = hartlap_factor*icov

    ndim = 3
    nwalkers = 64
    
    np.random.seed(123)
    p_logalpha  = 0         +np.random.uniform(-0.1,0.1, nwalkers) 
    p_logmh     = 12.428    +np.random.uniform(-0.1,0.1, nwalkers)
    p_c         = 1.0       +np.random.uniform(-0.1,0.1, nwalkers)

    p_0         = np.transpose([p_logalpha, p_logmh, p_c])

   # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins, data, icov, mm])

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


