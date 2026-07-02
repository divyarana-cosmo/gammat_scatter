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

#def lnprior(x):
#    logalpha, logmh, cfac = x
#    if -2<logalpha<2 and 9<logmh<16 and  cfac>0:
#        return 0.0 + np.log(gauss(cfac,mean=1.0, sigma=0.16))
#    return -np.inf

def lnprior(x):
    logalpha, logmh, cfac = x
    #logmh, cfac = x
    if 0.1<(10**logalpha)<5 and 9<logmh<16 and  0<cfac<20:
    #if  9<logmh<16 and  0<cfac<20:
        return 0.0 #+ np.log(gauss(cfac,mean=1.0, sigma=0.16))
    return -np.inf



def lnprob(x, rbins, data, icov, mm):
    lp = 0 
    for ii in range(len(mm)):
        xx = [x[0], x[2*ii+1], x[2*ii+2]]
        lp += lnprior(xx)

    if not np.isfinite(lp):
       dirt = 5*np.ones(int(len(data)*len(rbins)) + 1)
       return -np.inf,dirt

    import time
    begin = time.time()
    # model prediction
    chisq   =   0
    pred    =   np.zeros(len(rbins)*len(mm))   
    for ii in range(len(mm)):
        xx = [x[0], x[2*ii+1], x[2*ii+2]]
        _esd =  mm[ii].esd(xx, rbins) 
        print('time_elaspsed', time.time() - begin)
        Delta = _esd - data[ii]
        chisq += np.dot(Delta, np.dot(icov[ii], Delta))
        x0  =   len(rbins)*ii; x1  =   len(rbins)*(ii +1)
        pred[int(x0):int(x1)] = _esd 

    blob = np.append(pred, chisq)
    print(np.shape(blob))
    print( 'logalpha, log_Mh, cfac, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq) :
        return -np.inf, 5*np.ones(len(pred) + 1)
    res = lp- chisq*0.5  
    return res,blob


def runchain(Ntotal,sampler,chainf,blobf,pos, nwalkers):
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
    parser.add_argument("--logMmin", help="minimum stellar mass", default=9.5, type=float)
    parser.add_argument("--logMmax", help="maximum stellar mass", default=11.0, type=float)
    parser.add_argument("--seed", help="seed", default=123, type=int)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # for the test case 
    H0          =  config['H0'] 
    Om0         =  config['Om0'] 
    lenstype    =  config['lenstype']     
    zlmin       =  config['zlmin'] 
    zlmax       =  config['zlmax'] 
    Njacks      =  config['Njacks'] 
    zdiff       =  config['zdiff'] 

    
    
    #creating modelling class instance
    logMstelmin = 9.5 + 0.1*np.arange(11)
    logMstelmax = 9.5 + 0.1*np.arange(1,12)
    
    # defining the dictionaries for the model instances, data and inverse covariance
    mm = {}; data = {}; icov = {}
    for ss,(logMmin, logMmax) in enumerate(zip(logMstelmin, logMstelmax)):
        mm[ss] = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

        rbins, data[ss], err, xdata, err    =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/desi_z_0.1_0.4_seed_%d/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(ss, logMmin, logMmax), unpack=1)
        cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/desi_z_0.1_0.4_seed_%d/cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(ss, logMmin, logMmax))                  
        cov     =   np.diag(np.diag(cov))
        _icov               = np.linalg.inv(cov*100)
        hartlap_factor      = (Njacks - len(rbins) - 2) * 1.0/(Njacks - 1)
        icov[ss]            = hartlap_factor*_icov

    #logMmin     =  float(args.logMmin)#config['logMmin'] 
    #logMmax     =  float(args.logMmax)#config['logMmax'] 
 
    ##taking above 0.004 h-1 Mpc
    #idx = rbins>0.01
    #rbins = rbins[idx]
    #data  = data[idx]
    #cov   = np.delete(cov, ~idx, axis=0)
    #cov   = np.delete(cov, ~idx, axis=1)
    
    
    #running with the first ten rbins
    outputdir       = 'output_mcmc_desi_runs' 
    os.system('mkdir -p %s'%outputdir)
    ## for full
    #icov = 1.5*icov

    ndim = 23
    nwalkers = 128
    initpos = np.zeros(( nwalkers, ndim))
    


    # setting up the initial walker position
    initpos[:,0]  = np.log10(np.random.uniform(0.2,4, nwalkers) )
    #initpos = np.array([p_logalpha])
    for ss,(logMmin, logMmax) in enumerate(zip(logMstelmin, logMstelmax)):
        rng = np.random.default_rng(ss*101010+123)
        p_logmh     =  rng.uniform(10,14, nwalkers)
        p_c         =  rng.uniform(0,20, nwalkers)
        initpos[:,2*ss+1]     =   p_logmh
        initpos[:,2*ss+2]     =   p_c

    p_0         = initpos
    print(np.all(np.isfinite(p_0)))

   # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins, data, icov, mm])

    print("Running burn-in...")
    Ntotal = 2000

    burnfile        =   './%s/burnfile_nfw_all_lmstelbin.dat_full_desixeuclid_seed_joint_analysis'%(outputdir)
    burnpredfile    =   './%s/burnpredfile_nfw_all_lmstelbin.dat_full_desixeuclid_seed_joint_analysis'%(outputdir)


    pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0, nwalkers)
    sampler.reset()

    print("Running production...")
    Ntotal = 4000
    chainfile = './%s/chainfile_nfw_all_lmstelbin.dat_full_desixeuclid_seed_joint_analysis'%(outputdir)
    predfile  = './%s/predfile_nfw_all_lmstelbin.dat_full_desixeuclid_seed_joint_analysis'%(outputdir)

    pos = runchain(Ntotal, sampler, chainfile, predfile, pos, nwalkers)
    print("Execution completed for", logMmin, logMmax, Rmin)
 
    pool.close()


