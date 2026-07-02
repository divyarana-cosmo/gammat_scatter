import sys
import os
import numpy as np
import argparse
import yaml
import emcee
import pandas as pd
from schwimmbad import MPIPool
from model_gnfw import model
from colossus.cosmology import cosmology
from colossus.halo import concentration
params = {'flat': True, 'H0': 100, 'Om0': 0.319, 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
cosmology.addCosmology('myCosmo', **params)
cosmo = cosmology.setCosmology('myCosmo')
 
import time

def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return ans

def lnprior(x):
    alpha, logmh, beta = x
    if 0.1<alpha<5 and 1e9<10**logmh<1e16  and 0<beta<3:
        return 0.0 
    return -np.inf



def lnprob(x, rbins, data, icov, mm, median_lzred, invsigc):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(len(data) + 1)
       return -np.inf,dirt
    
    begin = time.time()
    
    lconc  =   concentration.concentration(10**x[1], '200m', median_lzred, model = 'diemer19')
    x = np.array([x[0],x[1],lconc,x[2]])

    # model prediction
    _, esd = mm.set_esd_spl(x, rbins, lzred=median_lzred)


    print('time_elaspsed', time.time() - begin)
    Delta = esd - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(esd, chisq)
    print( 'alpha, log_Mh, cfac, beta, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq) :
        return -np.inf, 5*np.ones(len(data) + 1)
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
    parser.add_argument("--Rpmin", help="minimum projected radius to be used for fitting in units of half-light radius", default=3.0, type=float)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # for the test case 
    H0          =  config['H0'] 
    Om0         =  config['Om0'] 
    lenstype    =  config['lenstype']     
    logMmin     =  float(args.logMmin)#config['logMmin'] 
    logMmax     =  float(args.logMmax)#config['logMmax'] 
    zlmin       =  config['zlmin'] 
    zlmax       =  config['zlmax'] 
    Njacks      =  config['Njacks'] 
    zdiff       =  config['zdiff'] 


    #creating modelling class instance
    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

    outdir = '../output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d'%(logMmin, logMmax, args.seed)
    rbins, data, err, xdata, err, stelesd, invsigc, invsigcsq = np.loadtxt(outdir + '_ovp_100/' + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax), unpack=1)
    cov  = np.loadtxt(outdir + '_ovp_1/' + 'cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMmin, logMmax))
    cov     =   np.diag(np.diag(cov))
 
    #fixing the minimum projected separation to be atleast 2*Re
    lens_metadata   = pd.read_csv('/net/dobbe/data2/github/gammat_scatter/lens_sample.dat', delim_whitespace=1)
    idx             = (lens_metadata['logMmin']==logMmin) &(lens_metadata['logMmax']==logMmax)
    Rmin            = lens_metadata['re_50'][idx] * args.Rpmin/1e3 # 1e3 factor to convert Mpc to kpc
    print(Rmin.values[:])
    # removing the first bin
    idx         = (rbins<0.1) & (rbins>Rmin.values[:])
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
    from scipy.interpolate import interp1d
    func = interp1d(np.cumsum(pzl), zlbins)
    median_lzred = func(0.5)
   
    #running with the first ten rbins
    outputdir       = 'output_mcmc_desi_runs' 
    os.system('mkdir -p %s'%outputdir)
    icov            = np.linalg.inv(cov)
    hartlap_factor  = (Njacks - len(data) - 2) * 1.0/(Njacks - 1)
    icov            = hartlap_factor*icov

    ndim = 3
    nwalkers = 32
    
    np.random.seed(123)

    p_alpha     =   np.random.uniform(0.9,    1.1, nwalkers) 
    p_logmh     =   np.log10(np.random.uniform(1e9, 1e16, nwalkers))
    p_beta      =   np.random.uniform(0.9, 1.1, nwalkers)

    p_0         = np.transpose([p_alpha, p_logmh, p_beta])

   # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins, data, icov, mm, median_lzred, invsigc])

    print("Running burn-in...")
    Ntotal = 4000

    burnfile        =   './%s/burnfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_seed_%d_fix_conc_gnfw_14kdr3'%(outputdir, logMmin, logMmax, args.seed)
    burnpredfile    =   './%s/burnpredfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_seed_%d_fix_conc_gnfw_14kdr3'%(outputdir, logMmin, logMmax, args.seed)


    pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0, nwalkers)
    sampler.reset()

    print("Running production...")
    Ntotal = 8000
    chainfile = './%s/chainfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_seed_%d_fix_conc_gnfw_14kdr3'%(outputdir, logMmin, logMmax, args.seed)
    predfile  = './%s/predfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_seed_%d_fix_conc_gnfw_14kdr3'%(outputdir, logMmin, logMmax, args.seed)


    pos = runchain(Ntotal,sampler, chainfile, predfile, pos, nwalkers)
    print("Execution completed for", logMmin, logMmax, Rmin)
 
    pool.close()

    ## setting up the splines
    #delta_sigma, sigma = mm.set_esd_spl(x, rbins, lzred=median_lzred)
    ## change Mpc to pc units as the invsigc is in pc units
    #sigma       =   sigma/1e12
    #delta_sigma =   delta_sigma/1e12

    ##sigma       = (x[0] * mm.sigma_s + mm.sigma_dm)/1e12
    ##delta_sigma = (x[0] * mm.esd_s + mm.esd_dm)/1e12
    #
    ## correction for the reduced shear
    #esd = delta_sigma * (1 + sigma*invsigc )

