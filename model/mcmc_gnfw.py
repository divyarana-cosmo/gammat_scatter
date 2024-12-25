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
    logmstel, log_re, logmh, cfac, beta = x
    # we are evaluating at redshift of 0.3
    lconc   = concentration.concentration(10**logmh, '200m', 0.3, model = 'diemer19')
    conc    =   cfac * lconc
    hp          = halo(logmh, conc, omg_m=Om0, beta=beta)
    stel        = stellar(logmstel, log_re=log_re)
    esd_s       = stel.esd_deVaucouleurs(rbins)
    esd_dm      = hp.esd_gnfw(rbins)
    return esd_s/1e12, esd_dm/1e12

def lnprior(x):
    logmstel, log_re, logmh, c, beta = x
    #if 7<=logmstel<=16  and np.log10(0.001)<log_re<np.log10(0.05) and 9<=logmh<=16 and 0<=beta<=5 and 0<c<5:
    if 7<=logmstel<=16  and np.log10(0.001)<log_re<np.log10(0.05) and 9<=logmh<=16 and 0<=beta<=5 and c>0:
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

    print( 'log_Mstel, log_re, log_Mh, c, beta, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq):
        return -np.inf, 5*np.ones(2*len(data) + 1)
    res = lp-1.8*chisq*0.5 #added 3 to scale micecat area to the whole euclid area 

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
    rbins, data , err, xdata, err   =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), unpack=1)
    cov     =   np.loadtxt('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/cov_dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax))                  

    outputdir = 'output_mcmc' 

    #with MPIPool() as pool:
    #    if not pool.is_master():
    #        pool.wait()
    #        print("error")
    #        sys.exit(0)

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)                 
        
    icov    =   np.linalg.inv(cov)
    hartlap_factor = (njacks - len(data) - 2) * 1.0/(njacks - 1)
    icov = hartlap_factor*icov

    
    from scipy.optimize import minimize
    np.random.seed(42)
    nll = lambda *args: -lnprob(*args)[0]
    initial = np.array([(logMmin + logMmax)*0.5, -2, (logMmin + logMmax)*0.5 + 2, 1.0, 1.0]) 
    soln = minimize(nll, initial, args=(rbins, data, icov))

    logmstel, log_re, logmh, c, beta = soln.x    



    ndim = 5
    nwalkers = 256
    
    np.random.seed(123)
    p_logmstel  = logmstel  + 0.01*np.random.uniform(-1, 1, nwalkers) 
    p_log_re    = log_re    + 0.01*np.random.uniform(-1, 1, nwalkers)     
    p_logmh     = logmh     + 0.01*np.random.uniform(-1, 1, nwalkers) 
    p_c         = c         + 0.01*np.random.uniform(-1, 1, nwalkers)  
    p_beta      = beta      + 0.01*np.random.uniform(-1, 1, nwalkers)  

    p_0         = np.transpose([p_logmstel, p_log_re, p_logmh, p_c, p_beta])
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[rbins,data,icov])

    print("Running burn-in...")
    Ntotal = 4000

    burnfile        =   './%s/burnfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    burnpredfile    =   './%s/burnpredfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, burnfile, burnpredfile, p_0)
    sampler.reset()

    print("Running production...")
    Ntotal = 4000
    chainfile = './%s/chainfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)
    predfile  = './%s/predfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(outputdir, logMmin, logMmax)

    pos = runchain(Ntotal,sampler, chainfile, predfile, pos)

    print("Execution completed")
    pool.close()





    #rbin    = np.array([])    
    #lzred   = np.array([])
    #szred   = np.array([])
    #etan    = np.array([])
    
    #cnt = 0
    #for data in dat:
    #    idx     = (0.01<data['proj_sep'].values[:]) & (data['proj_sep'].values[:]<0.5)
    #    data    = data[idx]
    #    rbin    = np.append(rbin, data['proj_sep'])
    #    lzred   = np.append(lzred, data['lzred'])
    #    szred   = np.append(szred, data['szred'])
    #    etan    = np.append(etan,  data['etan_obs'])
    #    cnt+=1
    #    print("chunk number", cnt)


    #data    = pd.read_csv(outputfilename, delim_whitespace=1, usecols=['proj_sep', 'lzred', 'szred', 'etan_obs'])
    #idx     = (0.01<data['proj_sep'].values[:]) & (data['proj_sep'].values[:]<0.5)
    #data    = data[idx]
    #rbin    = data['proj_sep']
    #lzred   = data['lzred']
    #szred   = data['szred']

    #data    = data['etan_obs']
    #gamma_s, gamma_dm, kappa_s, kappa_dm =  ss._get_esd(logmstel, logmh, lconc, Rarr)
    #spl_gamma_s     =   interp1d(np.log10(Rarr), np.log10(gamma_s),  kind='cubic')      
    #spl_gamma_dm    =   interp1d(np.log10(Rarr), np.log10(gamma_dm), kind='cubic')
    #spl_kappa_s     =   interp1d(np.log10(Rarr), np.log10(kappa_s),  kind='cubic')
    #spl_kappa_dm    =   interp1d(np.log10(Rarr), np.log10(kappa_dm), kind='cubic')
    #print("interpolation done")


    #hp   = halo(logmh, lconc, omg_m=Om0)
    #stel = stellar(logmstel)

    #gamma_s    = stel.esd_pointmass(rbin)     * inv_crit_arr
    #gamma_dm   = hp.esd_nfw(rbin)             * inv_crit_arr
    #kappa_s    = stel.sigma_pointmass(rbin)   * inv_crit_arr
    #kappa_dm   = hp.sigma_nfw(rbin)           * inv_crit_arr

#def model(x, rbin, lzred, szred):
#    global cnt, inv_crit_arr
#    logmstel, logmh, beta = x
#    lconc    = concentration.concentration(10**logmh, '200m', np.median(lzred), model = 'diemer19')
#    #logmstel, logmh, lconc = x
#    if cnt==0:
#        inv_crit_arr = ss._get_sigma_crit_inv(lzred, szred)
#        cnt+=1
#        
#    print("val of cnt", cnt)
#    hp             = halo(logmh, lconc, omg_m=Om0, beta=beta)
#
#    log_re      = (0.774 + 0.977 *(np.log10(10**logmh / 0.7) - 11.4)) #check arxiv:1811.04934
#    log_re      = np.log10(10**log_re * 0.7/1e3) #h-1 kpc to h-1 Mpc
#
#    stel        = stellar(logmstel, log_re=log_re)
#    esd_s       = stel.esd_deVaucouleurs(rbin)
#    esd_dm      = hp.esd_gnfw(rbin)
#    sigma_s     = stel.sigma_deVaucouleurs(rbin)
#    sigma_dm    = hp.sigma_gnfw(rbin)
#
#    #Rarr = np.logspace(-3,1,100)
#    #gamma_s, gamma_dm, kappa_s, kappa_dm =  ss._get_esd(logmstel, logmh, lconc, rbin)
#    gamma_s     =   esd_s  * inv_crit_arr
#    gamma_dm    =   esd_dm * inv_crit_arr
#    kappa_s     =   sigma_s  * inv_crit_arr
#    if np.any(np.isnan(kappa_s)):
#        kappa_s[np.isnan(kappa_s)] = 0.0
#    kappa_dm    =   sigma_dm * inv_crit_arr
#
#
#    gtan = (gamma_s + gamma_dm)/(1 - (kappa_s + kappa_dm))
#    return gtan


