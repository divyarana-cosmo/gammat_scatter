import sys
sys.path.append('/home/rana/github/gammat_scatter/src/')
from distort import simshear
from halopy import halo
from stellarpy import stellar
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import emcee
import pandas as pd
from schwimmbad import MPIPool
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import concentration

Om0 =   0.308
H0  =   67.7
params = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', **params)




def gauss(x,mean,sigma):
    ans = np.exp(-(x-mean)**2/(2*sigma**2))
    ans = ans/(sigma * (2*np.pi)**0.5)
    return

def model(x, rbin):
    logmstel, log_re, logmh, cfac, beta = x
    lconc   = concentration.concentration(10**logmh, '200m', np.median(lzred), model = 'diemer19')
    conc    =   cfac * lconc
    hp          = halo(logmh, conc, omg_m=Om0, beta=beta)
    stel        = stellar(logmstel, log_re=log_re)
    esd_s       = stel.esd_deVaucouleurs(rbin)
    esd_dm      = hp.esd_gnfw(rbin)
    return esd_s, esd_dm

def lnprior(x):
    logmstel, log_re, logmh, c, beta = x
    if 8<=logmstel<=13  and 0.001<log_re<0.2 and 9<=logmh<=16 and 0<=beta<=5:
        return 0.0 + np.log(gauss(c,mean=1.0, sigma=0.2))
    return -np.inf

def lnprob(x, rbin, data, cov):
    lp = lnprior(x)
    if not np.isfinite(lp):
       dirt = 5*np.ones(2*len(data))
       return -np.inf,dirt
    mod = model(x, rbin)
    Delta = (mod[0] + mod[1]) - data
    chisq = np.dot(Delta, np.dot(icov, Delta))

    blob = np.append(mod[0],mod[1])

    print( 'log_Mstel, log_Mh, c, beta, chisq')
    print( x,chisq)
    if chisq<0 or np.isnan(chisq):
        return -np.inf
    res = lp-chisq*0.5 #added 3 to scale micecat area to the whole euclid area 

    return res,blob


def runchain(Ntotal,sampler,chainf,blobf,pos):
    blnk=[""];
    fchain=open(chainf,"w");
    fblob=open(blobf,"w");
    iterno=1;
    # Store chainfile and prednfile in the same format as before
    for result in sampler.sample(pos, iterations=Ntotal, store=0):
        posn,probn,staten = result;
        posn,probn,staten,blobsn = result;
        posn,probn,staten = result;
        for i in range(nwalkers):
            np.savetxt(fchain,posn[i],newline=' ');
            np.savetxt(fchain,[-2.*probn[i]],newline=' ');
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
    outputfilename = sys.argv[1]#'../simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_w_jacks_pairs'
    Rpin    = float(sys.argv[2]) 
    Rpmax   = float(sys.argv[3])
    rot90   = int(sys.argv[4])


    outputdir = outputfilename.split('/')[-2] 
    from subprocess import call
    call("mkdir -p %s" % (outputdir), shell=1)


    dat = pd.read_csv(outputfilename, delim_whitespace=1, usecols=['proj_sep', 'lzred', 'szred', 'etan_obs', 'llogmstel', 'llogmh',	'lconc', 'etan','r90et'], comment='#')
    print("data reading done")

    idx     = (dat['proj_sep'].values[:]>Rpin) & (dat['proj_sep'].values[:]<Rpmax)
    dat     = dat[idx]
    rbin    = dat['proj_sep']
    lzred   = dat['lzred']
    szred   = dat['szred']
    if rot90:
        etan    = dat['r90et']
    else:
        etan    = dat['etan_obs']
    avg_llogmstel   = np.log10(np.mean(10**dat['llogmstel']))
    avg_llogmh      = np.log10(np.mean(10**dat['llogmh']))
    avg_lconc       = np.mean(dat['lconc'])
    
    data    = etan
    print("data reading done")
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            print("error")
            sys.exit(0)

        ndim = 3
        nwalkers = 10
        
        np.random.seed(123)
        p_log_Mstel = avg_llogmstel + np.random.uniform(-0.1, 0.1, nwalkers)
        p_log_Mh    = avg_llogmh    + np.random.uniform(-0.1, 0.1, nwalkers)
        p_beta      = np.random.uniform(0.5, 1.5, nwalkers)

        p_0         = np.transpose([p_log_Mstel, p_log_Mh, p_beta])
        print(p_0)
 
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[data, rbin, lzred, szred])
        print(outputdir,"yay")
        #exit()

        print("Running burn-in...")

        Ntotal = 4000
        if rot90:
            pos = runchain(Ntotal,sampler, './%s/%s_Rpin_%s_Rpmax_%s_Burnfile_fixed_conc_gnfw.dat_r90'%( outputdir, outputfilename.split('/')[-1], Rpin, Rpmax), p_0)
        else:
            pos = runchain(Ntotal,sampler, './%s/%s_Rpin_%s_Rpmax_%s_Burnfile_fixed_conc_gnfw.dat'%( outputdir, outputfilename.split('/')[-1], Rpin, Rpmax), p_0)
        sampler.reset()

        print("Running production...")
        Ntotal = 4000
        if rot90:
            pos = runchain(Ntotal,sampler, './%s/%s_Rpin_%s_Rpmax_%s_Chainfile_fixed_conc_gnfw.dat_r90'%( outputdir, outputfilename.split('/')[-1], Rpin, Rpmax), pos)
        else:
            pos = runchain(Ntotal,sampler, './%s/%s_Rpin_%s_Rpmax_%s_Chainfile_fixed_conc_gnfw.dat'%( outputdir, outputfilename.split('/')[-1], Rpin, Rpmax), pos)



        print("Execution completed")
        pool.close()


    print("avg_logmstel:",avg_llogmstel)
    print("avg_logmh:", avg_llogmh)   




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


