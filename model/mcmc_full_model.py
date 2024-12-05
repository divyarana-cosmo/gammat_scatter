#model based on sonnenfeld 2018 - bayesian ggl paper
#arxiv:1710.00007
import sys
from full_model import halo
import numpy as np
import emcee
import pandas as pd
from schwimmbad import MPIPool

from mcmc_free_conc import lnprior as _ilnprior


def ilnprior(x):
    if np.isscalar(x):
        x = np.array(x)
    ans = 0.0*x[:,0]
    for ii in range(len(x[:,0])):
        ans[ii] = _ilnprior(x[ii,:])
    return ans    


def lnprior(x):
    mu_h0, sigma_h, beta_h, mu_stel, sigma_stel, alpha_stel, mu_c0, sigma_c, beta_c= x
    if 11<mu_h0<15 and 0<sigma_h<2 and -3<beta_h<3 and 10<mu_stel<12 and 0<sigma_stel<2 and -1<alpha_stel<1 and 0<mu_c0<2 and 0<sigma_c<1 and -1<beta_c<1 :
        return 0.0
    return -np.inf

def lnprob(x, data):
    lp = lnprior(x)
    if not np.isfinite(lp):
       return -np.inf#,dirt
    
    #intermediate chain dataset
    logMstel   =   data[:,0]
    logMh      =   data[:,1]
    logch      =   np.log10(data[:,2])


    mu_h0, sigma_h, beta_h, mu_stel, sigma_stel, alpha_stel, mu_c0, sigma_c, beta_c = x
    # plug in the probabilities for the full model
    hp = halo(mu_h0, sigma_h, beta_h, mu_stel, sigma_stel, alpha_stel, mu_c0, sigma_c, beta_c)
    #intermediate prior
    ans = np.mean(hp.P_Mstel_Mh(logMstel, logMh)*hp.P_c_Mh(logch, logMh)/10**ilnprior(data))

    print( 'mu_h0, sigma_h, beta_h, mu_stel, sigma_stel, alpha_stel, mu_c0, sigma_c, beta_c, chisq_eff')
    print( x, -2*np.log(ans))
    if ans<0 or np.isnan(ans):
        return -np.inf
    res = lp + np.log(ans)

    return res


def runchain(Ntotal,sampler,chainf,pos):
    blnk=[""];
    fchain=open(chainf,"w");
    #fblob=open(blobf,"w");
    iterno=1;
    # Store chainfile and prednfile in the same format as before
    for result in sampler.sample(pos, iterations=Ntotal, store=0):
        posn,probn,staten = result;
        #posn,probn,staten,blobsn = result;
        #posn,probn,staten = result;
        for i in range(nwalkers):
            np.savetxt(fchain,posn[i],newline=' ');
            np.savetxt(fchain,[-2.*probn[i]],newline=' ');
            #np.savetxt(fchain,[sampler.acceptance_fraction[i],-2.*probn[i]],newline=' ');
            #np.savetxt(fblob,blobsn[i],newline=' ');
            np.savetxt(fchain,blnk,fmt='%s');
            #np.savetxt(fblob,blnk,fmt='%s');
        print("Iteration number: %d of %d done"%(iterno,Ntotal));
        iterno=iterno+1;
        posnew=result[0];

    fchain.close();
    #fblob.close();
    return posnew;

if __name__ == "__main__":
    import sys
    #provide chainfile from the previous intermediate mcmc run
    ichainfilename = sys.argv[1]

    dat = pd.read_csv(ichainfilename, delim_whitespace=1, header=None, comment='#')
    print("data reading done")
    #log_Mstel, log_Mh, c
    data = dat.values[:,:3]
    #print(10**ilnprior(data))
    #exit()
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            print("error")
            sys.exit(0)

        ndim = 9
        nwalkers = 20
        
        #initializing the walkers positions

        np.random.seed(123)
        p_mu_h0       =   np.random.uniform(11, 15, nwalkers) 
        p_sigma_h     =   np.random.uniform(0, 2  , nwalkers) 
        p_beta_h      =   np.random.uniform(-3, 3 , nwalkers) 
        p_mu_stel     =   np.random.uniform(10, 12, nwalkers) 
        p_sigma_stel  =   np.random.uniform(0, 2  , nwalkers) 
        p_alpha_stel  =   np.random.uniform(-1, 1 , nwalkers) 
        p_mu_c0       =   np.random.uniform(0, 2  , nwalkers) 
        p_sigma_c     =   np.random.uniform(0, 1  , nwalkers)    
        p_beta_c      =   np.random.uniform(-1, 1 , nwalkers) 

        p_0 = np.transpose([p_mu_h0, p_sigma_h, p_beta_h, p_mu_stel, p_sigma_stel, p_alpha_stel, p_mu_c0, p_sigma_c, p_beta_c])     

 
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=[data])

        print("Running burn-in...")

        Ntotal = 4000
        pos = runchain(Ntotal,sampler, '%s_full_model_Burnfile.dat'%(ichainfilename), p_0)
        sampler.reset()

        print("Running production...")
        Ntotal = 4000
        pos = runchain(Ntotal,sampler, '%s_full_model_Chainfile.dat'%(ichainfilename), pos)

        print("Execution completed")
        pool.close()




