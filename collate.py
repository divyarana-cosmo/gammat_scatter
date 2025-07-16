import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/net/dobbe/data2/github/gammat_scatter/model/')
sys.path.append('/net/dobbe/data2/github/gammat_scatter/src/')
from model import model
from distort_com import simshear

def get_meas(outdir, logMstelarr):
    from glob import glob
   
    for logMstelmin, logMstelmax in zip(logMstelarr[:-1], logMstelarr[1:]):
        print(logMstelmin, logMstelmax)
        #flist = glob(outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_*'%(logMstelmin, logMstelmax) + '_fast')
        flist = glob(outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_*'%(logMstelmin, logMstelmax) + '_fast')
        Njacks = int(len(flist))
        print('Number of jackknifes samples', Njacks)
 
        rbin      = np.array([])
        dsigmaarr = np.array([])
        xdsigmaarr = np.array([])
        for jk in range(Njacks):
            num=0; xnum=0; den=0
            for ii in range(Njacks): # leave out one jackknife region
                if ii==jk :
                    continue
                file = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_%d'%(logMstelmin, logMstelmax, ii) + '_fast' 
                #0-rmin/2+rmax/2 1-gammat 2-gammatsq 3-sigma_gammat 4-SN_Errgammat 5-gammax 6-gammaxsq 7-sigma_gammax 8-SN_Errgammax 9-gammat_inp 10-gammat_inp_bary 11-gammat_inp_dm 12-sumd_wls 13-dsigma 14-dsigmasq 15-SN_Errdsigmat 16-dsigmax 17-dsigmaxsq 18-SN_Errdsigmax 19-dsigmat_inp 20-dsigmat_inp_bary 21-dsigmat_inp_dm 22-sumd_dsigma_wls 23-sumd_dsigma_num 24-sumd_dsigma_den
                data = pd.read_csv(file, delim_whitespace=1) 
                #idx = data['0-rmin/2+rmax/2']*1e3 > 4.0
                #data = data[idx]
                
                num +=data['23-sumd_dsigma_num'].values[:]
                xnum+=data['24-sumd_dsigmax_num'].values[:]
                den +=data['25-sumd_dsigma_den'].values[:]
                rbin = data['0-rmin/2+rmax/2'].values[:]
            dsigmaarr  = np.append(dsigmaarr,num/den)
            xdsigmaarr = np.append(xdsigmaarr,xnum/den)
    
        dsigmaarr   = dsigmaarr.reshape(Njacks,-1)
        xdsigmaarr  = xdsigmaarr.reshape(Njacks,-1)
        dsigma      = np.mean(dsigmaarr,axis=0)        
        xdsigma     = np.mean(xdsigmaarr,axis=0)        
        
        cov = np.zeros((len(rbin), len(rbin)))
        xcov = np.zeros((len(rbin), len(rbin)))
        # calculating the covariances
        for ii in range(len(rbin)):
            for jj in range(len(rbin)):
                cov[ii,jj] = np.mean((dsigmaarr[:,ii] - dsigma[ii])*(dsigmaarr[:,jj] - dsigma[jj]))
                xcov[ii,jj] = np.mean((xdsigmaarr[:,ii] - xdsigma[ii])*(xdsigmaarr[:,jj] - xdsigma[jj]))
        
        cov *=(Njacks -1) # correction for the jackknife
        xcov *=(Njacks -1)
        
        dsigmaerr       = np.diag(cov)**0.5
        xdsigmaerr      = np.diag(xcov)**0.5
        # saving the output
        np.savetxt(outdir + './dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), np.transpose([rbin, dsigma, dsigmaerr, xdsigma, xdsigmaerr]), header='Rp[h-1 Mpc] dsigma dsigmaerr xdsigma xdsigmaerr')
        np.savetxt(outdir + './cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), cov)
        np.savetxt(outdir + './xcov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), xcov)
        return 0



def make_plots(outdir, logMstelarr):
    # dsigma plots
    for logMstelmin, logMstelmax in zip(logMstelarr[:-1], logMstelarr[1:]):
        rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr = np.loadtxt(outdir + './dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), unpack=1)
        cov  = np.loadtxt(outdir + './cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))
        xcov = np.loadtxt(outdir + './xcov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))
        dsigmaerr   = np.diag(cov)**0.5        
        xdsigmaerr  = np.diag(xcov)**0.5        
 
        plt.subplot(3,3,1)
        icov  = np.linalg.inv(cov)#*(Njacks - len(rbin) -2)/(Njacks -1)
        snr = (np.dot(dsigma, np.dot(icov, dsigma)))**0.5
        print('SNR',snr) 
        plt.errorbar(rbins*1e3, dsigma, yerr=dsigmaerr, fmt='.', capsize=3 ,label='%2.2f-%2.2f'%(logMstelmin, logMstelmax))

        # model predictions for the test sample

        ss                  = simshear(H0=100, Om0=0.25)
        lzred       =   0.22326649
        logmstel    =   10.47462749
        logre      =   -2.38450355
        logmh       =   12.42861652
        lconc       =   9.67904423
        szred = 0.9

        proj_sep            = rbins
        sigma_crit_inv      = ss._get_sigma_crit_inv_scalar(lzred=lzred, szred=szred) 
        esd_s, sigma_s      = ss._get_esd_s(logmstel, logre,proj_sep)
        esd_dm, sigma_dm    = ss._get_esd_dm(logmh, lconc, proj_sep)

        esd = (esd_s+esd_dm)/1e12
        esd = esd/(1-(sigma_s + sigma_dm)*sigma_crit_inv)

        ## for the test case 
        #H0          =   100
        #Om0         =   0.25
        #lenstype    =   'test_desi'    
        #logMmin     =   9.5
        #logMmax     =   11.0
        #zlmin       =   0.1
        #zlmax       =   0.4
        #Njacks      =   50
        #zdiff       =   0.1

        #mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

        #logmstel    =   10.47462749
        #log_re      =   -2.38450355
        #logmh       =   12.42861652
        #cfac        =   1.0

        #x = [logmstel, log_re, logmh, cfac]

        #red_esd     = mm.esd( x, rbins)
        #gamma_esd   = mm.esd( x, rbins, reduced=False)

        flist = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_0'%(logMstelmin, logMstelmax) + '_fast'
        df= pd.read_csv(flist, delim_whitespace=1) 
        meas_inp = df['19-dsigmat_inp'].values[:]
        #plt.plot(rbins*1e3, df['19-dsigmat_inp'].values[:], '--r',label='input')
        plt.plot(rbins*1e3, esd,':k', label='total')
        #plt.plot(rbins*1e3, mm.esd_s/1e12, label='stellar')
        #plt.plot(rbins*1e3, mm.esd_dm/1e12, label='dark matter')

        plt.legend()
        #plt.xlim(2.0,5)
        #plt.ylim(200,1000)
        plt.ylabel(r'$\Delta \Sigma[{\rm h M_{\odot} pc^{-2}}]$' )
        plt.xlabel(r'${\rm R_{\rm p} [h^{-1}kpc]}$')
        plt.xscale('log')
        plt.yscale('log')
        

        plt.subplot(3,3,2)
        #plt.plot(rbins*1e3, (esd-dsigma)/dsigmaerr,'-k')
        plt.plot(rbins*1e3, (esd-dsigma)/dsigmaerr,'-k')
        plt.plot(rbins*1e3, (meas_inp-dsigma)/dsigmaerr,'-r')
        plt.axhline(0.0,ls='--',color='grey')
        plt.legend()
        plt.ylabel(r'(inp-meas)/error' )
        #plt.ylabel(r'(inp-meas)' )
        plt.xlabel(r'${\rm R_{\rm p} [h^{-1}kpc]}$')
        plt.xscale('log')
 

    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.savefig(outdir + 'dsigma-signals-fast.pdf', dpi=300)
    #plt.clf()
    ## covariance plots
    #for logMstelmin, logMstelmax in zip(logMstelarr[:-1], logMstelarr[1:]):
    #    cov  = np.loadtxt(outdir + './cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))
    #    corr = 0.0*cov
    #    for ii in range(len(cov[0,:])):
    #        for jj in range(len(cov[0,:])):
    #            corr[ii,jj] = cov[ii,jj]/(cov[ii,ii]*cov[jj,jj])**0.5

    #    plt.subplot(3,3,1)
    #    plt.imshow(corr, origin='lower', vmin=-1, vmax=1)
    #    plt.colorbar()
    #    plt.ylabel(r'$\Delta \Sigma(R_{p, i})$' )
    #    plt.xlabel(r'$\Delta \Sigma(R_{\rm p, i})$' )

    #    plt.savefig(outdir + 'corr-signals-fast_lmstelmin_%2.2f_lmstelmax_%2.2f.pdf'%(logMstelmin, logMstelmax), dpi=300)
    #    plt.clf()

#logMstelarr = [9.5,10.0,10.5,11.0]
logMstelarr = [9.5,11.0]

for ii in range(9):
    outdir = 'output/test_desi_z_0.1_0.4_seed_%d/'%ii
    get_meas(outdir, logMstelarr)
#make_plots(outdir, logMstelarr)




#for ii in range(10):
#    outdir = 'output/test_desi_z_0.1_0.4_seed_%d/'%ii
#    get_meas(outdir, logMstelarr)
#make_plots(outdir, logMstelarr)







#plt.subplot(2,2,2)
#icov  = np.linalg.inv(xcov)*(Njacks - len(rbin) -2)/(Njacks -1)
#chisq = np.dot(xdsigma, np.dot(icov, xdsigma))
##chisq = sum(xdsigma**2 / xdsigmaerr**2)#np.dot(xdsigma, np.dot(icov, xdsigma))
#from scipy.stats import chi2
#print(rbin)
#print(chisq, int(len(rbin)))
#pval = chi2.sf(chisq,int(len(rbin)))
#plt.errorbar(rbin*1e3, xdsigma*rbin, yerr=xdsigmaerr*rbin, fmt='.', capsize=3, label='%2.2f'%pval)
#plt.axhline(0.0, color='black')
#plt.ylabel(r'$R_{\rm p}\Delta \Sigma_{\times}$' )
#plt.xlabel(r'${\rm R_{\rm p} [h^{-1}Kpc]}$')
#plt.xscale('log')
