import numpy as np
import matplotlib.pyplot as plt


Njacks =100


#logMstelarr = [9.0, 9.5, 10.0,10.5]
logMstelarr = [9.0,10.5]
for logMstelmin, logMstelmax in zip(logMstelarr[:-1], logMstelarr[1:]):
    print(logMstelmin, logMstelmax)
    dsigmaarr = np.array([])
    xdsigmaarr = np.array([])
    for jk in range(Njacks):
        num=0; xnum=0; den=0
        for ii in range(Njacks):
            if ii==jk :
                continue
            outdir = 'output/test_debug_z_0.1_0.4/'
            file = 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_test_case_w_jacks_jk_%d'%(logMstelmin, logMstelmax, ii)
            #file = 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_%d'%(logMstelmin, logMstelmax, ii)
            data = np.loadtxt(outdir + file, skiprows=1, usecols=[0,13,16,22])
            idx = data[:,0]*1e3 > 1.0
            data = data[idx]
            #0-rmin/2+rmax/2 1-gammat 2-gammatsq 3-sigma_gammat 4-SN_Errgammat 5-gammax 6-gammaxsq 7-sigma_gammax 8-SN_Errgammax 9-gammat_inp 10-gammat_inp_bary 11-gammat_inp_dm 12-sumd_wls 13-dsigma 14-dsigmasq 15-SN_Errdsigmat 16-dsigmax 17-dsigmaxsq 18-SN_Errdsigmax 19-dsigmat_inp 20-dsigmat_inp_bary 21-dsigmat_inp_dm 22-sumd_dsigma_wls 23-sumd_dsigma_num 24-sumd_dsigma_den
            
            num +=data[:,1]*data[:,3]
            xnum+=data[:,2]*data[:,3]
            den +=data[:,3]
            rbin = data[:,0]*1e3
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
    
    cov *=(Njacks -1)
    xcov *=(Njacks -1)
    dsigmaerr   = np.diag(cov)**0.5        
    xdsigmaerr  = np.diag(xcov)**0.5        
    
    # saving the output
    np.savetxt(outdir + './test_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), np.transpose([rbin/1e3, dsigma, dsigmaerr, xdsigma, xdsigmaerr]), header='Rp[h-1 Mpc] dsigma dsigmaerr xdsigma xdsigmaerr')
    np.savetxt(outdir + './cov_test_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), cov)
    np.savetxt(outdir + './xcov_test_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), xcov)

    plt.subplot(2,2,1)
    plt.errorbar(rbin, dsigma, yerr=dsigmaerr, fmt='.', capsize=3 ,label='%2.2f-%2.2f'%(logMstelmin, logMstelmax))
    plt.legend()
    plt.ylabel(r'$\Delta \Sigma[{\rm h M_{\odot} pc^{-2}}]$' )
    plt.xlabel(r'${\rm R_{\rm p} [h^{-1}Kpc]}$')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(2,2,2)
    icov  = np.linalg.inv(xcov)*(Njacks - len(rbin) -2)/(Njacks -1)
    chisq = np.dot(xdsigma, np.dot(icov, xdsigma))
    #chisq = sum(xdsigma**2 / xdsigmaerr**2)#np.dot(xdsigma, np.dot(icov, xdsigma))
    from scipy.stats import chi2
    pval = chi2.sf(chisq,len(rbin))
    plt.errorbar(rbin, xdsigma*rbin/1e3, yerr=xdsigmaerr*rbin/1e3, fmt='.', capsize=3, label='%2.2f'%pval)
    plt.axhline(0.0, color='black')
    plt.ylabel(r'$\Delta \Sigma_{\times}[{\rm h M_{\odot} pc^{-2}}]$' )
    plt.xlabel(r'${\rm R_{\rm p} [h^{-1}Kpc]}$')
    plt.xscale('log')
    plt.legend()

plt.tight_layout()
plt.savefig('test_dsigma-signals.png', dpi=300)
