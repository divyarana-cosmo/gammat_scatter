import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/net/dobbe/data2/github/gammat_scatter/model/')
sys.path.append('/net/dobbe/data2/github/gammat_scatter/src/')
from model import model
from distort_com import simshear
from scipy.interpolate import interp1d

def get_meas(outdir, logMstelarr, nover_samp=1):
    # reading the data
    from glob import glob
    print(logMstelarr)   
    for logMstelmin, logMstelmax in zip(logMstelarr[:-1], logMstelarr[1:]):
        print(logMstelmin, logMstelmax)
        #flist = glob(outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_using_shear_w_jacks_jk_*'%(logMstelmin, logMstelmax) + '_fast')
        flist = glob(outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_*'%(logMstelmin, logMstelmax) + '_fast')
        Njacks = int(len(flist))
        print('Number of jackknifes samples', Njacks)
 
        rbin      = np.array([])
        dsigmaarr = np.array([])
        xdsigmaarr = np.array([])
        inp_gammat_arr = np.array([])

        
        for jk in range(Njacks):
            stel_num =0
            stel_den =0
            avg_inv_sigc_num =  0
            avg_inv_sigc_den =  0

            file = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_%d'%(logMstelmin, logMstelmax, jk) + '_fast' 
            data = pd.read_csv(file, delim_whitespace=1, skipfooter=1, engine='python') 
            stel_num += data['22-dsigmat_inp_bary'].values[:]*data['27-sumd_dsigma_den']
            stel_den += data['27-sumd_dsigma_den']
            avg_inv_sigc_num += data['28-sumd_wls_by_sigmac']
            avg_inv_sigc_den += data['14-sumd_wls']

            num=0; xnum=0; den=0
            for ii in range(Njacks): # leave out one jackknife region
                if ii==jk :
                    continue
                #file = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_using_shear_w_jacks_jk_%d'%(logMstelmin, logMstelmax, ii) + '_fast' 
                file = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_%d'%(logMstelmin, logMstelmax, ii) + '_fast' 
                data = pd.read_csv(file, delim_whitespace=1, skipfooter=1, engine='python') 
                #idx = data['0-rmin/2+rmax/2']*1e3 > 4.0
                #data = data[idx]
                num +=data['25-sumd_dsigma_num'].values[:]
                xnum+=data['26-sumd_dsigmax_num'].values[:]
                den +=data['27-sumd_dsigma_den'].values[:]
                rbin = data['2-rmin/2+rmax/2'].values[:]
                inp_gammat_arr = data['11-gammat_inp'].values[:]
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
                cov[ii,jj]  = np.mean((dsigmaarr[:,ii] - dsigma[ii])*(dsigmaarr[:,jj] - dsigma[jj]))
                xcov[ii,jj] = np.mean((xdsigmaarr[:,ii] - xdsigma[ii])*(xdsigmaarr[:,jj] - xdsigma[jj]))
        #correction for jackknife and oversampling factor        
        cov  *=(nover_samp*(Njacks -1))         
        xcov *=(nover_samp*(Njacks -1))
        
        dsigmaerr       = np.diag(cov)**0.5
        xdsigmaerr      = np.diag(xcov)**0.5

        
        #stellar and sigma c part
        stel_dsigmaarr = stel_num/stel_den 
        avg_inv_sigcarr    = avg_inv_sigc_num/avg_inv_sigc_den
        avg_inv_sigcarrsq   = stel_den/avg_inv_sigc_den
                         
        # saving the output
        #np.savetxt(outdir + '/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(logMstelmin, logMstelmax), np.transpose([rbin, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigmaarr, avg_inv_sigcarr, avg_inv_sigcarrsq]), header='Rp[h-1 Mpc] dsigma dsigmaerr xdsigma xdsigmaerr stel_dsigma avg_sigc avg_sigc_sq')
        ##np.savetxt(outdir + '/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), np.transpose([rbin, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigmaarr, avg_inv_sigcarr, avg_inv_sigcarrsq]), header='Rp[h-1 Mpc] dsigma dsigmaerr xdsigma xdsigmaerr stel_dsigma avg_sigc avg_sigc_sq')
        np.savetxt(outdir + '/cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), cov)
        np.savetxt(outdir + '/xcov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), xcov)
    return 0

def make_plots(pltdir, logMstelarr, Njacks=100):
    #fixing the minimum projected separation to be atleast 2*Re
    lens_metadata   = pd.read_csv('/net/dobbe/data2/github/gammat_scatter/lens_sample.dat', delim_whitespace=1)
    #pushing the fitting xx,yy in an array
    xfit = np.array([])
    yfit = np.array([])


    # remove first radial bin
    ss=1
    ax = plt.subplot(3,3,ss)
    colorcnt=0
    inp_gammat_arr =0
    # dsigma plots
    for ii, (logMstelmin, logMstelmax)in enumerate(zip(logMstelarr[:-1], logMstelarr[1:])):
        if logMstelmin>9.5 and logMstelmin<10.5:
            continue
        if logMstelmin>10.5 and logMstelmin<11.4:
            continue
        if logMstelmin>11.4:
            continue

        outdir = 'output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/'%(logMstelmin, logMstelmax, ii)
        file = outdir + 'simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_%d'%(logMstelmin, logMstelmax, 10) + '_fast' 
        data = pd.read_csv(file, delim_whitespace=1) 
        rbin = data['2-rmin/2+rmax/2'].values[:]
        inp_gammat_arr = data['11-gammat_inp'].values[:]
        from scipy.interpolate import interp1d
        idx = np.isfinite(inp_gammat_arr)
        interpfunc = interp1d(inp_gammat_arr[idx], rbin[idx], kind='cubic')
        
        try:
            rpvt = interpfunc(0.3)
        except ValueError:
            rpvt = min(rbin)

        rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigma, avg_sigc, avg_sigc_sq = np.loadtxt(outdir + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(logMstelmin, logMstelmax), unpack=1)

        #rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr 
        cov  = np.loadtxt(outdir + 'cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))


        idx             = (lens_metadata['logMmin']==logMstelmin) &(lens_metadata['logMmax']==logMstelmax)
        Rmin            = lens_metadata['re_50'].values[idx] * 2.5/1e3 # 1e3 factor to convert Mpc to kpc

        
        if Rmin<rbins[0]:
            Rmin = rbins[0]
            xfit = np.append(xfit, Rmin)
            yfit = np.append(yfit, dsigma[0])
        
        else:
            func = interp1d(rbins, dsigma, kind='cubic')
            xfit = np.append(xfit, Rmin)
            yfit = np.append(yfit, func(Rmin))


        ##removing nans
        #idx = np.isfinite(dsigma) & (rbins>np.min(rbins))
        #rbins       =   rbins[idx]
        #dsigma      =   dsigma[idx]

        #cov         =   np.delete(cov, ~idx, axis=0)
        #cov         =   np.delete(cov, ~idx, axis=1)
        
        #print(sum(idx), cov)
        dsigmaerr   = np.diag(cov)**0.5         
 
        icov  = np.linalg.inv(cov)*(Njacks - len(rbins) -2)/(Njacks -1)
        snr = (np.dot(dsigma, np.dot(icov, dsigma)))**0.5
        interpfunc = interp1d(rbins,dsigma)
        print(logMstelmin, logMstelmax, rpvt)

        #ax.plot(rpvt*1e3, interpfunc(rpvt), 'k.', zorder=20)
        #print(dsigma)
        ax.errorbar(rbins*1e3, dsigma, yerr=dsigmaerr, fmt='.', capsize=3 ,label='(%2.1f, %2.1f)'%(logMstelmin, logMstelmax), color='C%d'%colorcnt)

        print(rbins)
        print(dsigma)
        print(dsigmaerr/dsigma * 100)

        ## reading the best fit model predictions
        #preddata    =   np.loadtxt('./model/output_mcmc_desi_runs/predfile_nfw_lmstelmin_%2.2f_lmstelmax_%2.2f.dat_full_desixeuclid_seed_%d_fixed_conc_14kdr3'%(logMstelmin, logMstelmax, ii))
        #predesd    =   preddata[np.argmin(preddata[:,-1]),:-1]
        #ax.plot(rbins*1e3, predesd, 'C%d'%colorcnt)    



        colorcnt +=1
    

    # make sure x values are sorted
    xfill = xfit * 1e3
    idx = np.argsort(xfill)
    xfill = xfill[idx]
    yfill = yfit[idx]
    
    ax = plt.gca()
    ymax = np.max(dsigma) * 3  # or any suitably large value
    
    ax.plot(xfill, yfill, '--', color='grey', zorder=10, label=r'$2.5 R_{\rm e}$')

    xshade = np.insert(xfill, 0, 5)
    yshade = np.insert(yfill, 0, yfill[0])  # horizontal extension
    
    ax.fill_between(
        xshade,
        yshade,
        ymax,
        color='grey',
        alpha=0.5,
        zorder=10
    )


    #plt.plot(xfit*1e3, yfit, '--', color='grey', zorder=10)
    #plt.xlim(5, 150)
    plt.ylabel(r'$\Delta \Sigma[{\rm h M_{\odot} pc^{-2}}]$' )
    plt.xlabel(r'${\rm R_{\rm p} [h^{-1}kpc]}$')
    plt.xscale('log')
    plt.yscale('log')
    leg = plt.legend(fontsize='xx-small')
    leg.set_zorder(100)

    plt.savefig(pltdir + 'dsigma-signals_paper.pdf')
    return 0

def plot_snr(pltdir, logMstelarr, Njacks=100):
    #fixing the minimum projected separation to be atleast 2*Re
    lens_metadata   = pd.read_csv('/net/dobbe/data2/github/gammat_scatter/lens_sample.dat', delim_whitespace=1)

    # dsigma plots
    logMstel_arr =   np.array([])
    snr_arr      =   np.array([])

    for ii, (logMstelmin, logMstelmax)in enumerate(zip(logMstelarr[:-1], logMstelarr[1:])):
        print(logMstelmin, logMstelmax)
        outdir = 'output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/'%(logMstelmin, logMstelmax, ii)
        #rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr = np.loadtxt(outdir + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), unpack=1)

        rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigma, avg_sigc, avg_sigc_sq = np.loadtxt(outdir + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(logMstelmin, logMstelmax), unpack=1)

        cov  = np.loadtxt(outdir + 'cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))
        cov  = np.diag(np.diag(cov))
        


        #removing nans
        idx       = (lens_metadata['logMmin']==logMstelmin) &(lens_metadata['logMmax']==logMstelmax)
        Rmin      = lens_metadata['re_50'].values[idx] * 2.5/1e3 # 1e3 factor to convert Mpc to kpc


        idx         = (rbins>Rmin)
        rbins       =   rbins[idx]
        dsigma      =   dsigma[idx]

        cov         =   np.delete(cov, ~idx, axis=0)
        cov         =   np.delete(cov, ~idx, axis=1)
        
        #print(sum(idx), cov)
        dsigmaerr   = np.diag(cov)**0.5         

        #print(logMstelmin, logMstelmax, dsigmaerr/dsigma * 100)
        #continue
 
        icov  = np.linalg.inv(cov)*(Njacks - len(rbins) -2)/(Njacks -1)
        snr = (np.dot(dsigma, np.dot(icov, dsigma)))**0.5
        print(snr)
        logMstel_arr =   np.append(logMstel_arr, [logMstelmin*0.5 + logMstelmax*0.5])
        snr_arr      =   np.append(snr_arr     , [snr])


    ax = plt.subplot(3,3,1)
    ax.plot(logMstel_arr, snr_arr,'-')
    #ax.set_yscale('log')
    ax.set_xlabel(r'$\log[M_*/{\rm h^{-1}M_\odot}]$')
    ax.set_ylabel(r'${\rm SNR}$')

    plt.savefig(pltdir + 'snr_paper.pdf', dpi=300)
    return 0


#ss=17
#logMmin =   11.20
#logMmax =   11.30
#outdir = 'output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/'%(logMmin, logMmax, ss)
#get_meas(outdir, [logMmin, logMmax], nover_samp=1000)
#




logMstelarr = 9.5 + 0.1*np.arange(21)
#for ss,(logMmin, logMmax) in enumerate(zip(logMstelarr[:-1], logMstelarr[1:])):
#    #if logMmin!=11.5:
#    #    continue
#    outdir = 'output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/'%(logMmin, logMmax, ss)
#    get_meas(outdir, [logMmin, logMmax], nover_samp=1)

##print('measurements done')
make_plots('./plots/', logMstelarr)
#plt.clf()
#logMstelarr = logMstelarr[logMstelarr<=11.6]
#plot_snr('./plots/', logMstelarr)

## plot only first and last stellar mass bin




