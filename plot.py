import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/net/dobbe/data2/github/gammat_scatter/model/')
sys.path.append('/net/dobbe/data2/github/gammat_scatter/src/')
from model import model
from distort_com import simshear
from scipy.interpolate import interp1d

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

        outdir = 'output/desi_z_0.0_0.4/iso_centrals_p_satellites/%2.2f_%2.2f_seed_%d'%(logMstelmin, logMstelmax, ii)

        rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigma, avg_sigc, avg_sigc_sq = np.loadtxt(outdir + '_ovp_100/' + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), unpack=1)

        cov  = np.loadtxt(outdir + '_ovp_1/' + 'cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))

        # removing the first radialbin
        rbins      =     rbins      [1:] 
        dsigma     =     dsigma     [1:] 
        dsigmaerr  =     dsigmaerr  [1:] 
        xdsigma    =     xdsigma    [1:] 
        xdsigmaerr =     xdsigmaerr [1:] 
        stel_dsigma=     stel_dsigma[1:] 
        avg_sigc   =     avg_sigc   [1:] 
        avg_sigc_sq=     avg_sigc_sq[1:]


        cov =   cov[1:,1:]

        idx             = (lens_metadata['logMmin']==logMstelmin) &(lens_metadata['logMmax']==logMstelmax)
        Rmin            = lens_metadata['re_50'].values[idx] * 3/1e3 # 1e3 factor to convert Mpc to kpc
        
        if Rmin<rbins[0]:
            Rmin = rbins[0]
            xfit = np.append(xfit, Rmin)
            yfit = np.append(yfit, dsigma[0])
        
        else:
            func = interp1d(rbins, dsigma, kind='cubic')
            xfit = np.append(xfit, Rmin)
            yfit = np.append(yfit, func(Rmin))

        dsigmaerr   = np.diag(cov)**0.5         

        ax.errorbar(rbins*1e3, dsigma, yerr=dsigmaerr, fmt='.', capsize=3 ,label='(%2.1f, %2.1f)'%(logMstelmin, logMstelmax), color='C%d'%colorcnt)

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
    
    ax.plot(xfill, yfill, '--', color='grey', zorder=10, label=r'$3 R_{\rm e}$')

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
        outdir = 'output/desi_z_0.0_0.4/iso_centrals_p_satellites/%2.2f_%2.2f_seed_%d/'%(logMstelmin, logMstelmax, ii)
        #rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr = np.loadtxt(outdir + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax), unpack=1)

        rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigma, avg_sigc, avg_sigc_sq = np.loadtxt(outdir + 'dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(logMstelmin, logMstelmax), unpack=1)

        cov  = np.loadtxt(outdir + 'cov_dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f'%(logMstelmin, logMstelmax))
        cov  = np.diag(np.diag(cov))
        


        #removing nans
        idx       = (lens_metadata['logMmin']==logMstelmin) &(lens_metadata['logMmax']==logMstelmax)
        Rmin      = lens_metadata['re_50'].values[idx] * 3/1e3 # 1e3 factor to convert Mpc to kpc


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
#outdir = 'output/desi_z_0.0_0.4/iso_centrals_p_satellites/%2.2f_%2.2f_seed_%d/'%(logMmin, logMmax, ss)
#get_meas(outdir, [logMmin, logMmax], nover_samp=1000)
#




logMstelarr = 9.5 + 0.1*np.arange(21)
#for ss,(logMmin, logMmax) in enumerate(zip(logMstelarr[:-1], logMstelarr[1:])):
#    #if logMmin!=11.5:
#    #    continue
#    outdir = 'output/desi_z_0.0_0.4/iso_centrals_p_satellites/%2.2f_%2.2f_seed_%d/'%(logMmin, logMmax, ss)
#    get_meas(outdir, [logMmin, logMmax], nover_samp=1)

##print('measurements done')
make_plots('./plots/', logMstelarr)
#plt.clf()
#logMstelarr = logMstelarr[logMstelarr<=11.6]
#plot_snr('./plots/', logMstelarr)

## plot only first and last stellar mass bin


















































"""
import sys
sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from halopy import halo
from stellarpy import stellar
from distort_com import simshear
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from weakpipe_select import lens_select
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.integrate import quad

from astropy.cosmology import FlatLambdaCDM
cc = FlatLambdaCDM(H0=100,Om0=0.25) 

def model(x, rbins, Om0=0.25):
    logmstel, log_re, logmh, cfac = x
    hp          = halo(logmh, cfac, omg_m=Om0)
    stel        = stellar(logmstel, log_re=log_re)
    sigma_s    = stel.sigma_deVaucouleurs(rbins) 
    sigma_dm   = hp.sigma_nfw(rbins)         

    esd_s       = stel.esd_deVaucouleurs(rbins)
    esd_dm      = hp.esd_nfw(rbins)

    sigma_s     =   sigma_s  
    sigma_dm    =   sigma_dm
    #ans         =   get_avg_sigmacritinv(sigma_s,sigma_dm,zred)
    return esd_s/1e12,  esd_dm/1e12, sigma_s, sigma_dm#, ans


outputfilename = './debug/simed_sources.dat_lmstelmin_9.00_lmstelmax_10.50_with_shape_noise_using_shear'
import pandas as pd
data = pd.read_csv(outputfilename, delim_whitespace=1)
#rbins, dsigma, dsigmaerr, dsigmax, dsigmaxerr = np.loadtxt(outputfilename, unpack=1)
#idx = rbins>0.004
rbins       = data['0-rmin/2+rmax/2'].values[:]  #rbins       [idx]
dsigma      = data['13-dsigma'].values[:]  #dsigma      [idx]
#dsigmaerr   = data['']  #dsigmaerr   [idx]

plt.subplot(2,2,1)
yy = dsigma 
plt.plot(rbins, yy,'-')
#plt.errorbar(rbins, yy, yerr=dsigmaerr, fmt='.')
x = [0.19999999999999998,10.0,-2.5,12.0,11.0]

zred = x[0]
esd_stel, esd_dm, sigma_stel, sigma_dm = model(x[1:], rbins=rbins)
print(yy)
print(esd_stel+esd_dm)
print(esd_stel)
print(data['20-dsigmat_inp_bary'])
print(esd_dm)
print(data['21-dsigmat_inp_dm'])

ss = simshear(H0=100, Om0=0.25)
data = ss._get_esd(10.0,-2.5,12.0,11.0, rbins)
print(data[0]/1e12)
print(data[1]/1e12)
plt.plot(rbins, esd_stel)
plt.plot(rbins, esd_dm)
plt.plot(rbins, (esd_stel + esd_dm))

#gamma_s, gamma_dm, kappa_s, kappa_dm =ss._get_g(logmstel=10.15, logre=-2.57, logmh=12.093, lconc=10.83, lzred=0.2, szred=0.9, proj_sep=rbins)
#
#yy =  (gamma_s + gamma_dm)/ss._get_sigma_crit_inv(lzred=0.2, szred=0.9)/1e12
#print(yy)
#plt.plot(rbins, yy)

#plt.plot(rbins, (esd_stel + esd_dm)*ans)
plt.xscale('log')
#plt.yscale('log')
plt.savefig('test.png', dpi=300)


#
#def plt_data(config, outputfilename):
#    "generate sanity plots for the test case as mentioned in simulate_aroundlens.py"
#
#
#    ss = simshear(H0= config['H0'],Om0 = config['Om0'])
#    colossus_cosmo  = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8 = ss.sigma8, ns = ss.ns, cosmo_name=ss.cosmo_name)
#
#    lensargs    = config["lens"]
#    #lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
#    llogmh = 12.09 #+ 0.0*llogmh
#    lzred = 0.2 #+ 0.0*lzred
#    lconc = 10.83#concentration.concentration(1e14, '200m', lzred, model = 'diemer19')
#    llogmstel   = 10.15  
#
#    rbins, dsigma, dsigmaerr, dsigmax, dsigmaxerr = np.loadtxt(outputfilename, unpack=1)
#    xx = rbins
#    yy = dsigma
#    yyerr = dsigmaerr
#
#    plt.subplot(3,3,1)
#    plt.errorbar(xx, yy, yerr=yyerr, fmt='.', capsize=3)
#    #plt.plot(dat[:,0], dat[:,1], '.', lw=0.0)
#    szred   = 0.8 + 0.0*rbins
#    print(llogmh,lconc)
#    gamma_s, gamma_dm, kappa_s, kappa_dm    = ss._get_g(llogmstel,0.005,llogmh, lconc, lzred, szred, proj_sep=rbins*1.2)
#    inv_crit = ss._get_sigma_crit_inv(0.2, 0.8)
#
#    #plt.errorbar(xx, yy/inv_crit/1e12, yerr=yyerr/inv_crit/1e12, fmt='.', capsize=3)
#    #plt.plot(rbins, kappa_dm, '--', color='C1')
#    #plt.plot(rbins, gamma_s/inv_crit, '--', color='C1', label='baryon')
#    #plt.plot(rbins, gamma_dm/inv_crit, '--', color='C2', label='dark matter')
#    #plt.plot(rbins, (gamma_s + gamma_dm)/inv_crit, '-k', label='total')
#
#    plt.plot(rbins, gamma_s/inv_crit/(1-kappa_s)/1e12, '--', color='C1', label='Baryon')
#    plt.plot(rbins, gamma_dm/inv_crit/(1-kappa_dm)/1e12, '--', color='C2', label='Dark matter')
#    plt.plot(rbins, (gamma_s + gamma_dm)/inv_crit/(1-(kappa_dm + kappa_s))/1e12, '-k', label='Total')
#
#
#    plt.xscale('log')
#    plt.yscale('log')
#    #plt.ylim(50,1e3)
#    plt.ylabel(r'$\Delta \Sigma[{\rm h M_{\odot} pc^{-2}}]$' )
#    #plt.ylabel(r'$g_{\rm t}$')
#    #plt.xticks([])
#    plt.xlabel(r'${\rm R_{\rm p} [h^{-1}Mpc]}$')
#    plt.legend(fontsize='small', frameon=0)
#
#    #plt.subplot(2,2,3)
#    #res = (yy - (gamma_s + gamma_dm)/(1-(kappa_dm + kappa_s)))/yyerr
#    #plt.plot(xx, res)
#    #plt.ylim(-4,4)
#    #plt.axhline(0.0, ls='--', color='grey')
#    #plt.xscale('log')
#    #plt.ylabel(r'$ (g_{\rm t, meas} - g_{\rm t, mod})/\sigma$')
#    #plt.xlabel(r'${\rm R [h^{-1}Mpc]}$')
# 
#    #plt.subplot(2,2,4)
#    #plt.plot(yy, (gamma_s + gamma_dm)/(1-(kappa_dm + kappa_s)), '.')
#    #         
#
#    #plt.subplot(2,2,2)
#    #plt.errorbar(xx,xx*yyx, yerr=xx*yyerrx, fmt='.', capsize=3)
#    #plt.axhline(0.0, ls='--', color='grey')
#    #plt.ylim(-3e-3,3e-3)
#    #plt.ylabel(r'$R g_{\rm \times}$')
#    #plt.xlabel(r'${\rm R [h^{-1}Mpc]}$')
#    #plt.xscale('log')
#    #
#    #plt.tight_layout()
#    plt.savefig('test.png', dpi=300)
#    return 0
#
#outputfilename = '/net/dobbe/data2/github/gammat_scatter/output/debug_z_0.1_0.4/test_dsigma.dat_lmstelmin_9.00_lmstelmax_10.50'
#with open('config', 'r') as ymlfile:
#    config = yaml.safe_load(ymlfile)
#
#
#plt_data(config, outputfilename)           

#rbins   = np.unique(dat[:,0])
#xx      = rbins
#yy      = 0.0*rbins
#yyerr    = 0.0*rbins
#yyx      = 0.0*rbins
#yyerrx   = 0.0*rbins

#for i in range(len(rbins)):
#    idx = dat[:,0]==rbins[i]
#    print(dat[idx,1])
#    yy[i]       = np.mean(dat[idx,1])
#    yyerr[i]    = (sum(idx) - 1)**0.5 * np.std(dat[idx,1])

#    yyx[i]       = np.mean(dat[idx,5])
#    yyerrx[i]    = (sum(idx) - 1)**0.5 * np.std(dat[idx,5])

#ss = simshear()
#
#def get_sigma_crit_inv(lzred, szred, cc=cc):
#    # some important constants for the sigma crit computations
#    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
#    cee = 3e5 #km s^-1
#    # sigma_crit_calculations for a given lense-source pair
#    sigma_crit_inv = cc.angular_diameter_distance(lzred).value * cc.angular_diameter_distance_z1z2(lzred, szred).value  * 1.0/cc.angular_diameter_distance(szred).value
#    sigma_crit_inv = sigma_crit_inv * 4*np.pi*gee*1.0/cee**2
#    return sigma_crit_inv
#
#
#def get_avg_sigmacritinv(sigma_s,sigma_dm,lzred):
#    "assigns redshifts respecting the distribution"
#    if np.isscalar(lzred):
#        lzred = np.array([lzred])
#    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
#    cee = 3e5 #km s^-1
#    z0 = 0.9/(2)**0.5
#    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
#    zmin = 0.4 + 0.2 
#    zmax = 3.0
#    zzarr = np.linspace(zmin,zmax,200)
#    xzzarr = np.linspace(0.0,zmax,400)
#
#    ans = 0.0*sigma_dm
#    for ii in range(len(ans)):
#        sigma = sigma_s[ii]+sigma_dm[ii]
#        #integrad = lambda zz: f(zz)* (get_sigma_crit_inv(lzred, zz)+ get_sigma_crit_inv(lzred, zz)**2)
#        integrad = lambda zz: f(zz)* 1/(1-sigma*(get_sigma_crit_inv(lzred, zz)*4*np.pi*gee*1.0/cee**2))
#
#        ans[ii]=  sum(integrad(zzarr))*(zzarr[1] -zzarr[0])/(sum(f(xzzarr))*(xzzarr[1] -xzzarr[0]))
#        #print(sigma,ans[ii])
#        #print(get_sigma_crit_inv(lzred, zzarr[:5])*4*np.pi*gee*1.0/cee**2)
#       #ans[ii]=  quad(integrad, zmin, zmax)[0]/quad(f, 0, zmax)[0]
#    return ans





"""


