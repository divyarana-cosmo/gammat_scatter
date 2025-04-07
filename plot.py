import sys
sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from halopy import halo
from stellarpy import stellar
from distort import simshear
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from get_data import lens_select
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.integrate import quad

from astropy.cosmology import FlatLambdaCDM
cc = FlatLambdaCDM(H0=100,Om0=0.25) 

ss = simshear()

def get_sigma_crit_inv(lzred, szred, cc=cc):
    # some important constants for the sigma crit computations
    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    cee = 3e5 #km s^-1
    # sigma_crit_calculations for a given lense-source pair
    sigma_crit_inv = cc.angular_diameter_distance(lzred).value * cc.angular_diameter_distance_z1z2(lzred, szred).value  * 1.0/cc.angular_diameter_distance(szred).value
    sigma_crit_inv = sigma_crit_inv * 4*np.pi*gee*1.0/cee**2
    return sigma_crit_inv


def get_avg_sigmacritinv(sigma_s,sigma_dm,lzred):
    "assigns redshifts respecting the distribution"
    if np.isscalar(lzred):
        lzred = np.array([lzred])
    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    cee = 3e5 #km s^-1
    z0 = 0.9/(2)**0.5
    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
    zmin = 0.4 + 0.2 
    zmax = 3.0
    zzarr = np.linspace(zmin,zmax,200)
    xzzarr = np.linspace(0.0,zmax,400)

    ans = 0.0*sigma_dm
    for ii in range(len(ans)):
        sigma = sigma_s[ii]+sigma_dm[ii]
        #integrad = lambda zz: f(zz)* (get_sigma_crit_inv(lzred, zz)+ get_sigma_crit_inv(lzred, zz)**2)
        integrad = lambda zz: f(zz)* 1/(1-sigma*(get_sigma_crit_inv(lzred, zz)*4*np.pi*gee*1.0/cee**2))

        ans[ii]=  sum(integrad(zzarr))*(zzarr[1] -zzarr[0])/(sum(f(xzzarr))*(xzzarr[1] -xzzarr[0]))
        #print(sigma,ans[ii])
        #print(get_sigma_crit_inv(lzred, zzarr[:5])*4*np.pi*gee*1.0/cee**2)
       #ans[ii]=  quad(integrad, zmin, zmax)[0]/quad(f, 0, zmax)[0]
    return ans


def model(x, zred, rbins, Om0=0.25):
    logmstel, log_re, logmh, cfac = x
    # we are evaluating at redshift of 0.3
    lconc   = 1.0#concentration.concentration(10**logmh, '200m', 0.2, model = 'diemer19')
    conc    =   cfac * lconc
    hp          = halo(logmh, conc, omg_m=Om0)
    stel        = stellar(logmstel, log_re=log_re + np.log10(1+zred))
    sigma_s    = stel.sigma_deVaucouleurs(rbins*(1+zred)) 
    sigma_dm   = hp.sigma_nfw(rbins*(1+zred))         
 
    esd_s       = stel.esd_deVaucouleurs(rbins*(1+zred))
    esd_dm      = hp.esd_nfw(rbins*(1+zred))

    sigma_s     =   sigma_s*(1+zred)**2  
    sigma_dm    =   sigma_dm*(1+zred)**2 
    ans         =   get_avg_sigmacritinv(sigma_s,sigma_dm,zred)
    return (1+zred)**2 * esd_s/1e12, (1+zred)**2 * esd_dm/1e12, sigma_s, sigma_dm, ans


outputfilename = '/net/dobbe/data2/github/gammat_scatter/output/test_debug_z_0.1_0.4/test_dsigma.dat_lmstelmin_9.00_lmstelmax_10.50'

rbins, dsigma, dsigmaerr, dsigmax, dsigmaxerr = np.loadtxt(outputfilename, unpack=1)
idx = rbins>0.004
rbins       =   rbins       [idx]
dsigma      =   dsigma      [idx]
dsigmaerr   =   dsigmaerr   [idx]

plt.subplot(2,2,1)
yy = dsigma * (1-0.26**2)
plt.errorbar(rbins, yy, yerr=dsigmaerr, fmt='.')
x = [0.2001691200597535, 10.054764747619629, -2.6006509190525486, 11.906508445739746, 11.369261778278597]
zred = x[0]
esd_stel, esd_dm, sigma_stel, sigma_dm, ans = model(x[1:], zred, rbins)
print(yy)
print(esd_stel+esd_dm)
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


