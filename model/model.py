import sys
import os
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
from colossus.halo import concentration

sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from stellarpy import stellar
from distort_com import simshear
#integration
from scipy.integrate import quad
#interpolation
from scipy.interpolate import InterpolatedUnivariateSpline as ius
#lens sample
sys.path.append('/home/rana/github_0/gammat_scatter/')
from weakpipe_select import lens_select

import gc

class model():
    def __init__(self, H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff, zsrcmin=0.0, zsrcmax=3.0):
        "initialization parameters"
        self.H0         = H0
        self.Om0        = Om0
        self.logMmin    = logMmin
        self.logMmax    = logMmax
        params          = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        self.cosmo      = cosmology.setCosmology('myCosmo', **params)
        self.ss         = simshear(H0=H0, Om0=Om0)
        
        # lense sample selection for redshift
        self.get_pzl()
        self.precompute_esd_s()


        # setting the integration limits over source redshift
        self.zlmax      =   zlmax
        self.zdiff      = 1e-10+zdiff
        self.zsrcmin    = zsrcmin
        self.zsrcmax    = zsrcmax

        # source redshift distribution normalization and 1/(1-kappa) averaging
        self.Norm = quad(self.nsrc, self.zlmax + self.zdiff, self.zsrcmax)[0]
        
        # values for the spl
        self.spl_rbins = np.logspace(-3, 0, 30)


 
    def get_pzl(self):
        "creates a probability distribution for the lens redshifts"
        fpath = './precompute/'+'pzl_%s_%s.dat'%(self.logMmin, self.logMmax)
        if not os.path.exists(fpath):
            print('please run the precompute first')
            exit()
 
        self.zlbins, self.pzl, mean_redshift = np.loadtxt(fpath, unpack=1) 
        self.mean_lzred         =   np.unique(mean_redshift)[0]
        return 0

    def precompute_esd_s(self):
        fpath = './precompute/'+'esd_s_%s_%s.dat'%(self.logMmin, self.logMmax)
        if not os.path.exists(fpath):
            print('please run the precompute first')
            exit()
        rarr, self.esd_s, self.sigma_s    = np.loadtxt(fpath, unpack=1)
        #rarr, esdarr, sigarr    = np.loadtxt(fpath, unpack=1)
        #self.spl_log_esd_s      = ius(np.log10(rarr), np.log10(esdarr))
        #self.spl_log_sigma_s    = ius(np.log10(rarr), np.log10(sigarr))
        #del rarr, esdarr, sigarr
        #gc.collect()
        print("putting splines on precomputations for stellar contribution done")
        return 0    


    def nsrc(self,z):
        "assigns redshifts respecting the distribution"
        z0 = 0.9/(2)**0.5
        f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
        return f(z)

    def set_esd_spl(self,x, reduced=True):
        logalpha, logmh, cfac = x
        rbins = self.spl_rbins
        
        cosmology.setCurrent(self.cosmo)
        self.lconc = cfac #* concentration.concentration(10**logmh, '200m', self.mean_lzred, model='diemer19')
    
        self.esd_dm, self.sigma_dm = self.ss._get_esd_dm(logmh=logmh, lconc=self.lconc, proj_sep=rbins)
        sigma       = 10**logalpha * self.sigma_s    + self.sigma_dm
        delta_sigma = 10**logalpha * self.esd_s      + self.esd_dm
    
        if not reduced:
            esd = delta_sigma / 1e12
        else: 
            # Normalize n(z) over valid source redshift range
            zmin = self.zlmax + self.zdiff
            def integrand_num(zs,n):
                if zs <= zmin: return 0.0
                nz = self.nsrc(zs)
                siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)
                return nz * np.dot(self.pzl, siginv**(n))

            n=1
            kappa = sigma *  quad(integrand_num, zmin, self.zsrcmax, args=(n,))[0] 

            kappa = kappa / self.Norm#quad(integrand_den, zmin, self.zsrcmax)[0]
            # Compute reduced ESD for each R bin
            ds_reduced =  delta_sigma/(1-kappa)            
            esd =  ds_reduced / 1e12
        
        return ius(np.log10(rbins), np.log10(esd))


    def esd(self, x, rbins, reduced=True):
        """
        Predicts ESD profile (in M_sun/pc^2). If `reduced=True`, returns g * sigma_crit using integrated correction.
        """
 
        loglogspl = self.set_esd_spl(x, reduced=reduced)

        logrbins = np.log10(rbins)
        yy      =   0.0*rbins

        logrdiff = logrbins[1] - logrbins[0]
        logrbins = np.append(logrbins-logrdiff, logrbins[-1] + logrdiff)

        for ii,(rmin,rmax) in enumerate(zip(10**logrbins[:-1], 10**logrbins[1:])):
            print(rmin, rmax)
            yy[ii] = quad(lambda x: 10**loglogspl(np.log10(x)) * x, rmin, rmax)[0]*2/((rmax**2 - rmin**2))
        
        return yy

if __name__ == "__main__":
    # for the test case 
    H0          =   100
    Om0         =   0.25
    lenstype    =   'desi'    
    logMmin     =   9.5
    logMmax     =   11.0
    zlmin       =   0.1
    zlmax       =   0.4
    Njacks      =   50
    zdiff       =   0.0

    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

    logmstel    =   10.47462749
    log_re      =   -2.38450355
    logmh       =   12.42861652
    #cfac        =   1.0
    cfac        =   0.8

    logalpha       =   0.0
    x = [logalpha, logmh, cfac]
    rbins = np.logspace(np.log10(0.002), np.log10(0.4),10)
    import time
    begin = time.time()
    red_esd     = mm.esd( x, rbins)
    print(time.time() - begin)
    begin = time.time()
    gamma_esd   = mm.esd( x, rbins, reduced=False)
    print(time.time() - begin)
    import matplotlib.pyplot as plt
    #print(red_esd)
    #print(gamma_esd)

    siginv = mm.ss._get_sigma_crit_inv(np.array([0.3]), 3.0)
    #print(mm.esd_dm/1e12)

    plt.subplot(2,2,1)
    plt.plot(rbins, red_esd, label='$g$')
    plt.plot(rbins, gamma_esd, label='$\gamma$')
    plt.xlabel('$R_p$')
    plt.ylabel('$\Delta \Sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('test.png', dpi=300)


#def esd(self, x, rbins, reduced=True):
#    """
#    Predicts ESD profile (in M_sun/pc^2). If `reduced=True`, returns g * sigma_crit using integrated correction.
#    """
#    logalpha, logmh, cfac = x
#    
#    cosmology.setCurrent(self.cosmo)
#    self.lconc = cfac #* concentration.concentration(10**logmh, '200m', self.mean_lzred, model='diemer19')
#
#    self.esd_dm, self.sigma_dm = self.ss._get_esd_dm(logmh=logmh, lconc=self.lconc, proj_sep=rbins)
#    sigma       = 10**logalpha * self.sigma_s    + self.sigma_dm
#    delta_sigma = 10**logalpha * self.esd_s      + self.esd_dm
#    #sigma       = 10**logalpha * 10**self.spl_log_sigma_s(np.log10(rbins))    + sigma_dm
#    #delta_sigma = 10**logalpha * 10**self.spl_log_esd_s(np.log10(rbins))  + self.esd_dm
#
#    if not reduced:
#        return delta_sigma / 1e12
#
#    # Normalize n(z) over valid source redshift range
#    zmin = self.zlmax + self.zdiff
#    def integrand_num(zs,n):
#        if zs <= zmin: return 0.0
#        nz = self.nsrc(zs)
#        siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)
#        return nz * np.dot(self.pzl, siginv**(n))

#    def integrand_den(zs):
#        if zs <= zmin: return 0.0
#        nz = self.nsrc(zs)
#        siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)
#        return nz * np.dot(self.pzl, siginv**2)

#    kappa = 0

#    for n in range(1,2):
#        kappa += sigma**n *  quad(integrand_num, zmin, self.zsrcmax, args=(n,))[0] 

#    kappa = kappa / self.Norm#quad(integrand_den, zmin, self.zsrcmax)[0]
#    # Compute reduced ESD for each R bin
#    ds_reduced =  delta_sigma*(1+kappa)            
#    return ds_reduced / 1e12

#def esd(self, x, rbins, reduced=True):
#    """
#    Predicts ESD profile (in M_sun/pc^2). If `reduced=True`, returns g * sigma_crit using integrated correction.
#    """

#    logrbins = np.log10(rbins)
#    logrdiff = logrbins[1] - logrbins[0]
#    logrbins = np.append(logrbins-logrdiff, logrbins[-1] + logrdiff)

#    logalpha, logmh, cfac = x
#    
#    cosmology.setCurrent(self.cosmo)
#    self.lconc = cfac #* concentration.concentration(10**logmh, '200m', self.mean_lzred, model='diemer19')
#
#    self.esd_dm, self.sigma_dm = self.ss._get_esd_dm(logmh=logmh, lconc=self.lconc, proj_sep=rbins)
#    sigma       = 10**logalpha * self.sigma_s    + self.sigma_dm
#    delta_sigma = 10**logalpha * self.esd_s      + self.esd_dm
#    #sigma       = 10**logalpha * 10**self.spl_log_sigma_s(np.log10(rbins))    + sigma_dm
#    #delta_sigma = 10**logalpha * 10**self.spl_log_esd_s(np.log10(rbins))  + self.esd_dm
#
#    if not reduced:
#        return delta_sigma / 1e12
#
#    # Normalize n(z) over valid source redshift range
#    zmin = self.zlmax + self.zdiff
#    def integrand(zs, i):
#        if zs <= zmin: return 0.0
#        nz = self.nsrc(zs)
#        siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)
#        gsc = delta_sigma[i] / (1 - sigma[i] * siginv)
#        return nz * np.dot(self.pzl, gsc)


#    # Compute reduced ESD for each R bin
#    ds_reduced = np.array([
#        quad(integrand, zmin, self.zsrcmax, args=(i,))[0] / self.Norm
#        for i in range(len(rbins))
#    ])
#
#    return ds_reduced / 1e12



