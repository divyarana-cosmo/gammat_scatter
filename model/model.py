import sys
import os
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
from colossus.halo import concentration

sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from distort_com import simshear
#integration
from scipy.integrate import quad
#lens sample
sys.path.append('/home/rana/github_0/gammat_scatter/')
from get_data import lens_select

import gc

class model():
    def __init__(self, H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff, zsrcmin=0.0, zsrcmax=3.0):
        "initialization parameters"
        self.H0         = H0
        self.Om0        = Om0
        params          = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        cosmo           = cosmology.setCosmology('myCosmo', **params)
        self.ss         = simshear(H0=H0, Om0=Om0)
        
        # lense sample selection for redshift
        self.get_pzl(lenstype, logMmin, logMmax, zlmin, zlmax, Njacks)


        # setting the integration limits over source redshift
        self.zdiff    = 1e-10+zdiff
        self.zsrcmin  = zsrcmin
        self.zsrcmax  = zsrcmax

        # source redshift distribution normalization and 1/(1-kappa) averaging
        self.Norm       = quad(self.nsrc, self.zsrcmin, self.zsrcmax)[0]
        self.addon      = quad(self.nsrc, 0.0, self.zlmax+self.zdiff)[0]


 
    def get_pzl(self, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, Nzlbins=101):
        "creates a probability distribution for the lens redshifts"
        lensargs = {}
        lensargs['type']        =   lenstype
        lensargs['logmstelmin'] =   logMmin
        lensargs['logmstelmax'] =   logMmax
        lensargs['zmin']        =   zlmin
        lensargs['zmax']        =   zlmax
        lensargs['Njacks']      =   Njacks
        lensargs['H0']          =   self.H0
        lensargs['Om0']         =   self.Om0
    
        lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs)
        zzbins                  = np.linspace(zlmin, zlmax, Nzlbins)
        nlens,binedgs           = np.histogram(lzred, bins=zzbins)
        self.mean_lzred         = np.mean(lzred)
        self.zlmax              = zlmax
        del lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        gc.collect()
        
        self.zlbins             =   binedgs[1:]*0.5 + binedgs[:-1]*0.5
        self.pzl                =   nlens/sum(nlens)
        return 0

    def nsrc(self,z):
        "assigns redshifts respecting the distribution"
        z0 = 0.9/(2)**0.5
        f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
        return f(z)


    def esd(self, x, rbins, reduced=True):
        "predicts the total esd profile after adding baryon+dark-matter"
        logmstel, log_re, logmh, cfac = x
        self.lconc   = cfac*concentration.concentration(10**logmh, '200m', self.mean_lzred, model = 'diemer19')

        # we are getting the esd profiles.
        self.esd_s, self.esd_dm, sigma_s, sigma_dm  = self.ss._get_esd(logmstel=logmstel, logre=log_re, logmh=logmh, lconc=self.lconc, proj_sep=rbins)
        esd     =   self.esd_s + self.esd_dm
        if not reduced:
            return esd/1e12
        sigma   =   sigma_s + sigma_dm
        # we are averaging over the 1/(1-kappa) to correct for the reduced shear
        for ii, rr in enumerate(rbins):
            integrand   =   lambda xx: self.nsrc(xx) * sum(self.pzl * 1/(1-(sigma[ii]*self.ss._get_sigma_crit_inv(lzred=self.zlbins, szred= 0.0*self.zlbins + xx))))

            esd[ii]     *= ((self.addon + quad(integrand, self.zlmax+self.zdiff, self.zsrcmax)[0])/self.Norm)
        return esd/1e12



if __name__ == "__main__":
    # for the test case 
    H0          =   100
    Om0         =   0.25
    lenstype    =   'test_desi'    
    logMmin     =   9.5
    logMmax     =   11.0
    zlmin       =   0.1
    zlmax       =   0.4
    Njacks      =   50
    zdiff       =   0.1

    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)

    logmstel    =   10.47462749
    log_re      =   -2.38450355
    logmh       =   12.42861652
    cfac        =   1.0

    x = [logmstel, log_re, logmh, cfac]
    rbins = np.logspace(-3,0,20)
    import time
    begin = time.time()
    red_esd     = mm.esd( x, rbins)
    gamma_esd   = mm.esd( x, rbins, reduced=False)
    print(time.time() - begin)
    import matplotlib.pyplot as plt

    plt.subplot(2,2,1)
    plt.plot(rbins, red_esd, label='$g$')
    plt.plot(rbins, gamma_esd, label='$\gamma$')
    plt.xlabel('$R_p$')
    plt.ylabel('$\Delta \Sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('test.png', dpi=300)


