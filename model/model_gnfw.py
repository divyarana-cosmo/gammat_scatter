import sys
import os
import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
from colossus.halo import concentration

sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from halopy import halo
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
        self.spl_rbins = np.logspace(-3, 0, 101)



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
        self.rbins_esd_s, self.esd_s, self.sigma_s, self.esd_s_sigma_s    = np.loadtxt(fpath, unpack=1)
        #rarr, esdarr, sigarr    = np.loadtxt(fpath, unpack=1)
        self.spl_log_esd_s      = ius(np.log10(self.rbins_esd_s), np.log10(self.esd_s))
        self.spl_log_sigma_s    = ius(np.log10(self.rbins_esd_s), np.log10(self.sigma_s))
        #del rarr, esdarr, sigarr
        #gc.collect()
        print("putting splines on precomputations for stellar contribution done")
        return 0


    def nsrc(self,z):
        "assigns redshifts respecting the distribution"
        z0 = 0.9/(2)**0.5
        f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
        return f(z)



    def set_esd_spl(self, x, rbins, lzred, reduced=True):
        alpha, logmh, cfac,beta = x
        
        hp = halo(log_mtot=logmh, con_par=cfac, omg_m=self.Om0 , beta=beta)

        #rbins = self.rbins_esd_s
        self.esd_dm     = hp.esd_gnfw(r=rbins)
        self.sigma_dm   = hp.sigma_gnfw(r=rbins)
        sigma       = alpha * 10**(self.spl_log_sigma_s(np.log10(rbins))) + self.sigma_dm
        delta_sigma = alpha * 10**(self.spl_log_esd_s(np.log10(rbins))) + self.esd_dm

        #return delta_sigma, sigma

        if not reduced:
            return rbins, delta_sigma / 1e12
        else:
            zmin = self.zlmax + self.zdiff

            # Precompute sigma^n for each order (shape: n_rbins)
            # We'll compute this inside the loop for each n

            correction = np.zeros_like(sigma)
            max_order = 5

            for n in range(1, max_order + 1):

                def integrand_kappa_power_n(zs):
                    """
                    Compute ??^n? for all radial bins at once (but returns scalar for quad)
                    We'll call this for each radial bin separately
                    """
                    if zs <= zmin:
                        return 0.0
                    nz = self.nsrc(zs) / self.Norm
                    siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)

                    # Average over lens redshifts: ??_crit^(-n)?
                    avg_siginv_n = np.dot(self.pzl, siginv ** n)

                    return nz * avg_siginv_n

                # Integrate to get ??_crit^(-n)? averaged over sources
                avg_siginv_n = quad(integrand_kappa_power_n, zmin, self.zsrcmax,
                                   epsabs=1e-10, epsrel=1e-8)[0]

                # Now multiply by ?^n for each radial bin
                kappa_n_avg = (sigma ** n) * avg_siginv_n

                correction += kappa_n_avg

            # Apply correction
            ds_reduced = delta_sigma * (1.0 + correction)
            return rbins, ds_reduced / 1e12



    def esd(self, x, rbins, reduced=True):
        """
        Predicts ESD profile (in M_sun/pc^2). If `reduced=True`, returns g * sigma_crit using integrated correction.
        """

        splrbins, splesd = self.set_esd_spl(x, reduced=reduced)
        return splesd[np.isin(splrbins, rbins)]

if __name__ == "__main__":
    # for the test case
    H0          =   100
    Om0         =   0.25
    lenstype    =   'desi'
    logMmin     =   10.5
    logMmax     =   11.0
    zlmin       =   0.1
    zlmax       =   0.5
    Njacks      =   50
    zdiff       =   0.0

    mm = model(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, zdiff)
    logmh       =   0.0#
    cfac        =   0.8
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)

    for ll in np.linspace(-1,1,5):
        alpha       =   ll
        x = [alpha, logmh, cfac]
        rbins = np.logspace(-2, -1, 10)
        import time
        begin = time.time()
        red_esd     = mm.esd( x, rbins)
        plt.plot(rbins, red_esd, label='$g$')

    plt.xlabel('$R_p$')
    plt.ylabel('$\Delta \Sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('test.png', dpi=300)




