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
        #self.spl_log_esd_s      = ius(np.log10(self.rbins_esd_s), np.log10(esdarr))
        #self.spl_log_sigma_s    = ius(np.log10(self.rbins_esd_s), np.log10(sigarr))
        #del rarr, esdarr, sigarr
        #gc.collect()
        print("putting splines on precomputations for stellar contribution done")
        return 0


    def nsrc(self,z):
        "assigns redshifts respecting the distribution"
        z0 = 0.9/(2)**0.5
        f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
        return f(z)





    def set_esd_spl(self, x, reduced=True):
        logalpha, logmh, cfac = x
        rbins = self.rbins_esd_s

        self.lconc = cfac
        self.esd_dm, self.sigma_dm = self.ss._get_esd_dm(logmh=logmh, lconc=self.lconc, proj_sep=rbins)

        sigma = 10**logalpha * self.sigma_s + self.sigma_dm
        delta_sigma = 10**logalpha * self.esd_s + self.esd_dm

        if not reduced:
            return delta_sigma / 1e12
        else:
            zmin = self.zlmax + self.zdiff

            # Precompute sigma^n for each order (shape: n_rbins)
            # We'll compute this inside the loop for each n

            correction = np.zeros_like(sigma)
            max_order = 5

            for n in range(1, max_order + 1):

                def integrand_kappa_power_n(zs):
                    """
                    Compute ⟨κ^n⟩ for all radial bins at once (but returns scalar for quad)
                    We'll call this for each radial bin separately
                    """
                    if zs <= zmin:
                        return 0.0
                    nz = self.nsrc(zs) / self.Norm
                    siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)

                    # Average over lens redshifts: ⟨Σ_crit^(-n)⟩
                    avg_siginv_n = np.dot(self.pzl, siginv ** n)

                    return nz * avg_siginv_n

                # Integrate to get ⟨Σ_crit^(-n)⟩ averaged over sources
                avg_siginv_n = quad(integrand_kappa_power_n, zmin, self.zsrcmax,
                                   epsabs=1e-10, epsrel=1e-8)[0]

                # Now multiply by Σ^n for each radial bin
                kappa_n_avg = (sigma ** n) * avg_siginv_n

                correction += kappa_n_avg

                # Optional: check convergence
                if n > 1 and np.all(np.abs(kappa_n_avg / correction) < 1e-4):
                    print(f"Series converged at order {n}")
                    break

            # Apply correction
            ds_reduced = delta_sigma * (1.0 + correction)

            return ds_reduced / 1e12


    def esd(self, x, rbins, reduced=True):
        """
        Predicts ESD profile (in M_sun/pc^2). If `reduced=True`, returns g * sigma_crit using integrated correction.
        """

        loglogspl = self.set_esd_spl(x, reduced=reduced)
        #return 10**loglogspl(np.log10(rbins))
        return loglogspl

        #logrbins = np.log10(rbins)
        #yy      =   0.0*rbins

        #logrdiff = logrbins[1] - logrbins[0]
        #logrbins = np.append(logrbins-logrdiff, logrbins[-1] + logrdiff)

        #for ii,(rmin,rmax) in enumerate(zip(10**logrbins[:-1], 10**logrbins[1:])):
        #    #print(rmin, rmax)
        #    yy[ii] = quad(lambda x: 10**loglogspl(np.log10(x)) * x, rmin, rmax)[0]*2/((rmax**2 - rmin**2))
        #
        #return yy

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
        logalpha       =   ll
        x = [logalpha, logmh, cfac]
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





"""
    def set_esd_spl(self,x, reduced=True):
        logalpha, logmh, cfac = x
        #rbins = self.spl_rbins
        rbins = self.rbins_esd_s

        #cosmology.setCurrent(self.cosmo)
        self.lconc = cfac #* concentration.concentration(10**logmh, '200m', self.mean_lzred, model='diemer19')

        self.esd_dm, self.sigma_dm = self.ss._get_esd_dm(logmh=logmh, lconc=self.lconc, proj_sep=rbins)
        sigma       = 10**logalpha * self.sigma_s    + self.sigma_dm
        delta_sigma = 10**logalpha * self.esd_s      + self.esd_dm
        #esd_sigma   = 10**(2*logalpha) * self.esd_s_sigma_s +  10**logalpha * self.esd_s  * self.sigma_dm + self.esd_dm * 10**logalpha * self.sigma_s +  self.esd_dm * self.sigma_dm

        if not reduced:
            esd = delta_sigma #/ 1e12
        else:
            # Normalize n(z) over valid source redshift range
            zmin = self.zlmax + self.zdiff
            def integrand_num(zs,n):
                if zs <= zmin: return 0.0
                nz = self.nsrc(zs)/self.Norm
                siginv = self.ss._get_sigma_crit_inv(self.zlbins, zs)
                return nz * np.dot(self.pzl, siginv**(n))

            kappa =0
            for ii in range(1,2):
                kappa += sigma**ii *  quad(integrand_num, zmin, self.zsrcmax, args=(ii,))[0]
            #ii=1
            #avg_inv_sigc = quad(integrand_num, zmin, self.zsrcmax, args=(ii,))[0]/self.Norm

            kappa = kappa #/ self.Norm#quad(integrand_den, zmin, self.zsrcmax)[0]
            # Compute reduced ESD for each R bin
            ds_reduced =  delta_sigma/(1-kappa)
            #ds_reduced =  delta_sigma*(1  + sigma * avg_inv_sigc)
            #ds_reduced =  delta_sigma / (1 - sigma * avg_inv_sigc)
            esd =  ds_reduced


        #return ius(np.log10(rbins), np.log10(esd/1e12))
        return esd/1e12





"""


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



