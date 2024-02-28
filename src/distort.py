# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper


import numpy as np
import matplotlib.pyplot as plt
#import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
#from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
#from mpi4py import MPI
#from subprocess import  call

class simshear():
    "simulated the shear for a given configuration of dark matter and stellar profiles"
    def __init__(self, H0=100, Om0=0.25, Ob0=0.044, Tcmb0=2.7255, Neff=3.046, sigma8=0.8, ns=0.95, lzredmin=0.0, lzredmax=1.0, szredmax=4.0):

        "initialize the parameters"
        #fixing the cosmology
        self.omg_m = Om0
        params = dict(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff)
        self.Astropy_cosmo = FlatLambdaCDM(**params)
        colossus_cosmo = cosmology.fromAstropy(self.Astropy_cosmo, sigma8 = sigma8, ns = ns, cosmo_name='my_cosmo')
        self.sigma8 = sigma8
        self.ns = ns
        self.cosmo_name ='my_cosmo'
        self.init_spl_sigma_crit_inv = False
        self.spl_Rmin = 0.001
        self.spl_Rmax = 10
        self.spl_Rbin = 100
        self.spl_Rarr = np.logspace(np.log10(self.spl_Rmin), np.log10(self.spl_Rmax), self.spl_Rbin)
        print("fixing cosmology \n")

    def get_xyz(self, ra,dec):
        theta = (90-dec)*np.pi/180
        phi = ra*np.pi/180
        z = np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        x = np.cos(phi)*np.sin(theta)
        return x,y,z  

    def _get_sigma_crit_inv(self, lzred, szred):
    #def get_sigma_crit_inv(self, lzred, szred):
        "evaluates the lensing efficency geometrical factor"
        sigm_crit_inv = 0.0*szred + 0.0*lzred
        idx =  szred>lzred   # if sources are in foreground then lensing is zero
        if np.isscalar(idx):
            lzred = np.array([lzred])
            szred = np.array([szred])
            idx = np.array([idx])
            sigm_crit_inv = np.array([sigm_crit_inv])

        # some important constants for the sigma crit computations
        gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
        cee = 3e5 #km s^-1
        # sigma_crit_calculations for a given lense-source pair
        sigm_crit_inv = self.Astropy_cosmo.angular_diameter_distance(lzred).value * self.Astropy_cosmo.angular_diameter_distance_z1z2(lzred, szred).value * 1.0/self.Astropy_cosmo.angular_diameter_distance(szred).value

        # If you want to work with comoving cooredinates
        #sigm_crit_inv = self.Astropy_cosmo.angular_diameter_distance(lzred).value * self.Astropy_cosmo.angular_diameter_distance_z1z2(lzred, szred).value * (1.0 + lzred)**2 * 1.0/self.Astropy_cosmo.angular_diameter_distance(szred).value

        sigm_crit_inv[~idx]=0.0 
        sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
        return sigm_crit_inv


    def _interp_get_sigma_crit_inv(self, lzred):
    #def _interp_get_sigma_crit_inv(self, szred):
        xx = np.linspace(lzred+1e-4,4.0, 100)
        #xx = np.linspace(0, szred, 50)
        #yy = self._get_sigma_crit_inv(xx, szred)
        yy = self._get_sigma_crit_inv(lzred, xx)
        return interp1d(xx, yy, kind='cubic')

    def _spl_get_esd(self, logmstel, logmh, lconc):
        self.hp             = halo(logmh, lconc, omg_m=self.omg_m)
        self.stel           = stellar(logmstel)
        #self.spl_Rarr       = np.logspace(-3,1,100)
        _esd_s              = self.stel.esd_deVaucouleurs(self.spl_Rarr)   
        _esd_dm             = self.hp.esd_nfw(self.spl_Rarr)           
        _sigma_s            = self.stel.sigma_deVaucouleurs(self.spl_Rarr) 
        _sigma_dm           = self.hp.sigma_nfw(self.spl_Rarr)         

        self.spl_esd_s      = interp1d(np.log10(self.spl_Rarr), np.log10(_esd_s),       kind='cubic')  
        self.spl_esd_dm     = interp1d(np.log10(self.spl_Rarr), np.log10(_esd_dm),      kind='cubic')  
        self.spl_sigma_s    = interp1d(np.log10(self.spl_Rarr), np.log10(_sigma_s),     kind='cubic')  
        self.spl_sigma_dm   = interp1d(np.log10(self.spl_Rarr), np.log10(_sigma_dm),    kind='cubic')
        return


    def _get_esd(self, logmstel, logmh, lconc, proj_sep):
        self._spl_get_esd(logmstel, logmh, lconc)
        gamma_s     =  -999 + 0.0*proj_sep      
        gamma_dm    =  -999 + 0.0*proj_sep
        kappa_s     =  -999 + 0.0*proj_sep
        kappa_dm    =  -999 + 0.0*proj_sep
        idx =   (proj_sep > self.spl_Rmin) & (proj_sep < self.spl_Rmax)
        gamma_s [idx]    = 10**self.spl_esd_s   (np.log10(proj_sep[idx]))  
        gamma_dm[idx]    = 10**self.spl_esd_dm  (np.log10(proj_sep[idx]))  
        kappa_s [idx]    = 10**self.spl_sigma_s (np.log10(proj_sep[idx]))
        kappa_dm[idx]    = 10**self.spl_sigma_dm(np.log10(proj_sep[idx]))         
        return gamma_s, gamma_dm, kappa_s, kappa_dm 



    def _get_g(self, logmstel, logmh, lconc, lzred, szred, proj_sep):
        if not self.init_spl_sigma_crit_inv:
            self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)
            self.init_spl_sigma_crit_inv = True
        if not np.isscalar(lzred) and not np.isscalar(szred):
            get_sigma_crit_inv =   self._get_sigma_crit_inv(lzred, szred)
        else:
            get_sigma_crit_inv = self.interp_get_sigma_crit_inv(szred) 

        get_sigma_crit_inv = self.interp_get_sigma_crit_inv(szred) 
        gamma_s, gamma_dm, kappa_s, kappa_dm =  self._get_esd(logmstel, logmh, lconc, proj_sep)
        idx = (gamma_s != -999) & (gamma_dm != -999) & (kappa_s != -999) & (kappa_dm != -999)
        #considering only tangential shear and adding both contributions
        gamma_s [idx]    =   gamma_s [idx]  * get_sigma_crit_inv[idx] 
        gamma_dm[idx]    =   gamma_dm[idx]  * get_sigma_crit_inv[idx]
        kappa_s [idx]    =   kappa_s [idx]  * get_sigma_crit_inv[idx]
        kappa_dm[idx]    =   kappa_dm[idx]  * get_sigma_crit_inv[idx]
        return gamma_s, gamma_dm, kappa_s, kappa_dm

    def get_g(self, lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred):
        "computes the g1 and g2 components for the reduced shear"
        lx, ly, lz = self.get_xyz(lra, ldec) 
        sx, sy, sz = self.get_xyz(sra, sdec) 

        #projected separation on the lense plane
        proj_sep = self.Astropy_cosmo.angular_diameter_distance(lzred).value * np.sqrt((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2) # in h-1 Mpc

        #if you want to use the comoving distance
        #proj_sep = self.Astropy_cosmo.comoving_distance(lzred).value * np.sqrt((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2) # in h-1 Mpc

        #considering only tangential shear and adding both contributions
        gamma_s, gamma_dm, kappa_s, kappa_dm = self._get_g(logmstel, logmh, lconc, lzred, szred, proj_sep)
        sflag = (gamma_s != -999) & (gamma_dm != -999) & (kappa_s != -999) & (kappa_dm != -999)
        if np.any(np.isnan(kappa_s))>0:
            kappa_s[np.isnan(kappa_s)] = 0.0

        gamma = gamma_s + gamma_dm
        kappa = kappa_s + kappa_dm

        g    = gamma/(1.0 - kappa) # reduced shear
        g_b  = gamma_s/(1.0 - kappa_s) # reduced shear
        g_dm = gamma_dm/(1.0 - kappa_dm) # reduced shear
        
        
        # phi to get the compute the tangential shear

        lra  = lra*np.pi/180
        ldec = ldec*np.pi/180
        sra  = sra*np.pi/180
        sdec = sdec*np.pi/180

        c_sra_lra = np.cos(sra)*np.cos(lra) + np.sin(lra)*np.sin(sra)
        s_sra_lra = np.sin(sra)*np.cos(lra) - np.cos(sra)*np.sin(lra)
        
        #angular separation between lens-source pairs
        c_theta = lx*sx + ly*sy + lz*sz
        #c_theta = np.cos(ldec)*np.cos(sdec)*c_sra_lra + np.sin(ldec)*np.sin(sdec)
        s_theta = np.sqrt(1-c_theta**2)

        sflag = sflag & (np.abs(s_theta)>np.sin(np.pi/180 * 1/3600)) & (np.abs(kappa)<0.5) & (np.abs(g)<1)  #weak lensing flag and proximity flag

        c_phi   =  np.cos(ldec)*s_sra_lra*1.0/s_theta
        s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*c_sra_lra*np.sin(sdec))*1.0/s_theta
        
        # tangential shear
        g_1     = - g*(2*c_phi**2 - 1)
        g_2     = - g*(2*c_phi * s_phi)

        return g_1, g_2, g, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm


    def shear_src(self, lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred, se1, se2):
        "apply shear on to the source galaxies with given intrinsic shapes"
        self.conc       = lconc#concentration.concentration(10**logmh, '200m', lzred, model = 'diemer19')
        self.hp         = halo(logmh, self.conc, omg_m=self.omg_m)
        self.stel       = stellar(logmstel)

        if not self.init_spl_sigma_crit_inv:
            self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)
            #self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(szred)
            self.init_spl_sigma_crit_inv = True



        g_1, g_2, etan, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm = self.get_g(lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred)
        g   = g_1 + 1j* g_2
        es  = se1 + 1j* se2 + 0.0*g  # intrinsic sizes
        e   = 0.0*es # sheared shapes
        #using the seitz and schnider 1995 formalism to shear the galaxy
        idx = np.abs(g)<=1
        e[idx] = (es[idx] + g[idx])/(1.0 + np.conj(g[idx])*es[idx])
        e[~idx] = (1 + g[~idx]*np.conj(es[~idx]))/(np.conj(es[~idx]) + np.conj(g[~idx])) # mod(g)>1
        return np.real(e), np.imag(e), etan, kappa, proj_sep, sflag, g_b, g_dm









if __name__ == "__main__":
    ss = simshear()

    proj_sep = np.logspace(-3,0,4)
    print(ss._get_g(logmstel=11, logmh=13, lconc=4, lzred=0.5, szred=1.0 + 0.0*proj_sep, proj_sep=proj_sep))


    print( ss._get_sigma_crit_inv(lzred=0.5, szred=1.0))

    plt.plot(proj_sep, ss.stel.esd_deVaucouleurs(proj_sep))
    plt.plot(proj_sep, ss.stel.esd_pointmass(proj_sep))
    plt.plot(proj_sep, ss.hp.esd_nfw(proj_sep)) 
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('test.png')


