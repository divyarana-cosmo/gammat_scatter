import numpy as np
import matplotlib.pyplot as plt
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d


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
        print("fixing cosmology \n")

    def get_xyz(self, ra,dec):
        theta = (90-dec)*np.pi/180
        phi = ra*np.pi/180
        z = np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        x = np.cos(phi)*np.sin(theta)
        return x,y,z  

    def _get_sigma_crit_inv(self, lzred, szred):
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
        sigm_crit_inv[~idx]=0.0 
        sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
        return sigm_crit_inv


    def _interp_get_sigma_crit_inv(self, lzred):
        xx = np.linspace(lzred+1e-4,4.0, 100)
        yy = self._get_sigma_crit_inv(lzred, xx)
        return interp1d(xx, yy, kind='cubic')

    def _get_esd(self, logmstel, logre, logmh, lconc, proj_sep):
        self.hp    = halo(logmh, lconc, omg_m=self.omg_m)
        self.stel  = stellar(logmstel, log_re = logre)
        esd_s      = self.stel.esd_deVaucouleurs(proj_sep)   
        esd_dm     = self.hp.esd_nfw(proj_sep)           
        sigma_s    = self.stel.sigma_deVaucouleurs(proj_sep) 
        sigma_dm   = self.hp.sigma_nfw(proj_sep)         
        return esd_s, esd_dm, sigma_s, sigma_dm 

    def _get_g(self, logmstel, logre, logmh, lconc, lzred, szred, proj_sep):
        if not self.init_spl_sigma_crit_inv:
            self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)
            self.init_spl_sigma_crit_inv = True
        if not np.isscalar(lzred) and not np.isscalar(szred):
            get_sigma_crit_inv =   self._get_sigma_crit_inv(lzred, szred)
        else:
            get_sigma_crit_inv = self.interp_get_sigma_crit_inv(szred) 

        get_sigma_crit_inv = self.interp_get_sigma_crit_inv(szred) 
        esd_s, esd_dm, sigma_s, sigma_dm =  self._get_esd(logmstel, logre, logmh, lconc, proj_sep)
        #considering only tangential shear and adding both contributions
        gamma_s     =   esd_s     * get_sigma_crit_inv 
        gamma_dm    =   esd_dm    * get_sigma_crit_inv
        kappa_s     =   sigma_s   * get_sigma_crit_inv
        kappa_dm    =   sigma_dm  * get_sigma_crit_inv
        return gamma_s, gamma_dm, kappa_s, kappa_dm

    def get_g(self, lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, use_shear=False, no_shear=False):
        "computes the g1 and g2 components for the reduced shear"
        lx, ly, lz = self.get_xyz(lra, ldec) 
        sx, sy, sz = self.get_xyz(sra, sdec) 
        #projected separation on the lense plane in physical units
        proj_sep = self.Astropy_cosmo.angular_diameter_distance(lzred).value * np.sqrt((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2) # in h-1 Mpc
       #considering only tangential shear and adding both contributions
        gamma_s, gamma_dm, kappa_s, kappa_dm = self._get_g(logmstel, logre, logmh, lconc, lzred, szred, proj_sep)
        sflag = (gamma_s != -999) & (gamma_dm != -999) & (kappa_s != -999) & (kappa_dm != -999)
        gamma = gamma_s + gamma_dm
        kappa = kappa_s + kappa_dm
        if use_shear:
            g    = gamma        # shear
            g_b  = gamma_s      # shear
            g_dm = gamma_dm     # shear
        else:
            g    = gamma/(1.0 - kappa)          # reduced shear
            g_b  = gamma_s/(1.0 - kappa_s)      # reduced shear
            g_dm = gamma_dm/(1.0 - kappa_dm)    # reduced shear

        sflag = sflag & (np.abs(kappa)<0.5) & (np.abs(g)<1)  #weak lensing flag 
        # phi to get the compute the tangential shear
        lra  = lra*np.pi/180
        ldec = ldec*np.pi/180
        sra  = sra*np.pi/180
        sdec = sdec*np.pi/180

        c_sra_lra = np.cos(sra)*np.cos(lra) + np.sin(lra)*np.sin(sra)
        s_sra_lra = np.sin(sra)*np.cos(lra) - np.cos(sra)*np.sin(lra)
        #angular separation between lens-source pairs
        c_theta = lx*sx + ly*sy + lz*sz
        s_theta = np.sqrt(1-c_theta**2)
        c_phi   =  np.cos(ldec)*s_sra_lra*1.0/s_theta
        s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*c_sra_lra*np.sin(sdec))*1.0/s_theta
        # tangential shear
        g_1     = - g*(2*c_phi**2 - 1)
        g_2     = - g*(2*c_phi * s_phi)
        if no_shear:
            g       = 0.0*g_1
            g_1     = 0.0*g_1
            g_2     = 0.0*g_1
        return g_1, g_2, g, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm


    def shear_src(self, lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, intse1, intse2, use_shear=False, no_shear=False):
        "apply shear on to the source galaxies with given intrinsic shapes"
        se1 = intse1;   se2 = intse2
        if not self.init_spl_sigma_crit_inv:
            self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)
            self.init_spl_sigma_crit_inv = True
        g_1, g_2, gtan, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm = self.get_g(lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, use_shear=use_shear, no_shear=no_shear)

        g   = g_1 + 1j* g_2
        es  = se1 + 1j* se2 + 0.0*g  # intrinsic sizes
        e   = 0.0*es # sheared shapes
        #using the seitz and schnider 1995 formalism to shear the galaxy
        idx = np.abs(g)<=1
        e[idx] = (es[idx] + g[idx])/(1.0 + np.conj(g[idx])*es[idx])
        e[~idx] = (1 + g[~idx]*np.conj(es[~idx]))/(np.conj(es[~idx]) + np.conj(g[~idx])) # mod(g)>1
        #observed quantities
        etan_obs = -np.real(e*(2*c_phi**2 - 1 - 1j *(2*c_phi*s_phi)))
        ex_obs   = -np.imag(e*(2*c_phi**2 - 1 - 1j *(2*c_phi*s_phi)))
        return np.real(e), np.imag(e), gtan, kappa, proj_sep, sflag, g_b, g_dm, etan_obs, ex_obs


if __name__ == "__main__":
    ss = simshear()
    proj_sep = np.logspace(np.log10(0.005), np.log10(0.3),10)
    
    from time import time
    begin = time()
    gamma_s, gamma_dm, kappa_s, kappa_dm = ss._get_g(logmstel=10, logre=-3, logmh=12, lconc=5.98, lzred=0.3, szred=0.8 + 0.0*proj_sep, proj_sep=proj_sep)
    
    g_s = gamma_s/(1-kappa_s)
    g_dm = gamma_dm/(1-kappa_dm)
    g_tot = (gamma_s + gamma_dm)/(1-kappa_s-kappa_dm)
    print( ss._get_sigma_crit_inv(lzred=0.5, szred=1.0))
    sigcrit_inv = ss._get_sigma_crit_inv(lzred=0.5, szred=1.0)
   
    plt.subplot(2,2,1)
    plt.plot(proj_sep, g_s/(sigcrit_inv*1e12))
    plt.plot(proj_sep, g_dm/(sigcrit_inv*1e12))
    plt.plot(proj_sep, g_tot/(sigcrit_inv*1e12))


    from halopy import halo
    from stellarpy import stellar

    hp = halo(log_mtot = 12, con_par=5.98, omg_m=0.25)
    stel = stellar(log_mstel=10, log_re=-3)

    esd_s              = stel.esd_deVaucouleurs(proj_sep)/1e12   
    esd_dm             = hp.esd_nfw(proj_sep)/1e12           
    plt.plot(proj_sep, esd_s, '.')
    plt.plot(proj_sep, esd_dm, '.')
    plt.plot(proj_sep, esd_s+esd_dm,'.')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.savefig('test.png', dpi=300)


