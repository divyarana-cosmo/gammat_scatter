#need to add generalized halo profile

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

class constants:
    """Useful constants"""
    G   = 4.300917270038e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0  = 100. #h km s-1 Mpc-1 hubble constant at present

class halo(constants):
    """Useful functions for weak lensing signal modelling"""
    def __init__(self,log_mtot, con_par, omg_m=0.3, beta=None, Rmin=0.0001, Rmax=5, Rbins=80):
        self.m_tot = 10**log_mtot # total mass of the halo
        self.c = con_par # concentration parameter
        self.omg_m = omg_m
        self.rho_crt = 3*self.H0**2/(8*np.pi*self.G) # rho critical
        self.rho_m  =   self.rho_crt*self.omg_m 
        self.r_200 = (3*self.m_tot/(4*np.pi*200*self.rho_m))**(1./3.) # radius defines size of the halo
        self.rho_0 = con_par**3 *self.m_tot/(4*np.pi*self.r_200**3 *(np.log(1+con_par)-con_par/(1+con_par)))

        self.spl_esd_rmin   = Rmin
        self.spl_esd_rmax   = Rmax
        self.spl_esd_rbins  = Rbins

        self.init_spl_esd_nfw = False
        self.init_spl_sigma_nfw = False


        self.init_spl_avg_sigma_gnfw = False
        self.init_spl_sigma_gnfw = False


        if beta is not None :
            self.beta  = beta
            self.r_s = self.r_200/self.c
            self.delta_c = (200*self.c**3)/(3*quad(lambda x: x**(2-self.beta)*(1+x)**(self.beta -3), 0.0, self.c)[0])
            
            self.rho0_gnfw = self.delta_c * self.rho_crt * self.omg_m             
            #self.rho0_gnfw = self.m_tot/(4*np.pi*quad(lambda r: r**2/((r/r_s)**self.beta * (1 + r/r_s)**(3 - self.beta)), 0.0, self.r_200)[0])

    def nfw(self,r):
        """given r, this gives nfw profile as per the instantiated parameters"""
        r_s = self.r_200/self.c
        value  = self.rho_0/((r/r_s)*(1+r/r_s)**2)
        return value

    def esd_nfw(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            r = np.array([r])
        sig = 0.0*r
        sig = self.avg_sigma_nfw(r) - self.sigma_nfw(r)
        return sig


    def avg_sigma_nfw(self,r):
        """analytical projection of NFW"""
        if np.isscalar(r):
            r = np.array([r])
        r_s = self.r_200/self.c
        k = 2*r_s*self.rho_0
        x = r/r_s
        value = 0.0*x
        idx =  (x < 1)
        value[idx] = np.arccosh(1/x[idx])/np.sqrt(1-x[idx]**2) + np.log(x[idx]/2.0)
        value[idx] = value[idx]*2.0/x[idx]**2
        idx = (x > 1)
        value[idx] = np.arccos(1/x[idx])/np.sqrt(x[idx]**2-1)  + np.log(x[idx]/2.0)
        value[idx] = value[idx]*2.0/x[idx]**2
        idx = (x == 1)
        value[idx] = 2*(1-np.log(2))
        sig = value*k
        return sig


    def sigma_nfw(self,r):
        """analytical projection of NFW"""
        if np.isscalar(r):
            r = np.array([r])

        r_s = self.r_200/self.c
        k = 2*r_s*self.rho_0
        value =0.0*r
        x = r/r_s
        idx = x < 1
        value[idx] = (1 - np.arccosh(1/x[idx])/np.sqrt(1-x[idx]**2))/(x[idx]**2-1)
        idx = x > 1
        value[idx] = (1 - np.arccos(1/x[idx])/np.sqrt(x[idx]**2-1))/(x[idx]**2-1)
        idx = x == 1
        value[idx] = 1./3.

        sig = value*k
        return sig


    def gnfw(self,r):
        """given r, this gives generalized nfw profile as per the instantiated parameters"""
        # following arxiv:0007354
        r_s = self.r_200/self.c
        value  = self.rho0_gnfw/((r/r_s)**self.beta*(1 + r/r_s)**(3 - self.beta))
        return value

    def esd_gnfw(self,r):
        """ESD profile from interpolated predictions"""
        if np.isscalar(r):
            r = np.array([r])
        sig = self.avg_sigma_gnfw(r) - self.sigma_gnfw(r)
        return sig


    def avg_sigma_gnfw(self,r):
        """projected average profile of generalized NFW"""
        if not self.init_spl_sigma_gnfw:
            self.get_spl_sigma_gnfw()
        if np.isscalar(r):
            r = np.array([r])
        _xx = r
        Sigmaarr = _xx*0.0
        for ii, x in enumerate(_xx):
            Sigmaarr[ii] = 2 * np.pi * quad((lambda t :t * 10**self.spl_sigma_gnfw(np.log10(t))), 0, x)[0]
            
        return Sigmaarr/(np.pi*r**2)


    def sigma_gnfw(self,r):
        """projected profile of generalized NFW"""
        if not self.init_spl_sigma_gnfw:
            self.get_spl_sigma_gnfw()
        if np.isscalar(r):
            r = np.array([r])
        return 10**self.spl_sigma_gnfw(np.log10(r))

    def get_spl_sigma_gnfw(self):
        """puts a log log interpolation scheme for the sigma"""
        xx = np.logspace(np.log10(self.spl_esd_rmin), np.log10(self.spl_esd_rmax), self.spl_esd_rbins)
        _xx = xx/self.r_s
        Sigmaarr = _xx*0.0
        for ii, x in enumerate(_xx):
            Sigmaarr[ii] = 2 * self.delta_c * self.rho_crt * self.omg_m * self.r_s * x**(1-self.beta) * quad((lambda theta : np.sin(theta)*(np.sin(theta) + x)**(self.beta -3)), 0, np.pi/2)[0]
 
        self.spl_sigma_gnfw = interp1d(np.log10(xx), np.log10(Sigmaarr), kind='cubic', fill_value = (np.log10(Sigmaarr[0]), -np.inf), bounds_error=False)
        self.init_spl_sigma_gnfw = True
        return 0

    def num_sigma(self, Rarr, func):
        """numerical test to the analytical part"""
        if np.isscalar(Rarr):
            return 2*quad((lambda z : func(np.sqrt(Rarr**2 + z**2))), 0, 100)[0]
        Sigmaarr = Rarr*0.0
        for ii, R in enumerate(Rarr):
            Sigmaarr[ii] = 2*quad((lambda z : func(np.sqrt(R**2 + z**2))), 0, 100)[0]
        return Sigmaarr

    def num_avg_sigma(self, R, func, extra):
        """numerical computation of mean sigma at R using log-log sigma spline and less than Rpmin integral"""
        if np.isscalar(R):
            return 2*np.pi*(extra + quad(lambda Rp: Rp*10**func(np.log10(Rp)), self.spl_esd_rmin, R)[0])/(np.pi*rr**2)

        value = 0.0*R
        #push in the spline of projected density
        for ii,rr in enumerate(R):
            value[ii] = 2*np.pi*(extra + quad(lambda Rp: Rp*10**func(np.log10(Rp)), self.spl_esd_rmin, rr)[0])/(np.pi*rr**2)
        return value


if __name__ == "__main__":
    omgm0 = 0.3

    from colossus.cosmology import cosmology
    params = {'flat': True, 'H0': 100, 'Om0': omgm0, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
    cosmology.addCosmology('myCosmo', **params)
    cosmo = cosmology.setCosmology('myCosmo')
        
    #zzarr = np.linspace(0,1,10)

    plt.subplot(2,2,1)
    #for zz in zzarr:
    for bb in [0.5,1.0,1.5]:
        hp = halo(log_mtot=14, con_par=3, omg_m=0.3, beta=bb)
        rr = np.logspace(-3,-1, 20)
        plt.plot(rr, (hp.esd_gnfw(rr) + 1e12/rr**2)/1e12, label=r'$\beta=%2.2f$'%(bb))
    
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')

    
    plt.savefig('test.png')







    #ax1 = plt.subplot(2,2,1)
    #ax2 = plt.subplot(2,2,2)
    #
    #rbin = np.logspace(-4,-1, int(30))
    #hp = halo(15,4)
    #print(hp.r_200)
    #ax1.plot(rbin, hp.esd_nfw(rbin)/(1e12), '-')
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')

    #ax2.plot(rbin, hp.sigma_nfw(rbin)/(1e12), '-')
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')

    #plt.savefig('test.png', dpi=300)

