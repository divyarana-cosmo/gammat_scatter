import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.special import gammainc

class constants:
    """Useful constants"""
    G = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0 = 100. #h kms-1 Mpc-1 hubble constant at present

class stellar(constants):
    """Useful functions for weak lensing signal modelling"""
    def __init__(self, log_mstel):
        self.log_mstel  = log_mstel # total mass of the halo
        
        #self.log_re     = 0.7*(0.774 + 0.977 *(log_mstel - np.log10(0.7) - 11.4)) #check arxiv:1811.04934
        # correcting for the h-1 factors and also kpc to Mpc
        self.log_re     = (0.774 + 0.977 *(np.log10(10**log_mstel / 0.7) - 11.4)) #check arxiv:1811.04934
        self.log_re     = np.log10(10**self.log_re * 0.7/1e3) #h-1 kpc to h-1 Mpc

    def esd_pointmass(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            r = np.array([r])
        val = self.avg_sigma_pointmass(r) - self.sigma_pointmass(r)
        #idx =  r<5e-3
        #val[idx] = 0.0
        return val

    def sigma_pointmass(self,r):
        "delta function projected profile"
        if np.isscalar(r):
            r = np.array([r])
        val = 0.0 * r
        #idx = r>0
        #if sum(idx)!=0:
        #    val[idx]=0

        #val[~idx] = np.inf
        return val

    def avg_sigma_pointmass(self,r):
        """analytical average projected of pointmass profile"""
        if np.isscalar(r):
            r = np.array([r])
        #idx = r<5e-3
        #r[idx] = 5e-3
        return 10**self.log_mstel*1.0/(np.pi*r**2)


    def esd_deVaucouleurs(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            r = np.array([r])
        val = self.avg_sigma_deVaucouleurs(r) - self.sigma_deVaucouleurs(r)
        return val

    def sigma_deVaucouleurs(self,r):
        "deVaucouleurs projected profile"
        if np.isscalar(r):
            r = np.array([r])
        
        b       = 7.669
        re      = 10**self.log_re
        sigma0  = 10**self.log_mstel * b**8/(8* np.pi * re**2 * gamma(8))

        val     = sigma0 * np.exp(-b*(r/re)**0.25)
        return val

    def avg_sigma_deVaucouleurs(self,r):
        """analytical average projected of deVaucouleurs profile"""
        if np.isscalar(r):
            r = np.array([r])

        b       = 7.669
        re      = 10**self.log_re
        sigma0  = 10**self.log_mstel * b**8/(8* np.pi * re**2 * gamma(8))
        
        t       = (b**4 * r/re)**0.25

        val     = 8*np.pi*sigma0*re**2/b**8 
        val     = val * gammainc(8.0, t)*gamma(8)
        return val/(np.pi*r**2)



if __name__ == "__main__":
    plt.subplot(2,2,1)
    rbin = np.logspace(-4,np.log10(5),10)
    hp = stellar(11)
    print(hp.log_re)
    #print hp.r_200
    yy = hp.esd_pointmass(rbin)/(1e12)
    print(yy)
    plt.plot(rbin, yy)
    yy = hp.esd_deVaucouleurs(rbin)/(1e12)
    print(yy)
    plt.plot(rbin, yy)
    
    from halopy import halo
    hp = halo(11.5,4)
    yy = hp.esd_nfw(rbin)/1e12
 
    plt.plot(rbin, yy)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma (R) [{\rm h M_\odot pc^{-2}}]$')
    plt.savefig('test.png', dpi=300)

