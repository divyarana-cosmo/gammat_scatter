import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

class constants:
    """Useful constants"""
    G = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0 = 100. #h kms-1 Mpc-1 hubble constant at present

class stellar(constants):
    """Useful functions for weak lensing signal modelling"""
    def __init__(self, log_mstel):
        self.log_mstel = log_mstel # total mass of the halo
        print("Intialing point mass parameter\n log_mstel = %s"%(log_mstel))

    def esd_pointmass(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            return self.esd_pointmass_scalar(r)
        else:
            val = 0.0 * r
            for ii,rr  in enumerate(r):
                val[ii] = self.esd_pointmass_scalar(rr)
            return val

    def sigma_pointmass(self,r):
        "delta function projected profile"
        if np.isscalar(r):
            return self.sigma_pointmass_scalar(r)
        else:
            val = 0.0 * r
            for ii,rr  in enumerate(r):
                val[ii] = self.sigma_pointmass_scalar(rr)
            return val



    def esd_pointmass_scalar(self,r):
        """ESD profile from analytical predictions"""
        val = self.avg_sigma_pointmass_scalar(r) - self.sigma_pointmass_scalar(r)
        return val

    def sigma_pointmass_scalar(self,r):
        "delta function projected profile"
        if r>0:
            return 0
        else:
            return np.inf

    def avg_sigma_pointmass_scalar(self,r):
        """analytical average projected of pointmass profile"""
        if r<5e-3:
            return 10**self.log_mstel*1.0/(np.pi*(5e-3)**2)
        else:
            return 10**self.log_mstel*1.0/(np.pi*r**2)

if __name__ == "__main__":
    plt.subplot(2,2,1)
    rbin = np.logspace(-2,np.log10(5),10)
    hp = stellar(10)
    #print hp.r_200
    yy = hp.esd_pointmass(rbin)/(1e12)
    plt.plot(rbin, yy, '-')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma (R) [{\rm h M_\odot pc^{-2}}]$')
    plt.savefig('test.png', dpi=300)
