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

    def sigma_pointmass(self,r):
        "delta function projected profile"
        val=0.0*r
        c=0
        for i in r:
            if i>0:
                val[c]=0
            else:
                val[c] = np.inf
            c+=1
        return val

    def avg_sigma_pointmass(self,r):
        """analytical average projected of pointmass profile"""
        val = 0.0*r
        c=0
        for i in r:
            if i<5e-3:
                val[c] = 10**self.log_mstel*1.0/(5e-3)**2
            else:
                val[c] = 10**self.log_mstel*1.0/r**2

            c=c+1
        return val

    def esd_pointmass(self,r):
        """ESD profile from analytical predictions"""
        val = self.avg_sigma_pointmass(r) - self.sigma_pointmass(r)
        return val

if __name__ == "__main__":
    plt.subplot(2,2,1)
    rbin = np.logspace(-2,np.log10(5),10)
    hp = halo(10**14.5,4)
    #print hp.r_200
    yy = hp.esd(rbin)/(1e12)
    plt.plot(rbin, yy, '-')
    #xx = rbin
    #yy = 0.0*xx

    #for ii,rr in enumerate(rbin):
    #    yy[ii] = hp.esd_scalar(rr)

    #plt.plot(xx, yy/(1e12), 's', lw=0.0)
    #plt.plot(xx, hp.esd(xx)/(1e12))

    #hp = halo(10**14,10)
    hp = halo(10**14.5,6)
    yy1 = hp.esd(rbin)/(1e12)
    plt.plot(rbin, yy1)

    #plt.plot(rbin, hp.num_delta_sigma(rbin)/(1e12), '.', lw=0.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma (R) [{\rm h M_\odot pc^{-2}}]$')

    plt.subplot(2,2,2)
    plt.plot(rbin, yy1*1.0/yy)
    plt.axhline(1.0, color='grey', ls='--')
    plt.xscale('log')
    plt.savefig('test.png', dpi=300)
