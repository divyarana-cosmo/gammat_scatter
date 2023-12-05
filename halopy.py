import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

class constants:
    """Useful constants"""
    G = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0 = 100. #h km s-1 Mpc-1 hubble constant at present

class halo(constants):
    """Useful functions for weak lensing signal modelling"""
    def __init__(self,log_mtot, con_par, omg_m=0.3):
        self.m_tot = 10**log_mtot # total mass of the halo
        self.c = con_par # concentration parameter
        self.omg_m = omg_m
        self.rho_crt = 3*self.H0**2/(8*np.pi*self.G) # rho critical
        self.r_200 = (3*self.m_tot/(4*np.pi*200*self.rho_crt*self.omg_m ))**(1./3.) # radius defines size of the halo
        self.rho_0 = con_par**3 *self.m_tot/(4*np.pi*self.r_200**3 *(np.log(1+con_par)-con_par/(1+con_par)))
        self.init_sigma = False
        #print("Intialing NFW parameters\n log_mtot = %s h-1 M_sun\nconc_parm = %s\nrho_0 = %s h-1 M_sun/(h-3 Mpc^3)\n r_s = %s h-1 Mpc"%(log_mtot,con_par,self.rho_0,self.r_200/self.c))
        print("Intialing NFW parameters\n log_mtot = %s\n conc_parm = %s"%(log_mtot, con_par))

    def nfw(self,r):
        """given r, this gives nfw profile as per the instantiated parameters"""
        r_s = self.r_200/self.c
        value  = self.rho_0/((r/r_s)*(1+r/r_s)**2)
        return value


    def esd_nfw(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            return self.esd_scalar(r)
        else:
            sig = 0.0*r
            for ii,rr in enumerate(r):
                sig[ii] = self.esd_scalar(rr)
            return sig

    def esd_scalar(self,r):
        """ESD profile from analytical predictions"""
        if r<5e-3:
            val = self.avg_sigma_nfw_scalar(5e-3) - self.sigma_nfw_scalar(5e-3)
        else:
            val = self.avg_sigma_nfw_scalar(r) - self.sigma_nfw_scalar(r)
        return val

    def sigma_nfw_scalar(self,r):
        """analytical projection of NFW"""
        r_s = self.r_200/self.c
        k = 2*r_s*self.rho_0

        x = r/r_s
        if x < 1:
            value = (1 - np.arccosh(1/x)/np.sqrt(1-x**2))/(x**2-1)
        elif x > 1:
            value = (1 - np.arccos(1/x)/np.sqrt(x**2-1))/(x**2-1)
        else:
            value = 1./3.
        sig = value*k

        return sig

    def avg_sigma_nfw_scalar(self,r):
        """analytical average projected of NFW"""
        r_s = self.r_200/self.c
        k = 2*r_s*self.rho_0
        x = r/r_s
        if x < 1:
            value = np.arccosh(1/x)/np.sqrt(1-x**2) + np.log(x/2.0)
            value = value*2.0/x**2
        elif x > 1:
            value = np.arccos(1/x)/np.sqrt(x**2-1)  + np.log(x/2.0)
            value = value*2.0/x**2
        else:
            value = 2*(1-np.log(2))
        sig = value*k
        return sig

    def num_sigma(self,Rarr):
        """numerical test to the analytical part"""
        if np.isscalar(Rarr):
            return 2*quad((lambda z : self.nfw(np.sqrt(Rarr**2 + z**2))), 0, np.inf)[0]

        Sigmaarr = Rarr*0.0
        for ii, R in enumerate(Rarr):
            Sigmaarr[ii] = 2*quad((lambda z : self.nfw(np.sqrt(R**2 + z**2))), 0, np.inf)[0]
        return Sigmaarr

    def num_delta_sigma(self,R):
        """num difference between mean sigma and average over the circle of radius R"""
        if np.isscalar(R):
            value =  2*np.pi*quad(lambda Rp: Rp*self.num_sigma(Rp), 0.0, R)[0]/(np.pi*R**2) - self.num_sigma(R)
        else:
            value = 0.0*R
            for ii,rr in enumerate(R):
                value[ii] = 2*np.pi*quad(lambda Rp: Rp*self.num_sigma(Rp), 0.0, rr)[0]/(np.pi*rr**2) - self.num_sigma(rr)
        return value


if __name__ == "__main__":
    plt.subplot(2,2,1)
    rbin = np.logspace(-2,np.log10(5),10)
    hp = halo(14,4)
    #print hp.r_200
    yy = hp.esd_nfw(rbin)/(1e12)
    plt.plot(rbin, yy, '-')
    plt.plot(rbin, hp.num_delta_sigma(rbin)/(1e12), '.', lw=0.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma (R) [{\rm h M_\odot pc^{-2}}]$')

    plt.savefig('test.png', dpi=300)

    #xx = rbin
    #yy = 0.0*xx

    #for ii,rr in enumerate(rbin):
    #    yy[ii] = hp.esd_scalar(rr)

    #plt.plot(xx, yy/(1e12), 's', lw=0.0)
    #plt.plot(xx, hp.esd(xx)/(1e12))

    #hp = halo(10**14,10)
    #hp = halo(14.5,6)
    #yy1 = hp.esd(rbin)/(1e12)
    #plt.plot(rbin, yy1)
    #def sigma_nfw(self,r):
    #    """analytical projection of NFW"""
    #    r_s = self.r_200/self.c
    #    k = 2*r_s*self.rho_0
    #    sig = 0.0*r
    #    c=0
    #    for i in r:
    #        if i<5e-3:
    #            sig[c] = self.sigma_nfw_scalar(5e-3)
    #        else:
    #            sig[c] = self.sigma_nfw_scalar(i)
    #        c=c+1

    #    return sig

    #def avg_sigma_nfw(self,r):
    #    """analytical average projected of NFW"""
    #    r_s = self.r_200/self.c
    #    k = 2*r_s*self.rho_0
    #    sig = 0.0*r
    #    c=0
    #    for i in r:
    #        if i<5e-3:
    #            sig[c] = self.avg_sigma_nfw_scalar(5e-3)
    #        else:
    #            sig[c] = self.avg_sigma_nfw_scalar(i)

    #        c=c+1

    #    return sig

