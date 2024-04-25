#need to add generalized halo profile

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
    def __init__(self,log_mtot, con_par, omg_m=0.3, beta=None, Rmin=0.001, Rmax=5, Rbins=80):
        self.m_tot = 10**log_mtot # total mass of the halo
        self.c = con_par # concentration parameter
        self.omg_m = omg_m
        self.rho_crt = 3*self.H0**2/(8*np.pi*self.G) # rho critical
        self.r_200 = (3*self.m_tot/(4*np.pi*200*self.rho_crt*self.omg_m ))**(1./3.) # radius defines size of the halo
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
            r_s = self.r_200/self.c
            self.rho0_gnfw = self.m_tot/(4*np.pi*quad(lambda r: r**2/((r/r_s)**self.beta * (1 + r/r_s)**(3 - self.beta)), 0.0, self.r_200)[0])

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
        if not self.init_spl_avg_sigma_gnfw:
            self.get_spl_avg_sigma_gnfw()
        if np.isscalar(r):
            r = np.array([r])
        return 10**self.spl_avg_sigma_gnfw(np.log10(r))


    def sigma_gnfw(self,r):
        """projected profile of generalized NFW"""
        if not self.init_spl_sigma_gnfw:
            self.get_spl_sigma_gnfw()
        if np.isscalar(r):
            r = np.array([r])
        return 10**self.spl_sigma_gnfw(np.log10(r))

    def get_spl_avg_sigma_gnfw(self):
        """puts a log log interpolation scheme for the average sigma"""
        if not self.init_spl_sigma_gnfw:
            self.get_spl_sigma_gnfw()
 
        extra =  quad(lambda Rp: Rp*self.num_sigma(Rp, self.gnfw), 0.0, self.spl_esd_rmin)[0]

        xx = np.logspace(np.log10(self.spl_esd_rmin), np.log10(self.spl_esd_rmax), self.spl_esd_rbins)
        yy = np.log10(self.num_avg_sigma(xx, self.spl_sigma_gnfw, extra))
        self.init_spl_avg_sigma_gnfw = True
        self.spl_avg_sigma_gnfw = interp1d(np.log10(xx), yy, kind='cubic')
        return 0

    def get_spl_sigma_gnfw(self):
        """puts a log log interpolation scheme for the sigma"""
        xx = np.logspace(np.log10(self.spl_esd_rmin), np.log10(self.spl_esd_rmax), self.spl_esd_rbins)
        yy = np.log10(self.num_sigma(xx, self.gnfw))
        self.init_spl_sigma_gnfw = True
        self.spl_sigma_gnfw = interp1d(np.log10(xx), yy, kind='cubic', fill_value = (np.log10(yy[0]), -np.inf))
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
    plt.subplot(2,2,1)
    rbin = np.logspace(np.log10(0.007),np.log10(0.8), int(1e6))
    hp = halo(13,4)
    print(hp.r_200)
    yy = hp.esd_nfw(rbin)/(1e12)
    #yy = hp.avg_sigma_nfw(rbin)/(1e12)
    plt.plot(rbin, yy, '-')


    hp = halo(13, 4, beta=1)
    print(hp.r_200)
    yy1 = hp.esd_gnfw(rbin)/(1e12)
    #yy1 = hp.avg_sigma_gnfw(rbin)/(1e12)
    plt.plot(rbin, yy1, '.')

    print(yy1 - yy)

    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('test.png')
 
    #mlist =  [12]
    #for mm in mlist:
    #    hp = halo(mm,4)
    #    rbin = np.logspace(-2,np.log10(1), int(1e6))
    #    yy = 0.0*rbin
    #    import time
    #    begin = time.time()
    #    yy = hp.esd_nfw(rbin)
    #    print(time.time() - begin)
    #    #yy0 = 0.0*yy
    #    #for ii, rr in enumerate(rbin):
    #    #    yy0[ii] = hp.esd_scalar(rr)
    #    #yy0 = hp.num_delta_sigma(rbin)
    #    plt.plot(rbin, yy)
    #    #plt.plot(rbin, yy0,'.')







    #plt.plot(rbin, hp.num_delta_sigma(rbin)/(1e12), '.', lw=0.0)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    #plt.ylabel(r'$M (<R)$')
    #plt.ylabel(r'$\Delta \Sigma (R) [{\rm h M_\odot pc^{-2}}]$')

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

    #def esd_scalar(self,r):
    #    """ESD profile from analytical predictions"""
    #    if r<5e-3:
    #        val = 0.0
    #    else:
    #        val = self.avg_sigma_nfw_scalar(r) - self.sigma_nfw_scalar(r)
    #    return val

    #def sigma_nfw_scalar(self,r):
    #    """analytical projection of NFW"""
    #    if r<5e-3:#cut at the 5h-1kpc
    #        r=5e-3

    #    r_s = self.r_200/self.c
    #    k = 2*r_s*self.rho_0

    #    x = r/r_s
    #    if x < 1:
    #        value = (1 - np.arccosh(1/x)/np.sqrt(1-x**2))/(x**2-1)
    #    elif x > 1:
    #        value = (1 - np.arccos(1/x)/np.sqrt(x**2-1))/(x**2-1)
    #    else:
    #        value = 1./3.
    #    sig = value*k

    #    return sig

    #def avg_sigma_nfw_scalar(self,r):
    #    """analytical average projected of NFW"""
    #    r_s = self.r_200/self.c
    #    k = 2*r_s*self.rho_0
    #    x = r/r_s


    #    if x < 1:
    #        value = np.arccosh(1/x)/np.sqrt(1-x**2) + np.log(x/2.0)
    #        value = value*2.0/x**2
    #    elif x > 1:
    #        value = np.arccos(1/x)/np.sqrt(x**2-1)  + np.log(x/2.0)
    #        value = value*2.0/x**2
    #    else:
    #        value = 2*(1-np.log(2))
    #    sig = value*k
    #    return sig


            #for ii,rr in enumerate(r):
            #    sig[ii] = self.sigma_nfw_scalar(rr)
            #return sig


