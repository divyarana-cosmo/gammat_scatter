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
    def __init__(self,log_mtot, con_par, omg_m=0.3, Rmin=0.005, Rmax=10, Rbins=100):
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
        #print("Intialing NFW parameters\n log_mtot = %s h-1 M_sun\nconc_parm = %s\nrho_0 = %s h-1 M_sun/(h-3 Mpc^3)\n r_s = %s h-1 Mpc"%(log_mtot,con_par,self.rho_0,self.r_200/self.c))
        #print("Intialing NFW parameters\n log_Mh = %s\n conc_parm = %s"%(log_mtot, con_par))

    def nfw(self,r):
        """given r, this gives nfw profile as per the instantiated parameters"""
        r_s = self.r_200/self.c
        value  = self.rho_0/((r/r_s)*(1+r/r_s)**2)
        return value

    def _esd_nfw(self,r):
        """ESD profile from analytical predictions"""
        if np.isscalar(r):
            r = np.array([r])
        sig = 0.0*r
        sig = self.avg_sigma_nfw(r) - self.sigma_nfw(r)

        #idx = r<5e-3
        #sig[idx]= 0.0
        return sig

    def _spl_esd_nfw(self):
        """ESD profile from analytical predictions"""
        xxarr = np.logspace(np.log10(self.spl_esd_rmin), np.log10(self.spl_esd_rmax), self.spl_esd_rbins) 
        yyarr = 0.0*xxarr
        for ii in range(len(yyarr)):
            yyarr[ii] = self._esd_nfw(xxarr[ii])
         
        self.spl_loglog_esd_nfw = interp1d(np.log10(xxarr), np.log10(yyarr), kind='cubic') 

        self.init_spl_esd_nfw = True
        return 

    def esd_nfw(self,r):
        """ESD profile from analytical predictions"""
        if not self.init_spl_esd_nfw:
            self._spl_esd_nfw()

        if np.isscalar(r):
            r = np.array([r])
        sig = 10**self.spl_loglog_esd_nfw(np.log10(r))
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
        if sum(idx)!=0:
            value[idx] = np.arccosh(1/x[idx])/np.sqrt(1-x[idx]**2) + np.log(x[idx]/2.0)
            value[idx] = value[idx]*2.0/x[idx]**2
        idx = (x > 1)
        if sum(idx)!=0:
            value[idx] = np.arccos(1/x[idx])/np.sqrt(x[idx]**2-1)  + np.log(x[idx]/2.0)
            value[idx] = value[idx]*2.0/x[idx]**2
        idx = (x == 1)
        if sum(idx)!=0:
            value[idx] = 2*(1-np.log(2))
        sig = value*k
        return sig


    def _sigma_nfw(self,r):
        """analytical projection of NFW"""
        if np.isscalar(r):
            r = np.array([r])
        #idx = r<5e-3#cut at the 5h-1kpc
        #r[idx]=5e-3

        r_s = self.r_200/self.c
        k = 2*r_s*self.rho_0
        value =0.0*r
        x = r/r_s
        idx = x < 1
        if sum(idx)!=0:
            value[idx] = (1 - np.arccosh(1/x[idx])/np.sqrt(1-x[idx]**2))/(x[idx]**2-1)
        idx = x > 1
        if sum(idx)!=0:
            value[idx] = (1 - np.arccos(1/x[idx])/np.sqrt(x[idx]**2-1))/(x[idx]**2-1)
        idx = x == 1
        if sum(idx)!=0:
            value[idx] = 1./3.
        sig = value*k
        return sig


    def _spl_sigma_nfw(self):
        """analytical projection of NFW"""
        xxarr = np.logspace(np.log10(self.spl_esd_rmin), np.log10(self.spl_esd_rmax), self.spl_esd_rbins) 
        yyarr = 0.0*xxarr
        for ii in range(len(yyarr)):
            yyarr[ii] = self._sigma_nfw(xxarr[ii])
         
        self.spl_loglog_sigma_nfw = interp1d(np.log10(xxarr), np.log10(yyarr), kind='cubic') 

        self.init_spl_sigma_nfw = True
 
        return
    def sigma_nfw(self,r):
        """ESD profile from analytical predictions"""
        if not self.init_spl_sigma_nfw:
            self._spl_sigma_nfw()

        if np.isscalar(r):
            r = np.array([r])
        sig = 10**self.spl_loglog_sigma_nfw(np.log10(r))
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
    mlist =  [12]
    for mm in mlist:
        hp = halo(mm,4)
        rbin = np.logspace(-2,np.log10(1),10)
        yy = 0.0*rbin
        yy = hp.esd_nfw(rbin)
        yy0 = 0.0*yy
        #for ii, rr in enumerate(rbin):
        #    yy0[ii] = hp.esd_scalar(rr)
        yy0 = hp.num_delta_sigma(rbin)
        plt.plot(rbin, yy)
        plt.plot(rbin, yy0,'.')

    #print hp.r_200
    #yy = hp.esd_nfw(rbin)/(1e12)
    #plt.plot(rbin, yy, '-')
    #plt.plot(rbin, hp.num_delta_sigma(rbin)/(1e12), '.', lw=0.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$R [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$M (<R)$')
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


