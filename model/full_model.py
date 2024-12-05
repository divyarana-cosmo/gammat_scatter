import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
class constants:
    """Useful constants"""
    G = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0 = 100. #h km s-1 Mpc-1 hubble constant at present

class halo(constants):
    """Useful functions for weak lensing signal modelling"""
    def __init__(self, mu_h0, sigma_h, beta_h, mu_stel, sigma_stel, alpha_stel, mu_c0, sigma_c, beta_c, logMstel_pvt=11.2, logMh_pvt=13.0):

        #put the hyperparameters here
        self.mu_h0      =   mu_h0 # amplitude of smhm
        self.sigma_h    =   sigma_h # scatter in smhm
        self.beta_h     =   beta_h  # slope of smhm
        self.mu_stel    =   mu_stel # mean of gaussian stellarmass distribution
        self.sigma_stel =   sigma_stel  # scatter of stellarmass distribution
        self.alpha_stel =   alpha_stel  # skewness in stellarmass distribution


        self.mu_c0      =   mu_c0   # amplitude of c-m relation
        self.sigma_c    =   sigma_c # scatter in c-m relation
        self.beta_c     =   beta_c  # slope of c-m relation


        self.logMstel_pvt   =   logMstel_pvt
        self.logMh_pvt      =   logMh_pvt

    def P_Mstel_Mh(self, logMstel, logMh):
        return self.H(logMh, logMstel) * self.S(logMstel)
    
    def P_c_Mh(self, logch, logMh):
        "modelling concentration mass relation"
        return np.exp(-(logch - self.cmh(logMh))**2/(2*self.sigma_c)**2)/(np.sqrt(2*np.pi)*self.sigma_c)

    def H(self, logMh, logMstel):
        "halo mass term"
        return np.exp(-(logMh - self.smhm(logMstel))**2/(2*self.sigma_h)**2)/(np.sqrt(2*np.pi)*self.sigma_h)
    
    def S(self,logMstel):
        "stellar mass term"
        ans =   np.exp(-(logMstel - self.mu_stel)**2/(2*self.sigma_stel)**2)/(np.sqrt(2*np.pi)*self.sigma_stel)
        ans =   ans*(1 + erf(self.alpha_stel*(logMstel - self.mu_stel)**2/(np.sqrt(2)*self.sigma_stel)))   
        return ans

    def cmh(self, logMh):
        "concentration mass relation"
        return self.mu_c0 + self.beta_c*(logMh - self.logMh_pvt)


    def smhm(self, logMstel):
        "stellar mass to halo mass relation"
        return self.mu_h0 + self.beta_h*(logMstel - self.logMstel_pvt)


if __name__ == "__main__":
    plt.subplot(2,2,1)

