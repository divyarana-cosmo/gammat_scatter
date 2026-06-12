import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as ius


class constants:
    """Useful constants"""
    G  = 4.301e-9   # km^2 Mpc M_sun^-1 s^-2 gravitational constant
    H0 = 100.       # h km s-1 Mpc-1 hubble constant at present


class halo(constants):
    """Useful functions for weak lensing signal modelling"""

    def __init__(self, log_mtot, con_par, omg_m=0.3, rtrunc=None, Roff=None,
                 beta=None, Rmin=0.0001, Rmax=5, Rbins=80):
        self.m_tot   = 10**log_mtot
        self.c       = con_par
        self.omg_m   = omg_m
        self.rho_crt = 3 * self.H0**2 / (8 * np.pi * self.G)
        self.r_200   = (3 * self.m_tot / (4 * np.pi * 200 * self.rho_crt * self.omg_m))**(1./3.)
        
        # Standard NFW Density Normalization
        self.rho_0   = (con_par**3 * self.m_tot
                        / (4 * np.pi * self.r_200**3
                           * (np.log(1 + con_par) - con_par / (1 + con_par))))

        self.spl_esd_rmin  = Rmin
        self.spl_esd_rmax  = Rmax
        self.spl_esd_rbins = Rbins

        # Hard truncation radius for 3-D profile (subhalo lensing)
        self.rtrunc = rtrunc * self.r_200 if rtrunc is not None else None

        # Off-centring radius
        self.Roff = Roff
        self.init_spl_esd_nfw    = False
        self.init_spl_sigma_nfw  = False
        self.init_spl_sigma_gnfw = False

        # Generalized NFW parameters
        self.beta = beta
        if beta is not None:
            self.r_s     = self.r_200 / self.c
            self.delta_c = (200 * self.c**3) / (
                3 * quad(lambda x: x**(2 - beta) * (1 + x)**(beta - 3), 0.0, self.c)[0])
            self.rho0_gnfw = self.delta_c * self.rho_crt * self.omg_m

    # ------------------------------------------------------------------
    # Dynamic Boundary Management
    # ------------------------------------------------------------------

    def _check_and_update_rmax(self, r):
        """Ensure the evaluation radius doesn't exceed the spline boundary while maintaining resolution."""
        max_r = np.max(np.atleast_1d(r))
        max_r_needed = max_r + self.Roff if self.Roff is not None else max_r
        
        if max_r_needed > self.spl_esd_rmax:
            # Calculate original resolution (bins per logarithmic decade)
            decades_old = np.log10(self.spl_esd_rmax / self.spl_esd_rmin)
            bins_per_decade = self.spl_esd_rbins / decades_old
            
            # Expand Rmax with a 10% safety buffer
            self.spl_esd_rmax = max_r_needed * 1.1
            
            # Scale up the number of bins to strictly maintain interpolation resolution
            decades_new = np.log10(self.spl_esd_rmax / self.spl_esd_rmin)
            self.spl_esd_rbins = int(bins_per_decade * decades_new)
            
            # Invalidate all state flags so splines rebuild automatically
            self.init_spl_esd_nfw    = False
            self.init_spl_sigma_nfw  = False
            self.init_spl_sigma_gnfw = False
            
            if hasattr(self, 'sigma_spl'):
                delattr(self, 'sigma_spl')

    # ------------------------------------------------------------------
    # Unified Public API (Auto-selects profile)
    # ------------------------------------------------------------------

    def sigma(self, r):
        """Master entry point to get the surface mass density Sigma(R)."""
        self._check_and_update_rmax(r)
        
        if self.Roff is not None:
            if not self.init_spl_sigma_nfw:
                self.create_spl()
            if np.isscalar(r):
                r = np.array([r])
            
            # Use spline. Mask out artificially small numbers from truncation
            val = 10**self.spl_log_log_sigma_nfw(np.log10(r))
            val[val < 1e-28] = 0.0
            return val
        else:
            return self.get_sigma_centered(r)

    def esd(self, r):
        """Master entry point to get the excess surface density Delta-Sigma(R)."""
        self._check_and_update_rmax(r)
        
        if self.Roff is not None:
            if not self.init_spl_esd_nfw:
                self.create_spl()
            if np.isscalar(r):
                r = np.array([r])
            return self.spl_esd_nfw(r)
        else:
            return self.get_esd_centered(r)

    # ------------------------------------------------------------------
    # 3-D density profiles
    # ------------------------------------------------------------------

    def nfw(self, r):
        r_s = self.r_200 / self.c
        return self.rho_0 / ((r / r_s) * (1 + r / r_s)**2)

    def gnfw(self, r):
        r_s = self.r_200 / self.c
        return self.rho0_gnfw / ((r / r_s)**self.beta * (1 + r / r_s)**(3 - self.beta))

    # ------------------------------------------------------------------
    # Analytical NFW projections
    # ------------------------------------------------------------------

    def _analytical_sigma_nfw(self, r):
        if np.isscalar(r):
            r = np.array([r])
        r_s   = self.r_200 / self.c
        k     = 2 * r_s * self.rho_0
        value = np.zeros_like(r, dtype=float)
        
        # Protect against exact zero querying
        r_safe = np.maximum(r, 1e-10)
        x      = r_safe / r_s

        idx = x < 1
        value[idx] = (1 - np.arccosh(1 / x[idx]) / np.sqrt(1 - x[idx]**2)) / (x[idx]**2 - 1)
        idx = x > 1
        value[idx] = (1 - np.arccos(1 / x[idx]) / np.sqrt(x[idx]**2 - 1)) / (x[idx]**2 - 1)
        idx = x == 1
        value[idx] = 1. / 3.
        return value * k

    def _analytical_avg_sigma_nfw(self, r):
        if np.isscalar(r):
            r = np.array([r])
        r_s   = self.r_200 / self.c
        k     = 2 * r_s * self.rho_0
        value = np.zeros_like(r, dtype=float)
        
        r_safe = np.maximum(r, 1e-10)
        x      = r_safe / r_s

        idx = x < 1
        value[idx] = (np.arccosh(1 / x[idx]) / np.sqrt(1 - x[idx]**2) + np.log(x[idx] / 2.0)) * 2.0 / x[idx]**2
        idx = x > 1
        value[idx] = (np.arccos(1 / x[idx]) / np.sqrt(x[idx]**2 - 1) + np.log(x[idx] / 2.0)) * 2.0 / x[idx]**2
        idx = x == 1
        value[idx] = 2 * (1 - np.log(2))
        return value * k

    # ------------------------------------------------------------------
    # Unified Centered Base Profiles (Handles Truncation & gNFW)
    # ------------------------------------------------------------------

    def _init_trunc_splines(self):
        """Initializes the numerical spline for a truncated profile."""
        if hasattr(self, 'sigma_spl'):
            return 
        
        rarr = np.logspace(np.log10(self.spl_esd_rmin),
                           np.log10(self.spl_esd_rmax),
                           self.spl_esd_rbins)
        
        func = self.gnfw if self.beta is not None else self.nfw
        sigma_arr = self.num_sigma(rarr, func)
        
        idx = rarr < self.rtrunc
        # Filter purely valid data for the spline to prevent IUS errors
        valid_idx = idx & (sigma_arr > 0)
        
        self.sigma_spl = ius(np.log10(rarr[valid_idx]), np.log10(np.maximum(sigma_arr[valid_idx], 1e-30)))

    def get_sigma_centered(self, r):
        if np.isscalar(r):
            r = np.array([r])
        value = np.zeros_like(r, dtype=float)

        if self.rtrunc is not None:
            self._init_trunc_splines()
            idx = r < self.rtrunc
            if np.any(idx):
                value[idx] = 10**self.sigma_spl(np.log10(r[idx]))
        elif self.beta is not None:
            value = self._sigma_gnfw_centered(r)
        else:
            value = self._analytical_sigma_nfw(r)
            
        return value

    def get_esd_centered(self, r):
        if np.isscalar(r):
            r = np.array([r])

        if self.rtrunc is not None:
            self._init_trunc_splines()
            avg_sig = self.num_avg_sigma(r, self.sigma_spl)
            sig = self.get_sigma_centered(r)
            return avg_sig - sig
        elif self.beta is not None:
            return self._esd_gnfw_centered(r)
        else:
            return self._analytical_avg_sigma_nfw(r) - self._analytical_sigma_nfw(r)

    # ------------------------------------------------------------------
    # Numerical projections (Matrix-Safe)
    # ------------------------------------------------------------------

    def num_sigma(self, Rarr, func):
        orig_shape = np.shape(Rarr)
        R_flat = np.atleast_1d(Rarr).flatten()
        Sigmaarr = np.zeros_like(R_flat, dtype=float)

        for ii, R in enumerate(R_flat):
            if self.rtrunc is not None:
                if R >= self.rtrunc:
                    Sigmaarr[ii] = 0.0
                    continue
                zmax = np.sqrt(self.rtrunc**2 - R**2)
            else:
                zmax = 100.0  

            Sigmaarr[ii] = 2 * quad(lambda z: func(np.sqrt(R**2 + z**2)), 0, zmax)[0]
            
        return Sigmaarr.reshape(orig_shape)
    
    def num_avg_sigma(self, R, sigma_spl):
        orig_shape = np.shape(R)
        R_flat = np.atleast_1d(R).flatten()
        value = np.zeros_like(R_flat, dtype=float)
        
        # EXACT SOLUTION: Mass contribution of the flat core from 0 to Rmin
        sigma_rmin = 10**sigma_spl(np.log10(self.spl_esd_rmin))
        extra = sigma_rmin * (self.spl_esd_rmin**2 / 2.0)
        
        for ii, rr in enumerate(R_flat):
            if rr <= self.spl_esd_rmin:
                value[ii] = sigma_rmin
            elif self.rtrunc is not None and rr >= self.rtrunc:
                integral = quad(lambda Rp: Rp * 10**sigma_spl(np.log10(Rp)), 
                                self.spl_esd_rmin, self.rtrunc)[0]
                value[ii] = 2 * (extra + integral) / rr**2
            else:
                integral = quad(lambda Rp: Rp * 10**sigma_spl(np.log10(Rp)), 
                                self.spl_esd_rmin, rr)[0]
                value[ii] = 2 * (extra + integral) / rr**2
                
        return value.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Off-centring integration
    # ------------------------------------------------------------------

    def create_spl(self):
        xx   = np.linspace(0, 2 * np.pi, 101)
        rarr = np.logspace(np.log10(self.spl_esd_rmin),
                           np.log10(self.spl_esd_rmax),
                           self.spl_esd_rbins)

        XX, RARR = np.meshgrid(xx, rarr, indexing='ij')
        rmat = np.sqrt(RARR**2 + self.Roff**2 + 2 * self.Roff * RARR * np.cos(XX))
        
        # HARDNESS FIX: Clamp rmat to Rmin to prevent 1/0 divergence for gNFW profiles
        rmat = np.maximum(rmat, self.spl_esd_rmin)

        # Averaged Sigma
        yy_sigma  = self.get_sigma_centered(rmat)
        val_sigma = simpson(yy_sigma, x=xx, axis=0) / (2 * np.pi)
        self.spl_log_log_sigma_nfw = ius(np.log10(rarr), np.log10(np.maximum(val_sigma, 1e-30)))

        # Averaged ESD
        alpha = np.arccos(np.clip((RARR + self.Roff * np.cos(XX)) / rmat, -1.0, 1.0))
        yy_esd = self.get_esd_centered(rmat) * np.cos(2 * alpha)
        val_esd = simpson(yy_esd, x=xx, axis=0) / (2 * np.pi)
        self.spl_esd_nfw = ius(rarr, val_esd)

        self.init_spl_esd_nfw   = True
        self.init_spl_sigma_nfw = True

    # ------------------------------------------------------------------
    # gNFW private handling
    # ------------------------------------------------------------------

    def _get_spl_sigma_gnfw(self):
        xx   = np.logspace(np.log10(self.spl_esd_rmin),
                           np.log10(self.spl_esd_rmax),
                           self.spl_esd_rbins)
        _xx  = xx / self.r_s
        Sigmaarr = np.zeros_like(_xx)
        for ii, x in enumerate(_xx):
            Sigmaarr[ii] = (2 * self.delta_c * self.rho_crt * self.omg_m
                            * self.r_s * x**(1 - self.beta)
                            * quad(lambda theta: np.sin(theta) * (np.sin(theta) + x)**(self.beta - 3),
                                   0, np.pi / 2)[0])

        self.spl_sigma_gnfw = interp1d(
            np.log10(xx), np.log10(Sigmaarr),
            kind='cubic',
            fill_value=(np.log10(Sigmaarr[0]), -np.inf),
            bounds_error=False)
        self.init_spl_sigma_gnfw = True

    def _sigma_gnfw_centered(self, r):
        if not self.init_spl_sigma_gnfw:
            self._get_spl_sigma_gnfw()
        if np.isscalar(r):
            r = np.array([r])
        
        # Clamp inputs to valid spline range to avoid bounds errors
        r_safe = np.maximum(r, self.spl_esd_rmin)
        return 10**self.spl_sigma_gnfw(np.log10(r_safe))

    def _avg_sigma_gnfw_centered(self, r):
        if not self.init_spl_sigma_gnfw:
            self._get_spl_sigma_gnfw()
        
        orig_shape = np.shape(r)
        r_flat = np.atleast_1d(r).flatten()
        Sigmaarr = np.zeros_like(r_flat, dtype=float)
        
        # EXACT SOLUTION: Mass contribution of the flat core from 0 to Rmin
        sigma_rmin = 10**self.spl_sigma_gnfw(np.log10(self.spl_esd_rmin))
        extra_integral = sigma_rmin * (self.spl_esd_rmin**2 / 2.0)
        
        for ii, x in enumerate(r_flat):
            if x <= self.spl_esd_rmin:
                local_extra = sigma_rmin * (x**2 / 2.0)
                Sigmaarr[ii] = 2 * np.pi * local_extra
            else:
                integral = quad(lambda t: t * 10**self.spl_sigma_gnfw(np.log10(t)), self.spl_esd_rmin, x)[0]
                Sigmaarr[ii] = 2 * np.pi * (extra_integral + integral)
            
        r_safe = np.maximum(r_flat, 1e-10)
        res = Sigmaarr / (np.pi * r_safe**2)
        return res.reshape(orig_shape)

    def _esd_gnfw_centered(self, r):
        orig_shape = np.shape(r)
        r_flat = np.atleast_1d(r).flatten()
        res = self._avg_sigma_gnfw_centered(r_flat) - self._sigma_gnfw_centered(r_flat)
        return res.reshape(orig_shape)



if __name__ == "__main__":
    
    rbin = np.logspace(-2, 1, 100)
    
    # 1. Standard NFW
    hp_nfw = halo(11.2, 4, omg_m=0.314)
    
    # 2. Generalized NFW (beta = 1.5)
    hp_gnfw = halo(11.2, 4, omg_m=0.314, beta=1.5)
    
    # 3. Truncated gNFW (beta = 1.5, rtrunc = 0.4)
    hp_trunc = halo(11.2, 4, omg_m=0.314, beta=1.5, rtrunc=0.4)
    
    # 4. ALL EFFECTS: gNFW + Truncated + Off-centered
    hp_all = halo(11.2, 4, omg_m=0.314, beta=None, rtrunc=None, Roff=0.01)

    # Plotting
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rbin, hp_nfw.esd(rbin) / 1e12, label='NFW')
    ax1.plot(rbin, hp_gnfw.esd(rbin) / 1e12, label='gNFW (beta=1.5)')
    ax1.plot(rbin, hp_trunc.esd(rbin) / 1e12, '--', label='Truncated gNFW')
    ax1.plot(rbin, hp_all.esd(rbin) / 1e12, ':', linewidth=2, label='NFW + Off-Center')
    
    if hp_trunc.rtrunc:
        ax1.axvline(hp_trunc.rtrunc, color='k', linestyle='-', alpha=0.3, label='Truncation Radius')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('R [Mpc]')
    ax1.set_ylabel(r'$\Delta\Sigma\ [10^{12}\ M_\odot\ \mathrm{Mpc}^{-2}]$')
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(rbin, hp_nfw.sigma(rbin) / 1e12, label='NFW')
    ax2.plot(rbin, hp_gnfw.sigma(rbin) / 1e12, label='gNFW (beta=1.5)')
    ax2.plot(rbin, hp_trunc.sigma(rbin) / 1e12, '--', label='Truncated gNFW')
    ax2.plot(rbin, hp_all.sigma(rbin) / 1e12, ':', linewidth=2, label='NFW + Off-Center')
    
    if hp_trunc.rtrunc:
        ax2.axvline(hp_trunc.rtrunc, color='k', linestyle='-', alpha=0.3, label='Truncation Radius')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('R [Mpc]')
    ax2.set_ylabel(r'$\Sigma\ [10^{12}\ M_\odot\ \mathrm{Mpc}^{-2}]$')
    ax2.legend()



    plt.savefig('test_profile.pdf')
    print("Test plot saved successfully.")


