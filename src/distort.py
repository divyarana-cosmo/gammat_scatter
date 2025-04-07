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
    """Simulates the shear for a given configuration of dark matter and stellar profiles with optimized vectorization"""
    def __init__(self, H0=100, Om0=0.25, Ob0=0.044, Tcmb0=2.7255, Neff=3.046, sigma8=0.8, ns=0.95, lzredmin=0.0, lzredmax=1.0, szredmax=4.0):
        """Initialize the parameters"""
        # Fixing the cosmology
        self.omg_m = Om0
        params = dict(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff)
        self.Astropy_cosmo = FlatLambdaCDM(**params)

        # Pre-compute interpolation tables for cosmological distances
        spl_zred_arr = np.linspace(0, szredmax, 1000)  # Increased resolution
        spl_d_com_arr = self.Astropy_cosmo.comoving_distance(spl_zred_arr).value
        self.cosmo_comoving_distance = interp1d(spl_zred_arr, spl_d_com_arr, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        spl_d_ang_arr = self.Astropy_cosmo.angular_diameter_distance(spl_zred_arr).value
        self.cosmo_angular_diameter_distance = interp1d(spl_zred_arr, spl_d_ang_arr, kind='cubic', bounds_error=False, fill_value="extrapolate")

        # Constants for sigma_crit calculations
        self.gee = 4.301e-9  # km^2 Mpc M_sun^-1 s^-2 gravitational constant
        self.cee = 3e5       # km s^-1

        colossus_cosmo = cosmology.fromAstropy(self.Astropy_cosmo, sigma8 = sigma8, ns = ns, cosmo_name='my_cosmo')
        self.sigma8 = sigma8
        self.ns = ns
        self.cosmo_name = 'my_cosmo'
        self.init_spl_sigma_crit_inv = False
        print("Cosmology initialized")

    def get_xyz(self, ra, dec):
        """Convert RA/Dec to Cartesian coordinates - fully vectorized"""
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        
        # Vectorized calculation
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)
        
        return x, y, z

    def _get_sigma_crit_inv(self, lzred, szred):
        """Evaluate the lensing efficiency geometrical factor - fully vectorized"""
        # Broadcast to handle scalar inputs
        if np.isscalar(lzred):
            lzred = np.full_like(szred, lzred)
        elif np.isscalar(szred):
            szred = np.full_like(lzred, szred)
            
        # Initialize output array
        sigma_crit_inv = np.zeros_like(szred, dtype=float)
        
        # Mask for valid source-lens pairs (source behind lens)
        valid_mask = szred > lzred
        
        if not np.any(valid_mask):
            return sigma_crit_inv
            
        # Vectorized computation for all valid pairs at once
        d_l = self.cosmo_comoving_distance(lzred[valid_mask])
        d_s = self.cosmo_comoving_distance(szred[valid_mask])
        
        # Calculate sigma_crit_inv in one operation
        sigma_crit_inv[valid_mask] = (d_l * (d_s - d_l) / (d_s * (1 + lzred[valid_mask])) * 
                                     (4 * np.pi * self.gee / self.cee**2))
        
        return sigma_crit_inv

    def _interp_get_sigma_crit_inv(self, lzred):
        """Create interpolator for sigma_crit_inv at a fixed lens redshift"""
        xx = np.linspace(lzred + 1e-4, 4.0, 200)  # Increased resolution
        yy = self._get_sigma_crit_inv(lzred, xx)
        return interp1d(xx, yy, kind='cubic', bounds_error=False, fill_value=0.0)

    def _get_esd(self, logmstel, logre, logmh, lconc, proj_sep):
        """Provides the ESD and sigma in comoving units - vectorized for multiple separations"""
        self.hp = halo(logmh, lconc, omg_m=self.omg_m)
        self.stel = stellar(logmstel, log_re=logre)
        
        # Vectorized calculations for all separations at once
        esd_s = self.stel.esd_deVaucouleurs(proj_sep)   
        esd_dm = self.hp.esd_nfw(proj_sep)           
        sigma_s = self.stel.sigma_deVaucouleurs(proj_sep) 
        sigma_dm = self.hp.sigma_nfw(proj_sep)         
        
        return esd_s, esd_dm, sigma_s, sigma_dm 

    def _get_g(self, logmstel, logre, logmh, lconc, lzred, szred, proj_sep):
        """Calculate gamma and kappa components - vectorized"""
        # Get sigma_crit_inv for all source-lens pairs at once
        get_sigma_crit_inv = self._get_sigma_crit_inv(lzred, szred) 
        
        # Converting from comoving to physical units
        physical_factor = (1 + lzred)
        physical_proj_sep = proj_sep * physical_factor
        physical_logre = logre + np.log10(physical_factor)
        
        # Get ESD and sigma for all separations at once
        esd_s, esd_dm, sigma_s, sigma_dm = self._get_esd(logmstel, physical_logre, logmh, lconc, physical_proj_sep)
        
        # Vectorized calculation of gamma and kappa
        factor = physical_factor**2 * get_sigma_crit_inv
        gamma_s = esd_s * factor
        gamma_dm = esd_dm * factor
        kappa_s = sigma_s * factor
        kappa_dm = sigma_dm * factor
        
        return gamma_s, gamma_dm, kappa_s, kappa_dm

    def get_g(self, lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, use_shear=False, no_shear=False):
        """Compute g1 and g2 components for the reduced shear - vectorized for multiple sources"""
        # Convert all positions to Cartesian coordinates at once
        lx, ly, lz = self.get_xyz(lra, ldec) 
        sx, sy, sz = self.get_xyz(sra, sdec)
        
        # Vectorized calculation of projected separation
        sep_vector = np.sqrt((sx - lx)**2 + (sy - ly)**2 + (sz - lz)**2)
        proj_sep = self.cosmo_angular_diameter_distance(lzred) * sep_vector
        
        # Get gamma and kappa components for all pairs at once
        gamma_s, gamma_dm, kappa_s, kappa_dm = self._get_g(logmstel, logre, logmh, lconc, lzred, szred, proj_sep)
        
        # Vectorized calculation of total gamma and kappa
        gamma = gamma_s + gamma_dm
        kappa = kappa_s + kappa_dm
        
        # Initialize flags for valid calculations
        sflag = (gamma_s != -999) & (gamma_dm != -999) & (kappa_s != -999) & (kappa_dm != -999)
        
        # Calculate reduced shear or use pure shear based on parameter
        if use_shear:
            g = gamma       # shear
            g_b = gamma_s   # shear - baryonic
            g_dm = gamma_dm # shear - dark matter
        else:
            # Vectorized division with clipping to avoid division by zero
            denom = 1.0 - kappa
            denom_s = 1.0 - kappa_s
            denom_dm = 1.0 - kappa_dm
            
            # Set small denominators to NaN to avoid numerical issues
            mask = np.abs(denom) < 1e-10
            if np.any(mask):
                denom[mask] = np.nan
                
            mask_s = np.abs(denom_s) < 1e-10
            if np.any(mask_s):
                denom_s[mask_s] = np.nan
                
            mask_dm = np.abs(denom_dm) < 1e-10
            if np.any(mask_dm):
                denom_dm[mask_dm] = np.nan
            
            # Calculate reduced shear
            g = gamma / denom          # reduced shear
            g_b = gamma_s / denom_s    # reduced shear - baryonic
            g_dm = gamma_dm / denom_dm # reduced shear - dark matter
        
        # Update flags for weak lensing regime
        sflag = sflag & (np.abs(kappa) < 0.5) & (np.abs(g) < 1)
        
        # Convert to radians for trigonometric calculations
        lra_rad = np.radians(lra)
        ldec_rad = np.radians(ldec)
        sra_rad = np.radians(sra)
        sdec_rad = np.radians(sdec)
        
        # Vectorized calculations for angular quantities
        c_sra_lra = np.cos(sra_rad) * np.cos(lra_rad) + np.sin(lra_rad) * np.sin(sra_rad)
        s_sra_lra = np.sin(sra_rad) * np.cos(lra_rad) - np.cos(sra_rad) * np.sin(lra_rad)
        
        # Angular separation between lens-source pairs
        c_theta = lx * sx + ly * sy + lz * sz
        s_theta = np.sqrt(1 - c_theta**2)
        
        # Prevent division by zero
        zero_mask = np.abs(s_theta) < 1e-10
        if np.any(zero_mask):
            s_theta[zero_mask] = 1e-10
            
        # Vectorized calculation of cosine and sine of phi
        c_phi = np.cos(ldec_rad) * s_sra_lra / s_theta
        s_phi = (-np.sin(ldec_rad) * np.cos(sdec_rad) + np.cos(ldec_rad) * c_sra_lra * np.sin(sdec_rad)) / s_theta
        
        # Tangential shear components
        g_1 = -g * (2 * c_phi**2 - 1)
        g_2 = -g * (2 * c_phi * s_phi)
        
        # Handle no_shear option
        if no_shear:
            g = np.zeros_like(g_1)
            g_1 = np.zeros_like(g_1)
            g_2 = np.zeros_like(g_1)
            
        return g_1, g_2, g, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm

    def shear_src(self, lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, intse1, intse2, use_shear=False, no_shear=False):
        """Apply shear to source galaxies with given intrinsic shapes - fully vectorized"""
        # Initialize output arrays
        se1 = intse1.copy()
        se2 = intse2.copy()
        
        # Initialize the interpolator for sigma_crit_inv if needed
        if not self.init_spl_sigma_crit_inv:
            self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)
            self.init_spl_sigma_crit_inv = True
            
        # Get shear components for all sources at once
        g_1, g_2, gtan, kappa, c_phi, s_phi, proj_sep, sflag, g_b, g_dm = self.get_g(
            lra, ldec, lzred, logmstel, logre, logmh, lconc, sra, sdec, szred, use_shear=use_shear, no_shear=no_shear
        )
        
        # Convert to complex numbers for easier manipulation
        g = g_1 + 1j * g_2
        es = intse1 + 1j * intse2
        
        # Initialize output array
        e = np.zeros_like(es, dtype=complex)
        
        # Apply Seitz & Schneider (1995) shear transformation
        idx = np.abs(g) <= 1
        
        # Vectorized calculation for |g| <= 1
        if np.any(idx):
            e[idx] = (es[idx] + g[idx]) / (1.0 + np.conj(g[idx]) * es[idx])
        
        # Vectorized calculation for |g| > 1
        if np.any(~idx):
            e[~idx] = (1 + g[~idx] * np.conj(es[~idx])) / (np.conj(es[~idx]) + np.conj(g[~idx]))
        
        # Calculate observed quantities
        phase_factor = (2 * c_phi**2 - 1) - 1j * (2 * c_phi * s_phi)
        etan_obs = -np.real(e * phase_factor)
        ex_obs = -np.imag(e * phase_factor)
        
        return np.real(e), np.imag(e), gtan, kappa, proj_sep, sflag, g_b, g_dm, etan_obs, ex_obs
