import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio
sys.path.append('/home/rana/github_0/gammat_scatter/utils/')
from lensutils import get_re
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import concentration
from scipy.interpolate import UnivariateSpline as usp
from colossus.halo import mass_defs


def lens_select(lensargs, seed=123, jk=None):
    if lensargs['type'] == "desi":
        # first assign the physical params 
        # assign the jackknife index
        # oversample to match the desi and minimize the sample bias (100 times more than sample size)
        # color cut file
        lmstel, zred = np.loadtxt('/home/rana/github_0/gammat_scatter/DataStore/flagship/color_cuts.dat', unpack=1)
        selfunc = usp(lmstel, zred)
    
        try:
            zlmin = lensargs['zlmin']
            zlmax = selfunc(lensargs['logmstelmin'])
        except:
            print("color_cuts are not in the same binning, please redo the color cuts\n")
            exit()
    
        fname = '/home/rana/github_0/gammat_scatter/DataStore/flagship/22984.fits'
        df = fits.getdata(fname)
        idx = (df['log_stellar_mass'] > lensargs['logmstelmin']) & (df['log_stellar_mass'] < lensargs['logmstelmax'])
        idx = idx & (df['kind'] == 0) & (df['observed_redshift_gal'] > zlmin) & (df['observed_redshift_gal'] < zlmax)
        idx = idx & (df['mag_r'] < 19.5)
        idx = idx & (np.isfinite(df['galaxy_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['log_stellar_mass'] > 0) & (df['m200b'] > 0)
        df = df[idx]
        
        # converting the units from h-2 to h-1
        df['log_stellar_mass'] = np.log10(10**df['log_stellar_mass'] / 0.677)
    
        # assigning the jackknife indices
        from numpy.random import default_rng
        rng     = default_rng(seed)
        rseed   = rng.integers(0, 2**64, size=1, dtype=np.uint64)
        rng     = default_rng(rseed)
    
        n_galaxies  = len(df)
        lseed       = rng.integers(0, 2**64, size=n_galaxies, dtype=np.uint64)
        lxjkreg     = rng.integers(0, lensargs['Njacks'], size=n_galaxies)
        
    
        # ............assigning the half-light radius and concentration.............#
        Astropy_cosmo = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        
        # Vectorized conversion of effective radius
        thetare     = get_re(df['log_stellar_mass'], df['observed_redshift_gal'])  # in units of arcsec
        thetare     = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre      = np.log10(thetare * Astropy_cosmo.comoving_distance(df['observed_redshift_gal']).value)  # in the units of h-1 Mpc
    
        # Vectorized calculation of concentration parameters
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
    
        lconc = -999 + np.zeros(n_galaxies)
        llogMh = -999 + np.zeros(n_galaxies)
    
        print(f'Converting mass definitions for {n_galaxies} galaxies...')
        for ii in range(n_galaxies):
            Mvir = df['mvir'][ii]
            cvir = df['conc_vir_halo'][ii]
            z = df['observed_redshift_gal'][ii]
            M200m, r200m, c200m = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
            llogMh[ii] = np.log10(M200m)
            lconc[ii] = c200m
            
            # Progress indicator every 1000 galaxies
            if (ii + 1) % 1000 == 0:
                print(f'  Processed {ii + 1}/{n_galaxies} galaxies')
        
        print('Mass definition conversion complete')
        
        # collecting all the parameters now
        lid     = df['galaxy_id']
        lra     = df['ra_gal']
        ldec    = df['dec_gal']
        lzred   = df['observed_redshift_gal']
        llogmh  = llogMh
        # lconc already assigned
        # lxjkreg already assigned
        lwgt = 1.0 + 0.0 * lra
        llogmstel = df['log_stellar_mass']
    
        # If no jackknife filtering requested, return all data
        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        
        # Filter by jackknife region
        jk_mask = (lxjkreg == jk)
        
        lid         = lid       [jk_mask]
        lra         = lra       [jk_mask]
        ldec        = ldec      [jk_mask]
        lzred       = lzred     [jk_mask]
        llogmh      = llogmh    [jk_mask]
        lconc       = lconc     [jk_mask]
        lxjkreg     = lxjkreg   [jk_mask]
        lwgt        = lwgt      [jk_mask]
        llogmstel   = llogmstel [jk_mask]
        llogre      = llogre    [jk_mask]
        
        n_jk_galaxies = len(lid)
        
        # Oversampling by 175 times (1.75 * 100)
        # 1.75 is to scale the area to full DESIxEuclid DR3
        # Sample with replacement from the filtered jackknife data
        n_samples = int(n_jk_galaxies * 1.75 * 1)
        oversample_idx = rng.choice(n_jk_galaxies, size=n_samples, replace=True)
        
        # Generate new seeds for oversampled data
        lseed = rng.integers(0, 2**64, size=n_samples, dtype=np.uint64)
        
        # Apply oversampling to all arrays
        lid         = lid       [oversample_idx]
        lra         = lra       [oversample_idx]
        ldec        = ldec      [oversample_idx]
        lzred       = lzred     [oversample_idx]
        llogmh      = llogmh    [oversample_idx]
        lconc       = lconc     [oversample_idx]
        lxjkreg     = lxjkreg   [oversample_idx]
        lwgt        = lwgt      [oversample_idx]
        llogmstel   = llogmstel [oversample_idx]
        llogre      = llogre    [oversample_idx]
        
        sys.stdout.write("Number of lenses after oversampling: %d \n" % n_samples)
        if 'sigma_Roff' in lensargs:
            rng     = default_rng(seed+jk)
            lRoff    = rng.rayleigh(scale=lensargs['sigma_Roff'],  size=int(len(lra)))
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg, lRoff
        else:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg


    if lensargs['type'] == "desi-smhm":
        # first assign the physical params 
        # assign the jackknife index
        # oversample to match the desi and minimize the sample bias (100 times more than sample size)
        # color cut file
        lmstel, zred = np.loadtxt('/home/rana/github_0/gammat_scatter/DataStore/flagship/color_cuts.dat', unpack=1)
        selfunc = usp(lmstel, zred)
    
        try:
            zlmin = lensargs['zlmin']
            zlmax = selfunc(lensargs['logmstelmin'])
        except:
            print("color_cuts are not in the same binning, please redo the color cuts\n")
            exit()
    
        fname = '/home/rana/github_0/gammat_scatter/DataStore/flagship/22984.fits'
        df = fits.getdata(fname)
        idx = (df['log_stellar_mass'] > lensargs['logmstelmin']) & (df['log_stellar_mass'] < lensargs['logmstelmax'])
        idx = idx & (df['kind'] == 0) & (df['observed_redshift_gal'] > zlmin) & (df['observed_redshift_gal'] < zlmax)
        idx = idx & (df['mag_r'] < 19.5)
        idx = idx & (np.isfinite(df['galaxy_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['log_stellar_mass'] > 0) & (df['m200b'] > 0)
        df = df[idx]
        
        # converting the units from h-2 to h-1
        df['log_stellar_mass'] = np.log10(10**df['log_stellar_mass'] / 0.677)
    
        # assigning the jackknife indices
        from numpy.random import default_rng
        rng     = default_rng(seed)
        rseed   = rng.integers(0, 2**64, size=1, dtype=np.uint64)
        rng     = default_rng(rseed)
    
        n_galaxies  = len(df)
        lseed       = rng.integers(0, 2**64, size=n_galaxies, dtype=np.uint64)
        lxjkreg     = rng.integers(0, lensargs['Njacks'], size=n_galaxies)
        
    
        # ............assigning the half-light radius and concentration.............#
        Astropy_cosmo = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        
        # Vectorized conversion of effective radius
        thetare     = get_re(df['log_stellar_mass'], df['observed_redshift_gal'])  # in units of arcsec
        thetare     = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre      = np.log10(thetare * Astropy_cosmo.comoving_distance(df['observed_redshift_gal']).value)  # in the units of h-1 Mpc
    
        # Vectorized calculation of concentration parameters
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
    
        lconc = -999 + np.zeros(n_galaxies)
        llogMh = -999 + np.zeros(n_galaxies)
    
        print(f'Converting mass definitions for {n_galaxies} galaxies...')
        for ii in range(n_galaxies):
            Mvir = df['mvir'][ii]
            cvir = df['conc_vir_halo'][ii]
            z = df['observed_redshift_gal'][ii]
            M200m, r200m, c200m = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
            llogMh[ii] = np.log10(M200m)
            lconc[ii] = c200m
            
            # Progress indicator every 1000 galaxies
            if (ii + 1) % 1000 == 0:
                print(f'  Processed {ii + 1}/{n_galaxies} galaxies')
        
        print('Mass definition conversion complete')
        
        # collecting all the parameters now
        lid     = df['galaxy_id']
        lra     = df['ra_gal']
        ldec    = df['dec_gal']
        lzred   = df['observed_redshift_gal']
        llogmh  = llogMh
        # lconc already assigned
        # lxjkreg already assigned
        lwgt = 1.0 + 0.0 * lra
        llogmstel = df['log_stellar_mass']
    
        # If no jackknife filtering requested, return all data
        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        
        # Filter by jackknife region
        jk_mask = (lxjkreg == jk)
        
        lid         = lid       [jk_mask]
        lra         = lra       [jk_mask]
        ldec        = ldec      [jk_mask]
        lzred       = lzred     [jk_mask]
        llogmh      = llogmh    [jk_mask]
        lconc       = lconc     [jk_mask]
        lxjkreg     = lxjkreg   [jk_mask]
        lwgt        = lwgt      [jk_mask]
        llogmstel   = llogmstel [jk_mask]
        llogre      = llogre    [jk_mask]
        
        n_jk_galaxies = len(lid)
        
        # 2.7 is to scale the area to full full Euclid DR3
        # Sample with replacement from the filtered jackknife data
        n_samples = int(n_jk_galaxies * 2.7 * 1)
        oversample_idx = rng.choice(n_jk_galaxies, size=n_samples, replace=True)
        
        # Generate new seeds for oversampled data
        lseed = rng.integers(0, 2**64, size=n_samples, dtype=np.uint64)
        
        # Apply oversampling to all arrays
        lid         = lid       [oversample_idx]
        lra         = lra       [oversample_idx]
        ldec        = ldec      [oversample_idx]
        lzred       = lzred     [oversample_idx]
        llogmh      = llogmh    [oversample_idx]
        lconc       = lconc     [oversample_idx]
        lxjkreg     = lxjkreg   [oversample_idx]
        lwgt        = lwgt      [oversample_idx]
        llogmstel   = llogmstel [oversample_idx]
        llogre      = llogre    [oversample_idx]
        
        sys.stdout.write("Number of lenses after oversampling: %d \n" % n_samples)
        if 'sigma_Roff' in lensargs:
            rng     = default_rng(seed+jk)
            lRoff    = rng.rayleigh(scale=lensargs['sigma_Roff'],  size=int(len(lra)))
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg, lRoff
        else:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg




    if lensargs['type'] == "desi_w_satellites":
        # first assign the physical params 
        # assign the jackknife index
        # oversample to match the desi and minimize the sample bias (100 times more than sample size)
        # color cut file
        
        lmstel, zred = np.loadtxt('/home/rana/github_0/gammat_scatter/DataStore/flagship/color_cuts.dat', unpack=1)
        selfunc = usp(lmstel, zred)
        
        try:
            zlmin = lensargs['zlmin']
            zlmax = selfunc(lensargs['logmstelmin'])
        except:
            print("color_cuts are not in the same binning, please redo the color cuts\n")
            sys.exit()
        
        fname = '/home/rana/github_0/gammat_scatter/DataStore/flagship/22984.fits'
        df = fits.getdata(fname)
        
        # Combined filtering to avoid creating multiple temporary boolean arrays
        idx = (
            (df['log_stellar_mass'] > lensargs['logmstelmin']) & 
            (df['log_stellar_mass'] < lensargs['logmstelmax']) &
            (df['observed_redshift_gal'] > zlmin) & 
            (df['observed_redshift_gal'] < zlmax) &
            (df['mag_r'] < 19.5) &
            np.isfinite(df['galaxy_id']) & 
            np.isfinite(df['ra_gal']) & 
            np.isfinite(df['dec_gal']) & 
            (df['log_stellar_mass'] > 0) & 
            (df['m200b'] > 0)
        )
        df = df[idx]
        print(f"Total galaxies after filtering: {len(df)}")
        
        # --- OPTIMIZED MATCHING LOGIC ---
        
        # 1. Isolate Centrals (kind == 0) and Satellites (kind == 1)
        cen_mask = df['kind'] == 0
        sat_mask = df['kind'] == 1
        
        # Extract the relevant arrays for quick access
        cen_z = df['observed_redshift_gal'][cen_mask]
        cen_mstel = df['log_stellar_mass'][cen_mask]
        cen_mvir = df['mvir'][cen_mask]
        cen_conc = df['conc_vir_halo'][cen_mask]
        
        sat_z = df['observed_redshift_gal'][sat_mask]
        sat_mstel = df['log_stellar_mass'][sat_mask]
        
        # Create result arrays filled with NaN by default
        sat_mvir_new = np.full(sat_mask.sum(), np.nan)
        sat_conc_new = np.full(sat_mask.sum(), np.nan)
        
        if len(cen_z) > 0 and len(sat_z) > 0:
            # 2. Build the KDTree for Central galaxies using [redshift, log_stellar_mass]
            cen_coords = np.column_stack((cen_z, cen_mstel))
            tree = cKDTree(cen_coords)
        
            # 3. Query the tree for Satellites
            sat_coords = np.column_stack((sat_z, sat_mstel))
            
            # query_ball_point with p=np.inf checks the maximum distance along any axis.
            # It acts exactly like: (z_diff <= 0.01) AND (mstel_diff <= 0.01)
            neighbors_list = tree.query_ball_point(sat_coords, r=0.01, p=np.inf)
        
            # 4. Resolve multiple matches by finding the one with the smallest stellar mass difference
            for i, neighbors in enumerate(neighbors_list):
                if not neighbors:
                    continue
                    
                # Extract the stellar masses of the valid central neighbors
                local_cen_mstels = cen_mstel[neighbors]
                
                # Find the neighbor index with the minimum difference to the satellite's mass
                best_local_idx = np.argmin(np.abs(local_cen_mstels - sat_mstel[i]))
                best_global_cen_idx = neighbors[best_local_idx]
                
                # Assign values
                sat_mvir_new[i] = cen_mvir[best_global_cen_idx]
                sat_conc_new[i] = cen_conc[best_global_cen_idx]
        
        # 5. Safely assign the mapped variables back to the main structured array
        df['mvir'][sat_mask] = sat_mvir_new
        df['conc_vir_halo'][sat_mask] = sat_conc_new

        # converting the units from h-2 to h-1
        df['log_stellar_mass'] = np.log10(10**df['log_stellar_mass'] / 0.677)
    
        # assigning the jackknife indices
        from numpy.random import default_rng
        rng     = default_rng(seed)
        rseed   = rng.integers(0, 2**64, size=1, dtype=np.uint64)
        rng     = default_rng(rseed)
    
        n_galaxies  = len(df)
        lseed       = rng.integers(0, 2**64, size=n_galaxies, dtype=np.uint64)
        lxjkreg     = rng.integers(0, lensargs['Njacks'], size=n_galaxies)
        
    
        # ............assigning the half-light radius and concentration.............#
        Astropy_cosmo = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        
        # Vectorized conversion of effective radius
        thetare     = get_re(df['log_stellar_mass'], df['observed_redshift_gal'])  # in units of arcsec
        thetare     = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre      = np.log10(thetare * Astropy_cosmo.comoving_distance(df['observed_redshift_gal']).value)  # in the units of h-1 Mpc
    
        # Vectorized calculation of concentration parameters
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
    
        lconc = -999 + np.zeros(n_galaxies)
        llogMh = -999 + np.zeros(n_galaxies)
    
        print(f'Converting mass definitions for {n_galaxies} galaxies...')
        for ii in range(n_galaxies):
            Mvir = df['mvir'][ii]
            cvir = df['conc_vir_halo'][ii]
            z = df['observed_redshift_gal'][ii]
            M200m, r200m, c200m = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
            llogMh[ii] = np.log10(M200m)
            lconc[ii] = c200m
            
            # Progress indicator every 1000 galaxies
            if (ii + 1) % 1000 == 0:
                print(f'  Processed {ii + 1}/{n_galaxies} galaxies')
        
        print('Mass definition conversion complete')




        
        # collecting all the parameters now
        lid     = df['galaxy_id']
        lra     = df['ra_gal']
        ldec    = df['dec_gal']
        lzred   = df['observed_redshift_gal']
        llogmh  = llogMh
        # lconc already assigned
        # lxjkreg already assigned
        lwgt = 1.0 + 0.0 * lra
        llogmstel = df['log_stellar_mass']
    
        # If no jackknife filtering requested, return all data
        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        
        # Filter by jackknife region
        jk_mask = (lxjkreg == jk)
        
        lid         = lid       [jk_mask]
        lra         = lra       [jk_mask]
        ldec        = ldec      [jk_mask]
        lzred       = lzred     [jk_mask]
        llogmh      = llogmh    [jk_mask]
        lconc       = lconc     [jk_mask]
        lxjkreg     = lxjkreg   [jk_mask]
        lwgt        = lwgt      [jk_mask]
        llogmstel   = llogmstel [jk_mask]
        llogre      = llogre    [jk_mask]
        
        n_jk_galaxies = len(lid)
        
        # Oversampling by 175 times (1.75 * 100)
        # 1.75 is to scale the area to full DESIxEuclid DR3
        # Sample with replacement from the filtered jackknife data
        n_samples = int(n_jk_galaxies * 1.75 * 100)
        oversample_idx = rng.choice(n_jk_galaxies, size=n_samples, replace=True)
        
        # Generate new seeds for oversampled data
        lseed = rng.integers(0, 2**64, size=n_samples, dtype=np.uint64)
        
        # Apply oversampling to all arrays
        lid         = lid       [oversample_idx]
        lra         = lra       [oversample_idx]
        ldec        = ldec      [oversample_idx]
        lzred       = lzred     [oversample_idx]
        llogmh      = llogmh    [oversample_idx]
        lconc       = lconc     [oversample_idx]
        lxjkreg     = lxjkreg   [oversample_idx]
        lwgt        = lwgt      [oversample_idx]
        llogmstel   = llogmstel [oversample_idx]
        llogre      = llogre    [oversample_idx]
        
        sys.stdout.write("Number of lenses after oversampling: %d \n" % n_samples)
        if 'sigma_Roff' in lensargs:
            rng     = default_rng(seed+jk)
            lRoff    = rng.rayleigh(scale=lensargs['sigma_Roff'],  size=int(len(lra)))
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg, lRoff
        else:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg




    if lensargs['type'] == "desi_z_binning":
        # first assign the physical params 
        # assign the jackknife index
        # oversample to match the desi and minimize the sample bias (100 times more than sample size)
        # color cut file
        lmstel, zred = np.loadtxt('/home/rana/github_0/gammat_scatter/DataStore/flagship/color_cuts.dat', unpack=1)
        selfunc = usp(lmstel, zred)
    
        try:
            zlmin = lensargs['zlmin']
            zlmax = selfunc(lensargs['logmstelmin'])
        except:
            print("color_cuts are not in the same binning, please redo the color cuts\n")
            exit()
    
        fname = '/home/rana/github_0/gammat_scatter/DataStore/flagship/22984.fits'
        df = fits.getdata(fname)
        idx = (df['log_stellar_mass'] > 10.5) & (df['log_stellar_mass'] < 11.5)
        idx = idx & (df['kind'] == 0) & (df['observed_redshift_gal'] > zlmin) & (df['observed_redshift_gal'] < zlmax)
        idx = idx & (df['mag_r'] < 19.5)
        idx = idx & (np.isfinite(df['galaxy_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['log_stellar_mass'] > 0) & (df['m200b'] > 0)
        df = df[idx]
        
        # converting the units from h-2 to h-1
        df['log_stellar_mass'] = np.log10(10**df['log_stellar_mass'] / 0.677)
    
        # assigning the jackknife indices
        from numpy.random import default_rng
        rng     = default_rng(seed)
        rseed   = rng.integers(0, 2**64, size=1, dtype=np.uint64)
        rng     = default_rng(rseed)
    
        n_galaxies  = len(df)
        lseed       = rng.integers(0, 2**64, size=n_galaxies, dtype=np.uint64)
        lxjkreg     = rng.integers(0, lensargs['Njacks'], size=n_galaxies)
    
        # ............assigning the half-light radius and concentration.............#
        Astropy_cosmo = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        
        # Vectorized conversion of effective radius
        thetare     = get_re(df['log_stellar_mass'], df['observed_redshift_gal'])  # in units of arcsec
        thetare     = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre      = np.log10(thetare * Astropy_cosmo.comoving_distance(df['observed_redshift_gal']).value)  # in the units of h-1 Mpc
    
        # Vectorized calculation of concentration parameters
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
    
        lconc = -999 + np.zeros(n_galaxies)
        llogMh = -999 + np.zeros(n_galaxies)
    
        print(f'Converting mass definitions for {n_galaxies} galaxies...')
        for ii in range(n_galaxies):
            Mvir = df['mvir'][ii]
            cvir = df['conc_vir_halo'][ii]
            z = df['observed_redshift_gal'][ii]
            M200m, r200m, c200m = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
            llogMh[ii] = np.log10(M200m)
            lconc[ii] = c200m
            
            # Progress indicator every 1000 galaxies
            if (ii + 1) % 1000 == 0:
                print(f'  Processed {ii + 1}/{n_galaxies} galaxies')
        
        print('Mass definition conversion complete')
        
        # collecting all the parameters now
        lid     = df['galaxy_id']
        lra     = df['ra_gal']
        ldec    = df['dec_gal']
        lzred   = df['observed_redshift_gal']
        llogmh  = llogMh
        # lconc already assigned
        # lxjkreg already assigned
        lwgt = 1.0 + 0.0 * lra
        llogmstel = df['log_stellar_mass']
    
        # If no jackknife filtering requested, return all data
        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        
        # Filter by jackknife region
        jk_mask = (lxjkreg == jk)
        
        lid         = lid       [jk_mask]
        lra         = lra       [jk_mask]
        ldec        = ldec      [jk_mask]
        lzred       = lzred     [jk_mask]
        llogmh      = llogmh    [jk_mask]
        lconc       = lconc     [jk_mask]
        lxjkreg     = lxjkreg   [jk_mask]
        lwgt        = lwgt      [jk_mask]
        llogmstel   = llogmstel [jk_mask]
        llogre      = llogre    [jk_mask]
        
        n_jk_galaxies = len(lid)
        
        # Oversampling by 175 times (1.75 * 100)
        # 1.75 is to scale the area to full DESIxEuclid DR3
        # Sample with replacement from the filtered jackknife data
        n_samples = int(n_jk_galaxies * 1.75 * 1)
        oversample_idx = rng.choice(n_jk_galaxies, size=n_samples, replace=True)
        
        # Generate new seeds for oversampled data
        lseed = rng.integers(0, 2**64, size=n_samples, dtype=np.uint64)
        
        # Apply oversampling to all arrays
        lid         = lid       [oversample_idx]
        lra         = lra       [oversample_idx]
        ldec        = ldec      [oversample_idx]
        lzred       = lzred     [oversample_idx]
        llogmh      = llogmh    [oversample_idx]
        lconc       = lconc     [oversample_idx]
        lxjkreg     = lxjkreg   [oversample_idx]
        lwgt        = lwgt      [oversample_idx]
        llogmstel   = llogmstel [oversample_idx]
        llogre      = llogre    [oversample_idx]
        
        sys.stdout.write("Number of lenses after oversampling: %d \n" % n_samples)
        
        return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg








    if lensargs['type'] == "micecatv2" :
        #fname   = './DataStore/micecatv2/micecatv2/19454.fits'
        fname   = './DataStore/micecatv2/micecatv2/20676.fits'
        df      = fitsio.read(fname, columns=['unique_gal_id','ra_gal','dec_gal','z_cgal_v', 'lmhalo', 'lmstellar', 'flag_central'])
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zmin']) & (df['z_cgal_v'] < lensargs['zmax'])

        df      = df[idx]

        idx = (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['lmstellar']>0) & (df['lmhalo']>0)#check if something is nan here

        df  = df[idx]
        lid         = df['unique_gal_id']   [:]
        lra         = df['ra_gal']          [:]
        ldec        = df['dec_gal']         [:]
        lzred       = df['z_cgal_v']        [:]
        llogmstel   = df['lmstellar']       [:]
        llogmh      = df['lmhalo']          [:]
        lwgt        = 1.0 + 0.0*lra
        rng = np.random.default_rng(123)
        lxjkreg = rng.integers(0,lensargs['Njacks'], size=len(lra))

        #............assigning the half-light radius and concentration.............#
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
        # Vectorized conversion of effective radius
        thetare = get_re(llogmstel, lzred)  # in units of arcsec
        thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        Astropy_cosmo  = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        llogre      = np.log10(thetare * Astropy_cosmo.comoving_distance(lzred).value)  # in the units of h-1 Mpc
        lid = np.arange(len(lid))
        # Vectorized calculation of concentration parameters
        xx = np.linspace(9, 16, 30)
        mean_lzred = np.mean(lzred)
        # Vectorize by creating a function that applies to each mass value
        masses  = 10**xx
        concs   = np.array([concentration.concentration(mh, '200m', mean_lzred, model='diemer19') for mh in masses])
        from scipy.interpolate import interp1d
        spl_c_mh    = interp1d(xx, concs)
        lconc       = spl_c_mh(llogmh)
        #............................................................................#
        if jk is None:
            return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lid[idx], lra[idx], ldec[idx], lzred[idx], lwgt[idx], llogmstel[idx], llogre[idx], llogmh[idx], lconc[idx], lxjkreg[idx]




    if lensargs['type'] == "test_desi" :
        fname   = '/home/rana/github_0/gammat_scatter/DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zlmin']) & (df['z_cgal_v'] < lensargs['zlmax'])
        idx     = idx & (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['lmstellar']>0) & (df['lmhalo']>0)#check if something is nan here
        df      = df[idx]

        # sampling the lenses in full euclidxdesi area-1.75
        rng = np.random.default_rng(123)

        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75)) 
        # full sample selection
        lra         =   df['ra_gal'] [idx]
        ldec        =   df['dec_gal'][idx]
        #lzred       =   np.mean(df['z_cgal_v']              [idx])  * np.ones(int(len(lra)))
        lzred       =   df['z_cgal_v'][idx]
        llogmstel   =   np.log10(np.mean(10**df['lmstellar'][idx])) * np.ones(int(len(lra)))
        llogmh      =   np.log10(np.mean(10**df['lmhalo']   [idx])) * np.ones(int(len(lra)))
        lwgt        =   np.ones(int(len(lra)))


        from numpy.random import default_rng, SeedSequence
        # Derive robust seed using entropy mixing
        ss = SeedSequence([seed, len(lra)])
        rng = default_rng(ss)
        
        lseed = rng.integers(0, 2**32, size=int(len(lra)), dtype=np.uint32)
        lxjkreg = rng.integers(0, lensargs['Njacks'], size=len(lra))

        #............assigning the half-light radius and concentration.............#
        Astropy_cosmo  = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
        # Vectorized conversion of effective radius
        thetare = get_re(llogmstel, lzred)  # in units of arcsec
        thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians

        llogre  = np.log10(thetare * Astropy_cosmo.comoving_distance(lzred).value)  # in the units of h-1 Mpc
        lconc       = np.ones(int(len(lra))) * concentration.concentration(10**np.unique(llogmh), '200m', np.mean(lzred), model='diemer19') 
        print("assigning concentration done")

        #............................................................................#
        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
        #picking a particular jackknife region
        lseed       =   lseed    [idx]   
        lra         =   lra      [idx]   
        ldec        =   ldec     [idx]   
        lzred       =   lzred    [idx]   
        lwgt        =   lwgt     [idx]   
        llogmstel   =   llogmstel[idx]   
        llogre      =   llogre   [idx]   
        llogmh      =   llogmh   [idx]   
        lconc       =   lconc    [idx]   
        lxjkreg     =   lxjkreg  [idx]   

        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg





    if lensargs['type'] == "desi-physical" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zmin']) & (df['z_cgal_v'] < lensargs['zmax'])
        df      = df[idx]
        idx = (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['lmstellar']>0) & (df['lmhalo']>0)#check if something is nan here

        df  = df[idx]
        rng = np.random.default_rng(123)
        # sampling the lenses in full euclidxdesi area-1.75
        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75)) 
        lid         = df['unique_gal_id']   [idx]
        lra         = df['ra_gal']          [idx]
        ldec        = df['dec_gal']         [idx]
        lzred       = df['z_cgal_v']        [idx]
        llogmstel   = df['lmstellar']       [idx]
        llogmh      = df['lmhalo']          [idx]
        lwgt        = 1.0 + 0.0*lra
        lxjkreg = rng.integers(0,lensargs['Njacks'], size=len(lra))

        #............assigning the half-light radius and concentration.............#
        Astropy_cosmo  = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
        # Vectorized conversion of effective radius
        thetare = get_re(llogmstel, lzred)  # in units of arcsec
        thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre  = np.log10(thetare * Astropy_cosmo.angular_diameter_distance(lzred).value)  # in the units of h-1 Mpc
        lid = np.arange(len(lid))
        # Vectorized calculation of concentration parameters
        xx = np.linspace(9, 16, 30)
        mean_lzred = np.mean(lzred)
        # Vectorize by creating a function that applies to each mass value
        masses  = 10**xx
        concs   = np.array([concentration.concentration(mh, '200m', mean_lzred, model='diemer19') for mh in masses])
        from scipy.interpolate import interp1d
        spl_c_mh    = interp1d(xx, concs)
        lconc       = spl_c_mh(llogmh)
        #............................................................................#
        if jk is None:
            return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lid[idx], lra[idx], ldec[idx], lzred[idx], lwgt[idx], llogmstel[idx], llogre[idx], llogmh[idx], lconc[idx], lxjkreg[idx]

    if lensargs['type'] == "test_case_physical" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zmin']) & (df['z_cgal_v'] < lensargs['zmax'])
        idx     =   idx &  (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) #check if something is nan here
        df      = df[idx]

        lid         = np.arange(len(df['unique_gal_id'][:]))
        lra         = df['ra_gal']          [:]
        ldec        = df['dec_gal']         [:]
        lzred       = df['z_cgal_v']        [:]
        llogmstel   = df['lmstellar']       [:]
        llogmh      = df['lmhalo']          [:]
        lwgt        = 1.0 + 0.0*lra
        rng = np.random.default_rng(123)
        lxjkreg = rng.integers(0,lensargs['Njacks'], size=len(lra))
        if jk is None:
            idx     =   (rng.uniform(size=len(lra))<0.2)
        else:    
            idx     =   (lxjkreg==jk) & (rng.uniform(size=len(lra))<1.0)
        rng = np.random.default_rng(123)
        # sampling the lenses in full euclidxdesi area-1.75
        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75)) 
 
        lid         = lid[idx]         
        lra         = 130.0 + 0.0*lra[idx]         
        ldec        = 0.0 + 0.0*ldec[idx]        
        lzred       = 0.0*lra + 0.2      
        llogmstel   = 0.0*lra + 10.0   
        llogre      = 0.0*lra + (-2.5) - np.log10(1+lzred)
        llogmh      = 0.0*lra + 12.0  
        lconc       = 0.0*lra + 10.0
        lwgt        = lwgt[idx]        
        lxjkreg     = lxjkreg[idx]  
        sys.stdout.write("Number of lenses: %d \n" % (len(lra)))
        return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg



    if lensargs['type'] == "test_case" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zmin']) & (df['z_cgal_v'] < lensargs['zmax'])
        idx     =   idx &  (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) #check if something is nan here
        df      = df[idx]

        lid         = np.arange(len(df['unique_gal_id'][:]))
        lra         = df['ra_gal']          [:]
        ldec        = df['dec_gal']         [:]
        lzred       = df['z_cgal_v']        [:]
        llogmstel   = df['lmstellar']       [:]
        llogmh      = df['lmhalo']          [:]
        lwgt        = 1.0 + 0.0*lra
        rng = np.random.default_rng(123)
        lxjkreg = rng.integers(0,lensargs['Njacks'], size=len(lra))
        if jk is None:
            idx     =   (rng.uniform(size=len(lra))<0.2)
        else:    
            idx     =   (lxjkreg==jk) & (rng.uniform(size=len(lra))<1.0)
        rng = np.random.default_rng(123)
        # sampling the lenses in full euclidxdesi area-1.75
        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75)) 
 
        lid         = lid[idx]         
        lra         = 130.0 + 0.0*lra[idx]         
        ldec        = 0.0 + 0.0*ldec[idx]        
        lzred       = 0.0*lra + 0.2      
        llogmstel   = 0.0*lra + 10.0   
        llogre      = 0.0*lra + (-2.5)
        llogmh      = 0.0*lra + 12.0  
        lconc       = 0.0*lra + 10.0
        lwgt        = lwgt[idx]        
        lxjkreg     = lxjkreg[idx]  
        sys.stdout.write("Number of lenses: %d \n" % (len(lra)))
        return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
    



        #xx = np.linspace(9, 16, 30)
        #med_lzred = np.median(lzred)
        ## Vectorize by creating a function that applies to each mass value
        #masses  = 10**xx
        #concs   = np.array([concentration.concentration(mh, '200m', med_lzred, model='diemer19') for mh in masses])
        #from scipy.interpolate import interp1d
        #spl_c_mh    = interp1d(xx, concs)
        #lconc       = spl_c_mh(llogmh) + 10**0.16 * rng.normal(0.0,1.0,size=len(llogmh))
        ##............................................................................#


"""    
if lensargs['type'] == "desi" :
        # first assign the physical params 
        # assign the jackknife index
        # oversample to match the desi and minimize the sample bias (100 times more than sample size)
        #color cut file
        lmstel, zred = np.loadtxt('/home/rana/github_0/gammat_scatter/DataStore/flagship/color_cuts.dat', unpack=1)
        selfunc = usp(lmstel, zred)

        try:
            zlmin   =   lensargs['zlmin']
            zlmax   =   selfunc(lensargs['logmstelmin'])
        except:
            print("color_cuts are not in the same binning, please redo the color cuts\n")
            exit()

        fname   = '/home/rana/github_0/gammat_scatter/DataStore/flagship/22984.fits'
        df      = fits.getdata(fname)
        idx     = (df['log_stellar_mass']>lensargs['logmstelmin']) & (df['log_stellar_mass']<lensargs['logmstelmax'])
        idx     = idx & (df['kind'] == 0) & (df['observed_redshift_gal'] > zlmin) & (df['observed_redshift_gal'] < zlmax)
        idx     = idx & (df['mag_r']<19.5)
        idx     = idx & (np.isfinite(df['galaxy_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['log_stellar_mass']>0) & (df['m200b']>0)#check if something is nan here
        df      = df[idx]
        # converting the units from h-2 to h-1
        df['log_stellar_mass'] =   np.log10(10**df['log_stellar_mass']/0.677)

        #assigning the jackknife indices
        from numpy.random import default_rng #, SeedSequence
        rng = default_rng(seed)
        rseed   = rng.integers(0, 2**64, size=1, dtype=np.uint64)
        rng = default_rng(rseed)

        lseed   = rng.integers(0, 2**64, size=int(sum(idx)), dtype=np.uint64)
        lxjkreg = rng.integers(0, lensargs['Njacks'], size=int(sum(idx)))

        #............assigning the half-light radius and concentration.............#
        Astropy_cosmo  = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        # Vectorized conversion of effective radius
        thetare = get_re(df['log_stellar_mass'], df['observed_redshift_gal'])  # in units of arcsec

        thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre  = np.log10(thetare * Astropy_cosmo.comoving_distance(df['observed_redshift_gal']).value)  # in the units of h-1 Mpc

        lid = np.arange(len(lid))

        # Vectorized calculation of concentration parameters
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.813, 'ns': 0.96}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')

        lconc   =   -999    +  np.zeros(int(sum(idx))) 
        llogMh  =   -999    +  np.zeros(int(sum(idx))) 

        for ii in range(int(sum(idx))):
            Mvir    =   df['mvir'][ii]
            cvir    =   df[ 'conc_vir_halo'][ii]
            z       =   df['observed_redshift_gal'][ii]
            M200m, r200m, c200m =  mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
            llogMh[ii]  =   np.log10(M200m)
            lconc[ii]   =   c200m
            print('Mass def changed from vir to 200m')
        
        #collecting all the parameters now
        lid         =   df['galaxy_id']               
        lra         =   df['ra_gal']                  
        ldec        =   df['dec_gal']                 
        lzred       =   df['observed_redshift_gal']   
        llogmh      =   llogMh
        lconc       =   lconc                                
        lxjkreg     =   lxjkreg
        lwgt        =   1.0 + 0.0*lra
        # need to be sure of the defination required by stellarpy                                
        llogmstel   = df['log_stellar_mass']        


        if jk is None:
            return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
 
        # sampling the lenses in full euclidxdesi area-1.75
        # oversampling by 100 times
        #rng = np.random.default_rng(123)
        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75*100)) 

        lseed   = rng.integers(0, 2**64, size=int(sum(idx)), dtype=np.uint64)
        lid        =    lid        [idx]     
        lra        =    lra        [idx]
        ldec       =    ldec       [idx]
        lzred      =    lzred      [idx]
        llogmh     =    llogmh     [idx]
        lconc      =    lconc      [idx]
        lxjkreg    =    lxjkreg    [idx]
        lwgt       =    lwgt       [idx]
        llogmstel  =    llogmstel  [idx]   
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg


"""
