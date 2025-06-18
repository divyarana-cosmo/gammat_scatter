import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio
sys.path.append('./utils/')
from lensutils import get_re
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import concentration

def lens_select(lensargs, jk=None):
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
        med_lzred = np.mean(lzred)
        # Vectorize by creating a function that applies to each mass value
        masses  = 10**xx
        concs   = np.array([concentration.concentration(mh, '200m', med_lzred, model='diemer19') for mh in masses])
        from scipy.interpolate import interp1d
        spl_c_mh    = interp1d(xx, concs)
        lconc       = spl_c_mh(llogmh)
        #............................................................................#
        if jk is None:
            return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lid[idx], lra[idx], ldec[idx], lzred[idx], lwgt[idx], llogmstel[idx], llogre[idx], llogmh[idx], lconc[idx], lxjkreg[idx]



    if lensargs['type'] == "desi" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
        idx     = (df['lmstellar']>lensargs['logmstelmin']) & (df['lmstellar']<lensargs['logmstelmax'])
        idx     = idx & (df['flag_central'] == 0) & (df['z_cgal_v'] > lensargs['zmin']) & (df['z_cgal_v'] < lensargs['zmax'])

        df      = df[idx]

        idx = (np.isfinite(df['unique_gal_id'])) & (np.isfinite(df['ra_gal'])) & (np.isfinite(df['dec_gal'])) & (df['lmstellar']>0) & (df['lmhalo']>0)#check if something is nan here

        df  = df[idx]
        # sampling the lenses in full euclidxdesi area-1.75
        rng = np.random.default_rng(123)
        idx = rng.choice(np.arange(int(sum(idx))), size=int(sum(idx)*1.75)) 
        lid         = df['unique_gal_id']   [idx]
        lra         = df['ra_gal']          [idx]
        ldec        = df['dec_gal']         [idx]
        lzred       = df['z_cgal_v']        [idx]
        llogmstel   = df['lmstellar']       [idx]
        llogmh      = df['lmhalo']          [idx]
        lwgt        = 1.0 + 0.0*lra
        lxjkreg     = rng.integers(0,lensargs['Njacks'], size=len(lra))

        #............assigning the half-light radius and concentration.............#
        Astropy_cosmo  = FlatLambdaCDM(H0=lensargs['H0'], Om0=lensargs['Om0'], Ob0=0.049)
        params = {'flat': True, 'H0': lensargs['H0'], 'Om0': lensargs['Om0'], 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        cosmology.addCosmology('myCosmo', **params)
        cosmo = cosmology.setCosmology('myCosmo')
        # Vectorized conversion of effective radius
        thetare = get_re(llogmstel, lzred)  # in units of arcsec
        thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
        llogre  = np.log10(thetare * Astropy_cosmo.comoving_distance(lzred).value)  # in the units of h-1 Mpc
        lid = np.arange(len(lid))
        # Vectorized calculation of concentration parameters
        xx = np.linspace(9, 16, 30)
        med_lzred = np.mean(lzred)
        # Vectorize by creating a function that applies to each mass value
        masses  = 10**xx
        concs   = np.array([concentration.concentration(mh, '200m', med_lzred, model='diemer19') for mh in masses])
        from scipy.interpolate import interp1d
        spl_c_mh    = interp1d(xx, concs)
        lconc       = spl_c_mh(llogmh)
        #............................................................................#
        if jk is None:
            return lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg
        idx     =   (lxjkreg==jk)
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lid[idx], lra[idx], ldec[idx], lzred[idx], lwgt[idx], llogmstel[idx], llogre[idx], llogmh[idx], lconc[idx], lxjkreg[idx]




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
        med_lzred = np.mean(lzred)
        # Vectorize by creating a function that applies to each mass value
        masses  = 10**xx
        concs   = np.array([concentration.concentration(mh, '200m', med_lzred, model='diemer19') for mh in masses])
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


