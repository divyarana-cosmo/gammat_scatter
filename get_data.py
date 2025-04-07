import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio

def lens_select(lensargs, jk=0):
    if lensargs['type'] == "desi" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
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
        idx     =   (lxjkreg==jk)
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return lid[idx], lra[idx], ldec[idx], lzred[idx], lwgt[idx], llogmstel[idx], llogmh[idx], lxjkreg[idx]



    if lensargs['type'] == "test_case" :
        fname   = './DataStore/micecatv2/micecatv2/mock_desi_bgs/combined_table.fits'
        df      = fits.getdata(fname)
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
        idx     =   (lxjkreg==jk) & (rng.uniform(size=len(lra))<1.0)

        lid         = lid[idx]         
        lra         = 130.0 + 0.0*lra[idx]         
        ldec        = 0.0 + 0.0*ldec[idx]        
        lzred       = 0.0*lra + np.mean(lzred)      
        llogmstel   = 0.0*lra + np.mean(llogmstel)   
        llogmh      = 0.0*lra + np.mean(llogmh)      
        lwgt        = lwgt[idx]        
        lxjkreg     = lxjkreg[idx]  
        sys.stdout.write("Number of lenses: %d \n" % (len(lra)))
        return lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg

