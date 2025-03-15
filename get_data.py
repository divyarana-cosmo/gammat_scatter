import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio

def lens_select(lensargs):
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
        np.random.seed(123)
        lxjkreg = np.random.randint(lensargs['Njacks'], size=len(lra))
        sys.stdout.write("Number of lenses: %d \n" % (len(df['ra_gal'])))
        return lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg





