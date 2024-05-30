import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio
from colossus.cosmology import cosmology
from colossus.halo import concentration


def lens_select(lensargs):
    if lensargs['type'] == "micecatv2" :
        #fname = './DataStore/micecatv2/15407.fits'
        #fname = './DataStore/micecatv2/15412.fits'
        #fname = './DataStore/micecatv2/15412.fits_mstelcut_10_centrals'
        fname = './DataStore/micecatv2/17281.fits'

        df = fitsio.FITS(fname)
        #taking only small area
        df = df[1][df[1].where('flag_central == 0 && lmstellar > %2.2f && lmstellar < %2.2f && z_cgal_v > %2.2f && z_cgal_v < %2.2f && ra_gal<10 && dec_gal<10'%(lensargs['logmstelmin'], lensargs['logmstelmax'], lensargs['zmin'], lensargs['zmax']))]
        #idx = np.random.uniform(size=len(df['ra_gal']))<0.1
        #df  = df[idx]

        lid         = df['unique_gal_id'][:]
        lra         = df['ra_gal'][:]  
        ldec        = df['dec_gal'][:] 
        lzred       = df['z_cgal_v'][:]
        llogmstel   = df['lmstellar'][:]
        llogMh      = df['lmhalo'][:]
        lwgt        = 1.0 + 0.0*lra

        #assigning the half-light radius - need to add a scatter later on
        llog_re     = (0.774 + 0.977 *(np.log10(10**llogmstel / 0.7) - 11.4)) #check arxiv:1811.04934
        llog_re     = np.log10(10**llog_re * 0.7/1e3) #h-1 kpc to h-1 Mpc
        
        #assigning the jackknife indices
        np.random.seed(123)
        Njacks = lensargs['Njacks']
        ljkreg     = np.random.randint(Njacks, size=len(lra))

        sys.stdout.write("Number of lenses: %d \n" % (len(lra)))
        return lra, ldec, lzred, lwgt, llogMh, llogmstel, llog_re, ljkreg


















