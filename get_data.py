import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
import fitsio
#import matplotlib.pyplot as plt
#cc = cosmology(omg_m0=0.31,omg_l0=0.69)
#cc = cosmology(omg_m0=0.315,omg_l0=0.685)


def get_file_list(region):
    flist = glob.glob("./lrgs_decals/%s/*.fits.dat.fits" % region)
    flist = np.sort(np.array(flist))
    return flist
from astropy.coordinates import SkyCoord
from astropy import units as u


def get_ar_jk(ra,dec):
    coor    = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galb    = coor.galactic.b.degree
    jckreg = -1 * np.ones(len(ra))

    ramin, ramax, decmin, decmax = np.loadtxt('./decals_randoms/NGP-regions.list', unpack=True)
    for i in range(len(ramin)):
           idx = (ra>ramin[i]) & (ra<=ramax[i]) & (dec>decmin[i]) &  (dec<=decmax[i]) & (galb>0)
           jckreg[idx] = i
    add = len(ramin)

    ramin, ramax, decmin, decmax = np.loadtxt('./decals_randoms/SGP-regions.list', unpack=True)
    ra = (ra + 80)%360
    for i in range(len(ramin)):
           idx = (ra>ramin[i]) & (ra<=ramax[i]) & (dec>decmin[i]) &  (dec<=decmax[i]) & (galb<0)
           jckreg[idx] = i + add

    return jckreg

def get_rands_wgts(rra,rdec,ra,dec):
    idx  = np.random.uniform(size=len(rra))<0.05
    rra  = rra[idx]
    rdec = rdec[idx]
    idx  = np.random.uniform(size=len(ra))<0.1
    ra  = ra[idx]
    dec = dec[idx]

    wgt   = 0.0*rra
    gcoor = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    rcoor = SkyCoord(ra=rra*u.degree, dec=rdec*u.degree, frame='icrs')
    gidx = (gcoor.galactic.b.degree>0) & (dec>32.375)
    ridx = (rcoor.galactic.b.degree>0) & (rdec>32.375)

    nnorm = np.sum(ridx)*1.0/np.sum(gidx)
    snorm = (len(rra)-np.sum(ridx))*1.0/(len(ra)-np.sum(gidx))
    wgt = snorm*1.0/nnorm

    return wgt


def lens_select(lensargs):
    if lensargs['type'] == "micecatv2" :
        fname = './DataStore/micecatv2/15407.fits'
        #fname = './DataStore/micecatv2/15412.fits'
        df = fitsio.FITS(fname)
        df = df[1][df[1].where('flag_central == 0 && lmstellar > %2.2f && lmstellar < %2.2f && z_cgal_v > %2.2f && z_cgal_v < %2.2f'%(lensargs['logmstelmin'], lensargs['logmstelmax'], lensargs['zmin'], lensargs['zmax']))]

        lid         = df['unique_gal_id']
        lra         = df['ra_gal']  
        ldec        = df['dec_gal'] 
        lzred       = df['z_cgal_v']
        lwgt        = lra/lra 
        llogmstel   = df['lmstellar']
        llogmh      = df['lmhalo']

        np.random.seed(123)
        Njacks = lensargs['Njacks']
        lxjkreg     = np.random.randint(Njacks, size=len(lra))
        sys.stdout.write("Number of lenses: %d \n" % (len(df['ra_gal'])))
        return lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg  



















