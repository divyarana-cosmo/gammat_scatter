import numpy as np
#from universe import cosmology
#import pyfits
import pandas
import sys
import glob
from astropy.io import fits
import healpy as hp
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


def lens_select(lensargs,jk=10000):
    if lensargs['type'] == "micecatv2" :
        fname = './Datastore/micecatv2/15407.fits'
        import fitsio

        df = fitsio.read(fname)
        idx  =  (df['lmstellar'] > lensargs['logmstelmin'])  & (df['lmstellar'] < lensargs['logmstelmax'])
        idx  =  idx & ((df['z_cgal_v'] > lensargs['zmin'])) & (df['z_cgal_v'] < lensargs['zmax'])
        idx  =  idx & (df['flag_central']==0)
        df = df[idx]

        #datalens = np.transpose([df['unique_gal_id'], df['ra_gal'], df['dec_gal'], df['z_cgal_v'], df['lmstellar'], df['lmhalo']])
        sys.stdout.write("Number of lenses: %d \n" % (sum(idx)))
        return df['unique_gal_id'], df['ra_gal'], df['dec_gal'], df['z_cgal_v'], df['lmstellar'], df['lmhalo']


def source_select(sourceargs, chunksize):
    np.random.seed(10)
    #njack = sourceargs['Njacks']
    """need a super edits"""
    if sourceargs['type'] == "euclid_like" and sourceargs['filetype'] == "ascii":
        itern = sourceargs['iter']
        if itern == 0:
            sourceargs['dfchunks'] = pandas.read_csv('./lrgs_decals_lowz/lowz_lrgs.dat', delim_whitespace=1, iterator=True, chunksize=chunksize)
            #sourceargs['fp'] = hp.read_map('./boss/boss_fp.fits')
            print('reading')
        try:
            data = sourceargs['dfchunks'].get_chunk()
            Ngal = data.ra.size
            status = 0
        except:
            datagal = 0
            Ngal = 0
            status = 1
        if status:
           return datagal, sourceargs, Ngal, status

        #nside = int(np.sqrt(len(sourceargs['fp'])/12))
        #ipix = hp.ang2pix(nside, data['ra'].values, data['dec'].values, lonlat=1)
        #idx = (sourceargs['fp'][ipix]==1.0)
        #idx = idx & (data['zmean'].values > 0) & (data['zstd'].values/(1.0+data['zmean']) < sourceargs['pcut'])
        #idx = (data['dec'].values>34.0)
        #idx = idx & (data['zmean'].values > 0) & (data['zstd'].values/(1.0+data['zmean']) < sourceargs['pcut'])
        idx = (data['zmean'].values > 0) & (data['zstd'].values/(1.0+data['zmean']) < sourceargs['pcut'])
        ra  = data['ra'].values[idx]
        dec = data['dec'].values[idx]
        z   = data['zmean'].values[idx]
        wgt = ra*1.0/ra

        jkreg = -1*np.ones(len(ra))

        datagal = np.transpose([ra, dec, z, wgt, jkreg])

        sourceargs['iter'] = sourceargs['iter'] + 1
        print(Ngal, status)

        return datagal, sourceargs, Ngal, status


