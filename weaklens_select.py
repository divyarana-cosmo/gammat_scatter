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

    if lensargs['type'] == "smp_boss_gwtc_north" :
        fname = './Datastore/gwtc_boss_north_catalog.dat'
        df = pandas.read_csv("%s"%fname, comment="#", delim_whitespace=True, header=None, names=(["sno", "ra", "dec", "dlum", "dlum_err", "loc_area", "area_ovlp", "prob_ovlp", "max_prob"]))

        mask = hp.read_map('./boss/boss_lowz_north_fp.fits', dtype=np.float)
        #mask = hp.read_map('./boss/boss_north_fp.fits', dtype=np.float)
        msknside = int(np.sqrt(len(mask)/12))
        pix = hp.pixelfunc.ang2pix(msknside, df.ra.values, df.dec.values, lonlat=True)
        idx = (df.loc_area.values <= lensargs['loc_area_cut']) & (mask[pix] > 0.5) & (df.dlum_err.values/df.dlum.values <= lensargs['dl_snr_cut']) & (df.prob_ovlp.values >= lensargs['ovlp_cut'])

        sel_gw = np.sum(idx) # number of GW events to choose to get the covariance
        ra  = np.zeros(sel_gw)
        dec = np.zeros(sel_gw)
        np.random.seed(1991)
        print('collecting sample no =%d'%jk)
        for cnt,ii in enumerate(df.sno.values[idx]):
            dat = np.loadtxt('./Datastore/loc_map_north/%d_loc_map.dat'%ii)
            pix = hp.pixelfunc.ang2pix(msknside, dat[:,0], dat[:,1], lonlat=True)
            loc_ra  = dat[mask[pix]>0.5,0]
            loc_dec = dat[mask[pix]>0.5,1]
            loc_wgt = dat[mask[pix]>0.5,2]
            pk = np.random.choice(len(loc_ra), size=50, p=loc_wgt*1.0/np.sum(loc_wgt))
            ra[cnt]  = loc_ra[pk[jk]]
            dec[cnt] = loc_dec[pk[jk]]

        ra  = ra
        dec = dec
        wgt = ra*1.0/ra
        sys.stdout.write("Number of real GW events: %d \n" % (len(ra)))
        return ra,dec,wgt

    if lensargs['type'] == "full_boss_gwtc_north":
        fname = './Datastore/gwtc_boss_north_catalog.dat'
        df = pandas.read_csv("%s"%fname, comment="#", delim_whitespace=True, header=None, names=(["sno", "ra", "dec", "dlum", "dlum_err", "loc_area", "area_ovlp", "prob_ovlp", "max_prob"]))

        mask = hp.read_map('./boss/boss_north_fp.fits', dtype=np.float)
        msknside = int(np.sqrt(len(mask)/12))
        pix = hp.pixelfunc.ang2pix(msknside, df.ra.values, df.dec.values, lonlat=True)
        idx = (df.loc_area.values <= lensargs['loc_area_cut']) & (mask[pix] > 0.5) & (df.dlum_err.values/df.dlum.values <= lensargs['dl_snr_cut']) & (df.prob_ovlp.values >= lensargs['ovlp_cut'])

        for cnt,ii in enumerate(df.sno.values[idx]):
            if cnt==0:
                dat = np.loadtxt('./Datastore/loc_map_north/%d_loc_map.dat'%ii)
                ra  = dat[:,0]
                dec = dat[:,1]
                wgt = dat[:,2]
            else:
                dat = np.loadtxt('./Datastore/loc_map_north/%d_loc_map.dat'%ii)
                ra  = np.append(ra,  dat[:,0])
                dec = np.append(dec, dat[:,1])
                wgt = np.append(wgt, dat[:,2])
        pix = hp.pixelfunc.ang2pix(msknside, ra, dec, lonlat=True)
        ra  = ra[mask[pix]>0]
        dec = dec[mask[pix]>0]
        #ra = df.ra.values[idx]
        #dec = df.dec.values[idx]
        wgt = wgt[mask[pix]>0]
        sys.stdout.write("Number of real GW events: %d \n" % (len(ra)))

        return ra, dec, wgt

    if lensargs['type'] == "boss_gwtc_north":
        fname = './Datastore/gwtc_boss_north_catalog.dat'
        df = pandas.read_csv("%s"%fname, comment="#", delim_whitespace=True, header=None, names=(["sno", "ra", "dec", "dlum", "dlum_err", "loc_area", "area_ovlp", "prob_ovlp", "max_prob"]))

        mask = hp.read_map('./boss/boss_lowz_north_fp.fits', dtype=np.float)
        #mask = hp.read_map('./boss/boss_north_fp.fits', dtype=np.float)
        msknside = int(np.sqrt(len(mask)/12))
        pix = hp.pixelfunc.ang2pix(msknside, df.ra.values, df.dec.values, lonlat=True)
        idx = (df.loc_area.values <= lensargs['loc_area_cut']) & (mask[pix] > 0.5) & (df.dlum_err.values/df.dlum.values <= lensargs['dl_snr_cut']) & (df.prob_ovlp.values >= lensargs['ovlp_cut'])

        ra = df.ra.values[idx]
        dec = df.dec.values[idx]
        wgt = ra*1.0/ra
        sys.stdout.write("Number of real GW events: %d \n" % (len(ra)))

        return ra, dec, wgt

    if lensargs['type'] == "cov_boss_gwtc_north":
        fname = './Datastore/gwtc_boss_north_catalog.dat'
        df = pandas.read_csv("%s"%fname, comment="#", delim_whitespace=True, header=None, names=(["sno", "ra", "dec", "dlum", "dlum_err", "loc_area", "area_ovlp", "prob_ovlp", "max_prob"]))

        mask = hp.read_map('./boss/boss_lowz_north_fp.fits', dtype=np.float)
        #mask = hp.read_map('./boss/boss_north_fp.fits', dtype=np.float)
        msknside = int(np.sqrt(len(mask)/12))

        pix = hp.pixelfunc.ang2pix(msknside, df.ra.values, df.dec.values, lonlat=True)

        idx = (df.loc_area.values <= lensargs['loc_area_cut']) & (mask[pix] > 0.5) & (df.dlum_err.values/df.dlum.values <= lensargs['dl_snr_cut']) & (df.prob_ovlp.values >= lensargs['ovlp_cut'])

        sel_gw = np.sum(idx) # number of GW events to choose to get the covariance

        ra = df.ra.values
        dec = df.dec.values

        print('collecting sample no =%d'%jk)
        np.random.seed(1991+jk)
        #np.random.seed(666+jk)
        chos = np.random.choice(len(ra), size=sel_gw*10000)
        ra = ra[chos] + np.random.uniform(size=sel_gw*10000)*360
        dec = dec[chos]
        rpix = hp.pixelfunc.ang2pix(msknside, ra, dec, lonlat=True)

        ra = ra[mask[rpix]>0.5]
        dec = dec[mask[rpix]>0.5]

        rchos = np.random.choice(len(ra), size=sel_gw)
        ra = ra[rchos]
        dec = dec[rchos]
        wgt = ra*1.0/ra
        sys.stdout.write("Number of real GW events: %d \n" % (len(ra)))
        return ra, dec, wgt


    if lensargs['type'] == "smp_test_gals_north" :
        data = fits.open('./boss/galaxy_DR12v5_LOWZ_North.fits.gz')[1].data
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz -1.0) # following eqn 48 in 1509.06529
        idx = (z>0.3) & (z<=0.31)
        ra  = ra[idx]
        dec = dec[idx]
        wgt = wgt[idx]
        dat = np.transpose([ra, dec, wgt])
        print(jk)
        np.random.seed(26+jk)
        chos = np.random.choice(len(ra), 10)
        return dat[chos,0],dat[chos,1],dat[chos,2]

    if lensargs['type'] == "cov_smp_test_gals_north" :
        import fitsio
        data = fitsio.read('./boss/random1_DR12v5_LOWZ_North.fits.gz')
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        idx = (z>0.3) & (z<=0.31)
        ra  = ra[idx]
        dec = dec[idx]
        wgt = ra*1.0/ra
        dat = np.transpose([ra, dec, wgt])
        print(jk)
        np.random.seed(26+jk)
        chos = np.random.choice(len(ra), 10)
        return dat[chos,0],dat[chos,1],dat[chos,2]


    if lensargs['type'] == "test_gals_north" :
        data = fits.open('./boss/galaxy_DR12v5_LOWZ_North.fits.gz')[1].data
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz -1.0) # following eqn 48 in 1509.06529
        idx = (z>0.3) & (z<=0.31)
        ra  = ra[idx]
        dec = dec[idx]
        wgt = wgt[idx]
        dat = np.transpose([ra, dec, wgt])
        np.random.seed(26)
        chos = np.random.choice(len(ra), 10, replace=False)
        return dat[chos,0],dat[chos,1],dat[chos,2]

    if lensargs['type'] == "test_gals" :
        data = fits.open('./boss/galaxy_DR12v5_LOWZ_North.fits.gz')[1].data
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz -1.0) # following eqn 48 in 1509.06529
        idx = (z>0.3) & (z<=0.31)
        ra  = ra[idx]
        dec = dec[idx]
        wgt = wgt[idx]
        dat = np.transpose([ra, dec, wgt])
        print(np.sum(idx))
        np.random.seed(111111)
        chos = np.random.choice(len(ra), 100, replace=False)

        return dat[chos,0],dat[chos,1],dat[chos,2]
        #return dat[:50,0],dat[:50,1],dat[:50,2]

    if lensargs['type'] == "test_decals" :
       data = pandas.read_csv('./lrgs_decals/all_lrgs.dat', delim_whitespace=1)
       idx = (data['zmean'].values>0.73) & (data['zmean'].values<=0.75)  & (data['zstd'].values/(1.0+data['zmean'].values)< 0.05)
       ra  = data['ra'][idx]
       dec = data['dec'][idx]
       z   = data['zmean'].values[idx]
       wgt = ra*1.0/ra

       dat = np.transpose([ra, dec, wgt])
       np.random.seed(2021)
       chos = np.random.choice(len(ra), 50, replace=False)
       return dat[chos,0],dat[chos,1],dat[chos,2]

    if lensargs['type'] == "test_decals_lowz" :
       data = pandas.read_csv('./lrgs_decals_lowz/lowz_lrgs.dat', delim_whitespace=1)
       #fp = hp.read_map('./boss/boss_fp.fits')
       #nside = int(np.sqrt(len(fp)/12))
       #ipix = hp.ang2pix(nside, data['ra'].values, data['dec'].values, lonlat=1)
       #idx = (fp[ipix]==1.0)

       #idx = idx & (data['zspec'].values>0.32) & (data['zspec'].values<=0.34)
       idx = (data['zspec'].values>0.32) & (data['zspec'].values<=0.34)
       ra  = data['ra'][idx]
       dec = data['dec'][idx]
       wgt = ra*1.0/ra
       dat = np.transpose([ra, dec, wgt])
       np.random.seed(1996)
       chos = np.random.choice(len(ra), 100, replace=False)
       sys.stdout.write("Number of gals: %d \n" % (len(dat[:,0])))
       return dat[chos,0],dat[chos,1],dat[chos,2]

    if lensargs['type'] == "test_decals_rand" :
       fil = './decals_randoms/dr9_sky/randoms-1-10.fits'
       import fitsio
       data = fitsio.read(fil, columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z','MASKBITS'])
       idx = ((data['MASKBITS'] & (2**1 + 2**5 + 2**6 + 2**7 + 2**11 + 2**12 + 2**13) ) == 0)
       idx = idx & (data['NOBS_G']>1) & (data['NOBS_R']>1) & (data['NOBS_Z']>1)
       fp = hp.read_map('./boss/boss_fp.fits')
       nside = int(np.sqrt(len(fp)/12))
       ipix = hp.ang2pix(nside, data['RA'], data['DEC'], lonlat=1)
       idx = idx & (fp[ipix]==1.0)

       data = data[idx]
       nrows = len(data['RA'])
       np.random.seed(1996)
       chos = np.random.choice(nrows, 100, replace=False)
       ra  = data['RA'][chos]
       dec = data['DEC'][chos]
       wgt = ra*1.0/ra
       return ra,dec,wgt

    if lensargs['type'] == "decals_gwtc":
       fname = './Datastore/gwtc_decals_catalog.dat_old'
       df = pandas.read_csv("%s"%fname, comment="#", delim_whitespace=True, header=None, names=(["sno", "ra", "dec", "dlum", "dlum_err", "loc_area", "area_ovlp", "prob_ovlp", "max_prob"]))

       mask = hp.read_map('./decals_randoms/decals_lrgs_dr9_msk.fits', dtype=np.float)
       msknside = int(np.sqrt(len(mask)/12))
       pix = hp.pixelfunc.ang2pix(msknside, df.ra.values, df.dec.values, lonlat=True)
       idx = (df.loc_area.values <= lensargs['loc_area_cut']) & (mask[pix] > 0.5) & (df.dlum_err.values*1.0/df.dlum.values <= lensargs['dl_snr_cut']) & (df.prob_ovlp.values >= lensargs['ovlp_cut'])

       ra = df.ra.values[idx]
       dec = df.dec.values[idx]
       wgt = ra*1.0/ra
       sys.stdout.write("Number of real GW events: %d \n" % (len(ra)))

       return ra, dec, wgt


def source_select(sourceargs, chunksize):
    np.random.seed(10)
    #njack = sourceargs['Njacks']

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

    if sourceargs['type'] == "decals_highz" and sourceargs['filetype'] == "ascii":
        itern = sourceargs['iter']
        if itern == 0:
            sourceargs['dfchunks'] = pandas.read_csv("./lrgs_decals/all_lrgs.dat", delim_whitespace=1, iterator=True, chunksize=chunksize)
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

        #idx = (data['zmean'].values > 0) & (data['zstd'].values/(1.0+data['zmean']) < 0.1)
        idx = (data['zmean'].values > 0) & (data['zstd'].values/(1.0+data['zmean']) < sourceargs['pcut'])
        ra  = data['ra'].values[idx]
        dec = data['dec'].values[idx]
        z   = data['zmean'].values[idx]
        #wgt = 1.0/data['zstd'].values[idx]
        wgt = ra*1.0/ra

        jkreg = -1*np.ones(len(ra))
        #jkreg = get_ar_jk(ra, dec)

        datagal = np.transpose([ra, dec, z, wgt, jkreg])

        sourceargs['iter'] = sourceargs['iter'] + 1
        print(Ngal, status)

        return datagal, sourceargs, Ngal, status

    if sourceargs['type'] == "decals_lowz_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            import fitsio
            fil = './decals_randoms/dr9_sky/randoms-1-%d.fits'%sourceargs['fno']
            print(fil)
            data = fitsio.read(fil, columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z','MASKBITS'])
            idx = ((data['MASKBITS'] & (2**1 + 2**5 + 2**6 + 2**7 + 2**11 + 2**12 + 2**13) ) == 0)
            idx = idx & (data['NOBS_G']>1) & (data['NOBS_R']>1) & (data['NOBS_Z']>1)
            #idx = idx & (data['DEC'] < 34.0)
            #idx = idx & (np.random.uniform(size=len(data['RA']))<0.5)

            sourceargs['fits'] = data[idx]
            #READ PANDAS GALAXY FILE FOR ASSIGNING REDSHIFTS
            dat = pandas.read_csv('./lrgs_decals_lowz/lowz_lrgs.dat', delim_whitespace=1)
            zcuts = (dat['zmean'].values > 0) &  (dat['zstd'].values/(1.0+dat['zmean']) < sourceargs['pcut'])
            zz = dat['zmean'][zcuts]

            #wgt = 1.0/dat['zstd'][zcuts]
            sourceargs['nrows'] = len(sourceargs['fits']['RA'])
            np.random.seed(123)
            ipick = np.random.choice(len(zz), size=sourceargs['nrows'], replace=True)
            sourceargs['zred'] = zz[ipick]
            #sourceargs['wgt'] = wgt[ipick]
            #sourceargs['wgt'] = get_rands_wgts(sourceargs['fits']['RA'], sourceargs['fits']['DEC'], dat['ra'][zcuts],dat['dec'][zcuts])
            print("reading done")

            #sourceargs['fp'] = hp.read_map('./boss/boss_fp.fits')
            sourceargs['fno'] = sourceargs['fno'] + 1

        status = (itern*chunksize>=sourceargs['nrows'])
        if ((itern+1)*chunksize>=sourceargs['nrows']) and sourceargs['fno']<1:
            sourceargs['iter']=-1

        datagal = 0
        Ngal = 0
        if status :
            return datagal, sourceargs, Ngal, status

        x0    = int(itern*chunksize)
        x1    = int((itern+1)*chunksize)
        print(sourceargs['nrows'],x1)

        data  = sourceargs['fits'][x0:x1]
        zred  = sourceargs['zred'][x0:x1]
        #wgt   = sourceargs['wgt'][x0:x1]
        #idx   = (data['DEC']>34)
        ra    = data['RA']
        dec   = data['DEC']
        z     = zred[idx]
        wgt   = ra*1.0/ra

        #nside = int(np.sqrt(len(sourceargs['fp'])/12))
        #ipix = hp.ang2pix(nside, ra, dec, lonlat=1)
        #idx = (sourceargs['fp'][ipix]==1.0)

        #ra    = ra[idx]
        #dec   = dec[idx]
        #z     = z[idx]
        #wgt   = ra*1.0/ra


        #coor = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        #idx = (coor.galactic.b.degree>0) & (dec>32.375)
        #wgt[idx] = sourceargs['wgt']

        jkreg = -1*np.ones(len(ra))
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        return datagal, sourceargs, Ngal, status

    if sourceargs['type'] == "decals_highz_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            import fitsio
            fil = './decals_randoms/dr9_sky/randoms-1-%d.fits'%sourceargs['fno']
            print(fil)
            data = fitsio.read(fil, columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z','MASKBITS'])
            idx = ((data['MASKBITS'] & (2**1 + 2**5 + 2**6 + 2**7 + 2**11 + 2**12 + 2**13) ) == 0)
            idx = idx & (data['NOBS_G']>1) & (data['NOBS_R']>1) & (data['NOBS_Z']>1)
            #idx = idx & (np.random.uniform(size=len(data['RA']))<0.5)

            sourceargs['fits'] = data[idx]
            #READ PANDAS GALAXY FILE FOR ASSIGNING REDSHIFTS
            dat = pandas.read_csv('./lrgs_decals/all_lrgs.dat', delim_whitespace=1)


            zcuts = (dat['zmean']>0) & (dat['zstd'].values/(1.0+dat['zmean']) < sourceargs['pcut'])
            #zcuts = (dat['zmean']>0) & (dat['zstd'].values/(1.0+dat['zmean']) < 0.1)
            zz = dat['zmean'][zcuts]
            #wgt = 1.0/dat['zstd'][zcuts]
            sourceargs['nrows'] = len(sourceargs['fits']['RA'])
            np.random.seed(123)
            ipick = np.random.choice(len(zz), size=sourceargs['nrows'], replace=True)
            sourceargs['zred'] = zz[ipick]
            #sourceargs['wgt'] = wgt[ipick]
            #sourceargs['wgt'] = get_rands_wgts(sourceargs['fits']['RA'], sourceargs['fits']['DEC'], dat['ra'][zcuts],dat['dec'][zcuts])
            print("reading done")

            sourceargs['fno'] = sourceargs['fno'] + 1

        status = (itern*chunksize>=sourceargs['nrows'])
        #if ((itern+1)*chunksize>=sourceargs['nrows']) and sourceargs['fno']<20:
        if ((itern+1)*chunksize>=sourceargs['nrows']) and sourceargs['fno']<1:
            sourceargs['iter']=-1

        datagal = 0
        Ngal = 0
        if status :
            return datagal, sourceargs, Ngal, status

        x0    = int(itern*chunksize)
        x1    = int((itern+1)*chunksize)
        print(sourceargs['nrows'],x1)

        data  = sourceargs['fits'][x0:x1]
        zred  = sourceargs['zred'][x0:x1]
        #wgt   = sourceargs['wgt'][x0:x1]

        ra    = data['RA']
        dec   = data['DEC']
        z     = zred
        wgt   = ra*1.0/ra

        #coor = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        #idx = (coor.galactic.b.degree>0) & (dec>32.375)
        #wgt[idx] = sourceargs['wgt']



        jkreg = -1*np.ones(len(ra))
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        return datagal, sourceargs, Ngal, status


#    if sourceargs['type'] == "decals_random" and sourceargs['filetype'] == "fits":
#        itern = sourceargs['iter']
#        if itern == 0:
#            import fitsio
#            data = fitsio.read('./decals_randoms/dr9_sky/randoms-1-0.fits', columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z','MASKBITS'])
#            idx = ((data['MASKBITS'] & (2**1 + 2**5 + 2**6 + 2**7 + 2**11 + 2**12 + 2**13) ) == 0)
#            idx = idx & (data['NOBS_G']>1) & (data['NOBS_R']>1) & (data['NOBS_Z']>1)
#            idx = idx & (np.random.uniform(size=np.sum(data['RA']))<0.5)
#
#            sourceargs['fits'] = data[idx]
#            #READ PANDAS GALAXY FILE FOR ASSIGNING REDSHIFTS
#            dat = pandas.read_csv('./lrgs_decals/all_lrgs.dat', delim_whitespace=1)
#            #numb = int(len(sourceargs['fits']['RA'])/len(dat['zmean'].values))
#            #print("we have %d times randoms"%numb)
#            #sourceargs['fits'] = sourceargs['fits'][:int(numb*len(dat['zmean'].values))]
#            #print(sourceargs['fits']['RA'])
#            sourceargs['nrows'] = len(sourceargs['fits']['RA'])
#            np.random.seed(123)
#            #sourceargs['zred'] = np.tile(dat['zmean'].values, numb)
#            sourceargs['zred'] = np.random.choice(dat['zmean'].values, size=sourceargs['nrows'], replace=True)
#            print("reading done")
#
#        datagal = 0
#        status = (itern*chunksize>=sourceargs['nrows'])
#        Ngal = 0
#        if status:
#            return datagal, sourceargs, Ngal, status
#
#        x0    = int(itern*chunksize)
#        x1    = int((itern+1)*chunksize)
#        print(sourceargs['nrows'],x1)
#
#        data  = sourceargs['fits'][x0:x1]
#        zred  = sourceargs['zred'][x0:x1]
#
#
#        ra    = data['RA']
#        dec   = data['DEC']
#        z     = zred
#        wgt   = ra*1.0/ra
#        jkreg = -1*np.ones(len(ra))#get_ar_jk(ra,dec)
#
#        datagal = np.transpose([ra, dec, z, wgt, jkreg])
#        sourceargs['iter'] = sourceargs['iter'] + 1
#        Ngal = len(ra)
#        return datagal, sourceargs, Ngal, status



    if sourceargs['type'] == "lowz_south" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/galaxy_DR12v5_LOWZ_South.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])
        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status
        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)

        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz - 1.0) # following eqn 48 in 1509.06529
        #np.random.seed(1996)
        #jkreg       = np.random.randint(njack, size=len(ra))

        jkreg = get_ar_jk(ra,dec)
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        return datagal, sourceargs, Ngal, status

    if sourceargs['type'] == "lowz_south_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:

            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/random0_DR12v5_LOWZ_South.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']

        idx = (np.random.uniform(size=len(ra))<0.5)
        ra  = ra[idx]
        dec = dec[idx]
        z   = z[idx]
        #jkreg  = np.random.randint(njack, size=len(ra))
        jkreg = get_ar_jk(ra,dec)

        wgt    = ra*1.0/ra

        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        return datagal, sourceargs, Ngal, status

    if sourceargs['type'] == "lowz_north" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:

            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/galaxy_DR12v5_LOWZ_North.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])
        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status
        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz - 1.0) # following eqn 48 in 1509.06529
        #np.random.seed(1996)
        #jkreg       = np.random.randint(njack, size=len(ra))

        jkreg = get_ar_jk(ra, dec)
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        return datagal, sourceargs, Ngal, status

    if sourceargs['type'] == "lowz_north_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            #from astropy.table import Table
            #sourceargs['fits'] = Table.read('./boss/random0_DR12v5_LOWZ_North.fits.gz', memmap=True)
            #sourceargs['nrows'] = len(sourceargs['fits'])
            import fitsio
            sourceargs['fits'] = fitsio.read('./boss/random0_DR12v5_LOWZ_North.fits.gz', columns=['RA','DEC','Z'])
            sourceargs['nrows'] = len(sourceargs['fits']['RA'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']

        np.random.seed(666)
        idx = (np.random.uniform(size=len(ra))<0.5)
        ra  = ra[idx]
        dec = dec[idx]
        z   = z[idx]
        #jkreg  = np.random.randint(njack, size=len(ra))
        jkreg = get_ar_jk(ra,dec)

        wgt    = ra*1.0/ra

        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)
        print("here")
        return datagal, sourceargs, Ngal, status


    if sourceargs['type'] == "cmasslowztot_north" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status
        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = wgt_systot*(wgt_cp + wgt_noz - 1.0) # following eqn 48 in 1509.06529

        jkreg = get_ar_jk(ra,dec)
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)

        return datagal, sourceargs, Ngal, status


    if sourceargs['type'] == "cmasslowztot_north_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/random1_DR12v5_CMASSLOWZTOT_North.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        print(sourceargs['nrows'],x1)
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']

        np.random.seed(1234)
        idx = (np.random.uniform(size=len(ra))<0.5)
        ra  = ra[idx]
        dec = dec[idx]
        z   = z[idx]
        #jkreg  = np.random.randint(njack, size=len(ra))
        jkreg = get_ar_jk(ra,dec)
        wgt    = ra*1.0/ra
        datagal = np.transpose([ra, dec, z, wgt, jkreg])
        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = len(ra)

        return datagal, sourceargs, Ngal, status



    if sourceargs['type'] == "cmass_south" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/galaxy_DR12v5_CMASS_South.fits.gz')
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = ra/ra#wgt_systot*(wgt_cp + wgt_noz -1) # following eqn 48 in 1509.06529

        np.random.seed(1996+itern)
        jkreg       = np.random.randint(njack, size=len(ra))


        try:
            datagal = np.transpose([ra, dec, z, wgt, jkreg])
        except:
            status = 1
            sourceargs['iter'] = sourceargs['iter'] + 1
            return datagal, sourceargs, Ngal, status

        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = np.shape(datagal)[0]

        return datagal, sourceargs, Ngal, status


    if sourceargs['type'] == "cmass_south_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/random1_DR12v5_CMASS_South.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]

        np.random.seed(1996+itern)
        jkreg  = np.random.randint(njack, size=len(data['RA']))
        wgt    = np.ones(len(data['RA']))
        try:
            datagal = np.transpose([data['RA'], data['DEC'], data['Z'], wgt, jkreg])
        except:
            status = 1
            sourceargs['iter'] = sourceargs['iter'] + 1
            return datagal, sourceargs, Ngal, status

        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = np.shape(datagal)[0]

        return datagal, sourceargs, Ngal, status


    if sourceargs['type'] == "cmasslowztot_south" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz')
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]
        ra          = data['RA']
        dec         = data['DEC']
        z           = data['Z']
        wgt_cp      = data['WEIGHT_CP']
        wgt_noz     = data['WEIGHT_NOZ']
        wgt_systot  = data['WEIGHT_SYSTOT']
        wgt         = ra/ra#wgt_systot*(wgt_cp + wgt_noz -1) # following eqn 48 in 1509.06529

        np.random.seed(1996+itern)
        jkreg       = np.random.randint(njack, size=len(ra))


        try:
            datagal = np.transpose([ra, dec, z, wgt, jkreg])
        except:
            status = 1
            sourceargs['iter'] = sourceargs['iter'] + 1
            return datagal, sourceargs, Ngal, status

        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = np.shape(datagal)[0]

        return datagal, sourceargs, Ngal, status


    if sourceargs['type'] == "cmasslowztot_south_random" and sourceargs['filetype'] == "fits":
        itern = sourceargs['iter']
        if itern == 0:
            from astropy.table import Table
            sourceargs['fits'] = Table.read('./boss/random1_DR12v5_CMASSLOWZTOT_South.fits.gz', memmap=True)
            sourceargs['nrows'] = len(sourceargs['fits'])

        datagal = 0
        status = (itern*chunksize>=sourceargs['nrows'])
        Ngal = 0
        if status:
            return datagal, sourceargs, Ngal, status

        x0 = int(itern*chunksize)
        x1 = int((itern+1)*chunksize)
        data = sourceargs['fits'][x0:x1]

        np.random.seed(1996+itern)
        jkreg  = np.random.randint(njack, size=len(data['RA']))
        wgt    = np.ones(len(data['RA']))
        try:
            datagal = np.transpose([data['RA'], data['DEC'], data['Z'], wgt, jkreg])
        except:
            status = 1
            sourceargs['iter'] = sourceargs['iter'] + 1
            return datagal, sourceargs, Ngal, status

        sourceargs['iter'] = sourceargs['iter'] + 1
        Ngal = np.shape(datagal)[0]

        return datagal, sourceargs, Ngal, status

