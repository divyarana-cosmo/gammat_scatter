import sys
sys.path.append('./src/')
sys.path.append('./utils/')
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d

def get_xyz(ra, dec):
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def get_interp_szred():
    "assigns redshifts respecting the distribution"
    z0 = 0.9/(2)**0.5
    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
    zmin = 0.0
    zmax = 3
    zarr = np.linspace(zmin, zmax, 20)
    xx  = 0.0 * zarr
    for ii in range(len(xx)):
        xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
    proj = interp1d(xx,zarr)
    return proj

interp_szred = get_interp_szred()

def create_sources(ra, dec, dismax, nsrc=30, sigell=0.27, mask=None, seed=123): #mask application for future
    "creates source around lens given angles in degrees"
    ramin = (ra - dismax*180/np.pi )*np.pi/180
    ramax = (ra + dismax*180/np.pi)*np.pi/180
    thetamax =(90 - (dec - dismax*180/np.pi))*np.pi/180
    thetamin =(90 - (dec + dismax*180/np.pi))*np.pi/180
    
    area    = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
    size    = round(nsrc * area)      # area of square in deg^2 --> arcmin^2
    #add the possion galaxy number density 
    rng     =   np.random.default_rng(seed) # fixing the seed of the random number generator
    size    =   rng.poisson(size) # number of sources
    print('samples source', size)
    cdec    =   rng.uniform(np.cos(thetamax), np.cos(thetamin), size=size)     
    sdec    =   (90.0 - np.arccos(cdec)*180/np.pi)
    sra     =   rng.uniform(ramin, ramax, size=size)*180/np.pi
    lx,ly,lz = get_xyz(ra, dec)
    sx,sy,sz = get_xyz(sra, sdec)
    #annulus aperture
    sep     =  ((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2)**0.5
    idx     =   (sep < dismax)
    sra     = sra[idx]
    sdec    = sdec[idx]

    # putting the interpolation for source redshift assignment
    szred   =   interp_szred(rng.random(size=len(sra)))
    se1     =   rng.normal(0.0, sigell, len(sra)) 
    se2     =   rng.normal(0.0, sigell, len(sra))
    wgal    =   sra/sra
    return sra, sdec, szred, wgal, se1, se2


