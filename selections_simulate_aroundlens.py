# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper
import sys
sys.path.append('./src/')
sys.path.append('./utils/')
from lensutils import get_re
from distort import simshear
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration

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


def run_pipe(config, outputfilename = 'gamma.dat', jksamp=0, outputpairfile=None):
    rmin    = config['Rmin'] 
    rmax    = config['Rmax'] 
    nbins   = config['Nbins']

    lensargs    = config["lens"]
    sourceargs  = config["source"]

    zdiff   =   sourceargs["zdiff"]
    #only working with H0 and omg0
    ss = simshear(H0= config['H0'],Om0 = config['Om0'])

    colossus_cosmo  = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8 = ss.sigma8, ns = ss.ns, cosmo_name=ss.cosmo_name)

    # set the projected radial binning in units of Mpc
    rmin  =  rmin
    rmax  =  rmax
    nbins = nbins #10 radial bins for our case
    rbins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff  = np.log10(rbins[1]*1.0/rbins[0])
    
    Njacks = int(lensargs['Njacks'])
    sumdgammat_num              = np.zeros(nbins)
    sumdgammat_inp_num          = np.zeros(nbins)
    sumdgammat_inp_bary_num     = np.zeros(nbins)
    sumdgammat_inp_dm_num       = np.zeros(nbins)
    sumdgammatsq_num            = np.zeros(nbins)
    sumdgammax_num              = np.zeros(nbins) 
    sumdgammaxsq_num            = np.zeros(nbins)
    sumdwls                     = np.zeros(nbins)
    sumddsigmat_num              = np.zeros(nbins)
    sumddsigmat_inp_num          = np.zeros(nbins)
    sumddsigmat_inp_bary_num     = np.zeros(nbins)
    sumddsigmat_inp_dm_num       = np.zeros(nbins)
    sumddsigmatsq_num            = np.zeros(nbins)
    sumddsigmax_num              = np.zeros(nbins) 
    sumddsigmaxsq_num            = np.zeros(nbins)
    sumdwls_by_sigcsq                = np.zeros(nbins)

    # getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    avg_logmstel    = np.log10(np.mean(10**llogmstel))
    avg_logmh       = np.log10(np.mean(10**llogmh))   
    avg_lzred       = np.mean(lzred)
    print("lens data read fully")
    thetare = get_re(llogmstel ,lzred)    # in units of arcsec
    thetare = thetare * np.pi/(180*60*60) # arcsec to radians
    llogre = np.log10(thetare * ss.Astropy_cosmo.angular_diameter_distance(lzred).value) # in the units of  h-1 Mpc
    lid = np.arange(len(lid))
    # picking a particular jksamp
    idx = (llogre != -999)
    lra         = lra       [idx]
    ldec        = ldec      [idx]
    lzred       = lzred     [idx]
    lwgt        = lwgt      [idx]
    llogmstel   = llogmstel [idx]
    llogmh      = llogmh    [idx]
    lid         = lid       [idx]
    llogre      = llogre    [idx]
    #fixed position 
    lra     = 130 + 0.0*lra
    ldec    = 0.0 + 0.0*ldec

    if config['test_case']:
        np.random.seed(123)
        idx         = (np.random.uniform(size=len(lra))<0.05)
        lra         = lra[idx]
        ldec        = ldec[idx]
        llogmh      = avg_logmh  + 0.0*llogmh[idx]
        lconc       = concentration.concentration(10**avg_logmh, '200m', avg_lzred, model = 'diemer19') + 0.0*lzred
        llogmstel   = avg_logmstel  + 0.0*llogmh
        print('log avg Mh:',avg_logmh, ' log avg mstel: ', avg_logmstel)        
        print('lzred:',avg_lzred, ' conc: ', lconc)        
    else:
        lconc = 0.0*lid
        xx = np.linspace(9,16,50)
        yy = 0.0*xx
        med_lzred = np.mean(lzred)

        for kk, mh in enumerate(10**xx):
            yy[kk]    = concentration.concentration(mh, '200m', med_lzred, model = 'diemer19')
        
        spl_c_mh = interp1d(xx,yy)
        lconc = spl_c_mh(llogmh)
    print("lens redshift -",med_lzred) 
    print("lens data read fully")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=111111001)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--use_shear", help="use shear or reduced shear for simulations", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=9.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=10.5)
    #parser.add_argument("--ten_percent", help="using ten percent of the lense sample", type=bool, default=False)

    parser.add_argument("--two_percent", help="using two percent of the lense sample", type=bool, default=False)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)


    config["outputdir"] = config["outputdir"] 

    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    config['test_case']                 = args.test_case
    config['seed']                      = args.seed
    config['lens']['two_percent']       = args.two_percent
    config['source']['use_shear']       = args.use_shear
    config['source']['no_shape_noise']  = args.no_shape_noise
    config['source']['no_shear']        = args.no_shear


    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
    if args.test_case:
        outputfilename = outputfilename + '_test_case'
    
    np.random.seed(args.seed)
    outputfilename = outputfilename + '_w_jacks'
    print(config)

    jk=0
    output_filename = outputfilename + '_jk_%d'%jk
    run_pipe(config, outputfilename = output_filename, jksamp=jk)           







