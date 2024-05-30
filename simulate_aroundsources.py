# split the lense sky in 100 chunks and assign sources via possion distribution
# run the code chunk by chunk 

# make a class which given the config values initialize up the arrays
# write a function to process lens
# write a function to process source
# write a function to put measured signal in a file everything in a file

import sys
sys.path.append('./src/')
from weakpipe import weakpipe
import numpy as np
import matplotlib.pyplot as plt
#from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
#from scipy.spatial import cKDTree
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call
#from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
#from welford import Welford
import time


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

def create_sources(ramin, ramax, decmin, decmax, sigell=0.27, nsrc=30, mask=None): #mask for future application
    "takes the angles in degrees and outputs in degrees"
    thetamin        = (90 - decmax) *np.pi/180
    thetamax        = (90 - decmin) *np.pi/180
    ramin           = ramin*np.pi/180
    ramax           = ramax*np.pi/180
    
    #samplying the poisson distributed sources
    area            = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
    avgNgal         = nsrc * area      # area of square in steradians --> arcmin^2
    Ngal            = round(np.random.poisson(avgNgal))
    print(avgNgal, Ngal)
 
    cdec            = np.random.uniform(np.cos(thetamax), np.cos(thetamin), size=Ngal)     
    sdec            = (90.0 - np.arccos(cdec)*180/np.pi)
    sra             = np.random.uniform(ramin, ramax, size=Ngal)*180/np.pi

    # putting the interpolation for source redshift assignment
    szred       =   interp_szred(np.random.uniform(size=Ngal))
    se1         =   np.random.normal(0.0, sigell, size=Ngal) 
    se2         =   np.random.normal(0.0, sigell, size=Ngal)
    # inverse variance weights for the sources
    wgal        =   1/sigell**2 + 0.0*sra
    return np.transpose([sra, sdec, szred, wgal, se1, se2])
    #return np.transpose([sra, sdec, szred, wgal, se1, se2])




def run_pipe(config, outputfilename):
    lensargs    = config["lens"]
    sourceargs  = config["source"]
    ##setting up cosmology and class instance

    params = {'flat': True, 'H0': 100, 'Om0': config['Om0'], 'Ob0': config['Ob0'], 'sigma8': config['sigma8'], 'ns': config['ns']}
    cosmo = cosmology.setCosmology('myCosmo', **params)
    
    # getting the lenses data and massaging it a bit
    lra, ldec, lzred, lwgt, llogMh, llogmstel, llog_re, ljkreg = lens_select(lensargs)
    lconc = 0.0*lra
    if config['test_case']:
        idx = (lra<10) & (ldec<10)
        lra  = lra[idx]
        ldec = ldec[idx]
        print("working with the test case")
        llogmstel   = 12.0 + 0.0*lra
        llog_re     = 0.02 + 0.0*lra
        llogMh      = 14.0 + 0.0*lra
        lconc       = 5.00 + 0.0*lra
        lzred       = 0.20 + 0.0*lra
    else:
        xx = np.linspace(9,16,100)
        yy = 0.0*xx
        med_lzred = np.median(lzred)
        for kk, mh in enumerate(10**xx):
            yy[kk]    = concentration.concentration(mh, '200m', med_lzred, model = 'diemer19')
        spl_c_mh = interp1d(xx,np.log10(yy))
        lconc = 10**spl_c_mh(llogMh)

    # initialize weakpipe
    wpipe = weakpipe(H0 = 100, Om0 = 0.25, Ob0 = 0.044, Tcmb0 = 2.7255, Neff = 3.046, sigma8 = 0.8, ns = 0.95, Rmin=0.02, Rmax=1.0, Nbins=10, Njacks=20, outputfilename=outputfilename)

    #process lens data and put a tree
    wpipe.process_lens(lra, ldec, lzred, lwgt, llogMh, lconc, llogmstel, llog_re, ljkreg)

    # make source catalog using the lens field dimensions, we are currently using a rectangular field 
    # but if we have a footprint we can easily generalize it
    # chopping up the lense field in the 10000 regions 
    Nchunks     = 500
    brickra     = np.linspace(lra.min(), lra.max(), Nchunks+1)
    lthetamax   = (90 - ldec.min())*np.pi/180
    lthetamin   = (90 - ldec.max())*np.pi/180
    brickdec    = np.linspace(np.cos(lthetamax), np.cos(lthetamin), Nchunks+1)     
    brickdec    = (90.0 - np.arccos(brickdec)*180/np.pi)
    ax1 = plt.subplot(2,2,2)
    for ii in range(Nchunks):  
        for jj in range(Nchunks):
            np.random.seed(ii*Nchunks+jj)
            print(brickra[ii], brickra[ii+1], brickdec[jj], brickdec[jj+1])
            datagal = create_sources(brickra[ii], brickra[ii+1], brickdec[jj], brickdec[jj+1], sigell=sourceargs['sigell'], nsrc = sourceargs['nsrc'])
            
            #processing the source galaxies
            for igal in range(len(datagal[:,0])):
                ragal, decgal, zphotgal, wgal, e1gal, e2gal = datagal[igal,:]
                wpipe.process_source(ragal, decgal, zphotgal, wgal, e1gal, e2gal, zdiff=sourceargs['zdiff'])

    wpipe.write2file()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=11.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=13.0)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    #make the directory for the output
    from subprocess import call
    #call("mkdir -p %s" % (config["outputdir"]), shell=1)

    outputfilename = '%s/dsigma.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax


    config['test_case'] = args.test_case

    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'


    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
    if args.test_case:
        outputfilename = outputfilename + '_test_case'
    np.random.seed(args.seed)

    run_pipe(config, outputfilename = outputfilename)           

