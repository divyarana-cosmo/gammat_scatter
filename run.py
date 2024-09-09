#!/usr/bin/python
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=9.0)
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
 
    lensargs    = config["lens"]
    sourceargs  = config["source"]
    ##setting up cosmology and class instance

    params = {'flat': True, 'H0': 100, 'Om0': config['Om0'], 'Ob0': config['Ob0'], 'sigma8': config['sigma8'], 'ns': config['ns']}
    cosmo = cosmology.setCosmology('myCosmo', **params)
    
    # getting the lenses data and massaging it a bit
    lra, ldec, lzred, lwgt, llogMh, llogmstel, llog_re, ljkreg = lens_select(lensargs)
    
    Nbins = 10
    perc = np.linspace(0,1, Nbins+1)*100
    print(len(llogmstel)/Nbins)

    for ii in range(len(perc)-1):
        print(ii, perc[ii],np.percentile(llogmstel, perc[ii]), np.percentile(llogmstel, perc[ii+1]))
        lmstelmin = np.percentile(llogmstel, perc[ii])
        lmstelmax = np.percentile(llogmstel, perc[ii + 1])
        call("python simulate_aroundsources.py --config config  --logmstelmin %2.2f --logmstelmax %2.2f >> logs/%d.out 2>&1 &"%(lmstelmin, lmstelmax, ii), shell=1)

    #from subprocess import  call
    #logmstelbinedgs = [9.5, 10.0, 10.5, 11.0]
    #for jk in range(len(logmstelbinedgs) - 1):
    #call("python simulate_aroundsources.py --config config  --logmstelmin %2.2f --logmstelmax %2.2f >> logs/%d.out 2>&1 &"%(logmstelbinedgs[jk], logmstelbinedgs[jk+1], jk), shell=1)



