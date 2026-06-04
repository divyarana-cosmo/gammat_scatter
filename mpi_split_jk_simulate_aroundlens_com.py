# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper
import sys
sys.path.append('./src/')
sys.path.append('./utils/')
from distort_com import simshear
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from weakpipe_select import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from create_sources import get_xyz, create_sources 
from weakpipe import weakpipe

def run_pipe(config, outputfilename='gamma.dat', jksamp=0):
    """Optimized pipeline function with proper NumPy vectorization"""
    wp = weakpipe(H0=config['H0'], Om0=config['Om0'], Rmin=config['Rmin'], Rmax=config['Rmax'], Nbins=config['Nbins'], outputfilename=outputfilename)

    lensargs    = config["lens"]
    # Getting the lenses data
    lensargs['H0']=config['H0']; lensargs['Om0']=config['Om0']
    if 'sigma_Roff' in lensargs:
        lseed, _lra, _ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg, lRoff = lens_select(lensargs, seed=config['seed'], jk=jksamp)
        wp.process_lensdata(lseed, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lRoff)
    else:    
        lseed, _lra, _ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs, seed=config['seed'], jk=jksamp)
        wp.process_lensdata(lseed, lzred, lwgt, llogmstel, llogre, llogmh, lconc)

    sourceargs  = config["source"]
    wp.weaklens_aroundlens(nsrc=sourceargs['nsrc'], sigell=sourceargs['sigell'], zmax=sourceargs['zmax'], zdiff=sourceargs['zdiff'], seed=config['seed'], use_shear=sourceargs['use_shear'],test_case=config['test_case'])
    wp.write2file()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=1)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--use_shear", help="use shear or reduced shear for simulations", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=9.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=10.5)
    parser.add_argument("--sigma_Roff", help="miscentering sigma for the rayleigh distribution in h-1 Mpc", type=float, default=-999)
    parser.add_argument("--two_percent", help="using two percent of the lense sample", type=bool, default=False)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)


    config["outputdir"] = config["outputdir"] 

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    config['test_case']                 = args.test_case
    config['seed']                      = int(5e11*args.seed)
    config['lens']['two_percent']       = args.two_percent
    config['source']['use_shear']       = args.use_shear
    config['source']['no_shape_noise']  = args.no_shape_noise
    config['source']['no_shear']        = args.no_shear

    if args.sigma_Roff != -999:
        config['lens']['sigma_Roff']        = args.sigma_Roff 
        config["outputdir"] += '%2.2f_%2.2f_seed_%d_sigma_off_%2.2f'%(args.logmstelmin, args.logmstelmax, args.seed, args.sigma_Roff)
    else:
        config["outputdir"] += '%2.2f_%2.2f_seed_%d'%(args.logmstelmin, args.logmstelmax, args.seed)
    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])
    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)



    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.use_shear:
        outputfilename = outputfilename + '_using_shear'
    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
    if args.test_case:
        outputfilename = outputfilename + '_test_case'
    
    outputfilename = outputfilename + '_w_jacks'
    print(config)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    


    for jk in range(config['lens']['Njacks']):
        if jk%size !=rank:
            continue
        output_filename = outputfilename + '_jk_%d'%jk+'_fast'
        run_pipe(config, outputfilename = output_filename, jksamp=jk)           

    comm.Barrier()






