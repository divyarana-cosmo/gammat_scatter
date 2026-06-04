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
from subprocess import call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from create_sources import get_xyz, create_sources 
from weakpipe import weakpipe

def run_pipe(config, outputfilename='gamma.dat', jksamp=0):
    """Optimized pipeline function with proper NumPy vectorization"""
    wp = weakpipe(H0=config['H0'], Om0=config['Om0'], Rmin=config['Rmin'], Rmax=config['Rmax'], Nbins=config['Nbins'], outputfilename=outputfilename)

    lensargs = config["lens"]
    lensargs['H0'] = config['H0']
    lensargs['Om0'] = config['Om0']
    
    # Getting the lenses data (Now capturing lkind after lconc)
    if 'sigma_Roff' in lensargs and lensargs['sigma_Roff'] != -999:
        lseed, _lra, _ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lkind, lxjkreg, lRoff = lens_select(lensargs, seed=config['seed'], jk=jksamp)
    else:    
        lseed, _lra, _ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lkind, lxjkreg = lens_select(lensargs, seed=config['seed'], jk=jksamp)
        lRoff = None

    # Apply Gaussian scatter to logRe if specified
    sigma_logre = config['lens'].get('sigma_logre', 0.0)
    if sigma_logre > 0.0:
        # Seeded locally per jackknife to guarantee deterministic MPI runs
        rng = np.random.default_rng(config['seed'] + jksamp)
        llogre = rng.normal(loc=llogre, scale=sigma_logre)

    # Pass lkind directly to the pipeline
    wp.process_lensdata(lseed, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lkind=lkind, lRoff=lRoff)

    sourceargs = config["source"]
    wp.weaklens_aroundlens(
        nsrc=sourceargs['nsrc'], 
        sigell=sourceargs['sigell'], 
        zmax=sourceargs['zmax'], 
        zdiff=sourceargs['zdiff'], 
        seed=config['seed'], 
        use_shear=sourceargs['use_shear'],
        test_case=config['test_case'],
        no_shape_noise=sourceargs['no_shape_noise'],
        no_shear=sourceargs['no_shear']
    )
    
    wp.write2file()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=float, default=1.0)
    
    parser.add_argument("--no_shape_noise", action="store_true", help="for removing shape noise-testing purpose")
    parser.add_argument("--no_shear", action="store_true", help="for removing shear-testing purpose")
    parser.add_argument("--test_case", action="store_true", help="testing the ideal case")
    parser.add_argument("--use_shear", action="store_true", help="use shear or reduced shear for simulations")
    parser.add_argument("--rot90", action="store_true", help="rotating intrinsic shapes by 90 degrees")
    parser.add_argument("--two_percent", action="store_true", help="using two percent of the lense sample")
    
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=9.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=10.5)
    parser.add_argument("--sigma_Roff", help="miscentering sigma for the rayleigh distribution in h-1 Mpc", type=float, default=-999)
    parser.add_argument("--sigma_logre", help="Gaussian scatter applied to llogre", type=float, default=0.0)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Populate config dictionary
    if 'logmstelmin' not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax' not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    config['test_case']                 = args.test_case
    config['seed']                      = int(5e11 * args.seed)
    config['lens']['two_percent']       = args.two_percent
    config['source']['use_shear']       = args.use_shear
    config['source']['no_shape_noise']  = args.no_shape_noise
    config['source']['no_shear']        = args.no_shear

    # Handle Naming and Directory Structure
    dir_name = '%2.2f_%2.2f_seed_%d' % (args.logmstelmin, args.logmstelmax, config['seed'])
    file_base = 'simed_sources'
    
    # Append Roff to name if active
    if args.sigma_Roff != -999:
        config['lens']['sigma_Roff'] = args.sigma_Roff 
        dir_name += '_sigma_off_%2.2f' % (args.sigma_Roff)
        file_base += '_Roff_%2.2f' % (args.sigma_Roff)
        
    # Append logre scatter to name if active
    if args.sigma_logre > 0.0:
        config['lens']['sigma_logre'] = args.sigma_logre
        dir_name += '_sigRe_%2.2f' % (args.sigma_logre)
        file_base += '_sigRe_%2.2f' % (args.sigma_logre)
        
    config["outputdir"] += dir_name
    
    # Ensure directory exists
    call("mkdir -p %s" % (config["outputdir"]), shell=True)

    # Build highly descriptive output filename
    outputfilename = '%s/%s_lmmin_%2.2f_lmmax_%2.2f' % (config['outputdir'], file_base, args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename += '_no_shape_noise'
    else:
        outputfilename += '_with_shape_noise'
        if args.rot90:
            outputfilename += '_with_90_rotation'

    if args.use_shear:
        outputfilename += '_using_shear'
    if args.no_shear:
        outputfilename += '_no_shear'
    if args.test_case:
        outputfilename += '_test_case'
    
    outputfilename += '_w_jacks'
    
    print("\n--- Pipeline Configuration ---")
    print(config)
    print("------------------------------\n")

    # MPI Parallelization across Jackknife regions
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    

    for jk in range(config['lens']['Njacks']):
        if jk % size != rank:
            continue
            
        print(f"Rank {rank} processing Jackknife {jk}...")
        output_filename_jk = outputfilename + '_jk_%d_fast' % jk
        run_pipe(config, outputfilename=output_filename_jk, jksamp=jk)            

    comm.Barrier()
    if rank == 0:
        print("All jackknife regions processed successfully.")
