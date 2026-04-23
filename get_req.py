import sys
sys.path.append('./src/')
sys.path.append('./utils/')
from lensutils import get_re
from distort_com import simshear
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from weakpipe_select import lens_select
from tqdm import tqdm
import argparse
import yaml
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from create_sources import get_xyz, create_sources 

def run_pipe(config, outputfilename='gamma.dat', jksamp=0, outputpairfile=None):
    """Optimized pipeline function with proper NumPy vectorization"""
    rmin = config['Rmin'] 
    rmax = config['Rmax'] 
    nbins = config['Nbins']
    
    # Set the projected radial binning in units of Mpc
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff = np.log10(rbins[1] / rbins[0])
 
    lensargs = config["lens"]
    sourceargs = config["source"]

    zdiff = sourceargs["zdiff"]
    
    # Initialize simshear object
    ss = simshear(H0=config['H0'], Om0=config['Om0'])

    #colossus_cosmo = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8=ss.sigma8, ns=ss.ns, cosmo_name=ss.cosmo_name)

    # Getting the lenses data
    lensargs['H0']=config['H0']; lensargs['Om0']=config['Om0']
    #lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs, jk=jksamp)
    return lens_select(lensargs)
    


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
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=14.5)
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
    
    outputfilename = outputfilename + '_w_jacks'
    print(config)
    bbins = 9.5 + 0.1*np.arange(21)
 
    #ax1 = plt.subplot(3,3,1)
    #Ngals = np.array([])
    #for logMmin, logMmax in zip(bbins[:-1], bbins[1:]):
    #    config['lens']['logmstelmin'] = logMmin
    #    config['lens']['logmstelmax'] = logMmax
    #
    #    lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = run_pipe(config, outputfilename)                   
    #    Ngals = np.append(Ngals, len(lid))

    #ax1.plot(bbins[:-1]*0.5 + bbins[1:]*0.5, Ngals*1.75)
    #ax1.set_yscale('log')
    #ax1.set_xlabel(r'$\log[M_{*}/{h^{-2} {\rm M_\odot}}]$')
    #ax1.set_ylabel(r'$N_{\rm gals}$')
    #plt.savefig('plots/lens_dist.pdf', dpi=300)
    #print('histogram done')
    #
    #plt.clf()

    
    from stellarpy import stellar
    from halopy import halo
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    lm_arr  =   np.array([]) 
    re_arr  =   np.array([]) 
    req_arr =   np.array([]) 
    
    omgm0 = 0.319
    for logMmin, logMmax in zip(bbins[:-1], bbins[1:]):
        config['lens']['logmstelmin'] = logMmin
        config['lens']['logmstelmax'] = logMmax
    
        lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = run_pipe(config, outputfilename)                   
 
        Nlens = int(len(lid))
        for jj in range(Nlens):
            #jj   = np.random.choice(len(lid), 1)
            bary = stellar(llogmstel[jj], llogre[jj])
            dark = halo(llogmh[jj], lconc[jj],omg_m=omgm0)

            rp = np.logspace(-5,0,10)

            esddiff = (bary.esd_deVaucouleurs(rp) - dark.esd_nfw(rp))/1e12
            spl = ius(np.log10(rp), esddiff)

            x = spl.roots()
            if len(x)==0:
                print('skipping')
                continue
            lm_arr      =   np.append(lm_arr, llogmstel[jj])
            req_arr     =   np.append(req_arr, 1e3 * 10**x)
            re_arr      =   np.append(re_arr, 1e3 * 10**llogre[jj])

    #idx = lm_arr>0
    print(len(lm_arr), len(req_arr), len(re_arr)) 
    ax1 = plt.subplot(3,3,1)

    #ax1.plot(lm_arr[idx] , req_arr[idx], '.', ms=1.0,c='C0', alpha=0.3, zorder=0)
    #ax1.plot(lm_arr[idx] , re_arr[idx], '.', ms=1.0,c='C1' , alpha=0.5, zorder=1)
    
    #putting a median line
    from scipy.stats import binned_statistic

    req_16, binedgs, b =  binned_statistic(lm_arr, req_arr, statistic=lambda y: np.percentile(y,16), bins=bbins)
    req_50, binedgs, b =  binned_statistic(lm_arr, req_arr, statistic=lambda y: np.percentile(y,50), bins=bbins)
    req_84, binedgs, b =  binned_statistic(lm_arr, req_arr, statistic=lambda y: np.percentile(y,84), bins=bbins)

    re_16, binedgs, b =  binned_statistic(lm_arr, re_arr, statistic=lambda y: np.percentile(y,16), bins=bbins)
    re_50, binedgs, b =  binned_statistic(lm_arr, re_arr, statistic=lambda y: np.percentile(y,50), bins=bbins)
    re_84, binedgs, b =  binned_statistic(lm_arr, re_arr, statistic=lambda y: np.percentile(y,84), bins=bbins)
    
    print(binedgs)
    print(req_50)
    print(re_50)
    
    #ax1.hist2d(lm_arr, req_arr, bins=[bbins, np.linspace(0.5,30,27)], cmap='Greys', cmin=1, zorder=0)

    ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, req_50, ls='-', color='C0', zorder=1, label=r'$R_{\rm eq}$')
    #ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, req_16, ls='--', color='C0', zorder=1)
    #ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, req_84, ls='--', color='C0', zorder=1)



    ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, re_50, ls='-', color='C1' , zorder=2, label=r'$R_{\rm e}$')
    #ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, re_16, ls='--', color='C1' , zorder=2)
    #ax1.plot(binedgs[:-1]*0.5 + binedgs[1:]*0.5, re_84, ls='--', color='C1' , zorder=2)

    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\log[M_{*}/{h^{-2}{\rm M_\odot}}]$')
    ax1.set_ylabel(r'$R_{\rm x}{[h^{-1}{\rm kpc}]}$')
    plt.legend()
    plt.savefig('plots/req_sample.pdf')   








