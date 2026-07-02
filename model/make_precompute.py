import sys
import os
import numpy as np
import pandas as pd
sys.path.append('/home/rana/github_0/gammat_scatter/src/')
from stellarpy import stellar
import gc
import argparse
import yaml
sys.path.append('/home/rana/github_0/gammat_scatter/')
from weakpipe_select import lens_select


def get_pzl(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, seed, Nzlbins=101):
    "creates a probability distribution for the lens redshifts"
    lensargs = {}
    lensargs['type']        =   lenstype
    lensargs['logmstelmin'] =   logMmin
    lensargs['logmstelmax'] =   logMmax
    lensargs['zlmin']        =   zlmin
    lensargs['zlmax']        =   zlmax
    lensargs['Njacks']      =   Njacks
    lensargs['H0']          =   H0
    lensargs['Om0']         =   Om0

    lseed, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs, seed=seed)
    zzbins                  = np.linspace(zlmin, zlmax, Nzlbins)
    nlens,binedgs           = np.histogram(lzred, bins=zzbins)

    print(np.mean(lzred)) 
    zlbins             =   binedgs[1:]*0.5 + binedgs[:-1]*0.5
    pzl                =   nlens/sum(nlens)
    mean_lzred         =   np.mean(lzred) + 0.0*zlbins
    del lseed, lra, ldec, lzred, lwgt, llogmh, lconc, lxjkreg
    gc.collect()
    return llogmstel, llogre, zlbins, pzl, mean_lzred


def precomputes(config):
    # for the test case 
    H0          =  config['H0'] 
    Om0         =  config['Om0'] 
    dsigma_fil  =  config['dsigma_fil']
    lenstype    =  config['lenstype']     
    logMmin     =  config['logMmin'] 
    logMmax     =  config['logMmax'] 
    zlmin       =  config['zlmin'] 
    zlmax       =  config['zlmax'] 
    Njacks      =  config['Njacks'] 
    zdiff       =  config['zdiff'] 
    seed        =  config['seed']
 
    rbins = np.loadtxt(dsigma_fil)[:,0]
    #creating modelling class instance
    llogmstel, llogre, zlbins, pzl, mean_lzred = get_pzl(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, Nzlbins=101, seed=seed)
    esd_s               = 0.0*rbins
    sigma_s             = 0.0*rbins
    prod_esd_sigma_s    = 0.0*rbins

    stel = stellar()
    print(len(llogmstel), np.mean(mean_lzred))
    print(len(llogre))
    stel.log_mstel  = llogmstel
    stel.log_re     = llogre   
    for ii in range(len(rbins)):
        print(len(stel.esd_deVaucouleurs(rbins[ii])))
        esd_s[ii]               = np.mean(stel.esd_deVaucouleurs(rbins[ii]))
        sigma_s[ii]             = np.mean(stel.sigma_deVaucouleurs(rbins[ii]))
        prod_esd_sigma_s[ii]    = np.mean(stel.esd_deVaucouleurs(rbins[ii]) * stel.sigma_deVaucouleurs(rbins[ii]))
    
    outdir = './precompute/'
    os.system('mkdir -p ./precompute/')
    np.savetxt(outdir+'esd_s_%s_%s.dat'%(logMmin, logMmax), np.transpose([rbins, esd_s, sigma_s, prod_esd_sigma_s]), header='rbins, esd_s, sigma_s, esd_s_sigma_s')
    np.savetxt(outdir+'pzl_%s_%s.dat'%(logMmin, logMmax), np.transpose([zlbins, pzl, mean_lzred]), header='zlbins, pzl, mean_zl')
    print("precomputations for stellar contribution done")
    return 0    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--logMmin", help="minimum stellar mass", default=9.5, type=float)
    parser.add_argument("--logMmax", help="maximum stellar mass", default=11.0, type=float)
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    config['logMmin']     =  args.logMmin 
    config['logMmax']     =  args.logMmax 
    config['seed']        = int(5e11*args.seed)

    config['dsigma_fil']  = '/home/rana/github_0/gammat_scatter/output/desi_z_0.0_0.4/iso_centrals/%2.2f_%2.2f_seed_%d/dsigma.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_ovpsamp_100'%(args.logMmin, args.logMmax, args.seed, args.logMmin, args.logMmax)  # we need this to fix the radial bins
    precomputes(config)
