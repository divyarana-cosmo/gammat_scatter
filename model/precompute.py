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
from get_data import lens_select


def get_pzl(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, Nzlbins=101):
    "creates a probability distribution for the lens redshifts"
    lensargs = {}
    lensargs['type']        =   lenstype
    lensargs['logmstelmin'] =   logMmin
    lensargs['logmstelmax'] =   logMmax
    lensargs['zmin']        =   zlmin
    lensargs['zmax']        =   zlmax
    lensargs['Njacks']      =   Njacks
    lensargs['H0']          =   H0
    lensargs['Om0']         =   Om0

    lid, lra, ldec, lzred, lwgt, llogmstel, llogre, llogmh, lconc, lxjkreg = lens_select(lensargs)
    zzbins                  = np.linspace(zlmin, zlmax, Nzlbins)
    nlens,binedgs           = np.histogram(lzred, bins=zzbins)

   
    zlbins             =   binedgs[1:]*0.5 + binedgs[:-1]*0.5
    pzl                =   nlens/sum(nlens)
    mean_lzred         =   np.mean(lzred) + 0.0*zlbins
    del lid, lra, ldec, lzred, lwgt, llogmh, lconc, lxjkreg
    gc.collect()
    return llogmstel, llogre, zlbins, pzl, mean_lzred


def precomputes(config):
    # for the test case 
    H0          =  config['H0'] 
    Om0         =  config['Om0'] 
    lenstype    =  config['lenstype']     
    logMmin     =  config['logMmin'] 
    logMmax     =  config['logMmax'] 
    zlmin       =  config['zlmin'] 
    zlmax       =  config['zlmax'] 
    Njacks      =  config['Njacks'] 
    zdiff       =  config['zdiff'] 

    #creating modelling class instance
    llogmstel, llogre, zlbins, pzl, mean_lzred = get_pzl(H0, Om0, lenstype, logMmin, logMmax, zlmin, zlmax, Njacks, Nzlbins=101)
    rbins = np.logspace(-4,0,51)
    esd_s      = 0.0*rbins
    sigma_s    = 0.0*rbins

    stel = stellar()
    stel.log_mstel  = llogmstel
    stel.log_re     = llogre   
    for ii in range(len(rbins)):
        esd_s[ii]   = np.mean(stel.esd_deVaucouleurs(rbins[ii]))
        sigma_s[ii] = np.mean(stel.sigma_deVaucouleurs(rbins[ii]))
    
    outdir = './precompute/'
    os.system('mkdir -p ./precompute/')
    np.savetxt(outdir+'esd_s_%s_%s.dat'%(logMmin, logMmax), np.transpose([rbins, esd_s, sigma_s]), header='rbins, esd_s, sigma_s')
    np.savetxt(outdir+'pzl_%s_%s.dat'%(logMmin, logMmax), np.transpose([zlbins, pzl, mean_lzred]), header='zlbins, pzl, mean_zl')
    print("precomputations for stellar contribution done")
    return 0    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    precomputes(config)
