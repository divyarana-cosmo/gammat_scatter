# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper
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
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from create_sources import get_xyz, create_sources 

def run_pipe(config, outputfilename='gamma.dat', outputpairfile=None):
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

    colossus_cosmo = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8=ss.sigma8, ns=ss.ns, cosmo_name=ss.cosmo_name)

    # Initialize arrays for accumulation
    Njacks = int(lensargs['Njacks'])
    sumdgammat_num              = np.zeros(nbins)
    sumdgammat_inp_num          = np.zeros(nbins)
    sumdgammat_inp_bary_num     = np.zeros(nbins)
    sumdgammat_inp_dm_num       = np.zeros(nbins)
    sumdgammatsq_num            = np.zeros(nbins)
    sumdgammax_num              = np.zeros(nbins) 
    sumdgammaxsq_num            = np.zeros(nbins)
    sumdwls                     = np.zeros(nbins)
    sumddsigmat_num             = np.zeros(nbins)
    sumddsigmat_inp_num         = np.zeros(nbins)
    sumddsigmat_inp_bary_num    = np.zeros(nbins)
    sumddsigmat_inp_dm_num      = np.zeros(nbins)
    sumddsigmatsq_num           = np.zeros(nbins)
    sumddsigmax_num             = np.zeros(nbins) 
    sumddsigmaxsq_num           = np.zeros(nbins)
    sumdwls_by_sigcsq           = np.zeros(nbins)

    # Getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg = lens_select(lensargs)
    
    # Vectorized conversion of effective radius
    thetare = get_re(llogmstel, lzred)  # in units of arcsec
    thetare = thetare * np.pi / (180 * 60 * 60)  # arcsec to radians
    llogre = -2.5 + 0.0*np.log10(thetare * ss.Astropy_cosmo.comoving_distance(lzred).value)  # in the units of h-1 Mpc
    
    lid = np.arange(len(lid))
    
    # Vectorized calculation of concentration parameters
    xx = np.linspace(9, 16, 50)
    med_lzred = np.mean(lzred)
    
    # Vectorize by creating a function that applies to each mass value
    masses = 10**xx
    concs = np.array([concentration.concentration(mh, '200m', med_lzred, model='diemer19') for mh in masses])
    
    spl_c_mh = interp1d(xx, concs)
    lconc = 11 + 0.0*spl_c_mh(llogmh)
    
    print("lens data read fully", np.min(lzred))
    # Calculate maximum angular separation based on minimum redshift
    dismax = config['Rmax'] / ss.Astropy_cosmo.comoving_distance(np.min(lzred)).value 
    print(np.min(lzred), 'thetamax', dismax) 
    
    # Handle shear option
    if sourceargs['use_shear']:
        print("using shear not reduced shear for the sims")
        outputfilename = outputfilename + '_using_shear'

    # Open pair output file if needed
    if outputpairfile is not None:
        fpairout = open(outputpairfile, "w")
        fpairout.write('jkid\tlra(deg)\tldec(deg)\tlzred\tllogmstel\tllogmh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\twls\tkappa\tintse1\tintse2\tr90se1\tr90se2\tr90et\tr90ex\tr90intse1\tr90intse2\n')
    
    # Process each lens
    for ii in tqdm(range(len(lra))):
        # Create sources for this lens - vectorized creation
        sra, sdec, szred, wgal, intse1, intse2 = create_sources(
            lra[ii], ldec[ii], dismax, 
            nsrc=sourceargs['nsrc'], 
            sigell=sourceargs['sigell'], 
            seed=int(config["seed"] * len(lra) + lid[ii])
        ) 
        
        # Handle test case option
        if config['test_case']:
            lra[ii]         =   130.0
            ldec[ii]        =   0.0   
            lzred[ii]       =   0.2 
            llogmstel[ii]   =   10.0
            llogre[ii]      =   -2.5
            llogmh[ii]      =   12.0
            lconc[ii]       =   10
            szred = np.full_like(sra, 0.9)
            
        # Handle shape noise option
        if sourceargs['no_shape_noise']:
            print("no shape noise")
            intse1 = np.zeros_like(intse1)
            intse2 = np.zeros_like(intse2)

        print("number of sources:", len(sra))
        
        # Vectorized source selection
        scut = szred > (lensargs['zmax'] + sourceargs['zdiff'])  # zdiff cut
        if np.sum(scut) == 0:
            continue

        # Apply cut to all source arrays at once
        sra     = sra[scut]
        sdec    = sdec[scut]        
        szred   = szred[scut]
        wgal    = wgal[scut]
        intse1  = intse1[scut]
        intse2  = intse2[scut]
        

        # Shear all sources at once
        se1, se2, etan, kappa, proj_sep, sflag, etan_b, etan_dm, et_obs, ex_obs = ss.shear_src(
            lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogre[ii], llogmh[ii], lconc[ii],
            sra, sdec, szred, intse1, intse2, 
            use_shear=sourceargs["use_shear"], 
            no_shear=sourceargs["no_shear"]
        )
        
        print('flagged sources', np.sum(sflag))
        
        # Handle no shear option
        if sourceargs['no_shear']:
            se1 = intse1
            se2 = intse2
            
        sl_sep  = proj_sep
        w_ls    = lwgt[ii] * wgal
        
        # Vectorized filtering of arrays
        idx = (sl_sep > rmin) & (sl_sep < rmax) & (sflag == 1)
        if np.sum(idx) == 0:
            continue
            
        # Apply filter to all arrays at once
        sl_sep  = sl_sep[idx]
        w_ls    = w_ls[idx]
        et_obs  = et_obs[idx]
        etan    = etan[idx]
        kappa   = kappa[idx]
        etan_b  = etan_b[idx]
        etan_dm = etan_dm[idx]
        ex_obs  = ex_obs[idx]
        se1     = se1[idx]
        se2     = se2[idx]
        sra     = sra[idx]
        sdec    = sdec[idx]
        szred   = szred[idx]

        # Write pairs to output file if needed
        if outputpairfile is not None:
            for jj in range(np.sum(idx)):
                fpairout.write(f'{lxjkreg[ii]}\t{lra[ii]}\t{ldec[ii]}\t{lzred[ii]}\t{llogmstel[ii]}\t{llogmh[ii]}\t{lconc[ii]}\t{sra[jj]}\t{sdec[jj]}\t{szred[jj]}\t{se1[jj]}\t{se2[jj]}\t{etan[jj]}\t{et_obs[jj]}\t{ex_obs[jj]}\t{sl_sep[jj]}\t{w_ls[jj]}\t{kappa[jj]}\t{intse1[jj]}\t{intse2[jj]}\t{r90se1[jj]}\t{r90se2[jj]}\t{r90et_obs[jj]}\t{r90ex_obs[jj]}\t{r90intse1[jj]}\t{r90intse2[jj]}\n')
        
        # Vectorized calculation of sigma critical terms
        w_ls_invsigmacritsq = np.zeros(int(np.sum(idx)))
        w_ls_invsigmacrit = np.zeros(int(np.sum(idx)))
        
        # Use vectorized operation instead of loop
        sigma_crit_inv = ss._get_sigma_crit_inv(lzred=lzred[ii], szred=szred) * 1e12 
        #sigma_crit_inv = np.array([ss._get_sigma_crit_inv(lzred=lzred[ii], szred=z) * 1e12 for z in szred])
        w_ls_invsigmacritsq = w_ls * sigma_crit_inv**2
        w_ls_invsigmacrit   = w_ls * sigma_crit_inv
        
        # Vectorized binning
        slrbins = (np.log10(sl_sep / rmin) // rdiff).astype(int)
        
        # Process each bin
        for rb in range(nbins):
            idx = slrbins == rb
            if np.sum(idx) == 0:
                continue
                
            # Vectorized accumulation for each bin
            sumdwls[rb]                     += np.sum(w_ls[idx])
            sumdwls_by_sigcsq[rb]           += np.sum(w_ls_invsigmacritsq[idx])

            sumdgammat_inp_num[rb]          += np.sum((w_ls * etan)[idx])
            sumdgammat_inp_bary_num[rb]     += np.sum((w_ls * etan_b)[idx])
            sumdgammat_inp_dm_num[rb]       += np.sum((w_ls * etan_dm)[idx])
                
            sumdgammat_num[rb]              += np.sum((w_ls * et_obs)[idx])
            sumdgammatsq_num[rb]            += np.sum(((w_ls * et_obs)**2)[idx])
            sumdgammax_num[rb]              += np.sum((w_ls * ex_obs)[idx])
            sumdgammaxsq_num[rb]            += np.sum(((w_ls * ex_obs)**2)[idx])
           
            sumddsigmat_inp_num[rb]         += np.sum((w_ls_invsigmacrit * etan)[idx])
            sumddsigmat_inp_bary_num[rb]    += np.sum((w_ls_invsigmacrit * etan_b)[idx])
            sumddsigmat_inp_dm_num[rb]      += np.sum((w_ls_invsigmacrit * etan_dm)[idx])

            sumddsigmat_num[rb]             += np.sum((w_ls_invsigmacrit * et_obs)[idx])
            sumddsigmatsq_num[rb]           += np.sum(((w_ls_invsigmacrit * et_obs)**2)[idx])
            sumddsigmax_num[rb]             += np.sum((w_ls_invsigmacrit * ex_obs)[idx])
            sumddsigmaxsq_num[rb]           += np.sum(((w_ls_invsigmacrit * ex_obs)**2)[idx])

    # Close pair output file if needed
    if outputpairfile is not None:
        fpairout.write("#OK")
        fpairout.close()
        
    # Calculate responsivity correction
    if sourceargs['no_shape_noise']:
        Resp = 1.0  
    else:    
        Resp = 1.0 - sourceargs['sigell']**2 
    
    # Create dictionary for results
    df = {}
    df["0-rmin/2+rmax/2"]           = rbins[:-1] * 0.5 + rbins[1:] * 0.5
    df["1-gammat"]                  = sumdgammat_num / sumdwls / Resp    
    df["2-gammatsq"]                = sumdgammatsq_num / sumdwls / Resp**2
    df["3-sigma_gammat"]            = np.sqrt(sumdgammatsq_num / sumdwls / Resp**2 - (sumdgammat_num / sumdwls / Resp)**2)
    df["4-SN_Errgammat"]            = np.sqrt(sumdgammatsq_num) / sumdwls / Resp
    df["5-gammax"]                  = sumdgammax_num / sumdwls / Resp
    df["6-gammaxsq"]                = sumdgammaxsq_num / sumdwls / Resp**2
    df["7-sigma_gammax"]            = np.sqrt(sumdgammaxsq_num / sumdwls / Resp**2 - (sumdgammax_num / sumdwls / Resp)**2)
    df["8-SN_Errgammax"]            = np.sqrt(sumdgammaxsq_num) / sumdwls / Resp
    df["9-gammat_inp"]              = sumdgammat_inp_num / sumdwls / Resp
    df["10-gammat_inp_bary"]        = sumdgammat_inp_bary_num / sumdwls / Resp
    df["11-gammat_inp_dm"]          = sumdgammat_inp_dm_num / sumdwls / Resp
    df["12-sumd_wls"]               = sumdwls
    df["13-dsigma"]                 = sumddsigmat_num / sumdwls_by_sigcsq / Resp
    df["14-dsigmasq"]               = sumddsigmatsq_num / sumdwls_by_sigcsq / Resp**2
    df["15-SN_Errdsigmat"]          = np.sqrt(sumddsigmatsq_num) / sumdwls_by_sigcsq / Resp
    df["16-dsigmax"]                = sumddsigmax_num / sumdwls_by_sigcsq / Resp
    df["17-dsigmaxsq"]              = sumddsigmaxsq_num / sumdwls_by_sigcsq / Resp**2
    df["18-SN_Errdsigmax"]          = np.sqrt(sumddsigmaxsq_num) / sumdwls_by_sigcsq / Resp
    df["19-dsigmat_inp"]            = sumddsigmat_inp_num / sumdwls_by_sigcsq / Resp
    df["20-dsigmat_inp_bary"]       = sumddsigmat_inp_bary_num / sumdwls_by_sigcsq / Resp
    df["21-dsigmat_inp_dm"]         = sumddsigmat_inp_dm_num / sumdwls_by_sigcsq / Resp
    df["22-sumd_dsigma_wls" ]       = sumdwls_by_sigcsq
    df["23-sumd_dsigma_num" ]       = sumddsigmat_num
    df["24-sumd_dsigmax_num" ]      = sumddsigmax_num
    df["25-sumd_dsigma_den" ]       = sumdwls_by_sigcsq*Resp

    import pandas as pd
    df = pd.DataFrame(df)
    #idx =  sumdwls!=0
    #df = df[idx]
    df.to_csv(outputfilename, index=False, sep=' ')
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=1119976)
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
    
    outputfilename = outputfilename 
    print(config)
    output_filename = outputfilename 
    run_pipe(config, outputfilename = output_filename)           







