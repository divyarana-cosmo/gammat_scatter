import numpy as np
import pandas as pd
import sys
import os
import argparse
import yaml

sys.path.append('/net/dobbe/data2/github/gammat_scatter/model/')
sys.path.append('/net/dobbe/data2/github/gammat_scatter/src/')
from model import model
from distort_com import simshear

def build_dir_path(base_dir, logMmin, logMmax, seed, oversamp, sigma_Roff=None, sigma_logre=None):
    """
    Reconstructs the directory path dynamically using the config and oversampling factor.
    Ensure your MPI run script uses this exact same naming convention!
    """
    dir_name = f'{logMmin:2.2f}_{logMmax:2.2f}_seed_{int(seed)}_ovp_{int(oversamp)}'
    
    if sigma_Roff and sigma_Roff != -999:
        dir_name += f'_sigma_off_{sigma_Roff:2.2f}'
    if sigma_logre and sigma_logre > 0.0:
        dir_name += f'_sigRe_{sigma_logre:2.2f}'
        
    return os.path.join(base_dir, dir_name) + '/'

def get_jk_leaves(outdir, logMmin, logMmax, Njacks):
    """
    Pre-loads all Jackknife files into RAM exactly ONCE per mass bin,
    then constructs the N-1 jackknife leaves in memory.
    """
    data_jacks = []
    
    # 1. Pre-load all jackknife files to prevent 10,000+ redundant I/O operations
    for jk in range(Njacks):
        # Assumes standard filename generation from the MPI script
        file = os.path.join(outdir, f'simed_sources_lmmin_{logMmin:2.2f}_lmmax_{logMmax:2.2f}_with_shape_noise_w_jacks_jk_{jk}_fast')
        try:
            data = pd.read_csv(file, delim_whitespace=True, skipfooter=1, engine='python')
            data_jacks.append(data)
        except Exception as e:
            print(f"  -> Missing or unreadable file: {file}")
            return None

    rbin = data_jacks[0]['2-rmin/2+rmax/2'].values
    
    # 2. Accumulate the total denominators for the Baryonic and SigC calculations
    stel_num_total = sum((d['22-dsigmat_inp_bary'].values * d['27-sumd_dsigma_den'].values) for d in data_jacks)
    stel_den_total = sum(d['27-sumd_dsigma_den'].values for d in data_jacks)
    avg_inv_sigc_num_total = sum(d['28-sumd_wls_by_sigmac'].values for d in data_jacks)
    avg_inv_sigc_den_total = sum(d['14-sumd_wls'].values for d in data_jacks)

    # 3. Construct the N-1 Jackknife leaves in memory
    dsigma_leaves = []
    xdsigma_leaves = []
    
    for jk in range(Njacks):
        num = 0; den = 0; xnum = 0
        for ii in range(Njacks):
            if ii == jk: 
                continue
            num  += data_jacks[ii]['25-sumd_dsigma_num'].values
            den  += data_jacks[ii]['27-sumd_dsigma_den'].values
            xnum += data_jacks[ii]['26-sumd_dsigmax_num'].values
            
        dsigma_leaves.append(num / den)
        xdsigma_leaves.append(xnum / den)

    stel_dsigmaarr = stel_num_total / stel_den_total
    avg_inv_sigcarr = avg_inv_sigc_num_total / avg_inv_sigc_den_total
    avg_inv_sigcarrsq = stel_den_total / avg_inv_sigc_den_total

    return np.array(dsigma_leaves), np.array(xdsigma_leaves), stel_dsigmaarr, avg_inv_sigcarr, avg_inv_sigcarrsq, rbin


def process_mass_bin(config, logMmin, logMmax, oversamp_sig, oversamp_cov):
    """
    Computes delta sigma using the highly oversampled signal directory, 
    but strictly computes the physical covariance matrix using the 1x directory.
    """
    print(f"\nProcessing Mass Bin: {logMmin:2.2f} - {logMmax:2.2f}")
    
    seed = config.get('seed', 1)
    Njacks = config['lens']['Njacks']
    sig_Roff = config['lens'].get('sigma_Roff', -999)
    sig_Re   = config['lens'].get('sigma_logre', 0.0)

    # Build precise directories using the single config + respective oversampling factors
    dir_sig = build_dir_path(config['outputdir'], logMmin, logMmax, seed, oversamp_sig, sig_Roff, sig_Re)
    dir_cov = build_dir_path(config['outputdir'], logMmin, logMmax, seed, oversamp_cov, sig_Roff, sig_Re)
    
    print(f"  Signal Dir : {dir_sig}")
    print(f"  Cov Dir    : {dir_cov}")

    # Get leaves for the signal and covariance
    res_signal = get_jk_leaves(dir_sig, logMmin, logMmax, Njacks)
    res_cov    = get_jk_leaves(dir_cov, logMmin, logMmax, Njacks)

    if res_signal is None or res_cov is None:
        print(f"  -> Skipping bin {logMmin:2.2f}-{logMmax:2.2f} due to missing data.")
        return

    dsigma_leaves_sig, xdsigma_leaves_sig, stel_dsigmaarr, avg_inv_sigcarr, avg_inv_sigcarrsq, rbin = res_signal
    dsigma_leaves_cov, xdsigma_leaves_cov, _, _, _, _ = res_cov

    # 1. SIGNAL: Evaluated from the heavily oversampled runs
    dsigma  = np.mean(dsigma_leaves_sig, axis=0)        
    xdsigma = np.mean(xdsigma_leaves_sig, axis=0)        
    
    # 2. COVARIANCE: Evaluated STRICTLY from the baseline (e.g. 1x) runs
    dsigma_cov_mean  = np.mean(dsigma_leaves_cov, axis=0)
    xdsigma_cov_mean = np.mean(xdsigma_leaves_cov, axis=0)
    
    cov = np.zeros((len(rbin), len(rbin)))
    xcov = np.zeros((len(rbin), len(rbin)))
    
    for ii in range(len(rbin)):
        for jj in range(len(rbin)):
            cov[ii,jj]  = np.mean((dsigma_leaves_cov[:,ii] - dsigma_cov_mean[ii]) * (dsigma_leaves_cov[:,jj] - dsigma_cov_mean[jj]))
            xcov[ii,jj] = np.mean((xdsigma_leaves_cov[:,ii] - xdsigma_cov_mean[ii]) * (xdsigma_leaves_cov[:,jj] - xdsigma_cov_mean[jj]))
            
    # Because we use the baseline 1x run, the physical area multiplier is standard.
    # Standard Jackknife prefactor applies.
    cov  *= (Njacks - 1)         
    xcov *= (Njacks - 1)
    
    dsigmaerr  = np.diag(cov)**0.5
    xdsigmaerr = np.diag(xcov)**0.5

    # Save the output back into the main Signal directory
    np.savetxt(dir_sig + f'dsigma.dat_lmstelmin_{logMmin:2.2f}_lmstelmax_{logMmax:2.2f}_ovpsamp_{int(oversamp_sig)}', 
               np.transpose([rbin, dsigma, dsigmaerr, xdsigma, xdsigmaerr, stel_dsigmaarr, avg_inv_sigcarr, avg_inv_sigcarrsq]), 
               header='Rp[h-1_Mpc] dsigma dsigmaerr xdsigma xdsigmaerr stel_dsigma avg_sigc avg_sigc_sq')
    
    np.savetxt(dir_sig + f'cov_dsigma.dat_lmstelmin_{logMmin:2.2f}_lmstelmax_{logMmax:2.2f}', cov)
    np.savetxt(dir_sig + f'xcov_dsigma.dat_lmstelmin_{logMmin:2.2f}_lmstelmax_{logMmax:2.2f}', xcov)
    
    print("  -> Successfully compiled and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML Configuration file")
    parser.add_argument("--oversamp_sig", type=float, default=100.0, help="Oversampling fraction used for the Signal extraction")
    parser.add_argument("--oversamp_cov", type=float, default=1.0, help="Oversampling fraction used for the Covariance extraction")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Convert seed to match the integer seed format used in MPI runs
    config['seed'] = int(5e11 * config.get('seed', 1.0)) if 'seed' not in config else config['seed']

    # Establish the Mass Bins you want to process
    logMstelarr = 9.5 + 0.1 * np.arange(21)
    
    print(f"--- Starting Collation ---")
    print(f"Signal Oversampling: {args.oversamp_sig}x")
    print(f"Covariance Oversampling: {args.oversamp_cov}x")
    
    # Process each bin
    for logMmin, logMmax in zip(logMstelarr[:-1], logMstelarr[1:]):
        process_mass_bin(config, logMmin, logMmax, args.oversamp_sig, args.oversamp_cov)

    print("\nMeasurements Complete. Proceeding to plots...")
    
    # make_plots('./plots/', logMstelarr)
    # plot_snr('./plots/', logMstelarr[logMstelarr <= 11.6])
