import sys
sys.path.append('/home/rana/github/gammat_scatter/src/')
from distort import simshear
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from get_data import lens_select
from colossus.cosmology import cosmology
from colossus.halo import concentration


def plt_data(config, outputfilename):
    "generate sanity plots for the test case as mentioned in simulate_aroundlens.py"

    ss = simshear(H0 = 100, Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'])

    colossus_cosmo  = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8 = ss.sigma8, ns = ss.ns, cosmo_name=ss.cosmo_name)

    lensargs    = config["lens"]
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    llogmh = 14 #+ 0.0*llogmh
    lzred = 0.4 #+ 0.0*lzred
    lconc = concentration.concentration(1e14, '200m', lzred, model = 'diemer19')
 
    dat = np.loadtxt(outputfilename)
    rbins   = np.unique(dat[:,0])
    xx      = rbins
    yy      = 0.0*rbins
    yyerr    = 0.0*rbins
    yyx      = 0.0*rbins
    yyerrx   = 0.0*rbins
 
    for i in range(len(rbins)):
        idx = dat[:,0]==rbins[i]
        yy[i]       = np.mean(dat[idx,1])
        yyerr[i]    = (sum(idx) - 1)**0.5 * np.std(dat[idx,1])

        yyx[i]       = np.mean(dat[idx,3])
        yyerrx[i]    = (sum(idx) - 1)**0.5 * np.std(dat[idx,3])

        
    #plt.figure(figsize=(8,8))
    
    plt.subplot(2,2,1)
    plt.errorbar(xx, yy, yerr=yyerr, fmt='.', capsize=3)
    #plt.plot(dat[:,0], dat[:,1], '.', lw=0.0)

    rbins   = np.unique(dat[:,0])
    szred   = 0.8
    print(llogmh,lconc)
    gamma_s, gamma_dm, kappa_s, kappa_dm    = ss._get_g(np.log10(np.mean(10**llogmstel)),llogmh, lconc, lzred, szred, rbins)

    #plt.plot(rbins, kappa_dm, '--', color='C1')
    plt.plot(rbins, gamma_s, '--', color='C1', label='baryon')
    plt.plot(rbins, gamma_dm/(1-kappa_dm), '--', color='C2', label='dark matter')
    plt.plot(rbins, (gamma_s + gamma_dm)/(1-kappa_dm), '-k', label='total')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-2,1)
    plt.ylabel(r'$g_{\rm t}$')
    plt.xticks([])
    #plt.xlabel(r'${\rm R [h^{-1}Mpc]}$')
    plt.legend(fontsize='small')

    plt.subplot(2,2,3)
    res = (yy - (gamma_s + gamma_dm)/(1-kappa_dm))/yyerr
    plt.plot(xx, res)
    plt.ylim(-2.5,2.5)
    plt.axhline(0.0, ls='--', color='grey')
    plt.xscale('log')
    plt.ylabel(r'$ (g_{\rm t, meas} - g_{\rm t, mod})/\sigma$')
    plt.xlabel(r'${\rm R [h^{-1}Mpc]}$')
 
    plt.subplot(2,2,2)
    plt.errorbar(xx,xx*yyx, yerr=xx*yyerrx, fmt='.', capsize=3)
    plt.axhline(0.0, ls='--', color='grey')
    plt.ylim(-3e-3,3e-3)
    plt.ylabel(r'$R g_{\rm \times}$')
    plt.xlabel(r'${\rm R [h^{-1}Mpc]}$')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('%s.png'%outputfilename, dpi=300)
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

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

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
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'

    if args.test_case:
        outputfilename = outputfilename + '_test_case'
 
    np.random.seed(args.seed)
    outputfilename = outputfilename + '_w_jacks'

    plt_data(config, outputfilename)           


