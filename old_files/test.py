import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
import yaml
from distort import simshear
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    #parser.add_argument("--logmh", help="dark matter halo mass", type=float, default=12.0)
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--no_shape_noise", help="scatter halo mass", type=bool, default=False)
    parser.add_argument("--ideal_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--rot90", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum", type=float, default=11.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum", type=float, default=13.0)
    parser.add_argument("--Njacks", help="number of jackknife samples", type=int, default=20)
    parser.add_argument("--Rmin", help="minimum projected separation", type=float, default=0.01)
    parser.add_argument("--Rmax", help="maximum projected separation", type=float, default=0.8)
    parser.add_argument("--Rbins", help="number of radial bins", type=int, default=6)
    parser.add_argument("--no_shear", help="scatter halo mass", type=bool, default=False)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)


    ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'])

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
 

    #picking up the lens data
    lensargs = config['lens']
    outputfilename = outputfilename + '_proc_*'
    etan_obs    = np.array([])
    ex_obs    = np.array([])
    sep         = np.array([])
    lid         = np.array([])
    
    # variables for the model predictions at the avg parameters
    logmstel    =   np.array([])
    logmh       =   np.array([])
    lzred       =   np.array([])
    szred       =   0.0

    flist = glob(outputfilename)
    #collecting the data
    for fil in flist:
        df = pd.read_csv(fil, delim_whitespace=1)
        df = df[(df['proj_sep']>args.Rmin) & ( df['proj_sep']<args.Rmax)]
        etan_obs    =   np.append(etan_obs, df['etan_obs'])
        ex_obs    =   np.append(ex_obs, df['ex_obs'])
        sep         =   np.append(sep, df['proj_sep'])
        lid         =   np.append(lid, df['lid'])
        #etan        =   np.append(etan, df['etan'])
        logmstel    =   np.append(logmstel,  df['llogmstel'])
        logmh       =   np.append(logmh,  df['llogmh'])
        lzred       =   np.append(lzred,  df['lzred'])
        szred       +=   sum(df['szred'])
        print(fil)


    np.random.seed(123)
    ulid, indx = np.unique(lid, return_index=True)
    logmstel =  logmstel[indx]
    logmh   =   logmh[indx]
    lzred   =   np.mean(lzred[indx])
    szred   /=  (len(etan_obs))

    # assigning the jackknife indices
    ulid, indices = np.unique(lid, return_inverse=True)
    jkreg = np.random.randint(args.Njacks, size=len(ulid))
    xjkreg = jkreg[indices]

  
    rbins = np.logspace(np.log10(args.Rmin), np.log10(args.Rmax), args.Rbins + 1)
    yy = np.zeros(( args.Rbins))
    yyx = np.zeros((args.Rbins))
    sigyy = np.zeros((args.Rbins))
    sigxyy = np.zeros((args.Rbins))
    #yyerr = np.zeros((len(rbins[:-1]), args.Njacks))
    
    for rr in range(args.Rbins):
        idx = (sep>rbins[rr]) & (sep<rbins[rr+1])
        yy [rr]  = np.mean(etan_obs[idx])
        yyx[rr] = np.mean(ex_obs[idx])

        idx = (sep>rbins[0]) & (sep<(rbins[rr] + rbins[rr+1])*0.5)
        #idx = (xjkreg !=ii) & (sep>rbins[0]) & (sep<rbins[rr+1])
        sigyy  [rr]  = np.std(etan_obs[idx])
        sigxyy [rr] = np.std(ex_obs[idx])

    rbins = (rbins[1:] + rbins[:-1])*0.5

    plt.subplot(2,2,2)
    plt.plot(rbins, sigyy,  label=r'$\gamma_{\rm t}$')
    plt.plot(rbins, sigxyy, label=r'$\gamma_{\rm \times}$')
 
    plt.axhline(0.27, ls='--', color='grey')
    plt.ylabel(r'$\sigma (< R)$')
    plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    plt.ylim(0.24, 0.34)
    
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.savefig('test.png', dpi=300)

