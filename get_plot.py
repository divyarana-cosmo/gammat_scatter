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
    yy = np.zeros((args.Njacks , args.Rbins))
    yyx = np.zeros((args.Njacks , args.Rbins))
    sigyy = np.zeros((args.Njacks , args.Rbins))
    sigxyy = np.zeros((args.Njacks , args.Rbins))
    #yyerr = np.zeros((len(rbins[:-1]), args.Njacks))
    
    for ii in range(args.Njacks):
        for rr in range(args.Rbins):
            idx = (xjkreg !=ii) & (sep>rbins[rr]) & (sep<rbins[rr+1])
            yy[ii] [rr]  = np.mean(etan_obs[idx])
            yyx[ii][rr] = np.mean(ex_obs[idx])

            idx = (xjkreg !=ii) & (sep>rbins[0]) & (sep<(rbins[rr] + rbins[rr+1])*0.5)
            #idx = (xjkreg !=ii) & (sep>rbins[0]) & (sep<rbins[rr+1])
            sigyy[ii] [rr]  = np.std(etan_obs[idx])
            sigxyy[ii] [rr] = np.std(ex_obs[idx])

    yyerr = np.sqrt(args.Njacks -1) * np.std(yy, axis=0)   #correcting for the jackknife method check norberg et al 2009
    yyxerr = np.sqrt(args.Njacks -1) * np.std(yyx, axis=0)   #correcting for the jackknife method check norberg et al 2009

    sigyyerr = np.sqrt(args.Njacks -1) * np.std(sigyy, axis=0)
    sigxyyerr = np.sqrt(args.Njacks -1) * np.std(sigxyy, axis=0)

    yy = np.mean(yy, axis=0)
    yyx = np.mean(yyx, axis=0)
    sigyy = np.mean(sigyy, axis=0)
    sigxyy = np.mean(sigxyy, axis=0)
    print(yyerr/yy)
    rbins = np.array((rbins[:-1] + rbins[1:])*0.5)
    print(rbins)
    #tgamma_s, tgamma_d, tkappa_s, tkappa_d = ss._get_g(logmstel , logmh, lzred, szred, rbins)
    

    #plt.subplot(2,2,1)
    #plt.errorbar(rbins, yy, yerr=yyerr, fmt='.', capsize=3)
    #plt.plot(rbins, tgamma_s, '--', label=r'Stellar, $\log[M_{\rm stel}/h^{-1}M_\odot] = %2.2f$'%logmstel)
    #plt.plot(rbins, tgamma_d, '--', label=r'Dark matter, $\log[M_{\rm h}/h^{-1}M_\odot] = %2.2f $'%(logmh))
    #plt.plot(rbins, tgamma_s + tgamma_d, '-k', label='total')
 
    #plt.ylabel(r'$\gamma_t$')
    #plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    #plt.ylim(1e-3, 1)
    #
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.legend()

    #plt.subplot(2,2,3)
    #plt.plot(rbins, (yy - tgamma_s + tgamma_d)/yyerr, '-k', label='total')
 
    #plt.ylabel(r'$\frac{\gamma^{\rm meas}_t - \gamma^{\rm mod}_t}{\sigma_{\gamma_{t}}}$')
    #plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    #plt.xscale('log')
    #plt.legend()


    plt.subplot(2,2,2)
    plt.errorbar(rbins, sigyy, yerr=sigyyerr, fmt='.', capsize=3, label=r'$\gamma_{\rm t}$')
    plt.errorbar(rbins, sigxyy, yerr=sigxyyerr, fmt='.', capsize=3, label=r'$\gamma_{\rm \times}$')
 
    plt.axhline(0.27, ls='--', color='grey')
    plt.ylabel(r'$\sigma (< R)$')
    plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    plt.ylim(0.24, 0.34)
    
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend()

    #plt.subplot(2,2,4)
    #plt.errorbar(rbins,yyx, yerr=yyxerr, fmt='.', capsize=3)
    #plt.axhline(0.0, ls='--', color='grey')
    #plt.ylabel(r'$\gamma_\times$')
    #plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    #plt.xscale('log')






    plt.tight_layout()
   
    plt.savefig(outputfilename.split('_proc_*')[0] + '.png', dpi=300)


#def getzred(zred):
#    n0=1.8048
#    a=0.417
#    b=4.8685
#    c=0.7841
#    return n0*(zred**a + zred**(a*b))/(zred**b + c)
#
#
#
#
#zmin = 0.0
#zmax = 2.5
#zarr = np.linspace(zmin, zmax, 20)
#
#xx  = 0.0 * zarr
#from scipy.integrate import quad
#
#for ii in range(len(xx)):
#    xx[ii] = quad(getzred, zmin, zarr[ii])[0]/quad(getzred, zmin, zmax)[0]
#
#from scipy.interpolate import interp1d
#proj = interp1d(xx,zarr)
#
#
#np.random.seed(123)
#xarr = np.random.uniform(size=100000)
#yarr = proj(xarr)
#
#plt.hist(yarr, histtype='step')
#
#plt.savefig('test.png', dpi=300)

exit()


#m = getzred(zarr)
#norm = np.sum(m)
#m = m/norm
#x0 = np.zeros(zarr)
#for i1 in range(1,len(zarr)):
#        df=np.sum(m[0:i1])
#        # x= (x1-x0)*int + x0
#        x0[i1]=((pdf)*(1.) + 0) # a is a array consists of (value, error)
##   print x0[0]
#
#   f=interp1d(x0,n,kind='linear')
#   #print max(x0)
#   #sample of uniform numbers
#   n0=1
#   xnew=np.random.random_sample(n0)
#    xnew=1.0
#    xnew=1.0
#   try:
#       ynew=f(xnew)
#   except ValueError:
#       print "Handling exeception"
#       if xnew<=np.min(x0):
#           ynew=0
#       if xnew>=np.max(x0):
#           ynew=NPIX-1
#   ynew = int(ynew)
#   new_pos = hp.pix2ang(NSIDE,ynew)
#
#
#
#
#zredarr = np.linspace(0,4, 100)
#plt.axhline(0.0)
#plt.plot(zredarr, getzred(zredarr))
#plt.savefig('test.png')
#exit()
#




#
#
#

