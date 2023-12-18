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
    #picking up the lens data
    lensargs = config['lens']
    outputfilename = outputfilename + '_proc_*'
    etan_obs    = np.array([])
    sep         = np.array([])
    etan        = np.array([])
    lid         = np.array([])
    
    # variables for the model predictions at the avg parameters
    logmstel    =   0.0# np.array([])
    logmh       =   0.0# np.array([])
    conc        =   0.0# np.array([])
    lzred       =   0.0# np.array([])
    szred       =   0.0# np.array([])

    flist = glob(outputfilename)
    #collecting the data
    for fil in flist:
        df = pd.read_csv(fil, delim_whitespace=1)
        df = df[(df['proj_sep']>args.Rmin) & ( df['proj_sep']<args.Rmax)]
        etan_obs    =   np.append(etan_obs, df['etan_obs'])
        sep         =   np.append(sep, df['proj_sep'])
        lid         =   np.append(lid, df['proj_sep'])
        #etan        =   np.append(etan, df['etan'])
        logmstel    +=   sum(df['llogmstel'])
        logmh       +=   sum(df['llogmh'])
        conc        +=   sum(df['lconc'])
        lzred       +=   sum(df['lzred'])
        szred       +=   sum(df['szred'])
        print(fil)
    
    #averaging here
    logmstel    /=   len(sep)    
    logmh       /=   len(sep)
    conc        /=   len(sep)
    lzred       /=   len(sep)
    szred       /=   len(sep)
   
    # assigning the jackknife indices
    np.random.seed(123)
    ulid, indices = np.unique(lid, return_inverse=True)
    jkreg = np.random.randint(args.Njacks, size=len(ulid))
    xjkreg = jkred[indices]

    rbins = np.logspace(np.log10(args.Rmin), np.log10(args.Rmax), args.Rbins + 1)
    yy = np.zeros((args.Njacks, args.Rbins))
    #yyerr = np.zeros((len(rbins[:-1]), args.Njacks))
    
    for ii in range(args.Njacks):
        for rr in range(args.Rbins):
            idx = (xjkreg !=ii) & (sep>rbins[rr]) & (sep<rbins[rr+1])
            #yy[ii, rr] = np.mean(etan[idx])
            yy[ii, rr] = np.mean(etan_obs[idx])

    yyerr = np.sqrt(args.Njacks -1) * np.std(yy, axis=0)   #correcting for the jackknife method check norberg et al 2009
    yy = np.mean(yy, axis=0)
    rbins = np.array((rbins[:-1] + rbins[1:])*0.5)
    tgamma_s, tgamma_d, tkappa_s, tkappa_d = ss._get_g(logmstel , logmh, lzred, szred, rbins)
 
    ax = plt.subplot(2,2,1)
    ax.errorbar(rbins, yy, yerr=yyerr, fmt='.', capsize=3)
    ax.plot(rbins, tgamma_s/(1-tkappa_s), '--', label=r'stellar, $\log M_{\rm stel} = %2.2f$'%logmstel)
    ax.plot(rbins, tgamma_d/(1-tkappa_d), '--', label='dark matter, $\log M_{\rm stel} = %2.2f'%(logmh))
    ax.plot(rbins, (tgamma_s + tgamma_d)/(1- (tkappa_s + tkappa_d)), '-k', label='total')
 
    plt.ylabel(r'$\gamma_t$')
    plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    plt.ylim(1e-3, 1)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
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

