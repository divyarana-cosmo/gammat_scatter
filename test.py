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

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)


    ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'])

    print(ss._get_g(10, 12, 0.3, 0.8, 0.1))
    exit()

    ax = plt.subplot(2,2,1)
    yd    = np.array([])
    sep = np.array([])
    etan    = np.array([])
    
    import pandas as pd
    from glob import glob
    flist = glob('./debug/simed_sources.dat_lmstelmin_11.00_lmstelmax_11.30_no_shape_noise_ideal_case_proc_*')
    for fil in flist:
        df = pd.read_csv(fil, delim_whitespace=1)
        yd    =   np.append(yd, df['etan_obs'])
        sep =  np.append(sep, df['proj_sep'])
        etan = np.append(etan, df['etan'])
    #added hack to cleanup numerical issues
    idx = (etan>0) & (etan<1.0)
    yd    =  yd[idx]
    sep  = sep[idx]
    etan  = etan[idx]
    
    print(sum(np.isnan(yd)))
    
    
    rbins = np.logspace(-2, np.log10(0.8), 6)
    yy = 0.0*rbins[:-1]
    yyerr = 0.0*rbins[:-1]
    
    yyt = 0.0*rbins[:-1]
    for rr in range(5):
        idx = (sep>rbins[rr]) & (sep<rbins[rr+1])
        yyerr[rr] = np.std(yd[idx])/np.sqrt(sum(idx))
        yy[rr] = np.mean(yd[idx])
        yyt[rr] = np.mean(etan[idx])
    
    print(yyerr/yy * 100)
    ax.errorbar((rbins[:-1] + rbins[1:])*0.5, yy, yerr=yyerr, fmt='.', capsize=3)
    ax.plot((rbins[:-1] + rbins[1:])*0.5, yyt, 'xk', lw=0.0)
    plt.ylabel(r'$\gamma_t$')
    plt.xlabel(r'$R[{\rm h^{-1} Mpc}]$')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('./debug/test.png', dpi=300)


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

