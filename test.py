import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob



def getzred(zred):
    n0=1.8048
    a=0.417
    b=4.8685
    c=0.7841
    return n0*(zred**a + zred**(a*b))/(zred**b + c)




zmin = 0.0
zmax = 2.5
zarr = np.linspace(zmin, zmax, 20)

xx  = 0.0 * zarr
from scipy.integrate import quad

for ii in range(len(xx)):
    xx[ii] = quad(getzred, zmin, zarr[ii])[0]/quad(getzred, zmin, zmax)[0]

from scipy.interpolate import interp1d
proj = interp1d(xx,zarr)


np.random.seed(123)
xarr = np.random.uniform(size=100000)
yarr = proj(xarr)

plt.hist(yarr, histtype='step')

plt.savefig('test.png', dpi=300)

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



#ax = plt.subplot(2,2,1)
#for ii in [12.0, 13.0, 14.0]:
#    data = pd.read_csv('./debug/simed_sources_logmh_%s.dat'%ii, delim_whitespace=1)
#    yyerr = np.std(data['etan_obs'])/np.sqrt(len(data['etan_obs']))
#    yy = np.mean(data['etan_obs'])
#
#    print(yyerr/yy * 100)
#    ax.errorbar(ii, yy, yerr=yyerr, fmt='.', capsize=3)
#    ax.plot(ii, np.mean(data['etan']), '.k', lw=0.0)
#
#plt.ylabel(r'$\gamma_t$')
#plt.xlabel(r'$\log ({\rm M_{\rm 200m}/[h^{-1}M_{\odot}]})$')
#
#plt.savefig('./debug/test.png', dpi=300)
#
#
#
#

