import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
import pandas as pd


logMmin =   float(sys.argv[1])
logMmax =   float(sys.argv[2])

flist   =   glob('/home/rana/github_0/gammat_scatter/output/debug_z_0.1_0.4/simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_jk_*'%(logMmin, logMmax))
Njacks  =   int(len(flist))


data    = np.array([])
xdata    = np.array([])
rbins   = np.array([])
#23-sumd_dsigma_num 24-sumd_dsigma_den
for cnt in range(Njacks):
    num     = 0.0
    den     = 0.0
    xnum    = 0.0
 
    for nn,fil in enumerate(flist):
        if nn==cnt:
            continue
        dat    =   pd.read_csv(fil, delim_whitespace=1)
        num    += dat['23-sumd_dsigma_num']
        den    += dat['24-sumd_dsigma_den']
        xnum   += (dat['16-dsigmax'] * dat['24-sumd_dsigma_den'])

        rbins               = dat['0-rmin/2+rmax/2'].values
    
    data  = np.append(data,num/den)
    xdata = np.append(xdata,xnum/den)



data = data.reshape(Njacks,len(rbins))
xdata = xdata.reshape(Njacks,len(rbins))


dsigma      =   np.mean(data, axis=0)
xdsigma      =   np.mean(xdata, axis=0)
cov         =   np.zeros((len(rbins), len(rbins)))
xcov         =   np.zeros((len(rbins), len(rbins)))

for ii in range(len(rbins)):
    for jj in range(len(rbins)):
        cov[ii,jj] = np.mean((data[:,ii]-dsigma[ii]) * (data[:,jj]-dsigma[jj]))
        xcov[ii,jj] = np.mean((xdata[:,ii]-xdsigma[ii]) * (xdata[:,jj]-xdsigma[jj]))

cov = cov * (Njacks - 1)
dsigmaerr = np.diag(cov)**0.5
xcov = xcov * (Njacks - 1)
xdsigmaerr = np.diag(xcov)**0.5


np.savetxt('output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), np.transpose([rbins, dsigma, dsigmaerr, xdsigma, xdsigmaerr]))
np.savetxt('output/debug_z_0.1_0.4/cov_dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), cov)


#plotting


plt.subplot(2,2,1)
plt.errorbar(rbins, dsigma, yerr=dsigmaerr, fmt='.', capsize=3)
plt.xscale('log')
plt.yscale('log')

plt.subplot(2,2,2)
plt.errorbar(rbins, xdsigma, yerr=xdsigmaerr, fmt='.', capsize=3)
plt.axhline(0.0, color='black')
plt.xscale('log')

plt.savefig('output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat.png'%(logMmin, logMmax),dpi=300)

