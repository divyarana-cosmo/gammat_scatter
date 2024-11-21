import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
import pandas as pd


logMmin =   float(sys.argv[1])
logMmax =   float(sys.argv[2])

flist   =   glob('output/debug_z_0.1_0.4/simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_test_case_w_jacks_jk_*'%(logMmin, logMmax))
Njacks  =   int(len(flist))

data    =   np.array([])
rbins   =   np.array([])
for fil in flist:
    df      =   pd.read_csv(fil, delim_whitespace=1)
    data    =   np.append(data,df['13-dsigma'].values)
    rbins    =   df['0-rmin/2+rmax/2'].values

data = data.reshape(Njacks,len(rbins))

dsigma  = np.mean(data, axis=0)
cov     =   np.zeros((len(rbins), len(rbins)))

for ii in range(len(rbins)):
    for jj in range(len(rbins)):
        cov[ii,jj] = np.mean(((data[ii,:]-dsigma[ii]) * (data[jj,:]-dsigma[jj])))

cov = cov * (Njacks - 1)

np.savetxt('output/debug_z_0.1_0.4/dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), np.transpose([rbins, dsigma]))
np.savetxt('output/debug_z_0.1_0.4/cov_dsigma_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax), cov)

