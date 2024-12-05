import pandas as pd
import pygtc
import sys
import numpy as np


dat0  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_no_shape_noise_test_case_w_jacks_pairs_Rpin_0.007_Rpmax_0.025_Chainfile_fixed_conc.dat', header=None, delim_whitespace=1)

dat1  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_w_jacks_pairs_Rpin_0.007_Rpmax_0.025_Chainfile_fixed_conc.dat', header=None, delim_whitespace=1)

dat2  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_with_90_rotation_test_case_w_jacks_pairs_Rpin_0.007_Rpmax_0.025_Chainfile_fixed_conc.dat', header=None, delim_whitespace=1)

#dat0  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_no_shape_noise_test_case_w_jacks_pairs_Rpin_0.007_Chainfile.dat', header=None, delim_whitespace=1)
#
#dat1  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_w_jacks_pairs_Rpin_0.007_Chainfile.dat', header=None, delim_whitespace=1)
#
#dat2  = pd.read_csv('./debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_with_90_rotation_test_case_w_jacks_pairs_Rpin_0.007_Chainfile.dat', header=None, delim_whitespace=1)


dat0 = dat0.values[:,:-1]
dat1 = dat1.values[:,:-1]
dat2 = dat2.values[:,:-1]


import matplotlib.pyplot as plt
#plt.rcParams['figure.dpi'] = 400
#dat = pd.read_csv(filename, header=None, delim_whitespace=1)
#print(np.shape(dat))
#pygtc.plotGTC(chains=[dat0, dat1, dat2], paramNames = [r'$\log({\rm M_{stel}[h^{-1}M_\odot]})$', r'$\log({\rm M_{h}[h^{-1}M_\odot]})$','$f_{c}$'], truths=((12, 14, 1)), plotName = './debug_full_z_0.1_0.5/test_case_corner_free_conc.pdf', figureSize='MNRAS_page', chainLabels=(r'no-shpnos', r'shpnos', r'shpnos-90'), filledPlots=False)

chns    =   [dat0, dat1, dat2]
params  =   [r'$\log({\rm M_{stel}[h^{-1}M_\odot]})$', r'$\log({\rm M_{h}[h^{-1}M_\odot]})$']
truths  =   ((12, 14))  
pltname =   './debug_full_z_0.1_0.5/test_case_corner_Rpmax_0.025.png'
figsize =   'MNRAS_page'
chnlab  =   (r'no-shpnos', r'shpnos', r'shpnos-90')

pygtc.plotGTC(chains=chns, paramNames =params, truths=truths, plotName =pltname, figureSize=figsize, chainLabels=chnlab, filledPlots=False)
#pygtc.plotGTC(chains=[dat0, dat1, dat2], paramNames = [r'$\log({\rm M_{stel}[h^{-1}M_\odot]})$', r'$\log({\rm M_{h}[h^{-1}M_\odot]})$', '$c$'], truths=((12, 14, 5.5)), plotName = 'test_case_corner.png', figureSize='MNRAS_page', chainLabels=(r'$R_{min}$=0.01', r'$R_{min}$=0.05', r'$R_{min}$=0.1'))
#pygtc.plotGTC(chains=[dat0, dat1, dat2], paramNames = [r'$\log({\rm M_{stel}[h^{-1}M_\odot]})$', r'$\log({\rm M_{h}[h^{-1}M_\odot]})$', '$c$'], truths=((12, 14, 5.5)), plotName = 'test_case_corner.png', figureSize='MNRAS_page', chainLabels=(r'$R_{min}$=0.01', r'$R_{min}$=0.05', r'$R_{min}$=0.1'))


