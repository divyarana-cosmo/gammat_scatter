import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

filename = sys.argv[1]
data = pd.read_csv(filename, delim_whitespace=1)

rbins = np.logspace(np.log10(0.02), 0, 9)
sigt = 0.0*rbins[:-1]
sigx = 0.0*rbins[:-1]
gammat = 0.0*rbins[:-1]
gammax = 0.0*rbins[:-1]
scattert = 0.0*rbins[:-1]
scatterx = 0.0*rbins[:-1]
true_gammat = 0.0*rbins[:-1]

for ii in range(len(rbins[:-1])):
    idx = (data['proj_sep'] > rbins[ii]) & (data['proj_sep'] < rbins[ii+1])
    gammat[ii] = np.mean(data['etan_obs'][idx])
    gammax[ii] = np.mean(data['ex_obs'][idx])
    sigt[ii] = np.std(data['etan_obs'][idx])/np.sqrt(sum(idx))
    sigx[ii] = np.std(data['ex_obs'][idx])/np.sqrt(sum(idx))
    true_gammat[ii] = np.mean(data['etan'][idx])

    idx = (data['proj_sep'] > rbins[0]) & (data['proj_sep'] < rbins[ii+1])
    scattert[ii] = np.std(data['etan_obs'][idx])
    scatterx[ii] = np.std(data['ex_obs'][idx])


xx = (rbins[:-1] + rbins[1:])*0.5
ax = plt.subplot(2,2,1)
ax.errorbar(xx, gammat, yerr=sigt, fmt='.', capsize=3, label=r'$\gamma_t$')
ax.errorbar(xx, gammax, yerr=sigx, fmt='.', capsize=3, label=r'$\gamma_\times$')
ax.plot(xx, true_gammat,'xk')
ax.axhline(0.0, ls='--', color='grey')
ax.set_ylim(-0.1, 0.2)
ax.set_xlabel(r'$R[h^{-1}Mpc]$')
ax.set_ylabel(r'$\gamma$')
ax.legend()
ax.set_xscale('log')
plt.savefig('%s.png'%filename, dpi=300)
