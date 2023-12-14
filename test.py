import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from astropy.cosmology import FlatLambdaCDM
from halopy import halo
from stellarpy import stellar

hp    = halo(12, 3.5, omg_m=0.27)
stel  = stellar(10)

#projected separation on the lense plane
proj_sep = np.logspace(np.log10(0.01), 0, 10) # in h-1 Mpc

#considering only tangential shear and adding both contributions
plt.plot(proj_sep, hp.esd_nfw(proj_sep))
plt.plot(proj_sep, stel.esd_pointmass(proj_sep))
plt.xscale('log')
plt.yscale('log')
plt.savefig('test.png')
exit()


ax = plt.subplot(2,2,1)
for ii in [12.0, 13.0, 14.0]:
    data = pd.read_csv('./debug/simed_sources_logmh_%s.dat'%ii, delim_whitespace=1)
    yyerr = np.std(data['etan_obs'])/np.sqrt(len(data['etan_obs']))
    yy = np.mean(data['etan_obs'])

    print(yyerr/yy * 100)
    ax.errorbar(ii, yy, yerr=yyerr, fmt='.', capsize=3)
    ax.plot(ii, np.mean(data['etan']), '.k', lw=0.0)

plt.ylabel(r'$\gamma_t$')
plt.xlabel(r'$\log ({\rm M_{\rm 200m}/[h^{-1}M_{\odot}]})$')

plt.savefig('./debug/test.png', dpi=300)


