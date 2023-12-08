import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

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


