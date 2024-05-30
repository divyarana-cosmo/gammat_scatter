import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./src/')

from halopy import halo
from stellarpy import stellar


dat = np.loadtxt('dsigma.dat')

logmh =  14; conc = 5.0
logmstel = 12; log_re = 0.02

hp = halo(logmh, conc)
stel = stellar(logmstel, log_re=log_re)

plt.subplot(2,2,1)
njacks = int(len(dat[:,0])/len(np.unique(dat[:,0])))
data = dat[:,1].reshape(njacks,-1)
#r90_data = dat[:,5].reshape(njacks,-1)


xx = np.unique(dat[:,0])
yy = np.mean(data, axis=0)
yyerr = (njacks - 1)**0.5 * np.std(data,axis=0)

#r90_yy = np.mean(r90_data, axis=0)
#r90_yyerr = (njacks - 1)**0.5 * np.std(r90_data,axis=0)




plt.errorbar(xx, yy, yerr=yyerr, fmt='.',capsize=3)
#plt.errorbar(xx, r90_yy, yerr=r90_yyerr, fmt='.',capsize=3)
plt.plot(xx, hp.esd_nfw(xx)/1e12 + stel.esd_deVaucouleurs(xx)/1e12)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{\rm p}$')
plt.ylabel(r'$\Delta \Sigma (R_{\rm p})$')
plt.savefig('test.png', dpi=300)


