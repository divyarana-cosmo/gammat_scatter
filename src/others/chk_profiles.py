from halopy import halo
from stellarpy import stellar
import numpy as np
import matplotlib.pyplot as plt

rbins = np.logspace(-3,0,10)

hp = halo(14,3)
ss = stellar(12)


plt.plot(rbins, hp.sigma_nfw(rbins))
plt.plot(rbins, ss.sigma_deVaucouleurs(rbins))

plt.xscale('log')
plt.yscale('log')

plt.savefig('test.png')
