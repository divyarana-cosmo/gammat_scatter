import numpy as np
import matplotlib.pyplot as plt
from simulate_aroundsources import create_sources

Nsrc = 10000
ra  = np.zeros(Nsrc)
dec = np.zeros(Nsrc)

for ii in range(Nsrc):
    dat = create_sources(ramin=0, ramax=np.pi/2, thetamin=0, thetamax=np.pi/2, sigell=0.27, mask=None)
    ra[ii]  = dat[0]
    dec[ii] = dat[1] 


plt.scatter(ra, np.sin(dec*np.pi/180), s=1.0)
plt.savefig('test.png', dpi=300)
