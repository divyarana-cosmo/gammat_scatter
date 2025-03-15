import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from astropy.table import Table


# we are taking the lenses in 4x4 ~ 16 deg^2
ramin = 0.0
ramax =  80.0 * np.pi/180
decmin = 0.0
decmax = 80.0 * np.pi/180

raarr       = np.linspace(ramin, ramax, 401)
sindecarr   = np.linspace(np.sin(decmin), np.sin(decmax), 401)
decarr  = np.arcsin(sindecarr)

area        = (raarr[1:] - raarr[:-1]) * (np.cos(np.pi/2 - decarr[1:]) - np.cos(np.pi/2 - decarr[:-1]))* (180*60/np.pi)**2

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.hist(area)
plt.savefig('test.png', dpi=300)
                                          

racen   = (raarr[:-1] + raarr[1:])*0.5
deccen  = np.arcsin((sindecarr[:-1] + sindecarr[1:])*0.5) 

fil = open('sim_tracts.dat','w')
fil.write("ramin(deg)\tramax(deg)\tdecmin(deg)\tdecmax(deg)\tracen(deg)\tdeccen(deg)\n")
for rr in range(len(raarr[:-1])):
    for dd in range(len(decarr[:-1])):
        fil.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(raarr[rr]*180/np.pi, raarr[rr+1]*180/np.pi, decarr[dd]*180/np.pi, decarr[dd+1]*180/np.pi, racen[rr]*180/np.pi, deccen[dd]*180/np.pi))
    





