import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

"assigns redshifts respecting the distribution"
z0 = 0.9/(2)**0.5
# taken from euclif prep arxiv:1910.09273, eqn 113
f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
zmin = 0.0
zmax = 3
zarr = np.linspace(zmin, zmax, 100)

plt.subplot(3,3,1)
plt.plot(zarr, f(zarr))
plt.xlabel(r'$z_{\rm src}$')
plt.ylabel(r'$n(z_{\rm src})$')
plt.savefig('euclid_nz.pdf', dpi=300)



 
