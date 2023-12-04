import numpy as np
import matplotlib.pyplot as plt
import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM

lzred = 0.0
log_mh  = np.arange(10,15)

params = dict(H0 = 100, Om0 = 0.3, Ob0 = 0.0457, Tcmb0 = 2.7255, Neff = 3.046)
sigma8 = 0.82
ns = 0.96
astropy_cosmo = FlatLambdaCDM(**params)
colossus_cosmo = cosmology.fromAstropy(astropy_cosmo, sigma8, ns, cosmo_name='my_cosmo')

conc = concentration.concentration(10**log_mh, '200m', lzred, model = 'diemer19')

plt.plot(log_mh, conc)
plt.yscale('log')
plt.savefig('test.png')


