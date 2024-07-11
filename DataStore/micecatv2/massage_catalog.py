import fitsio
import matplotlib.pyplot as plt
import numpy as np


def get_re(logMstel):
    # Mstel in units of Msun
    Mstel   =   10**logMstel
    rp      =   3.8
    Mp      =   10**10.3
    alpha   =   0.09
    beta    =   0.37
    #we use equation 2 from arxiv:1901.05014
    re = rp*(Mstel/Mp)**alpha * (0.5 * (1 + (Mstel/Mp)**6))**((beta-alpha)/6)
    
    idx = Mstel<10**9
    re[idx] = -999
    return re



h=0.7


from astropy.io import fits
from astropy.table import Table, Column

fname = 'selected_samp.fits'
with fits.open(fname, mode='update') as hdul:

    data = Table(hdul[1].data)

    # log(M^*/M-sun h^-1)
    df      = fitsio.read(fname)
    log_re  = np.log10(get_re(data['lmstellar'] - np.log10(h)))

    new_col = Column(name='log_re(kpc)', data=log_re, format='E')
    data.add_column(new_col)

    # Create a new PrimaryHDU (Header Data Unit)
    primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
    
    # Create a new TableHDU from the modified table data
    table_hdu = fits.BinTableHDU(data, header=hdul[1].header)

    # Create a new HDUList object with the primary and table HDUs
    new_hdul = fits.HDUList([primary_hdu, table_hdu])
    
    # Write the new HDUList to a new FITS file
    new_hdul.writeto('massaged_selected_samp.fits', overwrite=True)



