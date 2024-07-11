import fitsio
import matplotlib.pyplot as plt
import numpy as np

fname = '17423.fits'
df = fitsio.read(fname)
idx = (df['flag_central']==0) & (df['ra_gal']<10) & (df['dec_gal']<10)
df = df[idx]
plt.subplot(3,3,1)
plt.scatter(df['z_cgal_v'], df['lmstellar'], s=0.1, lw=0.0)

idx = (df['z_cgal_v']>0.1) & (df['z_cgal_v']<0.2) 
idx = idx & (df['lmstellar']>8.5) & (df['lmstellar']<10.5)
plt.plot(np.linspace(0.1,0.2,10), 9*np.ones(10) ,       'r-', lw=0.8)
plt.plot(np.linspace(0.1,0.2,10), 10.5*np.ones(10) ,    'r-', lw=0.8)
plt.plot(0.1*np.ones(10), np.linspace(9, 10.5,10) ,     'r-', lw=0.8)
plt.plot(0.2*np.ones(10), np.linspace(9, 10.5,10) ,     'r-', lw=0.8)

plt.xlim(0,0.7)
plt.xlabel('$z$')
plt.ylabel(r'$\log(M_{\rm stel}/[h^{-1}M_\odot])$')

#plt.legend()

df = df[idx]

fitsio.write('selected_samp.fits', df)

#plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300)

plt.clf()

#plt.subplot(3,3,2)
#xx      = np.linspace(11,14,11) #logmh bins
#yy      = 0.0*xx[:-1]
#yyerr   = 0.0*xx[:-1]
#
#for ii in range(len(xx) -1):
#    idx = (df['lmhalo']>xx[ii]) & (df['lmhalo']<xx[ii+1])
#    yy[ii]      = np.mean(df['lmstellar'][idx])
#    yyerr[ii]   = np.std(df['lmstellar'][idx])/(sum(idx))**0.5
#
#plt.errorbar(xx[:-1]*0.5 + xx[1:]*0.5, yy, yerr=yyerr, fmt='.', capsize=3)
#
#plt.ylabel(r'$\langle \log(M_{\rm stel}/[h^{-1} M_\odot]) \rangle$')
#plt.xlabel(r'$\log(M_{\rm h}/ [h^{-1} M_\odot])$')
#
#plt.savefig('scatter_plot_1.png', dpi=300)


#df = fitsio.FITS(fname)
#df = df[1][df[1].where('flag_central == 0  && lmstellar > 10.0')]
#fitsio.write('%s_mstelcut_10_centrals'%fname, df)

#idx = (np.random.uniform(size=len(df['lmhalo']))<0.01)
#llogmstel   = df['lmstellar'][idx]
#llogmh      = df['lmhalo'][idx]
#lzred       = df['z_cgal_v'][idx]
#
#idx = (lzred>0.1) & (lzred<0.5)
#print(np.median(lzred[idx]), lzred.min())
#

