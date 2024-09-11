import numpy as np
import matplotlib.pyplot as plt
from glob import glob


flist = glob('dsigma.dat_lmstelmin_9.00_lmstelmax_9.25_with_shape_noisejk_*')

xx     = np.loadtxt(flist[0])[:,0]
nbins  = int(len(xx))
Njacks = int(len(flist))
dsig       = np.zeros((Njacks, nbins))
xdsig      = np.zeros((Njacks, nbins))

#collecting the data
for jk in range(Njacks):
    num     = 0
    numx    = 0
    den     = 0
    for nn in range(Njacks):
        if nn==jk:
            continue
        data = np.loadtxt('dsigma.dat_lmstelmin_9.00_lmstelmax_9.25_with_shape_noisejk_%d'%nn)

        num  += data[:,13]
        numx += data[:,14]
        den  += data[:,12]
    
    dsig[jk,:] = num/den
    xdsig[jk,:] = numx/den

plt.subplot(2,2,1)

plt.errorbar(xx, np.mean(dsig,axis=0), yerr=np.sqrt(Njacks - 1)*np.std(dsig, axis=0)  , fmt='.', capsize=3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{\rm p [h^{-1} Mpc]}$')
plt.ylabel(r'$\Delta \Sigma[{\rm h\,M_\odot pc^{-2}}]$')

plt.subplot(2,2,2)
plt.errorbar(xx, np.mean(xdsig,axis=0), yerr=np.sqrt(Njacks - 1)*np.std(xdsig, axis=0), fmt='.', capsize=3)
plt.xscale('log')

plt.xlabel(r'$R_{\rm p [h^{-1} Mpc]}$')
plt.ylabel(r'$\Delta \Sigma_{\times}[{\rm h\,M_\odot pc^{-2}}]$')

plt.tight_layout()
plt.savefig('esd_test.png', dpi=300)

# get the covariances
cov  = np.zeros((int(len(xx)), int(len(xx))))
xcov = np.zeros((int(len(xx)), int(len(xx))))

for ii in range(int(len(xx))):
    for jj in range(int(len(xx))):
        cov[ii,jj]      = np.mean((dsig[:,ii] - np.mean(dsig[:,ii]))*(dsig[:,jj] - np.mean(dsig[:,jj])))
        xcov[ii, jj]    =  np.mean((xdsig[:,ii] - np.mean(xdsig[:,ii]))*(xdsig[:,jj] - np.mean(xdsig[:,jj])))
 

cov     =   (Njacks - 1) * cov
xcov    =   (Njacks - 1) * xcov
corr    =   0.0*cov
xcorr   =   0.0*xcov
for ii in range(int(len(xx))):
    for jj in range(int(len(xx))):
        corr[ii,jj]     = cov[ii,jj]/(cov[ii,ii]*cov[jj,jj])**0.5
        xcorr[ii,jj]    = xcov[ii,jj]/(xcov[ii,ii]*xcov[jj,jj])**0.5

plt.clf()
#def tlab(i,j):
#    a = r'$\Delta\Sigma_{%d,%d}$'%(i,j)
#    return a
#
#for i in range(1,8):
#    if i==1:
#        lticks =[tlab(i,1),tlab(i,6)]
#    else:
#        lticks = np.concatenate((lticks,[tlab(i,1),tlab(i,6)]))
#
#cb = plt.colorbar(fraction=0.046,pad=0.04)
##cb = plt.colorbar(label=r"$r_{\rm ij}$",fraction=0.046,pad=0.04)
#cb.set_label(label=r"$r_{\rm ij}$",size=15)
##cb.set_label(label=r"$r^{\rm JK}_{\rm ij}$",size=15)
#cb.ax.tick_params(direction='in',length=2.5,labelsize=10)
#plt.xticks(np.arange(0,70,5),lticks,fontsize=8,rotation=90)
#plt.yticks(np.arange(0,70,5),lticks,fontsize=8)


plt.subplot(2,2,1)
plt.imshow(corr, vmin=-1, vmax=1, aspect='equal', origin='lower')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(2,2,2)
plt.imshow(xcorr, vmin=-1, vmax=1, aspect='equal', origin='lower')
plt.colorbar(fraction=0.046, pad=0.04)

plt.savefig('corr_test.png', dpi=300)
