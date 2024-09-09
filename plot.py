import sys
import numpy as np
import matplotlib.pyplot as plt

def plt_data(outputfilename):
    "generate sanity plots for the test case as mentioned in simulate_aroundsource.py"
    dat     = np.loadtxt(outputfilename)
    rbins   = np.unique(dat[:,0])
    njacks  = int(len(dat[:,0])/len(rbins))
    
    cov = np.zeros((len(rbins), len(rbins)))
    xcov = np.zeros((len(rbins), len(rbins)))

    dsigarr = np.array([])
    dsigxarr = np.array([])

    for i in range(len(rbins)):
        idx = dat[:,0]==rbins[i]
        dsigarr     = np.append(dsigarr, dat[idx,1])
        dsigxarr    = np.append(dsigxarr, dat[idx,3])

    dsigarr     = dsigarr.reshape(-1, njacks) 
    dsigxarr    = dsigxarr.reshape(-1, njacks)


    mean_dsig  = np.mean(dsigarr, axis=1)
    mean_dsigx = np.mean(dsigxarr, axis=1)

    
    for i in range(len(rbins)):
        for j in range(len(rbins)):
            cov[i,j]    = np.mean((dsigarr[i,:] - mean_dsig[i]) *(dsigarr[j,:] - mean_dsig[j]))
            xcov[i,j]   = np.mean((dsigxarr[i,:] - mean_dsigx[i]) *(dsigxarr[j,:] - mean_dsigx[j]))

    
    cov *=(njacks - 1)
    xcov *=(njacks - 1)

    np.savetxt('%s_cov'%outputfilename, cov)
    np.savetxt('%s_xcov'%outputfilename, xcov)

    xx       = rbins
    yy       = mean_dsig
    yyerr    = np.diag(cov)**0.5
    yyx      = mean_dsigx
    yyerrx   = np.diag(xcov)**0.5

    np.savetxt('%s_meas'%outputfilename, np.transpose([xx,yy,yyerr]), header= 'rad \t dsigma \t dsigmaerr')
    np.savetxt('%s_xmeas'%outputfilename, np.transpose([xx,yyx,yyerrx]), header= 'rad \t dsigmax \t dsigmaxerr')


    plt.subplot(2,2,1)
    plt.errorbar(xx, yy, yerr=yyerr, fmt='.', capsize=3)
    plt.xlabel(r'$R[{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma[{\rm h M_{\odot} pc^{-2}}]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.1,200)

    plt.subplot(2,2,2)
    plt.errorbar(xx, yyx, yerr=yyerrx, fmt='.', capsize=3)
    plt.xlabel(r'$R[{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$\Delta \Sigma_{\times}[{\rm h M_{\odot} pc^{-2}}]$')
    plt.xscale('log')
    plt.axhline(0.0, ls='--', color='grey')

    plt.ylim(-50,50)

    plt.tight_layout()
    plt.savefig('%s_signal.png'%outputfilename, dpi=300)

    plt.clf()

    plt.subplot(2,2,1)
    corr = 0.0*cov
    for i in range(len(xx)):
        for j in range(len(yy)):
            corr[i,j] = cov[i,j]/(cov[i,i]*cov[j,j])**0.5
    plt.imshow(corr, origin='lower', aspect= 'equal', vmin=-1, vmax =1)
    plt.colorbar()

    
    plt.savefig('%s_cov.png'%outputfilename, dpi=600)

    return 0

if __name__ == "__main__":
    from glob import glob
    flist = glob('./small_ggl/dsigma.dat*shape_noise')

    for fil in flist:
        print(fil)
        plt_data(outputfilename = fil)       
