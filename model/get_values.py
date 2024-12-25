import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getdist import plots, MCSamples
import getdist


def make_corner(logMmin, logMmax):
    _ff = './output_mcmc/chainfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax)
    #_ff = './output_mcmc/chainfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax)
    chainfile = pd.read_csv(_ff,header=None, delim_whitespace=1).values[:,:]
    #predfile = pd.read_csv('./output_mcmc/predfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat'%(logMmin, logMmax),header=None, delim_whitespace=1).values[:,:]


    #names = ['logmstel',  'logre', 'logmh', 'c']
    names = ['logmstel',  'logre', 'logmh', 'c', 'beta']
    #labels = [r'\log M_{\rm stel}', r'\log R_{\rm e}', r'\log M_{\rm h}', r'c']
    labels = [r'\log M_{\rm stel}', r'\log R_{\rm e}', r'\log M_{\rm h}', r'c', r'\beta']
    #samples = MCSamples(samples=chainfile[:,:4], names=names, labels=labels)
    samples = MCSamples(samples=chainfile[:,:5], names=names, labels=labels)

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.title_limit_fontsize = 14
    g.triangle_plot([samples], ['logmstel', 'logre', 'logmh', 'c', 'beta'],
        filled=True, 
        title_limit=1, # first title limit (for 1D plots) is 68% by default
        )
    plt.savefig('%s.png'%_ff,dpi=600)
    plt.clf()
    return 0

def make_scatter(logMmin, logMmax):
    _ff = './output_mcmc/chainfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax)
    #_ff = './output_mcmc/chainfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax)
    chainfile = pd.read_csv(_ff,header=None, delim_whitespace=1).values[:,:]
    #predfile = pd.read_csv('./output_mcmc/predfile_nfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax),header=None, delim_whitespace=1).values[:,:]
    predfile = pd.read_csv('./output_mcmc/predfile_gnfw_logMmin_%2.2f_logMmax_%2.2f.dat_full_desixeuclid'%(logMmin, logMmax),header=None, delim_whitespace=1).values[:,:]

    Nparams = len(chainfile[0,:]) - 2
    #label = [r'$\log M_{\rm stel}$', r'$\log R_{\rm e}$', r'$\log M_{\rm h}$', r'$c$']
    label = [r'$\log M_{\rm stel}$', r'$\log R_{\rm e}$', r'$\log M_{\rm h}$', r'$c$',r'$\beta$']
    for pp in range(Nparams):
        plt.subplot(3,3,pp+1)

        plt.scatter(chainfile[:,pp], predfile[:,-1], s=0.1)
        plt.ylabel('$\chi^2$')
        plt.xlabel(label[pp])
    
    plt.tight_layout()
    plt.savefig('%s_scatter.png'%_ff,dpi=300)
    plt.clf()
    return 0



logMstelarr = [9.0,9.25,9.5,9.75,10.0,10.25,10.5]
for ll in range(len(logMstelarr) -1):
    make_scatter(logMstelarr[ll], logMstelarr[ll+1])
    make_corner(logMstelarr[ll], logMstelarr[ll+1])
    print(ll)


