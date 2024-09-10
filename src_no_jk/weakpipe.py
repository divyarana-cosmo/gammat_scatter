# This code is adapted from iagrg's notes
import sys
from distort import simshear
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
#from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from welford import Welford
import time

def get_xyz(ra, dec):
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z



class weakpipe():
    def __init__(self,H0 = 100, Om0 = 0.25, Ob0 = 0.044, Tcmb0 = 2.7255, Neff = 3.046, sigma8 = 0.8, ns = 0.95, Rmin=0.1, Rmax=1.0, Nbins=10, outputfilename='dsigma.dat'):

        self.H0              = H0 
        self.Om0             = Om0
        self.Ob0             = Ob0 
        self.Tcmb0           = Tcmb0 
        self.Neff            = Neff 
        self.sigma8          = sigma8 
        self.ns              = ns 
        self.rmin            = Rmin       
        self.rmax            = Rmax           
        self.nbins           = Nbins          
        self.outputfilename  = outputfilename 

        
        self.initialize_arrays()
        print('arrays are initialized')
        #setting up cosmology and class instance
        self.ss = simshear(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff, sigma8 = sigma8, ns = ns)
        self.colossus_cosmo  = cosmology.fromAstropy(self.ss.Astropy_cosmo, sigma8 = self.ss.sigma8, ns = self.ss.ns, cosmo_name=self.ss.cosmo_name)


        # set the projected radial binning
        self.rbins   = np.logspace(np.log10(Rmin), np.log10(Rmax), Nbins + 1)
        self.rdiff   = np.log10(self.rbins[1]*1.0/self.rbins[0])



    def initialize_arrays(self):
        # initializing arrays for signal compuations
        self.sumd_dsigmat_num              = np.zeros(self.nbins)
        self.sumd_dsigmatsq_num            = np.zeros(self.nbins)
        self.sumd_dsigmax_num              = np.zeros(self.nbins)
        self.sumd_dsigmaxsq_num            = np.zeros(self.nbins)

        self.r90_sumd_dsigmat_num          = np.zeros(self.nbins)
        self.r90_sumd_dsigmatsq_num        = np.zeros(self.nbins)
        self.r90_sumd_dsigmax_num          = np.zeros(self.nbins)
        self.r90_sumd_dsigmaxsq_num        = np.zeros(self.nbins)


        self.sumdwls                       = np.zeros(self.nbins)
        self.r90_sumdwls                   = np.zeros(self.nbins)
        self.pair_counts                   = np.zeros(self.nbins)


        self.sumd_dsigmat_inp_num          = np.zeros(self.nbins)
        self.sumd_dsigmat_inp_bary_num     = np.zeros(self.nbins)
        self.sumd_dsigmat_inp_dm_num       = np.zeros(self.nbins)
        return 0
 
 
    def process_lens(self, lra, ldec, lzred, lwgt, llogMh, lconc, llogmstel, llog_re):
        self.lra         = lra      
        self.ldec        = ldec     
        self.lzred       = lzred    
        self.lwgt        = lwgt     
        self.llogMh      = llogMh   
        self.lconc       = lconc    
        self.llogmstel   = llogmstel
        self.llog_re     = llog_re

        # convert lense ra and dec into x,y,z cartesian coordinates
        lx, ly, lz = get_xyz(lra, ldec)
        # putting kd tree around the lenses
        self.lens_tree = cKDTree(np.array([lx, ly, lz]).T)
        print('lenses tree is ready\n')
        # setting maximum search radius
        self.dcommin = self.ss.Astropy_cosmo.angular_diameter_distance(np.min(lzred)).value
        self.dismax  = (self.rmax*1.0/(self.dcommin))
        return 0
 

    
    def process_source(self, ragal, decgal, zphotgal, wgal, e1gal, e2gal, zdiff): 
        "takes the intrinsic shapes and redshifts to get the Delta sigma, Delta sigma cross"
        r90_e1gal = -e1gal
        r90_e2gal = -e2gal
        # ra and dec to x,y,z for sources
        sx, sy, sz = get_xyz(ragal, decgal)
        # query in a ball around individual sources and collect the lenses ids with a maximum radius
        lidx = np.array(self.lens_tree.query_ball_point(np.transpose([sx, sy, sz]), self.dismax))
            
        #lidx = np.array(slidx[igal])
        # removing sources which doesn't have any lenses around them
        if len(lidx)==0:
            return 

        # selecting a cleaner background
        zcut = (np.max(self.lzred) < (zphotgal - zdiff)) #only taking the foreground lenses

        # again skipping the onces which doesn't satisfy the above criteria
        if zcut==0.0:
            return 
        # collecting the  data of lenses around individual source
        lidx   = lidx[zcut] # this will catch the array indices for our lenses
        sra    = ragal
        sdec   = decgal
        
        #lid, lra, ldec, lzred, lwgt, logmstel, logMh, xjkreg 
        #l_id        = lid[lidx]
        l_ra        = self.lra[lidx]
        l_dec       = self.ldec[lidx]
        l_zred      = self.lzred[lidx]
        l_wgt       = self.lwgt[lidx]
        l_logmstel  = self.llogmstel[lidx]
        l_log_re    = self.llog_re[lidx]
        l_logMh     = self.llogMh[lidx]
        l_conc      = self.lconc[lidx]
        
        e1gal, e2gal, etan, kappa, proj_sep, sflag, etan_b, etan_dm, etan_obs, ex_obs = self.ss.shear_src(l_ra, l_dec, l_zred, l_logmstel, l_log_re, l_logMh, l_conc, ragal, decgal, zphotgal, e1gal, e2gal)

        e1gal, e2gal, etan, kappa, proj_sep, sflag, etan_b, etan_dm, r90_etan_obs, r90_ex_obs = self.ss.shear_src(l_ra, l_dec, l_zred, l_logmstel, l_log_re, l_logMh, l_conc, ragal, decgal, zphotgal, r90_e1gal, r90_e2gal)
        

        # tangential shear
        sl_sep  = proj_sep
        w_ls    = l_wgt*wgal*self.ss.get_sigma_crit_inv**2
 
        #cure the arrays a bin
        idx = (sl_sep>self.rmin) & (sl_sep<self.rmax) & (sflag==1)
        sl_sep      = sl_sep[idx]
        sigma_crit  = 1/self.ss.get_sigma_crit_inv[idx]
        w_ls        = w_ls[idx]
        
        # measured values
        etan_obs        = etan_obs[idx]
        ex_obs          = ex_obs[idx]  
        r90_etan_obs    = r90_etan_obs[idx]
        r90_ex_obs      = r90_ex_obs[idx]  
        
        # input shear values
        etan        = etan[idx]   
        etan_b      = etan_b[idx]
        etan_dm     = etan_dm[idx]

        # getting the radial separations for a lense source pair
        slnbins = np.log10(sl_sep*1.0/self.rmin)//self.rdiff
        
        for rb in range(self.nbins):
            idx  = slnbins==rb 
            if sum(idx)==0:
                continue
            #print(np.shape(w_ls * etan_obs * sigma_crit))
            #print(np.shape(idx))
            #exit()

            self.sumd_dsigmat_num          [rb]     +=sum((w_ls * etan_obs * sigma_crit)[idx])
            self.sumd_dsigmatsq_num        [rb]     +=sum(((w_ls* etan_obs * sigma_crit)**2)[idx])
            self.sumd_dsigmax_num          [rb]     +=sum((w_ls * ex_obs * sigma_crit)[idx])
            self.sumd_dsigmaxsq_num        [rb]     +=sum(((w_ls* ex_obs * sigma_crit)**2)[idx])
            self.sumdwls                   [rb]     +=sum(w_ls[idx])


            self.r90_sumd_dsigmat_num      [rb]     +=sum((w_ls * r90_etan_obs * sigma_crit)[idx])      
            self.r90_sumd_dsigmatsq_num    [rb]     +=sum(((w_ls* r90_etan_obs * sigma_crit)**2)[idx])  
            self.r90_sumd_dsigmax_num      [rb]     +=sum((w_ls * r90_ex_obs * sigma_crit)[idx])      
            self.r90_sumd_dsigmaxsq_num    [rb]     +=sum(((w_ls* r90_ex_obs * sigma_crit)**2)[idx])  
            self.r90_sumdwls               [rb]     +=sum(w_ls[idx])

            self.pair_counts               [rb]     +=sum(idx)
  

            self.sumd_dsigmat_inp_num      [rb]     +=sum((sigma_crit * etan)[idx])
            self.sumd_dsigmat_inp_bary_num [rb]     +=sum((sigma_crit * etan_b)[idx])
            self.sumd_dsigmat_inp_dm_num   [rb]     +=sum((sigma_crit * etan_dm)[idx])
        return 0                


    def write2file(self):
        fout = open(self.outputfilename, "w")
        fout.write("# 0:rmin/2+rmax/2 1:dsigt 2:SN_Errdsigt 3:dsigx 4:SN_Errdsigx 5:r90_dsigt 6:r90_SN_Errdsigt 7:r90_dsigx 8:r90_SN_Errdsigx 9:true_dsig_bary 10:true_dsig_dm 11:true_dsig 12:sumdwls 13:jkreg\n")

        for i in range(self.nbins):
            rmin = self.rbins[i]
            rmax = self.rbins[i+1]
            rr                  =   rmin/2.0 + rmax/2.0 
            dsig                =   self.sumd_dsigmat_num[i]*1.0/self.sumdwls[i]
            SN_Errdsigt         =   np.sqrt(self.sumd_dsigmatsq_num[i])*1.0/self.sumdwls[i]
            dsigx               =   self.sumd_dsigmax_num[i]*1.0/self.sumdwls[i]
            SN_Errdsigx         =   np.sqrt(self.sumd_dsigmaxsq_num[i])*1.0/self.sumdwls[jk*self.nbins + i]


            r90_dsig            =   self.r90_sumd_dsigmat_num[i]*1.0/self.r90_sumdwls[i]
            r90_SN_Errdsigt     =   np.sqrt(self.r90_sumd_dsigmatsq_num[i])*1.0/self.r90_sumdwls[i]
            r90_dsigx           =   self.r90_sumd_dsigmax_num[i]*1.0/self.r90_sumdwls[i]
            r90_SN_Errdsigx     =   np.sqrt(self.r90_sumd_dsigmaxsq_num[i])*1.0/self.r90_sumdwls[i]

            true_dsig_bary      =   self.sumd_dsigmat_inp_bary_num      [i]/self.pair_counts[i] 
            true_dsig_dm        =   self.sumd_dsigmat_inp_dm_num [i]/self.pair_counts[i] 
            true_dsig           =   self.sumd_dsigmat_inp_num   [i]/self.pair_counts[i] 
            fout.write("%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\n"%(rr, dsig, SN_Errdsigt, dsigx, SN_Errdsigx, r90_dsig, r90_SN_Errdsigt, r90_dsigx, r90_SN_Errdsigx, true_dsig_bary, true_dsig_dm, true_dsig, self.sumdwls[jk*self.nbins + i]))
        fout.write("#OK")
        fout.close()

        return 0

if __name__ == "__main__":
    #section for testing purposes
    print("hello")



#def get_interp_szred():
#    "assigns redshifts respecting the distribution"
#    z0 = 0.9/(2)**0.5
#    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
#    zmin = 0.0
#    zmax = 3
#    zarr = np.linspace(zmin, zmax, 20)
#    xx  = 0.0 * zarr
#    for ii in range(len(xx)):
#        xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
#    proj = interp1d(xx,zarr)
#    return proj
#
#interp_szred = get_interp_szred()
#
#def create_sources(ramin, ramax, thetamin, thetamax, sigell=0.27, mask=None): #mask application for future
#    cdec    = np.random.uniform(np.cos(thetamax), np.cos(thetamin))     
#    sdec    = (90.0 - np.arccos(cdec)*180/np.pi)
#    sra     = np.random.uniform(ramin, ramax)*180/np.pi
#
#    # putting the interpolation for source redshift assignment
#    szred   =   interp_szred(np.random.uniform())
#    se1     =   np.random.normal(0.0, sigell) 
#    se2     =   np.random.normal(0.0, sigell)
#    wgal    =   sra/sra
#    return np.transpose([sra, sdec, szred, wgal, se1, se2])

#def get_et_ex(lra, ldec, sra, sdec, se1, se2):
#    "measures the etan and ecross for a given  lens-source pair"
#    lra  = lra*np.pi/180
#    ldec = ldec*np.pi/180
#    sra  = sra*np.pi/180
#    sdec = sdec*np.pi/180
#
#    c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
#    s_theta = np.sqrt(1-c_theta**2)
#
#    c_phi   =  np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
#    s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta
#
#    # tangential shear
#    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)
#    e_x     =  se1*(2*c_phi * s_phi) - se2*(2*c_phi**2 -1)
#
#    return e_t, e_x


