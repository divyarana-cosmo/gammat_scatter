# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper
import sys
sys.path.append('./src/')
sys.path.append('./utils/')
from distort_com import simshear

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration
from create_sources import get_xyz, create_sources 
from scipy.interpolate import interp1d
from scipy.integrate import quad


class source_select():
   def __init__(self):
       #getting the source redshift interpolation for sampling
       self.get_interp_szred()

    def get_xyz(self, ra, dec):
        ra = ra*np.pi/180.
        dec = dec*np.pi/180.
        x = np.cos(dec)*np.cos(ra)
        y = np.cos(dec)*np.sin(ra)
        z = np.sin(dec)
        return x, y, z
    
    def get_interp_szred(self):
        "assigns redshifts respecting the distribution"
        z0 = 0.9/(2)**0.5
        f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
        zmin = 0.0
        zmax = 3
        zarr = np.linspace(zmin, zmax, 20)
        xx  = 0.0 * zarr
        for ii in range(len(xx)):
            xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
        self.interp_szred = interp1d(xx,zarr)
        return 0
    
    
    def create_sources(self, ra, dec, dismax, nsrc=30, sigell=0.27, seed=123): #mask application for future
        "creates source around lens given angles in degrees"
        print('using seed - ', seed)
        ramin = (ra - dismax*180/np.pi )*np.pi/180
        ramax = (ra + dismax*180/np.pi)*np.pi/180
        thetamax =(90 - (dec - dismax*180/np.pi))*np.pi/180
        thetamin =(90 - (dec + dismax*180/np.pi))*np.pi/180
        
        area    = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
        size    = round(nsrc * area)      # area of square in deg^2 --> arcmin^2
        #add the possion galaxy number density 
        rng     =   np.random.default_rng(seed) # fixing the seed of the random number generator
        size    =   rng.poisson(size) # number of sources
        print('samples source', size)
        cdec    =   rng.uniform(np.cos(thetamax), np.cos(thetamin), size=size)     
        sdec    =   (90.0 - np.arccos(cdec)*180/np.pi)
        sra     =   rng.uniform(ramin, ramax, size=size)*180/np.pi
        lx,ly,lz = get_xyz(ra, dec)
        sx,sy,sz = get_xyz(sra, sdec)
        #annulus aperture
        sep     =  ((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2)**0.5
        idx     =   (sep < dismax)
        sra     = sra[idx]
        sdec    = sdec[idx]
    
        # putting the interpolation for source redshift assignment
        szred   =   self.interp_szred(rng.random(size=len(sra)))
        se1     =   rng.normal(0.0, sigell, len(sra)) 
        se2     =   rng.normal(0.0, sigell, len(sra))
        wgal    =   sra/sra

        return sra, sdec, szred, wgal, se1, se2





class weakpipe():
    def __init__(self, H0=100, Om0=0.25, Rmin=0.004, Rmax=0.4, Nbins=10, outputfilename='dsigma.dat', outputpairfile=False):
        self.Rmin           =   Rmin 
        self.Rmax           =   Rmax 
        self.Nbins          =   Nbins
        self.Rbins          =   np.logspace(np.log10(Rmin), np.log10(Rmax), Nbins + 1)
        self.Rdiff          =   np.log10(self.Rbins[1] / self.Rbins[0])
 
        self.selsrc         =   source_select()
        self.ss             =   simshear(H0=H0, Om0=Om0)
        self.outputfilename =   outputfilename
        self.outputpairfile =   outputpairfile

        # Open pair output file if needed
        if self.outputpairfile:
            self.fpairout = open(self.outputfilename + '_pairs', "w")
            self.fpairout.write('jkid\tlra(deg)\tldec(deg)\tlzred\tllogmstel\tllogmh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\twls\tkappa\tintse1\tintse2\tr90se1\tr90se2\tr90et\tr90ex\tr90intse1\tr90intse2\n')

        

    def init_array(self):
        # Initialize arrays for accumulation
        self.sumdgammat_num              = np.zeros(self.Nbins)
        self.sumdgammat_inp_num          = np.zeros(self.Nbins)
        self.sumdgammat_inp_bary_num     = np.zeros(self.Nbins)
        self.sumdgammat_inp_dm_num       = np.zeros(self.Nbins)
        self.sumdgammatsq_num            = np.zeros(self.Nbins)
        self.sumdgammax_num              = np.zeros(self.Nbins) 
        self.sumdgammaxsq_num            = np.zeros(self.Nbins)
        self.sumdwls                     = np.zeros(self.Nbins)
        self.sumddsigmat_num             = np.zeros(self.Nbins)
        self.sumddsigmat_inp_num         = np.zeros(self.Nbins)
        self.sumddsigmat_inp_bary_num    = np.zeros(self.Nbins)
        self.sumddsigmat_inp_dm_num      = np.zeros(self.Nbins)
        self.sumddsigmatsq_num           = np.zeros(self.Nbins)
        self.sumddsigmax_num             = np.zeros(self.Nbins) 
        self.sumddsigmaxsq_num           = np.zeros(self.Nbins)
        self.sumdwls_by_sigcsq           = np.zeros(self.Nbins)
        return 0

    def process_lensdata(self, lid, lzred, lwgt, llogmstel, llogre, llogmh, lconc):
        self.lid        =  lid
        self.lzred      =  lzred 
        self.lwgt       =  lwgt 
        self.llogmstel  =  llogmstel 
        self.llogre     =  llogre
        self.llogmh     =  llogmh 
        self.lconc      =  lconc 
        print("lens data read fully", np.min(lzred))
        # Calculate maximum angular separation based on minimum redshift
        self.dismax = self.Rmax / self.ss.Astropy_cosmo.comoving_distance(np.min(lzred)).value 
        print(np.min(lzred), 'thetamax', self.dismax) 
        return 0
    
    def weaklens_aroundlens(self, nsrc=30, sigell=0.26, zmax=0.4, zdiff=0.1, seed=123, use_shear=False,test_case=False):
        lra = 130.0; ldec=0.0
        for ii in tqdm(range(len(self.lid))):
            # Create sources for this lens - vectorized creation
            sra, sdec, szred, wgal, intse1, intse2 = self.selsrc.create_sources(
                lra, ldec, self.dismax, nsrc=nsrc, sigell=sigell, seed=int(seed + self.lid[ii])) 
                        # Handle test case option
            if test_case:
                szred = np.full_like(sra, 0.9)
 
            print("number of sources:", len(sra))
            
            # Vectorized source selection
            scut = szred > (zmax + zdiff)  # zdiff cut
            if np.sum(scut) == 0:
                continue

            # Apply cut to all source arrays at once
            sra     = sra[scut]
            sdec    = sdec[scut]
            szred   = szred[scut]
            wgal    = wgal[scut]
            intse1  = intse1[scut]
            intse2  = intse2[scut]
            
            # Shear all sources at once
            se1, se2, etan, kappa, proj_sep, sflag, etan_b, etan_dm, et_obs, ex_obs = self.ss.shear_src(
                lra, ldec, self.lzred[ii], self.llogmstel[ii], self.llogre[ii], self.llogmh[ii], self.lconc[ii], sra, sdec, szred, intse1, intse2, use_shear=use_shear)
            
            print('flagged sources', np.sum(sflag))
            sl_sep  = proj_sep
            w_ls    = self.lwgt[ii] * wgal
            
            # Vectorized filtering of arrays
            idx = (sl_sep > self.Rmin) & (sl_sep < self.Rmax) & sflag
            if np.sum(idx) == 0:
                continue
                
            # Apply filter to all arrays at once
            sl_sep  = sl_sep[idx]
            w_ls    = w_ls[idx]
            et_obs  = et_obs[idx]
            etan    = etan[idx]
            kappa   = kappa[idx]
            etan_b  = etan_b[idx]
            etan_dm = etan_dm[idx]
            ex_obs  = ex_obs[idx]
            se1     = se1[idx]
            se2     = se2[idx]
            sra     = sra[idx]
            sdec    = sdec[idx]
            szred   = szred[idx]

            # Write pairs to output file if needed
            if self.outputpairfile :
                for jj in range(np.sum(idx)):
                    self.fpairout.write(f'{lxjkreg[ii]}\t{lra[ii]}\t{ldec[ii]}\t{lzred[ii]}\t{llogmstel[ii]}\t{llogmh[ii]}\t{lconc[ii]}\t{sra[jj]}\t{sdec[jj]}\t{szred[jj]}\t{se1[jj]}\t{se2[jj]}\t{etan[jj]}\t{et_obs[jj]}\t{ex_obs[jj]}\t{sl_sep[jj]}\t{w_ls[jj]}\t{kappa[jj]}\t{intse1[jj]}\t{intse2[jj]}\t{r90se1[jj]}\t{r90se2[jj]}\t{r90et_obs[jj]}\t{r90ex_obs[jj]}\t{r90intse1[jj]}\t{r90intse2[jj]}\n')
            
       
            # Use vectorized operation instead of loop
            sigma_crit_inv      = self.ss._get_sigma_crit_inv(lzred=self.lzred[ii], szred=szred) * 1e12
            w_ls_invsigmacritsq = w_ls * sigma_crit_inv**2
            w_ls_invsigmacrit   = w_ls * sigma_crit_inv
            
            # Vectorized binning
            slrbins = (np.log10(sl_sep / self.Rmin) // self.Rdiff).astype(int)
                   
            np.add.at(self.sumdwls                   ,slrbins    ,w_ls)
            np.add.at(self.sumdwls_by_sigcsq         ,slrbins    ,w_ls_invsigmacritsq)

            np.add.at(self.sumdgammat_inp_num        ,slrbins    ,w_ls * etan)
            np.add.at(self.sumdgammat_inp_bary_num   ,slrbins    ,w_ls * etan_b)
            np.add.at(self.sumdgammat_inp_dm_num     ,slrbins    ,w_ls * etan_dm)
            
            np.add.at(self.sumdgammat_num            ,slrbins    ,w_ls * et_obs)
            np.add.at(self.sumdgammatsq_num          ,slrbins    ,(w_ls * et_obs)**2)
            np.add.at(self.sumdgammax_num            ,slrbins    ,w_ls * ex_obs)
            np.add.at(self.sumdgammaxsq_num          ,slrbins    ,(w_ls * ex_obs)**2)

            np.add.at(self.sumddsigmat_inp_num       ,slrbins    ,w_ls_invsigmacrit * etan)
            np.add.at(self.sumddsigmat_inp_bary_num  ,slrbins    ,w_ls_invsigmacrit * etan_b)
            np.add.at(self.sumddsigmat_inp_dm_num    ,slrbins    ,w_ls_invsigmacrit * etan_dm)

            np.add.at(self.sumddsigmat_num           ,slrbins    ,w_ls_invsigmacrit * et_obs)
            np.add.at(self.sumddsigmatsq_num         ,slrbins    ,(w_ls_invsigmacrit * et_obs)**2)
            np.add.at(self.sumddsigmax_num           ,slrbins    ,w_ls_invsigmacrit * ex_obs)
            np.add.at(self.sumddsigmaxsq_num         ,slrbins    ,(w_ls_invsigmacrit * ex_obs)**2)

        return 0


    def write2file(self):
         if self.outputpairfile:
             self.fpairout.write("#OK")
             self.fpairout.close()
             
         # Calculate responsivity correction
         Resp = 1.0
         # Create dictionary for results
         df = {}
         df["-2-rmin"]                   = self.Rbins[:-1]
         df["-1-rmax"]                   = self.Rbins[1:]
         df["0-rmin/2+rmax/2"]           = self.Rbins[:-1] * 0.5 + self.Rbins[1:] * 0.5
         df["1-gammat"]                  = self.sumdgammat_num / self.sumdwls / Resp    
         df["2-gammatsq"]                = self.sumdgammatsq_num / self.sumdwls / Resp**2
         df["3-sigma_gammat"]            = np.sqrt(self.sumdgammatsq_num / self.sumdwls / Resp**2 - (self.sumdgammat_num / self.sumdwls / Resp)**2)
         df["4-SN_Errgammat"]            = np.sqrt(self.sumdgammatsq_num) / self.sumdwls / Resp
         df["5-gammax"]                  = self.sumdgammax_num / self.sumdwls / Resp
         df["6-gammaxsq"]                = self.sumdgammaxsq_num / self.sumdwls / Resp**2
         df["7-sigma_gammax"]            = np.sqrt(self.sumdgammaxsq_num / self.sumdwls / Resp**2 - (self.sumdgammax_num / self.sumdwls / Resp)**2)
         df["8-SN_Errgammax"]            = np.sqrt(self.sumdgammaxsq_num) / self.sumdwls / Resp
         df["9-gammat_inp"]              = self.sumdgammat_inp_num / self.sumdwls / Resp
         df["10-gammat_inp_bary"]        = self.sumdgammat_inp_bary_num / self.sumdwls / Resp
         df["11-gammat_inp_dm"]          = self.sumdgammat_inp_dm_num / self.sumdwls / Resp
         df["12-sumd_wls"]               = self.sumdwls
         df["13-dsigma"]                 = self.sumddsigmat_num / self.sumdwls_by_sigcsq / Resp
         df["14-dsigmasq"]               = self.sumddsigmatsq_num / self.sumdwls_by_sigcsq / Resp**2
         df["15-SN_Errdsigmat"]          = np.sqrt(self.sumddsigmatsq_num) / self.sumdwls_by_sigcsq / Resp
         df["16-dsigmax"]                = self.sumddsigmax_num / self.sumdwls_by_sigcsq / Resp
         df["17-dsigmaxsq"]              = self.sumddsigmaxsq_num / self.sumdwls_by_sigcsq / Resp**2
         df["18-SN_Errdsigmax"]          = np.sqrt(self.sumddsigmaxsq_num) / self.sumdwls_by_sigcsq / Resp
         df["19-dsigmat_inp"]            = self.sumddsigmat_inp_num / self.sumdwls_by_sigcsq / Resp
         df["20-dsigmat_inp_bary"]       = self.sumddsigmat_inp_bary_num / self.sumdwls_by_sigcsq / Resp
         df["21-dsigmat_inp_dm"]         = self.sumddsigmat_inp_dm_num / self.sumdwls_by_sigcsq / Resp
         df["22-sumd_dsigma_wls" ]       = self.sumdwls_by_sigcsq
         df["23-sumd_dsigma_num" ]       = self.sumddsigmat_num
         df["24-sumd_dsigmax_num" ]      = self.sumddsigmax_num
         df["25-sumd_dsigma_den" ]       = self.sumdwls_by_sigcsq*Resp

         import pandas as pd
         df = pd.DataFrame(df)
         df.to_csv(self.outputfilename, index=False, sep=' ')
         return 0

if __name__ == "__main__":

