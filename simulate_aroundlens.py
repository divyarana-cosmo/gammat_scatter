# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper

import sys
sys.path.append('./src/')
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
from mpi4py import MPI
from subprocess import  call
from scipy import stats
from colossus.cosmology import cosmology
from colossus.halo import concentration



def get_xyz(ra, dec):
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def get_et_ex(lra, ldec, sra, sdec, se1, se2):
    "measures the etan and ecross for a given  lens-source pair"
    lra  = lra*np.pi/180
    ldec = ldec*np.pi/180
    sra  = sra*np.pi/180
    sdec = sdec*np.pi/180

    c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
    s_theta = np.sqrt(1-c_theta**2)

    c_phi   =  np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
    s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta

    # tangential shear
    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)
    e_x     =  se1*(2*c_phi * s_phi) - se2*(2*c_phi**2 -1)

    return e_t, e_x

def get_interp_szred():
    "assigns redshifts respecting the distribution"
    n0  = 1.8048
    a   = 0.417
    b   = 4.8685
    c   = 0.7841
    f = lambda zred: n0*(zred**a + zred**(a*b))/(zred**b + c)
    zmin = 0.0
    zmax = 3
    zarr = np.linspace(zmin, zmax, 20)
    xx  = 0.0 * zarr
    for ii in range(len(xx)):
        xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
    proj = interp1d(xx,zarr)
    return proj


interp_szred = get_interp_szred()

def create_sources(ra, dec, dismax, nsrc=30, sigell=0.27, mask=None): #mask application for future
    "creates source around lens given angles in degrees"
    ramin = (ra - dismax*180/np.pi )*np.pi/180
    ramax = (ra + dismax*180/np.pi)*np.pi/180
    thetamax =(90 - (dec - dismax*180/np.pi))*np.pi/180
    thetamin =(90 - (dec + dismax*180/np.pi))*np.pi/180

    area    = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
    size    = round(nsrc * area)      # area of square in deg^2 --> arcmin^2
    cdec    = np.random.uniform(np.cos(thetamax), np.cos(thetamin), size=size)     
    sdec    = (90.0 - np.arccos(cdec)*180/np.pi)
    sra     = np.random.uniform(ramin, ramax, size=size)*180/np.pi
    lx,ly,lz = get_xyz(ra, dec)
    sx,sy,sz = get_xyz(sra, sdec)
    #circular aperture
    idx = ((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2)**0.5 < dismax
    sra  = sra[idx]
    sdec = sdec[idx]

    # putting the interpolation for source redshift assignment
    szred   =   interp_szred(np.random.uniform(size=len(sra)))
    se1     =   np.random.normal(0.0, sigell, len(sra)) 
    se2     =   np.random.normal(0.0, sigell, len(sra))
    wgal    =   sra/sra
    return sra, sdec, szred, wgal, se1, se2




def run_pipe(config, outputfile = 'gamma.dat', outputpairfile=None):
    rmin    = config['Rmin'] 
    rmax    = config['Rmax'] 
    nbins   = config['Nbins']

    lensargs    = config["lens"]
    sourceargs  = config["source"]

    zdiff   = sourceargs["zdiff"]

    #setting up cosmology and class instance
    ss = simshear(H0 = 100, Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'])

    colossus_cosmo  = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8 = ss.sigma8, ns = ss.ns, cosmo_name=ss.cosmo_name)


    # set the projected radial binning
    rmin  =  rmin
    rmax  =  rmax
    nbins = nbins #10 radial bins for our case
    rbins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff  = np.log10(rbins[1]*1.0/rbins[0])

    sumdgammat_num      = np.zeros(nbins)
    sumdgammat_inp_num  = np.zeros(nbins)
    sumdgammat_inp_bary_num  = np.zeros(nbins)
    sumdgammat_inp_dm_num  = np.zeros(nbins)
    sumdgammatsq_num    = np.zeros(nbins)
    sumdgammax_num      = np.zeros(nbins) 
    sumdgammaxsq_num    = np.zeros(nbins)
    sumwls              = np.zeros(nbins)



    # getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    lconc = 0.0*lid
    xx = np.linspace(9,16,100)
    yy = 0.0*xx
    med_lzred = np.median(lzred)

    for kk, mh in enumerate(10**xx):
        yy[kk]    = concentration.concentration(mh, '200m', med_lzred, model = 'diemer19')
    spl_c_mh = interp1d(xx,np.log10(yy))

    lconc = 10**spl_c_mh(llogmh)
    print("lens data read fully")

    np.random.seed(123)
    #..................................#

    for ii in tqdm(range(len(lra))):
        # fixing the simulation aperture
        dismax = config['Rmax']/ss.Astropy_cosmo.comoving_distance(lzred[ii]).value 
        sra, sdec, szred, wgal, se1, se2 = create_sources(lra[ii], ldec[ii], dismax, nsrc=sourceargs['nsrc'], sigell=sourceargs['sigell']) 
        print("number of sources: ", len(sra))
        # selecting cleaner background
        scut    = (szred>(lzred[ii] + sourceargs['zdiff']))
        sra     = sra[scut  ]  
        sdec    = sdec[scut]
        szred   = szred[scut]
        wgal    = wgal[scut]
        se1     = 0.0*se1[scut]
        se2     = 0.0*se2[scut]

        # add a section of stellar and dark matter
        se1, se2, etan, proj_sep, sflag, etan_b, etan_dm = ss.shear_src(lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogmh[ii], lconc[ii], sra, sdec, szred, se1, se2)

        et, ex  = get_et_ex(lra = lra[ii], ldec = ldec[ii], sra = sra, sdec = sdec, se1 = se1,  se2 = se2)
        sl_sep  = proj_sep
        w_ls    = lwgt[ii]*wgal
        #cure the arrays a bin
        idx = (sl_sep>rmin) & (sl_sep<rmax) & (sflag==1)
        sl_sep      = sl_sep[idx]
        w_ls        = w_ls[idx]
        et          = et[idx]
        etan        = etan[idx]   
        etan_b      = etan_b[idx]
        etan_dm     = etan_dm[idx]
        ex          = ex[idx]  

        slrbins = np.log10(sl_sep*1.0/rmin)//rdiff
        for rb in range(nbins):
            idx = slrbins==rb
            sumdgammat_num[rb]              +=sum((w_ls * et)[idx])
            sumdgammat_inp_num[rb]          +=sum((w_ls * etan)[idx])
            sumdgammat_inp_bary_num[rb]     +=sum((w_ls * etan_b)[idx])
            sumdgammat_inp_dm_num[rb]       +=sum((w_ls * etan_dm)[idx])
            sumdgammatsq_num[rb]            +=sum(((w_ls* et)**2)[idx])
            sumdgammax_num[rb]              +=sum((w_ls * ex)[idx])
            sumdgammaxsq_num[rb]            +=sum(((w_ls* ex)**2)[idx])
            sumwls[rb]                      +=sum(w_ls[idx])








        #for ll,sep in enumerate(sl_sep):
        #    if sep<rmin or sep>rmax or sflag[ll]==0:
        #        continue
        #    rb = int(np.log10(sep*1.0/rmin)*1/rdiff)

        #    # get tangantial components given positions and shapes

        #    # following equations given in the surhud's lectures
        #    w_ls    = lwgt[ii] * wgal[ll]

        #    # separate numerator and denominator computation
        #    sumdgammat_num[rb]              += w_ls  * et[ll]
        #    sumdgammat_inp_num[rb]          += w_ls  * etan[ll]
        #    sumdgammat_inp_bary_num[rb]     += w_ls  * etan_b[ll]
        #    sumdgammat_inp_dm_num[rb]       += w_ls  * etan_dm[ll]
        #    sumdgammatsq_num[rb]            += (w_ls * et[ll])**2
        #    sumdgammax_num[rb]              += w_ls  * ex[ll]
        #    sumdgammaxsq_num[rb]            += (w_ls * ex[ll])**2
        #    sumwls[rb]                      += w_ls


    fout = open(outputfilename, "w")
    fout.write("# 0:rmin/2+rmax/2 1:gammat 2:SN_Errgammat 3:gammax 4:SN_Errgammax 5:truegamma 6:gammat_inp 7:gammat_inp_bary 8:gammat_inp_dm  \n")
    for i in range(len(rbins[:-1])):
        rrmin = rbins[i]
        rrmax = rbins[i+1]
       #Resp = sumwls_resp[i]*1.0/sumwls[i]

        fout.write("%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdgammat_num[i]*1.0/sumwls[i], np.sqrt(sumdgammatsq_num[    i])*1.0/sumwls[i], sumdgammax_num[i]*1.0/sumwls[i], np.sqrt(sumdgammaxsq_num[i])*1.0/sumwls[i], sumdgammat_inp_num[i]*1.0/sumwls[i], sumdgammat_inp_num[i]/sumwls[i], sumdgammat_inp_bary_num[i]/sumwls[i], sumdgammat_inp_dm_num[i]/sumwls[i])    )
        #fout.write("%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdsig_num[i]*1.0/sumwls[i]/2./Resp, np.sqrt(sumdsigsq_num[i])*1.0/sumw    ls[i]/2./Resp))
    fout.write("#OK")
    fout.close()
    return 0








if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--ideal_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=11.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=13.0)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
    
    np.random.seed(args.seed)

    run_pipe(config, outputfile = outputfilename)           

