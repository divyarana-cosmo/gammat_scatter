# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper
import sys
sys.path.append('./src/')
sys.path.append('./utils/')
from lensutils import get_re
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


def get_interp_szred():
    "assigns redshifts respecting the distribution"
    z0 = 0.9/(2)**0.5
    f = lambda zred: (zred/z0)**2 * np.exp(-(zred/z0)**(3/2)) #taken from euclid prep 2020 page 22
    zmin = 0.0
    zmax = 3
    zarr = np.linspace(zmin, zmax, 20)
    xx  = 0.0 * zarr
    for ii in range(len(xx)):
        xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
    proj = interp1d(xx,zarr)
    return proj

interp_szred = get_interp_szred()

def create_sources(ra, dec, dismax, nsrc=30, sigell=0.27, mask=None, seed=123): #mask application for future
    "creates source around lens given angles in degrees"
    ramin = (ra - dismax*180/np.pi )*np.pi/180
    ramax = (ra + dismax*180/np.pi)*np.pi/180
    thetamax =(90 - (dec - dismax*180/np.pi))*np.pi/180
    thetamin =(90 - (dec + dismax*180/np.pi))*np.pi/180

    area    = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
    size    = round(nsrc * area)      # area of square in deg^2 --> arcmin^2
    #add the possion galaxy number density 
    rng     =   np.random.default_rng(seed) # fixing the seed of the random number generator

    size    =   rng.poisson(size) # number of sources
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
    szred   =   interp_szred(rng.random(size=len(sra)))
    se1     =   rng.normal(0.0, sigell, len(sra)) 
    se2     =   rng.normal(0.0, sigell, len(sra))
    wgal    =   sra/sra
    return sra, sdec, szred, wgal, se1, se2


def run_pipe(config, outputfilename = 'gamma.dat', jksamp=0, outputpairfile=None):
    rmin    = config['Rmin'] 
    rmax    = config['Rmax'] 
    nbins   = config['Nbins']

    lensargs    = config["lens"]
    sourceargs  = config["source"]

    zdiff   = sourceargs["zdiff"]

    #only working with H0 and omg0
    ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = 0.044, Tcmb0 = 2.7255, Neff = 3.046, sigma8 = 0.8, ns = 0.95)

    colossus_cosmo  = cosmology.fromAstropy(ss.Astropy_cosmo, sigma8 = ss.sigma8, ns = ss.ns, cosmo_name=ss.cosmo_name)

    # set the projected radial binning
    rmin  =  rmin
    rmax  =  rmax
    nbins = nbins #10 radial bins for our case
    rbins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff  = np.log10(rbins[1]*1.0/rbins[0])
    
    Njacks = int(lensargs['Njacks'])
    sumdgammat_num              = np.zeros(nbins)
    sumdgammat_inp_num          = np.zeros(nbins)
    sumdgammat_inp_bary_num     = np.zeros(nbins)
    sumdgammat_inp_dm_num       = np.zeros(nbins)
    sumdgammatsq_num            = np.zeros(nbins)
    sumdgammax_num              = np.zeros(nbins) 
    sumdgammaxsq_num            = np.zeros(nbins)
    sumdwls                     = np.zeros(nbins)
    sumddsigmat_num              = np.zeros(nbins)
    sumddsigmat_inp_num          = np.zeros(nbins)
    sumddsigmat_inp_bary_num     = np.zeros(nbins)
    sumddsigmat_inp_dm_num       = np.zeros(nbins)
    sumddsigmatsq_num            = np.zeros(nbins)
    sumddsigmax_num              = np.zeros(nbins) 
    sumddsigmaxsq_num            = np.zeros(nbins)
    sumddsigmawls                = np.zeros(nbins)

    # getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    NNlens = int(len(lid))
    lid = np.arange(len(lid))
    idx = (lxjkreg != jksamp)
    lra         = lra       [idx]
    ldec        = ldec      [idx]
    lzred       = lzred     [idx]
    lwgt        = lwgt      [idx]
    llogmstel   = llogmstel [idx]
    llogmh      = llogmh    [idx]
    lid         = lid       [idx]
    #fixed position 
    lra     = 130 + 0.0*lra
    ldec    = 0.0 + 0.0*ldec

    if config['test_case']:
        np.random.seed(123)
        idx         = (np.random.uniform(size=len(lra))<0.1)
        lra         = lra[idx]
        ldec        = ldec[idx]
        llogmh      = 12.0  + 0.0*llogmh[idx]
        lzred       = 0.3   + 0.0*lzred[idx]
        lconc       = concentration.concentration(10**14, '200m', 0.3, model = 'diemer19') + 0.0*lzred
        llogmstel   = 10.0  + 0.0*llogmh
    else:
        lconc = 0.0*lid
        xx = np.linspace(9,16,50)
        yy = 0.0*xx
        med_lzred = np.median(lzred)

        for kk, mh in enumerate(10**xx):
            yy[kk]    = concentration.concentration(mh, '200m', med_lzred, model = 'diemer19')
        
        spl_c_mh = interp1d(xx,yy)
        lconc = spl_c_mh(llogmh)
 
    lzredmax = np.max(lzred)

    print("lens data read fully")
    llogre = get_re(llogmstel - np.log10(config['H0']/100)) -3 # converting Kpc to Mpc
    dismax = config['Rmax']/ss.Astropy_cosmo.angular_diameter_distance(np.min(lzred)).value 
    
    if sourceargs['use_shear']:
        print("using shear not reduced shear for the sims")
        outputpairfile = outputpairfile + '_using_shear'
        outputfilename = outputfilename + '_using_shear'

    if outputpairfile != None:
        fpairout = open(outputpairfile, "w")
        fpairout.write('jkid\tlra(deg)\tldec(deg)\tlzred\tllogmstel\tllogmh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\twls\tkappa\tintse1\tintse2\tr90se1\tr90se2\tr90et\tr90ex\tr90intse1\tr90intse2\n')

    #..................................#
    for ii in tqdm(range(len(lra))):
        # simulating the sources
        sra, sdec, szred, wgal, intse1, intse2 = create_sources(lra[ii], ldec[ii], dismax, nsrc=sourceargs['nsrc'], sigell=sourceargs['sigell'], seed = config["seed"]*NNlens+ lid[ii]) 
       
        if config['test_case']:
            szred = 0.8 + 0.0*sra
        if sourceargs['no_shape_noise']:
            print("no shape noise")
            intse1 = 0.0*intse1
            intse2 = 0.0*intse2

        print("number of sources: ", len(sra))
        # selecting cleaner background
        scut    = (szred>(lzredmax + sourceargs['zdiff'])) # zdiff cut
        if sum(scut)==0:
            continue
 
        sra         =   sra[scut]  
        sdec        =   sdec[scut]
        szred       =   szred[scut]
        wgal        =   wgal[scut]
        intse1      =   intse1[scut]
        intse2      =   intse2[scut]

        # shearing the sources
        se1, se2, etan, kappa, proj_sep, sflag, etan_b, etan_dm, et_obs, ex_obs = ss.shear_src(lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogre[ii], llogmh[ii], lconc[ii], sra, sdec, szred, intse1, intse2, use_shear=sourceargs["use_shear"], no_shear=sourceargs["no_shear"])
        
       
        if sourceargs['no_shear']:
            se1 = intse1; se2 = intse2
        
        sl_sep  = proj_sep
        w_ls    = lwgt[ii]*wgal
    
        #cure the arrays a bin
        idx = (sl_sep>rmin) & (sl_sep<rmax) & (sflag==1)
        if sum(idx)==0.0:
            continue
        sl_sep      = sl_sep[idx]
        w_ls        = w_ls[idx]
        et_obs      = et_obs[idx]

        etan        = etan[idx]   
        kappa       = kappa[idx]   
        etan_b      = etan_b[idx]
        etan_dm     = etan_dm[idx]
        ex_obs      = ex_obs[idx]  
        se1         = se1[idx]
        se2         = se2[idx]
        sra         = sra[idx]
        sdec        = sdec[idx]
        szred       = szred[idx]


        if outputpairfile != None:
            for jj in range(sum(idx)):
                fpairout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lxjkreg[ii], lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogmh[ii], lconc[ii], sra[jj], sdec[jj], szred[jj], se1[jj], se2[jj], etan[jj], et_obs[jj], ex_obs[jj], sl_sep[jj], w_ls[jj], kappa[jj], intse1[jj], intse2[jj], r90se1[jj], r90se2[jj], r90et_obs[jj], r90ex_obs[jj], r90intse1[jj], r90intse2[jj]))
        
        w_ls_invsigmacritsq = np.zeros(int(sum(idx)))
        w_ls_invsigmacrit   = np.zeros(int(sum(idx)))
        #get the sigma critical
        for zz in range(sum(idx)):
            w_ls_invsigmacritsq[zz] = w_ls[zz] *(ss._get_sigma_crit_inv(lzred=lzred[ii], szred=szred[zz])*1e12)**2
            w_ls_invsigmacrit[zz]   = w_ls[zz] *ss._get_sigma_crit_inv(lzred=lzred[ii], szred=szred[zz])*1e12
        
        #exit()
        slrbins = np.log10(sl_sep*1.0/rmin)//rdiff

        for rb in range(nbins):
            idx = slrbins==rb
            if sum(idx)==0:
                continue
            
            sumdwls                 [rb] +=sum(w_ls[idx])
            sumddsigmawls           [rb] +=sum(w_ls_invsigmacritsq[idx])


            sumdgammat_inp_num      [rb] +=sum((w_ls * etan)[idx])
            sumdgammat_inp_bary_num [rb] +=sum((w_ls * etan_b)[idx])
            sumdgammat_inp_dm_num   [rb] +=sum((w_ls * etan_dm)[idx])
                
            sumdgammat_num          [rb] +=sum((w_ls * et_obs)[idx])
            sumdgammatsq_num        [rb] +=sum(((w_ls* et_obs)**2)[idx])
            sumdgammax_num          [rb] +=sum((w_ls * ex_obs)[idx])
            sumdgammaxsq_num        [rb] +=sum(((w_ls* ex_obs)**2)[idx])
           
            sumddsigmat_inp_num      [rb] +=sum(( w_ls_invsigmacrit * etan)[idx])
            sumddsigmat_inp_bary_num [rb] +=sum(( w_ls_invsigmacrit * etan_b)[idx])
            sumddsigmat_inp_dm_num   [rb] +=sum(( w_ls_invsigmacrit * etan_dm)[idx])

            sumddsigmat_num          [rb] +=sum(( w_ls_invsigmacrit * et_obs)[idx])
            sumddsigmatsq_num        [rb] +=sum(((w_ls_invsigmacrit * et_obs)**2)[idx])
            sumddsigmax_num          [rb] +=sum(( w_ls_invsigmacrit * ex_obs)[idx])
            sumddsigmaxsq_num        [rb] +=sum(((w_ls_invsigmacrit * ex_obs)**2)[idx])

    if outputpairfile != None:
        fpairout.write("#OK")
        fpairout.close()
    
    #need to clean this up
    print(sumdwls)
    df = {}
    df["0-rmin/2+rmax/2"    ]           =   rbins[:-1] *0.5 + rbins[1:]*0.5
    df["1-gammat"           ]           =   sumdgammat_num[:] * 1.0 / sumdwls[:]
    df["2-gammatsq"         ]           =   sumdgammatsq_num[:] * 1.0 / sumdwls[:]
    df["3-sigma_gammat"     ]           =   np.sqrt(sumdgammatsq_num[:] * 1.0 / sumdwls[:] - (sumdgammat_num[:] * 1.0 / sumdwls[:])**2)
    df["4-SN_Errgammat"     ]           =   np.sqrt(sumdgammatsq_num[:]) * 1.0 / sumdwls[:]
    df["5-gammax"           ]           =   sumdgammax_num[:] * 1.0 / sumdwls[:]
    df["6-gammaxsq"         ]           =   sumdgammaxsq_num[:] * 1.0 / sumdwls[:]
    df["7-sigma_gammax"     ]           =   np.sqrt(sumdgammaxsq_num[:] * 1.0 / sumdwls[:] - (sumdgammax_num[:] * 1.0 / sumdwls[:])**2)
    df["8-SN_Errgammax"     ]           =   np.sqrt(sumdgammaxsq_num[:]) * 1.0 / sumdwls[:]
    df["9-gammat_inp"      ]            =   sumdgammat_inp_num[:] / sumdwls[:]
    df["10-gammat_inp_bary" ]           =   sumdgammat_inp_bary_num[:] / sumdwls[:]
    df["11-gammat_inp_dm"   ]           =   sumdgammat_inp_dm_num[:] / sumdwls[:]
    df["12-sumd_wls"        ]           =   sumdwls
    df["13-dsigma"           ]          =   sumddsigmat_num[:] * 1.0 / sumddsigmawls[:]
    df["14-dsigmasq"         ]          =   sumddsigmatsq_num[:] * 1.0 / sumddsigmawls[:]
    df["15-SN_Errdsigmat"     ]         =   np.sqrt(sumddsigmatsq_num[:]) * 1.0 / sumddsigmawls[:]
    df["16-dsigmax"           ]         =   sumddsigmax_num[:] * 1.0 / sumddsigmawls[:]
    df["17-dsigmaxsq"         ]         =   sumddsigmaxsq_num[:] * 1.0 / sumddsigmawls[:]
    df["18-SN_Errdsigmax"     ]         =   np.sqrt(sumddsigmaxsq_num[:]) * 1.0 / sumddsigmawls[:]
    df["19-dsigmat_inp"      ]          =   sumddsigmat_inp_num[:] / sumddsigmawls[:]
    df["20-dsigmat_inp_bary" ]          =   sumddsigmat_inp_bary_num[:] / sumddsigmawls[:]
    df["21-dsigmat_inp_dm"   ]          =   sumddsigmat_inp_dm_num[:] / sumddsigmawls[:]
    df["22-sumd_dsigma_wls"        ]    =   sumddsigmawls[:]
    import pandas as pd
    df = pd.DataFrame(df)
    #idx =  sumdwls!=0
    #df = df[idx]
    df.to_csv(outputfilename, index=False, sep=' ')
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--no_shape_noise", help="for removing shape noise-testing purpose", type=bool, default=False)
    parser.add_argument("--no_shear", help="for removing shear-testing purpose", type=bool, default=False)
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--use_shear", help="use shear or reduced shear for simulations", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=11.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=13.0)
    #parser.add_argument("--ten_percent", help="using ten percent of the lense sample", type=bool, default=False)

    parser.add_argument("--two_percent", help="using two percent of the lense sample", type=bool, default=False)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)


    config["outputdir"] = config["outputdir"] 

    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax

    config['test_case']                 = args.test_case
    config['seed']                      = args.seed
    config['lens']['two_percent']       = args.two_percent
    config['source']['use_shear']       = args.use_shear
    config['source']['no_shape_noise']  = args.no_shape_noise
    config['source']['no_shear']        = args.no_shear


    outputfilename = outputfilename + '_lmstelmin_%2.2f_lmstelmax_%2.2f'%(args.logmstelmin, args.logmstelmax)

    if args.no_shape_noise:
        outputfilename = outputfilename + '_no_shape_noise'
    else:
        outputfilename = outputfilename + '_with_shape_noise'
        if args.rot90:
            outputfilename = outputfilename + '_with_90_rotation'

    if args.no_shear:
        outputfilename = outputfilename + '_no_shear'
    if args.test_case:
        outputfilename = outputfilename + '_test_case'
    
    np.random.seed(args.seed)
    outputfilename = outputfilename + '_w_jacks'
    print(config)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size    


    for jk in range(config['lens']['Njacks']):
        if jk%size !=rank:
            continue
        output_filename = outputfilename + '_jk_%d'%jk
        run_pipe(config, outputfilename = output_filename, jksamp=jk)           

    comm.Barrier()






