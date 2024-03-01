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
from welford import Welford


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

    
    c_sra_lra = np.cos(sra)*np.cos(lra) + np.sin(lra)*np.sin(sra)
    s_sra_lra = np.sin(sra)*np.cos(lra) - np.cos(sra)*np.sin(lra)

    c_theta = np.cos(ldec)*np.cos(sdec)*c_sra_lra + np.sin(ldec)*np.sin(sdec)
    s_theta = np.sqrt(1-c_theta**2)

    c_phi   =  np.cos(ldec)*s_sra_lra*1.0/s_theta
    s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*c_sra_lra*np.sin(sdec))*1.0/s_theta

    #phi     = 2*np.pi*np.random.uniform(size=len(sra))
    #c_phi   = np.cos(phi)
    #s_phi   = np.sin(phi)

    # tangential shear
    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)
    e_x     =  se1*(2*c_phi * s_phi) - se2*(2*c_phi**2 -1)

    return e_t, e_x

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
    #annulus aperture
    sep =  ((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2)**0.5
    idx =   (sep < dismax)
    sra  = sra[idx]
    sdec = sdec[idx]

    # putting the interpolation for source redshift assignment
    szred   =   interp_szred(np.random.uniform(size=len(sra)))
    se1     =   np.random.normal(0.0, sigell, len(sra)) 
    se2     =   np.random.normal(0.0, sigell, len(sra))
    wgal    =   sra/sra
    return sra, sdec, szred, wgal, se1, se2


def run_pipe(config, outputfilename = 'gamma.dat', outputpairfile=None):
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
    
    Njacks = int(lensargs['Njacks'])
    sumdgammat_num              = np.zeros(nbins*Njacks)
    sumdgammat_inp_num          = np.zeros(nbins*Njacks)
    sumdgammat_inp_bary_num     = np.zeros(nbins*Njacks)
    sumdgammat_inp_dm_num       = np.zeros(nbins*Njacks)
    sumdgammatsq_num            = np.zeros(nbins*Njacks)
    sumdgammax_num              = np.zeros(nbins*Njacks) 
    sumdgammaxsq_num            = np.zeros(nbins*Njacks)
    sumdwls                     = np.zeros(nbins*Njacks)


    # getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    if sum(np.isnan(lwgt))>0:
        idx = np.isnan(lwgt)
        print(lra[idx], lwgt[idx])
        print(lwgt) 
        print("lense data is screwed!!")

        exit()





    lra     = 130 + 0.0*lra
    ldec    = 0.0 + 0.0*ldec


    lzredmax = np.max(lzred)
    lconc = 0.0*lid
    xx = np.linspace(9,16,100)
    yy = 0.0*xx
    med_lzred = np.median(lzred)

    for kk, mh in enumerate(10**xx):
        yy[kk]    = concentration.concentration(mh, '200m', med_lzred, model = 'diemer19')
    
    spl_c_mh = interp1d(xx,yy)
    lconc = spl_c_mh(llogmh)
 
    print("lens data read fully")
    if config['test_case']:
        llogmh      = 14.0  + 0.0*llogmh
        lzred       = 0.3   + 0.0*lzred
        lconc       = 5.5   + 0.0*lzred
        llogmstel   = 12.0  + 0.0*llogmh

 
    #variables defs for welford approx sigma calculations
    M2 = 0.0
    mean = 0
    count = 0
    
#    np.random.seed(444)
    weldict = {}
    weldictx = {}

    dismax = config['Rmax']/ss.Astropy_cosmo.angular_diameter_distance(np.min(lzred)).value 
    #dismax = config['Rmax']/ss.Astropy_cosmo.comoving_distance(np.min(lzred)).value 
    
    if outputpairfile != None:
        fpairout = open(outputpairfile, "w")
        fpairout.write('jkid\tlra(deg)\tldec(deg)\tlzred\tllogmstel\tllogmh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\twls\tkappa\n')

    #..................................#
    for ii in tqdm(range(len(lra))):
        #dismax = config['Rmax']/ss.Astropy_cosmo.comoving_distance(lzred[ii]).value 
        # fixing the simulation aperture
        sra, sdec, szred, wgal, se1, se2 = create_sources(lra[ii], ldec[ii], dismax, nsrc=sourceargs['nsrc'], sigell=sourceargs['sigell']) 

        if sum(np.isnan(wgal))>0:
            print(wgal)
            print("source data is screwed!!")
            exit()


        if config['test_case']:
            szred = 0.8 + 0.0*sra
        if sourceargs['rot90']:
            se1 = -1*se1
            se2 = -1*se2
        if sourceargs['no_shape_noise']:
            se1 = 0.0*se1
            se2 = 0.0*se2
            


        print("number of sources: ", len(sra))
        # selecting cleaner background
        scut    = (szred>(lzredmax + sourceargs['zdiff'])) # zdiff cut
        if sum(scut)==0:
            continue
 
        sra         = sra[scut]  
        sdec        = sdec[scut]
        szred       = szred[scut]
        wgal        = wgal[scut]
        intse1      = se1[scut]
        intse2      = se2[scut]

        # add a section of stellar and dark matter

        se1, se2, etan, kappa, proj_sep, sflag, etan_b, etan_dm = ss.shear_src(lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogmh[ii], lconc[ii], sra, sdec, szred, intse1, intse2)
        
        if sourceargs['no_shear']:
            se1 = intse1; se2 = intse2
        

        et, ex  = get_et_ex(lra = lra[ii], ldec = ldec[ii], sra = sra, sdec = sdec, se1 = se1,  se2 = se2)

        sl_sep  = proj_sep
        w_ls    = lwgt[ii]*wgal
        #cure the arrays a bin
        idx = (sl_sep>rmin) & (sl_sep<rmax) & (sflag==1)
        if sum(idx)==0.0:
            continue
        sl_sep      = sl_sep[idx]
        w_ls        = w_ls[idx]
        et          = et[idx]
        etan        = etan[idx]   
        kappa       = kappa[idx]   
        etan_b      = etan_b[idx]
        etan_dm     = etan_dm[idx]
        ex          = ex[idx]  
        se1         = se1[idx]
        se2         = se2[idx]
        sra         = sra[idx]
        sdec        = sdec[idx]
        szred       = szred[idx]


        if outputpairfile != None:
            for jj in range(sum(idx)):
                fpairout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lxjkreg[ii], lra[ii], ldec[ii], lzred[ii], llogmstel[ii], llogmh[ii], lconc[ii], sra[jj], sdec[jj], szred[jj], se1[jj], se2[jj], etan[jj], et[jj], ex[jj], sl_sep[jj], w_ls[jj], kappa[jj]))


        #exit()
        slrbins = np.log10(sl_sep*1.0/rmin)//rdiff

        for jk in range(Njacks):
            if lxjkreg[ii] == jk:
                continue

            for rb in range(nbins):
                idx = slrbins==rb
                if sum(idx)==0:
                    continue
                try:
                    weldict[jk*nbins + rb].add_all(np.array(w_ls * et)[idx])
                    weldictx[jk*nbins + rb].add_all(np.array(w_ls * ex)[idx])
                except:
                    weldict[jk*nbins + rb]  = Welford(np.array(w_ls * et)[idx])
                    weldictx[jk*nbins + rb] = Welford(np.array(w_ls * ex)[idx])
                    
                sumdgammat_num          [jk*nbins + rb] +=sum((w_ls * et)[idx])
                sumdgammat_inp_num      [jk*nbins + rb] +=sum((w_ls * etan)[idx])
                sumdgammat_inp_bary_num [jk*nbins + rb] +=sum((w_ls * etan_b)[idx])
                sumdgammat_inp_dm_num   [jk*nbins + rb] +=sum((w_ls * etan_dm)[idx])
                sumdgammatsq_num        [jk*nbins + rb] +=sum(((w_ls* et)**2)[idx])
                sumdgammax_num          [jk*nbins + rb] +=sum((w_ls * ex)[idx])
                sumdgammaxsq_num        [jk*nbins + rb] +=sum(((w_ls* ex)**2)[idx])
                sumdwls                 [jk*nbins + rb] +=sum(w_ls[idx])

    

    if outputpairfile != None:
        fpairout.write("#OK")
        fpairout.close()

    fout = open(outputfilename, "w")
    fout.write("# 0:rmin/2+rmax/2 1:gammat 2:gammatsq 3:SN_Errgammat 4:gammax 5:gammaxsq 6:SN_Errgammax 7:truegamma 8:gammat_inp 9:gammat_inp_bary 10:gammat_inp_dm 11:sumd_wls 12:welford_gammat_mean 13:welford_gammat_std 14:welford_counts 15:welford_gammax_mean 16:welford_gammax_std 17:Jkid \n")
    for jk in range(Njacks):
        for i in range(nbins):
            rrmin = rbins[i]
            rrmax = rbins[i+1]
            if np.isnan(sumdwls[jk*nbins + i]):
                print('error', jk, i, sumdwls[jk*nbins + i])
                exit()
           #Resp = sumdwls_resp[i]*1.0/sumdwls[i]
            try:
                fout.write("%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdgammat_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammatsq_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], np.sqrt(sumdgammatsq_num[jk*nbins + i])*1.0/sumdwls[jk*nbins + i], sumdgammax_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammaxsq_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], np.sqrt(sumdgammaxsq_num[jk*nbins + i])*1.0/sumdwls[jk*nbins + i], sumdgammat_inp_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammat_inp_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdgammat_inp_bary_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdgammat_inp_dm_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdwls[jk*nbins + i], weldict[jk*nbins + i].mean, weldict[jk*nbins + i].var_p**0.5, weldict[jk*nbins + i].count, weldictx[jk*nbins + i].mean, weldictx[jk*nbins + i].var_p**0.5, jk)    )
            
            except KeyError:
                fout.write("%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdgammat_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammatsq_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], np.sqrt(sumdgammatsq_num[jk*nbins + i])*1.0/sumdwls[jk*nbins + i], sumdgammax_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammaxsq_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], np.sqrt(sumdgammaxsq_num[jk*nbins + i])*1.0/sumdwls[jk*nbins + i], sumdgammat_inp_num[jk*nbins + i]*1.0/sumdwls[jk*nbins + i], sumdgammat_inp_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdgammat_inp_bary_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdgammat_inp_dm_num[jk*nbins + i]/sumdwls[jk*nbins + i], sumdwls[jk*nbins + i], -999, -999, -999, -999, -999, jk))


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
    parser.add_argument("--test_case", help="testing the ideal case", type=bool, default=False)
    parser.add_argument("--rot90", help="rotating intrinsic shapes by 90 degrees", type=bool, default=False)
    parser.add_argument("--logmstelmin", help="log stellar mass minimum-lense selection", type=float, default=11.0)
    parser.add_argument("--logmstelmax", help="log stellar mass maximum-lense selection", type=float, default=13.0)
    parser.add_argument("--ten_percent", help="using ten percent of the lense sample", type=bool, default=False)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)


    config['lens']['ten_percent'] = args.ten_percent
    
    if args.ten_percent:
        config["outputdir"] = config["outputdir"] + "_ten_percent"



    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)

    outputfilename = '%s/simed_sources.dat'%(config['outputdir'])

    if 'logmstelmin'not in config:
        config['lens']['logmstelmin'] = args.logmstelmin
    if 'logmstelmax'not in config:
        config['lens']['logmstelmax'] = args.logmstelmax
        
    config['source']['rot90'] = args.rot90
    config['source']['no_shape_noise'] = args.no_shape_noise
    config['source']['no_shear'] = args.no_shear

    config['test_case'] = args.test_case

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
    #run_pipe(config, outputfile = outputfilename, outputpairfile = outputfilename + '_pairs')           
    run_pipe(config, outputfilename = outputfilename)           

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
        #    sumdwls[rb]                      += w_ls

