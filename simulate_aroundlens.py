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



def create_sources(ramin, ramax, thetamin, thetamax, sigell=0.27, size=100000, mask=None): #mask application for future
    cdec    = np.random.uniform(np.cos(thetamax), np.cos(thetamin), size=size)     
    sdec    = (90.0 - np.arccos(cdec)*180/np.pi)
    sra     = np.random.uniform(ramin, ramax, size=size)*180/np.pi

    # putting the interpolation for source redshift assignment
    interp_szred = get_interp_szred()
    szred   =   interp_szred(np.random.uniform(size=size))
    se1     =   np.random.normal(0.0, sigell, size) 
    se2     =   np.random.normal(0.0, sigell, size)
    wgal    =   sra/sra
    return np.transpose([sra, sdec, szred, wgal, se1, se2])




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


    # getting the lenses data
    lid, lra, ldec, lzred, lwgt, llogmstel, llogmh, lxjkreg   = lens_select(lensargs)
    lconc = 0.0*lid
    for kk, mh in enumerate(10**llogmh):
        lconc[kk]    = concentration.concentration(mh, '200m', lzred[kk], model = 'diemer19')


    # initializing arrays for signal compuations
    sumdgammat_num      = np.zeros(len(rbins[:-1]))
    sumdgammat_inp_num  = np.zeros(len(rbins[:-1]))
    sumdgammatsq_num    = np.zeros(len(rbins[:-1]))
    sumdgammax_num      = np.zeros(len(rbins[:-1]))
    sumdgammaxsq_num    = np.zeros(len(rbins[:-1]))
    sumwls              = np.zeros(len(rbins[:-1]))


    # convert lense ra and dec into x,y,z cartesian coordinates
    lx, ly, lz = get_xyz(lra, ldec)

    # putting kd tree around the lenses
    lens_tree = cKDTree(np.array([lx, ly, lz]).T)


    print('lenses tree is ready\n')

    # setting maximum search radius
    dcommin = ss.Astropy_cosmo.comoving_distance(np.min(lzred)).value
    dismax  = (rmax*1.0/(dcommin))
    print(dismax)

    # remember to provide angles in degrees
    thetamax = (90 - np.min(ldec))* np.pi/180
    thetamin = (90 - np.max(ldec))* np.pi/180
    ramin   = np.min(lra)*np.pi/180
    ramax   = np.max(lra)*np.pi/180
    area    = (ramax - ramin) * (np.cos(thetamin) - np.cos(thetamax))* (180*60/np.pi)**2
    nsrc = sourceargs['nsrc']
    Ngal    = round(nsrc * area) # area of square in strradians --> arcmin^2

    # Ready to pounce on the source data
    print("%s like sources created with sources:"%sourceargs['type'], Ngal)
    # various columns in sources
    # sra, sdec, szred, wgal, se1, se2
    # looping over all the galaxies
    
    Nchunks = 10000
    print('chunksize: ', int(Ngal/Nchunks))
    chkbinedgs = np.linspace(1, Ngal, Nchunks+1, endpoint=True)
    for cc in range(Nchunks):
        chunksize = int(chkbinedgs[cc+1]) - int(chkbinedgs[cc])
        datagal = create_sources(ramin, ramax, thetamin, thetamax, sigell=sourceargs['sigell'], size=chunksize)
        #print(igal)
        # first two entries are ra and dec for the sources
        allragal  = datagal[:,0]
        alldecgal = datagal[:,1]
        # ra and dec to x,y,z for sources
        allsx, allsy, allsz = get_xyz(allragal, alldecgal)
        # query in a ball around individual sources and collect the lenses ids with a maximum radius
        slidx = lens_tree.query_ball_point(np.transpose([allsx, allsy, allsz]), dismax)
 
        
        for igal in range(chunksize):
            ragal    = datagal[igal, 0]
            decgal   = datagal[igal, 1]
            zphotgal = datagal[igal, 2]
            wgal     = datagal[igal, 3]
            e1gal    = datagal[igal, 4]
            e2gal    = datagal[igal, 5]

            lidx = np.array(slidx[igal])
            # removing sources which doesn't have any lenses around them
            if len(lidx)==0:
                continue

            # selecting a cleaner background
            zcut = (lzred[lidx] < (zphotgal - zdiff)) #only taking the foreground lenses
            # again skipping the onces which doesn't satisfy the above criteria
            if np.sum(zcut)==0.0:
                continue
            # collecting the  data of lenses around individual source
            lidx   = lidx[zcut] # this will catch the array indices for our lenses
            sra    = ragal
            sdec   = decgal
            
            #lid, lra, ldec, lzred, lwgt, logmstel, logmh, xjkreg 
            
            l_id        = lid[lidx]
            l_ra        = lra[lidx]
            l_dec       = ldec[lidx]
            l_zred      = lzred[lidx]
            l_wgt       = lwgt[lidx]
            l_logmstel  = llogmstel[lidx]
            l_logmh     = llogmh[lidx]
            l_conc      = lconc[lidx]
            l_xjkreg    = lxjkreg[lidx]
            
            e1gal, e2gal, etan, proj_sep, sflag = ss.shear_src(l_ra, l_dec, l_zred, l_logmstel, l_logmh, l_conc, ragal, decgal, zphotgal, e1gal, e2gal)

            # getting the radial separations for a lense source pair
            sl_sep = proj_sep 
            for ll,sep in enumerate(sl_sep):
                if sep<rmin or sep>rmax or sflag[ll]==0:
                    continue
                rb = int(np.log10(sep*1.0/rmin)*1/rdiff)

                # get tangantial components given positions and shapes
                e_t, e_x = get_et_ex(lra = l_ra[ll], ldec = l_dec[ll], sra = sra, sdec = sdec, se1 = e1gal[ll],  se2 = e2gal[ll])

                # following equations given in the surhud's lectures
                w_ls    = l_wgt[ll] * wgal 

                # separate numerator and denominator computation
                sumdgammat_num[rb]       += w_ls  * e_t
                sumdgammat_inp_num[rb]   += w_ls  * etan[ll]
                sumdgammatsq_num[rb]     += (w_ls * e_t)**2
                sumdgammax_num[rb]       += w_ls  * e_x
                sumdgammaxsq_num[rb]     += (w_ls * e_x)**2
                sumwls[rb]               += w_ls
            

       
        print('done with chunk no-', cc)
        #exit()
    fout = open(outputfile, "w")
    fout.write("# 0:rmin/2+rmax/2 1:gammat 2:SN_Errgammat 3:gammax 4:SN_Errgammax 5:truegamma\n")
    for i in range(len(rbins[:-1])):
        rrmin = rbins[i]
        rrmax = rbins[i+1]
       #Resp = sumwls_resp[i]*1.0/sumwls[i]

        fout.write("%le\t%le\t%le\t%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdgammat_num[i]*1.0/sumwls[i], np.sqrt(sumdgammatsq_num[i])*1.0/sumwls[i], sumdgammax_num[i]*1.0/sumwls[i], np.sqrt(sumdgammaxsq_num[i])*1.0/sumwls[i], sumdgammat_inp_num[i]*1.0/sumwls[i]))
        #fout.write("%le\t%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumdsig_num[i]*1.0/sumwls[i]/2./Resp, np.sqrt(sumdsigsq_num[i])*1.0/sumwls[i]/2./Resp))
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

