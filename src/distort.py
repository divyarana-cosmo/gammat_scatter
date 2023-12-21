# have to add the responsivity part
# the psf of Euclid part -- airy disk or check the preparation paper


import numpy as np
import matplotlib.pyplot as plt
#import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d
from get_data import lens_select
from tqdm import tqdm
import argparse
import yaml
from mpi4py import MPI
from subprocess import  call

class simshear():
    "simulated the shear for a given configuration of dark matter and stellar profiles"
    def __init__(self, H0, Om0, Ob0, Tcmb0, Neff, sigma8, ns, lzredmin=0.0, lzredmax=1.0, szredmax=4.0):
        "initialize the parameters"
        #fixing the cosmology
        self.omg_m = Om0
        params = dict(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff)
        self.Astropy_cosmo = FlatLambdaCDM(**params)
        colossus_cosmo = cosmology.fromAstropy(self.Astropy_cosmo, sigma8 = sigma8, ns = ns, cosmo_name='my_cosmo')
        self.sigma8 = sigma8
        self.ns = ns
        self.cosmo_name='my_cosmo'
        print("fixing cosmology \n")

    def get_xyz(self, ra,dec):
        theta = (90-dec)*np.pi/180
        phi = ra*np.pi/180
        z = np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        x = np.cos(phi)*np.sin(theta)
        return x,y,z  

    def _get_sigma_crit_inv(self, lzred, szred):
        "evaluates the lensing efficency geometrical factor"
        sigm_crit_inv = 0.0*szred + 0.0*lzred
        idx =  szred>lzred   # if sources are in foreground then lensing is zero
        if np.isscalar(idx):
            lzred = np.array([lzred])
            szred = np.array([szred])
            idx = np.array([idx])
            sigm_crit_inv = np.array([sigm_crit_inv])

        # some important constants for the sigma crit computations
        gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
        cee = 3e5 #km s^-1
        # sigma_crit_calculations for a given lense-source pair
        #sigm_crit_inv[idx] = self.Astropy_cosmo.angular_diameter_distance(lzred).value * self.Astropy_cosmo.angular_diameter_distance_z1z2(lzred, szred[idx]).value * (1.0 + lzred)**2 * 1.0/self.Astropy_cosmo.angular_diameter_distance(szred[idx]).value
        sigm_crit_inv = self.Astropy_cosmo.angular_diameter_distance(lzred).value * self.Astropy_cosmo.angular_diameter_distance_z1z2(lzred, szred).value * (1.0 + lzred)**2 * 1.0/self.Astropy_cosmo.angular_diameter_distance(szred).value
        sigm_crit_inv[~idx]=0.0 
        sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
        return sigm_crit_inv


    def _interp_get_sigma_crit_inv(self, lzred):
        xx = np.linspace(lzred+1e-4,4.0, 150)
        yy = np.log10(self._get_sigma_crit_inv(lzred, xx))
        return interp1d(xx, yy, kind='cubic')


    def _get_g(self,logmstel, logmh, lconc, lzred, szred, proj_sep):
        self.interp_get_sigma_crit_inv = self._interp_get_sigma_crit_inv(lzred)

        #colossus_cosmo  = cosmology.fromAstropy(self.Astropy_cosmo, sigma8 = self.sigma8, ns = self.ns, cosmo_name=self.cosmo_name)
        self.conc       = lconc#concentration.concentration(10**logmh, '200m', lzred, model = 'diemer19')
        self.hp         = halo(logmh, self.conc, omg_m=self.omg_m)
        self.stel       = stellar(logmstel)
        get_sigma_crit_inv =10**self.interp_get_sigma_crit_inv(szred) 

        #considering only tangential shear and adding both contributions
        gamma_s     = (self.stel.esd_pointmass(proj_sep))   * get_sigma_crit_inv
        gamma_dm    = (self.hp.esd_nfw(proj_sep))           * get_sigma_crit_inv
        kappa_s     = (self.stel.sigma_pointmass(proj_sep)) * get_sigma_crit_inv
        kappa_dm    = (self.hp.sigma_nfw(proj_sep))         * get_sigma_crit_inv
        return gamma_s, gamma_dm, kappa_s, kappa_dm

    def get_g(self, lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred):
        "computes the g1 and g2 components for the reduced shear"
        lx, ly, lz = self.get_xyz(lra, ldec) 
        sx, sy, sz = self.get_xyz(sra, sdec) 

        #projected separation on the lense plane
        proj_sep = self.Astropy_cosmo.comoving_distance(lzred).value * np.sqrt((sx-lx)**2 + (sy-ly)**2 + (sz-lz)**2) # in h-1 Mpc

        #considering only tangential shear and adding both contributions
        gamma_s, gamma_dm, kappa_s, kappa_dm = self._get_g(logmstel, logmh, lconc, lzred, szred, proj_sep)
        gamma = gamma_s + gamma_dm
        kappa = kappa_s + kappa_dm

        g    = gamma/(1.0 - kappa) # reduced shear
        g_b  = gamma_s/(1.0 - kappa_s) # reduced shear
        g_dm = gamma_dm/(1.0 - kappa_dm) # reduced shear

        # phi to get the compute the tangential shear

        lra  = lra*np.pi/180
        ldec = ldec*np.pi/180
        sra  = sra*np.pi/180
        sdec = sdec*np.pi/180

        c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
        #if sum(c_theta>1)>0 or sum(c_theta<-1)>0:
        #    print('trigon screwed')
        #    print(c_theta)
        #    exit()
        s_theta = np.sqrt(1-c_theta**2)

        sflag = (s_theta!=0) & (np.abs(kappa)<0.5)   #weak lensing flag and proximity flag

        c_phi   =  np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
        s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta
        
        # tangential shear
        g_1     = - g*(2*c_phi**2 - 1)
        g_2     = - g*(2*c_phi * s_phi)
        
        return g_1, g_2, g, c_phi, s_phi, proj_sep, sflag, g_b, g_dm


    def shear_src(self, lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred, se1, se2):
        "apply shear on to the source galaxies with given intrinsic shapes"
        g_1, g_2, etan, c_phi, s_phi, proj_sep, sflag, g_b, g_dm = self.get_g(lra, ldec, lzred, logmstel, logmh, lconc, sra, sdec, szred)
        g   = g_1 + 1j* g_2
        es  = se1 + 1j* se2 + 0.0*g  # intrinsic sizes
        e   = 0.0*es # sheared shapes
        #using the seitz and schnider 1995 formalism to shear the galaxy
        idx = np.abs(g)<1
        e[idx] = (es[idx] + g[idx])/(1.0 + np.conj(g[idx])*es[idx])
        e[~idx] = (1 + g[~idx]*np.conj(es[~idx]))/(np.conj(es[~idx]) + np.conj(g[~idx])) # mod(g)>1
        return np.real(e), np.imag(e), etan, proj_sep, sflag, g_b, g_dm





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

def getszred():
    "assigns redshifts respecting the distribution"
    n0=1.8048
    a=0.417
    b=4.8685
    c=0.7841
    f = lambda zred: n0*(zred**a + zred**(a*b))/(zred**b + c)
    zmin = 0.0
    zmax = 2.5
    zarr = np.linspace(zmin, zmax, 20)
    xx  = 0.0 * zarr
    for ii in range(len(xx)):
        xx[ii] = quad(f, zmin, zarr[ii])[0]/quad(f, zmin, zmax)[0]
    proj = interp1d(xx,zarr)
    return proj






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
    parser.add_argument("--Njacks", help="number of jackknifes", type=int, default=30)


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
            
    #picking up the lens data
    lensargs = config['lens']
    sourceargs = config['source']

    lid, lra, ldec, lzred, logmstel, logmh, xjkreg   = lens_select(lensargs, Njacks=args.Njacks)


    np.random.seed(666)

    if args.ideal_case:
        logmstel = np.mean(logmstel) + np.random.normal(0,0.1, size=len(lra))
        logmh = np.mean(logmh) + 0.0*logmh
        outputfilename = outputfilename + '_ideal_case'

    # putting the interpolation for source redshift assignment
    interp_szred = getszred()



    #creating class instance
    ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'])

    fdata = open(outputfilename,'w')
    fdata.write('lid\txjkreg\tlra(deg)\tldec(deg)\tlzred\tllogmstel\tllogmh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\n')

    for ii in tqdm(range(len(lra))):

        #np.random.seed(666 + lid[ii])
        # fixing the simulation aperture
        cc      = FlatLambdaCDM(H0=100, Om0 = config['Om0'])
        thetamax = lensargs['Rmax']/cc.comoving_distance(lzred[ii]).value * 180/np.pi
        numbsrc = round(sourceargs['nsrc'] * (2*thetamax)**2*60**2)      # area of square in deg^2 --> arcmin^2
        print('number of sources: ', numbsrc)

        if numbsrc==0:
            continue
        cdec    = np.random.uniform(np.cos((90 - (ldec[ii] - thetamax))*np.pi/180), np.cos((90 - (ldec[ii] + thetamax))*np.pi/180), numbsrc) # uniform over the sphere
        sdec    = (90.0 - np.arccos(cdec)*180/np.pi)
        sra     = lra[ii] + np.random.uniform(-thetamax, thetamax, numbsrc)
        # selecting cleaner background
        szred = interp_szred(np.random.uniform(size=numbsrc))
        sra   = sra[  (szred>(lzred[ii] + sourceargs['zdiff']))]  
        sdec  = sdec[ (szred>(lzred[ii] + sourceargs['zdiff']))]
        szred = szred[(szred>(lzred[ii] + sourceargs['zdiff']))]

        # intrinsic shapes
        if args.no_shape_noise:
            se1 = 0.0*sra
            se2 = 0.0*sra
        else:
            se1 = np.random.normal(0.0, sourceargs['sige'], int(len(sra)))
            se2 = np.random.normal(0.0, sourceargs['sige'], int(len(sra)))
            if args.rot90:
                se1*=-1
                se2*=-1

               

        s1, s2, etan, proj_sep, sflag = ss.shear_src(lra[ii], ldec[ii], lzred[ii], logmstel[ii], logmh[ii], sra, sdec, szred, se1, se2)

        if len(s1)==0:
            continue

        if args.no_shear:
            s1 = se1
            s2 = se2
        et, ex = get_et_ex(lra[ii], ldec[ii], sra, sdec, s1, s2)

        for jj in range(len(sra)):
            if (sflag[jj]!=0) & (proj_sep[jj]<lensargs['Rmin']) & (proj_sep[jj]>lensargs['Rmax']):
                continue
            fdata.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lid[ii], xjkreg[ii], lra[ii], ldec[ii], lzred[ii], logmstel[ii], logmh[ii], ss.conc, sra[jj], sdec[jj], szred[jj], s1[jj], s2[jj], etan[jj], et[jj], ex[jj], proj_sep[jj]))

    fdata.close()


