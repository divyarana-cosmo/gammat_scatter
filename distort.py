# check all the input equations to be sure of the computations
# Please be cautions of all the units used in angles, distances and masses
# start with only circular source galaxies with all same sizes
# put in the intrinsic galaxy shapes -- need some galsim gimicks
# make it config based
# the psf of Euclid part -- airy disk or check the preparation paper
# have to implement smhm properly with a scatter
# put the argsparse in it

# remember to clip the sin and cos to -1,1

import numpy as np
import matplotlib.pyplot as plt
import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
import argparse
import yaml

class simshear():
    "simulated the shear for a given configuration of dark matter and stellar profiles"
    def __init__(self, H0, Om0, Ob0, Tcmb0, Neff, sigma8, ns, log_mstel, log_mh, lra, ldec, lzred):
    #def __init__(self, H0 = 100, Om0 = 0.3, Ob0 = 0.0457, Tcmb0 = 2.7255, Neff = 3.046, sigma8=0.82, ns=0.96, log_mstel, log_mh, lra, ldec, lzred):
        "initialize the parameters"
        #fixing the cosmology
        params = dict(H0 = H0, Om0 = Om0, Ob0 = Ob0, Tcmb0 = Tcmb0, Neff = Neff)
        self.cc = FlatLambdaCDM(**params)

        self.log_mstel  = log_mstel # total stellar pointmass
        self.log_mh     = log_mh    # total mass of the halo
        self.omg_m      = Om0
        self.lra        = lra
        self.ldec       = ldec
        self.lzred      = lzred

        # we are using diemer19 cocentration-mass relation
        colossus_cosmo = cosmology.fromAstropy(self.cc, sigma8 = sigma8, ns = ns, cosmo_name='my_cosmo')
        self.conc = concentration.concentration(10**self.log_mh, '200m', self.lzred, model = 'diemer19')

        print("Intialing  parameters\n omgM = %s\n log(Mstel / [h-1 Msun]) = %s\n log(M200m / [h-1 Msun]) = %s\n conc = %s"%(self.omg_m, self.log_mstel, self.log_mh, self.conc))
        self.hp         = halo(self.log_mh, self.conc, omg_m=self.omg_m)
        self.stel       = stellar(self.log_mstel)


    def get_sigma_crit_inv(self, lzred, szred):
        "evaluates the lensing efficency geometrical factor"
        if lzred>szred:
            return 0.0

        # some important constants for the sigma crit computations
        gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
        cee = 3e5 #km s^-1
        # sigma_crit_calculations for a given lense-source pair
        sigm_crit_inv = self.cc.angular_diameter_distance(lzred).value * self.cc.angular_diameter_distance_z1z2(lzred, szred).value * (1.0 + lzred)**2 * 1.0/self.cc.angular_diameter_distance(szred).value
        sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
        #print(4*np.pi*gee*1.0/cee**2)
        #sigm_crit_inv = 1e12*sigm_crit_inv #esd'-]s are in pc not in Mpc
        return sigm_crit_inv

    def get_g(self, sra, sdec, lzred, szred):
        "computes the g1 and g2 components for the reduced shear"
        # need to supply the angles in degrees
        lra  = self.lra*np.pi/180
        ldec = self.ldec*np.pi/180
        sra  = sra*np.pi/180
        sdec = sdec*np.pi/180

        c_theta = np.clip(np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec), -1, 1)
        s_theta = np.sqrt(1-c_theta**2)

        #projected separation on the lense plane
        proj_sep = self.cc.comoving_distance(lzred).value * s_theta/c_theta # in h-1 Mpc

        #considering only tangential shear and adding both contributions
        gamma = (self.hp.esd_nfw(proj_sep) + self.stel.esd_pointmass(proj_sep))*self.get_sigma_crit_inv(lzred, szred)
        kappa = (self.hp.sigma_nfw(proj_sep) + self.stel.sigma_pointmass(proj_sep))*self.get_sigma_crit_inv(lzred, szred)

        g = gamma/(1.0 - kappa)

        sflag = (s_theta!=0) & (np.abs(kappa)<1.0)   #strong lensing flag and proximity flag

        # phi to get the compute the tangential shear
        c_phi   = np.clip( np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta, -1, 1 )
        s_phi   = np.clip((-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta, -1, 1)


        # tangential shear
        g_1     = - g*(2*c_phi**2 - 1)
        g_2     = - g*(2*c_phi * s_phi)

        return g_1, g_2, g, c_phi, s_phi, proj_sep, sflag


    def shear_src(self, sra, sdec, se1, se2, lzred, szred):
        "apply shear on to the source galaxies with given intrinsic shapes"
        g_1, g_2, etan, c_phi, s_phi, proj_sep, sflag = self.get_g(sra, sdec, lzred, szred)

       #weeding out to the strong lensing systems

        g   = g_1 + 1j* g_2
        es  = se1 + 1j* se2  # intrinsic sizes

        e   = 0.0*es # sheared shapes
        #using the seitz and schnider 1995 formalism to shear the galaxy
        idx = np.abs(g)<1
        e[idx] = (es[idx] + g[idx])/(1.0 + np.conj(g[idx])*es[idx])
        e[~idx] = (1 + g[~idx]*np.conj(es[~idx]))/(np.conj(es[~idx]) + np.conj(g[~idx])) # mod(g)>1
        return np.real(e), np.imag(e), etan, proj_sep, sflag





def get_et_ex(lra, ldec, sra, sdec, se1, se2):
    "measures the etan and ecross for a given  lens-source pair"
    lra  = lra*np.pi/180
    ldec = ldec*np.pi/180
    sra  = sra*np.pi/180
    sdec = sdec*np.pi/180

    c_theta = np.clip(np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec), -1, 1)
    s_theta = np.sqrt(1-c_theta**2)

    # phi to get the compute the tangential shear
    c_phi   = np.clip(np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta, -1, 1)
    s_phi   = np.clip((-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta, -1, 1)
    # tangential shear
    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)
    e_x     =  se1*(2*c_phi * s_phi) - se2*(2*c_phi**2 -1)

    return e_t, e_x




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    #parser.add_argument("--log_mh", help="dark matter halo mass", type=float, default=12.0)
    parser.add_argument("--seed", help="seed for sampling the source intrinsic shapes", type=int, default=123)
    parser.add_argument("--sig0", help="scatter in stellar or halo mass", type=float, default=0.2)
    parser.add_argument("--scatter_stel", help="scatter stellar mass", type=bool, default=False)
    parser.add_argument("--scatter_halo", help="scatter halo mass", type=bool, default=False)
    parser.add_argument("--no_shape_noise", help="scatter halo mass", type=bool, default=False)


    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    print(config)

    #make the directory for the output
    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)

    if  "seed" not in config:
        config["seed"] = args.seed
    np.random.seed(config['seed']) #fixing the seed for reproducibility

    nlens = 1000
    logscat = np.random.normal(0.0,args.sig0, nlens)

    if args.scatter_stel:
        config['lens']['log_mstel'] +=logscat
        config['lens']['log_mh'] =config['lens']['log_mh'] + 0.0*logscat
        outputfilename = '%s/simed_sources_scatter_stel_sig0_%s.dat'%(config['outputdir'], args.sig0)
    if args.scatter_halo:
        config['lens']['log_mh'] +=logscat
        config['lens']['log_mstel'] =config['lens']['log_mstel'] + 0.0*logscat
        outputfilename = '%s/simed_sources_scatter_halo_sig0_%s.dat'%(config['outputdir'], args.sig0)

    if args.no_shape_noise:
        outputfilename = outputfilename + 'no_shape_noise'
    else:
        outputfilename = outputfilename + 'with_shape_noise'

    #config['lens']['log_mh'] = args.log_mh

    logmstelarr = config['lens']['log_mstel']
    logmharr    = config['lens']['log_mh']
    # lets say lens are in gama like survey
    lra = config['lens']['lra']
    ldec = config['lens']['ldec']
    lzred = config['lens']['lzred']

    #fixing the simulation aperture
    cc      = FlatLambdaCDM(H0=100, Om0 = config['Om0'])
    thetamax = config['lens']['Rmax']/cc.comoving_distance(lzred).value * 180/np.pi

    fdata = open(outputfilename,'w')
    fdata.write('lra(deg)\tldec(deg)\tlzred\tllog_mstel\tllog_mh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\tex_obs\tproj_sep\n')

    for ii in tqdm(range(len(logmharr))):
        ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'], log_mstel = logmstelarr[ii], log_mh = logmharr[ii], lra = config['lens']['lra'], ldec = config['lens']['ldec'], lzred = config['lens']['lzred'])

        # sampling 1000 sources with in the 1 h-1 Mpc comoving separation
        nsrcs   = int(1e4)
        cdec    = np.random.uniform(np.cos((90+thetamax)*np.pi/180), np.cos((90-thetamax)*np.pi/180), nsrcs) # uniform over the sphere
        sdec    = (90.0-np.arccos(cdec)*180/np.pi)
        sra     = lra + np.random.uniform(-thetamax, thetamax, nsrcs)
        szred = 0.8
        # intrinsic shapes
        if args.no_shape_noise:
            se = 0.0*np.random.normal(0.0, 0.27, int(2*len(sra))).reshape((-1,2))
        else:
            se = np.random.normal(0.0, 0.27, int(2*len(sra))).reshape((-1,2))

        se1 = se[:,0]
        se2 = se[:,1]

        s1, s2, etan, proj_sep, sflag = ss.shear_src(sra, sdec, se1, se2, lzred, szred)
        et, ex = get_et_ex(lra, ldec, sra, sdec, s1, s2)

        for jj in range(len(sra)):
            #weeding out to the strong lensing systems
            if sflag[jj] !=1.0:
                continue

            fdata.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lra, ldec, lzred, logmstelarr[ii], logmharr[ii], ss.conc, sra[jj], sdec[jj], szred, s1[jj], s2[jj], etan[jj], et[jj], ex[jj], proj_sep[jj]))


    fdata.close()


    #df = {'lra(deg)': lra, 'ldec(deg)': ldec, 'lzred': lzred*np.ones(len(sra)) , 'log_mstel': config['lens']['log_mstel']*np.ones(len(sra)), 'log_mh': config['lens']['log_mh']*np.ones(len(sra)), 'conc': ss.conc*np.ones(len(sra)), 'sra(deg)': sra, 'sdec(deg)': sdec, 'szred': szred, 'se1': se1, 'se2': se2, 'etan': etan, 'etan_obs': et, 'ex_obs': ex, 'proj_sep': proj_sep}

    #import pandas as pd
    #df = pd.DataFrame(df)


    #fname = '%s/simed_sources_logmh_%s.dat'%(config['outputdir'], config['lens']['log_mh'])
    #df.to_csv(fname, index=0, sep=' ')

    #fdata = open(fname,'w')
    #fdata.write('\tldec(deg)\tlzred\tllog_mstel\tllog_mh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\tetan\tetan_obs\t ex_obs\n')
    #from tqdm import tqdm
    #for ii in tqdm(range(len(sra))):
    #    s1, s2, etan = ss.shear_src(sra[ii], sdec[ii], se1[ii], se2[ii], lzred, szred)
    #    if np.isnan(s1):
    #        continue
    #    et, ex = get_et_ex(lra, ldec, sra[ii], sdec[ii], s1, s2)

    #    fdata.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lra, ldec, lzred, config['lens']['log_mstel'], config['lens']['log_mh'], ss.conc, sra[ii], sdec[ii], szred, s1, s2, etan, et, ex))

    #fdata.close()



