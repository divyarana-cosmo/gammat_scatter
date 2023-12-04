# check all the input equations to be sure of the computations
# Please be cautions of all the units used in angles, distances and masses
# start with only circular source galaxies with all same sizes
# put in the intrinsic galaxy shapes -- need some galsim gimicks
# make it config based
# the psf of Euclid part -- airy disk or check the preparation paper
# have to implement smhm properly with a scatter
# put the argsparse in it


import numpy as np
import matplotlib.pyplot as plt
import galsim
from halopy import halo
from stellarpy import stellar
from colossus.cosmology import cosmology
from colossus.halo import concentration
from astropy.cosmology import FlatLambdaCDM

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

        print("Intialing  parameters\n omgM = %s\n log(mstel / [h-1 Msun]) = %s\n log(mh200m / [h-1 Msun]) = %s\n conc = %s"%(self.omg_m, self.log_mstel, self.log_mh, self.conc))
        self.hp         = halo(self.log_mh, self.conc, omg_m=self.omg_m)
        self.stel       = stellar(self.log_mstel)


    def get_sigma_crit_inv(self, lzred, szred):
        "evaluates the lensing efficency geometrical factor"
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

        c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
        s_theta = np.sqrt(1-c_theta**2)

        #projected separation on the lense plane
        proj_sep = self.cc.comoving_distance(lzred).value * s_theta/c_theta # in h-1 Mpc
        proj_sep = np.array([proj_sep])

        #considering only tangential shear and adding both contributions
        #hp = halo(log_mh, conc, omg_m=self.omg_m)
        #stel = stellar(log_mstel)
        gamma = (self.hp.esd_nfw(proj_sep) + self.stel.esd_pointmass(proj_sep))*self.get_sigma_crit_inv(lzred, szred)
        kappa = (self.hp.sigma_nfw(proj_sep) + self.stel.sigma_pointmass(proj_sep))*self.get_sigma_crit_inv(lzred, szred)

        g = gamma/(1.0 - kappa)

        # phi to get the compute the tangential shear
        c_phi   = np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
        s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta

        # tangential shear
        g_1     = - g*(2*c_phi**2 - 1)
        g_2     = - g*(2*c_phi * s_phi)
        return g_1, g_2



    def shear_src(self, sra, sdec, se1, se2, lzred, szred):
        "apply shear on to the source galaxies with given intrinsic shapes"
        g_1, g_2 = self.get_g(sra, sdec, lzred, szred)
        g   = complex(g_1,g_2)
        es  = complex(se1, se2)
        if np.abs(g)<1:
            e = (es + g)/(1.0 + np.conj(g)*es)
        if np.abs(g)>1:
            e = (1 + g*np.conj(es))/(np.conj(es) + np.conj(g))
        return np.real(e), np.imag(e)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--outdir", help="Output filename with pairs information", default="debug")
    parser.add_argument("--log_mh", help="dark matter halo mass", type=float, default=13.0)

    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    config['lens']['log_mh'] = args.log_mh
    print(config)
    ss = simshear(H0 = config['H0'], Om0 = config['Om0'], Ob0 = config['Ob0'], Tcmb0 = config['Tcmb0'], Neff = config['Neff'], sigma8 = config['sigma8'], ns = config['ns'], log_mstel = config['lens']['log_mstel'], log_mh = config['lens']['log_mh'], lra = config['lens']['lra'], ldec = config['lens']['ldec'], lzred = config['lens']['lzred'])

    # lets say lens are in gama like survey
    lra = config['lens']['lra']
    ldec = config['lens']['ldec']
    lzred = config['lens']['lzred']

    # sampling 1000 sources with in the 1 h-1 Mpc comoving separation
    cc      = FlatLambdaCDM(H0=100, Om0 = config['Om0'])
    thetamax = config['lens']['Rmax']/cc.comoving_distance(lzred).value * 180/np.pi

    np.random.seed(123)
    nsrcs = int(1e5)
    spos = np.random.uniform(-thetamax/2.0, thetamax/2.0, nsrcs).reshape((-1,2)) # it has to be over sphere please correct.
    sra = lra + spos[:,0]
    sdec = ldec + spos[:,1]
    szred = 0.8
    # intrinsic shapes
    se = np.random.normal(1.0, 0.27, int(2*len(sra))).reshape((-1,2))
    se1 = se[:,0]
    se2 = se[:,1]


    from subprocess import call
    call("mkdir -p %s" % (config["outputdir"]), shell=1)
    fdata = open('%s/simed_sources_logmh_%s.dat'%(config['outputdir'], config['lens']['log_mh']),'w')

    fdata.write('lra(deg)\tldec(deg)\tlzred\tllog_mstel\tllog_mh\tlconc\tsra(deg)\tsdec(deg)\tszred\tse1\tse2\n')
    for ii in range(len(sra)):
        s1, s2 = ss.shear_src(sra[ii], sdec[ii], se1[ii], se2[ii], lzred, szred)
        fdata.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(lra, ldec, lzred, config['lens']['log_mstel'], config['lens']['log_mh'], ss.conc, sra[ii], sdec[ii], szred, s1, s2))

    fdata.close()


