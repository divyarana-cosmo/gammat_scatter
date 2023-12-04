import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM



def get_xyz(ra, dec):
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def get_sigma_crit_inv(lzred, szred, cc):
    # some important constants for the sigma crit computations
    gee = 4.301e-9 #km^2 Mpc M_sun^-1 s^-2 gravitational constant
    cee = 3e5 #km s^-1
    # sigma_crit_calculations for a given lense-source pair
    sigm_crit_inv = cc.angular_diameter_distance(lzred).value * cc.angular_diameter_distance_z1z2(lzred, szred).value * (1.0 + lzred)**2 * 1.0/cc.angular_diameter_distance(szred).value
    sigm_crit_inv = sigm_crit_inv * 4*np.pi*gee*1.0/cee**2
    #sigm_crit_inv = 1e12*sigm_crit_inv #esd's are in pc not in Mpc

    return sigm_crit_inv



@np.vectorize
def get_et_ex(lra, ldec, sra, sdec, se1, se2):
    "computes the etan and ecross for a given  lens-source pair"
    lra  = lra*np.pi/180
    ldec = ldec*np.pi/180
    sra  = sra*np.pi/180
    sdec = sdec*np.pi/180

    c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
    s_theta = np.sqrt(1-c_theta**2)

    # phi to get the compute the tangential shear
    c_phi   = np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
    s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta
    # tangential shear
    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)
    e_x     = - se1*(2*c_phi * s_phi) + se2*(2*c_phi**2 -1)

    return e_t, e_x

def get_earr(file):
    import pandas as pd
    data = pd.read_csv(file, delim_whitespace=2)
    et = 0.0*data['lra(deg)']
    ex = 0.0*data['lra(deg)']
    cc      = FlatLambdaCDM(H0=100, Om0=0.27, Ob0=0.0457)
    #cc      = FlatLambdaCDM(H0=100, Om0=0.27)
    #for ii in range(len(et)):
    for ii in range(50000):
        thetamax = 1/cc.comoving_distance(data['lzred'][ii]).value * 180/np.pi
        l_ra   = data['lra(deg)'][ii]
        l_dec  = data['ldec(deg)'][ii]
        l_zred = data['lzred'][ii]

        sra   = data['sra(deg)'][ii]
        sdec  = data['sdec(deg)'][ii]

        sx, sy, sz = get_xyz(sra,sdec) # individual galaxy ra,dec-->x,y,z
        lx, ly, lz = get_xyz(l_ra,l_dec) # individual galaxy ra,dec-->x,y,z

        # getting the radial separations for a lense source pair functionality for later use
        sl_sep = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - sz)**2)
        sl_sep = sl_sep * cc.comoving_distance(l_zred).value
        if sl_sep>1.0 or sl_sep<0.01:
            continue

        et[ii], ex[ii] = get_et_ex(l_ra, l_dec, sra, sdec, data['se1'][ii], data['se2'][ii])
    #et = et/get_sigma_crit_inv(0.4, 0.8, cc)
    return et,ex


if __name__ == "__main__":
    from halopy import halo
    from stellarpy import stellar
    import pandas as pd
    plt.subplot(2,2,1)
    from glob import glob
    flist = np.sort(glob('./debug/simed_sources_logmh_*.dat'))
    for fil in flist:
        et,ex = get_earr(fil)
        mm  = float(fil.split('_')[-1].split('.dat')[0])
        print(np.mean(et))
        plt.errorbar(mm, np.mean(et), yerr=np.std(et)/np.sqrt(len(et)), fmt='.', capsize=3)
        dat = pd.read_csv(fil, delim_whitespace=1)
        hp      = halo(dat['llog_mh'][0], dat['lconc'][0], omg_m=0.27)
        stel    = stellar(dat['llog_mstel'][0])
        from scipy.integrate import quad
        rbin = np.linspace(0.005,1,30)
        esd = hp.esd_nfw(rbin) + stel.esd_pointmass(rbin)
        esd = esd
        #print(esd)
        from  scipy.interpolate import interp1d
        f = interp1d(rbin, esd)
        pred = quad(lambda x: 2*x*f(x), 0.01, 1)[0]

        cc  = FlatLambdaCDM(H0=100, Om0=0.27, Ob0=0.0457)
        val = pred/((1 - 1e-4)) * get_sigma_crit_inv(0.4, 0.8, cc)

        plt.plot(mm, val,'.k')

        print('factor', np.mean(et)/val)

    plt.yscale('log')
    plt.ylabel(r'$\gamma_t$')
    plt.xlabel(r'$\log(M_h / [h^{-1} M_\odot])$')

    #plt.legend()
    plt.savefig('test.png', dpi=300)

    #a,b,c = plt.hist(et, histxype='step', bins=20, label=r'$e_t$')
    #plt.axvline(np.mean(et), x='C0')


