"""This file is used for generate gamma-ray spectrum for DM annihilation"""
import numpy as np
import abc
from scipy.interpolate import interp2d, interp1d


rsun = 8.127 # [kpc]
s_max = 300
kpc2cm = 3.0857 * np.power(10., 21)
rho_local = 0.4  # GeV/cm^3

ew_dict = {"e": 4, "mu": 7, "tau": 10, "bb": 13, "tt": 14, "WW": 17, "ZZ": 20, "gamma": 22, "h": 23}
noew_dict = {"e": 2, "mu": 3, "tau": 4, "bb": 7, "tt": 8, "WW": 9, "ZZ": 10, "gamma": 12, "h": 13}
data_folder = "/home/jguo/workspace/data/pppc4spectrum"
particle = "gammas"


class DMProfile(abc.ABC):
    
    @abc.abstractclassmethod
    def rho_dm(self, r):
        return NotImplemented
    
    def r_theta_s(self, theta, s):
        rr = (np.square(s) + np.square(rsun) -
            2 * s * rsun * np.cos(theta))
        r = np.sqrt(rr)
        return r
        
    def get_jfactor_theta(self, theta_1, theta_2, npoint):
        """Calculate the j factor between theta_1 and theta_2.
        Args:
            theta_1: start angle of the anuli, double, in radians.
            theta_2: end angle of the anuli, double, in radians.
            npoint: int, number of point for the integration.
        Return:
            j_val: double, the j-factor values in the r.o.i.
        """
        thetas = np.linspace(theta_1, theta_2, npoint)
        ss = np.linspace(0, s_max, npoint)
        vtheta, vs =  np.meshgrid(thetas, ss, indexing='ij')
        vr = self.r_theta_s(vtheta, vs)
        j_tensor = 2 * np.pi * np.square(self.rho_dm(vr)) * np.sin(vtheta)
        j_val = np.trapz(j_tensor, ss, axis=1)
        j_val = np.trapz(j_val, thetas)
        j_val = j_val * kpc2cm
        return j_val


class NFWProfile(DMProfile):
    
    def __init__(self, gamma, rs, rhos, rho_local=rho_local):
        self.gamma = gamma
        self.rs = rs
        self.rhos = rhos
        self.rho_local = rho_local
        
    def _rho_dm_temp(self, r):
        rho_val = self.rhos #  * np.power(2., 3-gamma)
        rho_val = rho_val * np.power(r/self.rs, -self.gamma)
        rho_val = rho_val / np.power(1+r/self.rs, 3-self.gamma)
        return rho_val

    def reset_rhos(self):
        rho_local_calc = self._rho_dm_temp(rsun)
        self.rhos = self.rho_local / rho_local_calc * self.rhos
        print("set rhos = {}".format(self.rhos))
    
    def rho_dm(self, r):
        self.reset_rhos()
        rho_val = self._rho_dm_temp(r)
        return rho_val


class FireProfile(DMProfile):

    def __init__(self, path, i, rho_local=rho_local):
        self.path = path
        self.i = i  # the number of the fire profile.
        self.rho_local = rho_local
        self.rho_fn = self.get_rho_fn

    @property
    def get_rho_fn(self):
        path = self.path + "/density_hydro_{}.npy".format(np.int_(self.i))
        fire = np.load(path)
        at = np.argmin((fire[::,0]-rsun)**2)
        rs = fire[::,0]
        rhos = fire[::,1] * self.rho_local / fire[::,1][at]
        rho_fn = interp1d(rs, rhos, bounds_error=False, fill_value=0)
        return rho_fn

    def rho_dm(self, r):
        rho_val = self.rho_fn(r)
        return rho_val



class DMSpectrum():
    """This class is used for gamma ray or positron spectrum `dN/dE`
    generation, based on PPPC4DM http://www.marcocirelli.net/PPPC4DMID.html.
    Args:
        data_folder: string, path to the pppc4 data folder.
        ewcorr: bool, electroweak corrections ('Yes' or 'No').
        particle: string, "gammas" or "positrons", the final particle we observe.
        channel_dict: dictionary, {"channel": int}.
    """
    
    def __init__(self, data_folder=data_folder, ewcorr=True, particle=particle, channel_dict=ew_dict):
        self.data_folder = data_folder
        self.ewcorr = ewcorr
        self.particle = particle
        self.channel_dict = channel_dict
        
    def get_table(self, channel):
        """get the mass_vec, log10x, flux_vec.
        Args:
            channel, string, the final state.
        """
        path = self.data_folder
        if self.ewcorr is False:
            path = path +"/AtProductionNoEW_{}.dat".format(self.particle)
        if self.ewcorr is True:
            path = path +"/AtProduction_{}.dat".format(self.particle)
        table = np.loadtxt(path, skiprows=1)
        mass_vec = table[:, 0]
        mass_vec = np.unique(mass_vec)
        log10x_vec = table[:, 1]
        log10x_vec = np.unique(log10x_vec)
        index_flux = self.channel_dict[channel]
        flux_vec = table[:, index_flux]
        return mass_vec, log10x_vec, flux_vec
    
    def get_dndx_fn(self, mdm, channel):
        """get the mass_vec, log10x, flux_vec.
        Args:
            mdm: double, DM mass, in the unit [GeV].
            channel: string, the final state.
        """
        mass_vec, log10x_vec, flux_vec = self.get_table(channel)
        f = interp2d(mass_vec, log10x_vec, flux_vec, kind='linear',
                     bounds_error=False, fill_value=0)
        return f
    
    def get_dnde_val(self, mdm, channel, e_vec):
        """Get the dnde value for certain channel and certain DM mass.
        Args:
            mdm: double, DM mass, in the unit [GeV].
            channel: string, the final state.
            e_vec: np.array, the energy array for the observed particle.
        Return:
            dnde_vals: np.array, the `dN/dE` values.
        """
        dndx_fn = self.get_dndx_fn(mdm, channel)
        x_e_vec = np.log10(e_vec/mdm)
        dndx_vals = dndx_fn(mdm, x_e_vec)
        dndx_vals = np.squeeze(dndx_vals)
        dnde_vals = e_vec * dndx_vals / (np.log(10.) * e_vec)
        return dnde_vals
