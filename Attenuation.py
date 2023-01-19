# Load standard modules
import numpy as np
from scipy.stats import lognorm
from matplotlib import pyplot as plt

import constants as cons


class attenuation:
    def __init__(self,
                 range_link,
                 h_sc,
                 h_ac,
                 heights: np.array,
                 zenith_angles: np.array,
                 wavelength = 1550E-9,
                 D_r = 0.1,
                 D_t = 0.1,
                 angle_pe = 5.0E-6,
                 r0 = 0.05,                 # HVB 5/7 model
                 angle_iso = 7.0E-6,        # HVB 5/7 model
                 att_coeff = 0.0025,         # Clear atmosphere: 0.0025 (1550nm, 690 nm), 0.1 (850nm), 0.13 (550nm)
                 P_thres = 1.0E-6,
                 P0 = 0.0,
                 refraction = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.05, 1.3))
                 ):

        # Received signal without turbulence effects
        self.P0 = P0

        # Range and height
        self.range_link = range_link
        self.zenith_angles = zenith_angles
        self.heights = heights
        self.h0 = h_ac
        self.h1 = h_sc
        self.h_limit = 100.0E3 #m
        self.H_scale = 6600.0

        # Absorption & Scattering coefficients
        # self.b_mol_0 = 5.0E-3
        self.b_sca_aer = 1.0
        self.b_abs_aer = 1.0
        self.b_sca_mol = 1.0
        self.b_abs_mol = 1.0
        self.b_v = att_coeff


    def std_atm(self):

        self.T_ext = np.exp( -self.b_v * self.H_scale * abs(1/np.cos(self.zenith_angles)) / 1000 *
                             (1 - np.exp(-self.range_link / (self.H_scale * abs(1/np.cos(self.zenith_angles))))) )

        return self.T_ext

    def Beer(self, turbulence_model = "Cn_HVB"):
        self.T_att = np.exp(-self.b_v * (40.0 - self.h_limit / 1000.0))
        return self.T_ext

    def coeff(self, s):
        beta_tot = self.b_sca_aer + self.b_abs_aer + self.b_sca_mol + self.b_abs_mol
        T_att = np.exp(-beta_tot * s)
        return T_att

    def plot(self):

        fig_ext = plt.figure(figsize=(6, 6), dpi=125)
        ax_ext = fig_ext.add_subplot(111)
        ax_ext.set_title('Transmission due to Atmospheric attenuation')
        ax_ext.plot(np.rad2deg(self.zenith_angles), self.T_ext, linestyle='-')
        ax_ext.set_xlabel('Zenith angle (deg')
        ax_ext.set_ylabel('Transmission')

    def print(self):

        print('------------------------------------------------')
        print('ATTENUATION/EXTINCTION ANALYSIS')
        print('------------------------------------------------')

        print('Extinction loss [dBW] : ', cons.dB2W(self.T_ext))







