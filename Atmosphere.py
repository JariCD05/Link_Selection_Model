from input import *
from helper_functions import *
from PDF import dist

import numpy as np
from scipy.stats import rice, rayleigh

from matplotlib import pyplot as plt


class turbulence:
    def __init__(self,
                 range,
                 link = 'up',
                 ):

        # Terminal properties
        self.D_r = D_r

        # Range and height
        self.range = range
        self.h_cruise = h_AC
        self.h_limit = 20.0E3 #* ureg.meter  # m
        self.h_sc = h_SC

        # Create a height profile for computation of r0, Cn and wind speed
        self.heights = np.linspace(self.h_cruise, self.h_sc, 500)
        self.heights = self.heights[self.heights<self.h_limit]
        if link == "down":
            np.flip(self.heights)

    # ------------------------------------------h_------------------------------
    # --------------------------TURUBLENCE-ENVIRONMENT------------------------
    # --------------------------Cn^2--windspeed--r0--Stehl-Ratio--------------
    def Cn_func(self, turbulence_model = "HVB", dim = 2):

        if turbulence_model == "HVB":
            self.Cn = 5.94E-53 * (self.windspeed_rms / 27) ** 2 * self.heights** 10 * \
                          np.exp(-self.heights / 1000.0) + \
                          2.75E-16 * np.exp(-self.heights/ 1500.0)


            # elif dim == 2:
            #     self.Cn = np.zeros(np.shape(self.heights))
            #     for i in range(len(self.windspeed_rms)):
            #         windspeed_rms = self.windspeed_rms[i]
            #         heights = self.heights[i]
            #         self.Cn[i] = 5.94E-53 * (windspeed_rms / 27) ** 2 * heights.to('meter').magnitude ** 10 * \
            #                         np.exp(-heights.to('meter').magnitude / 1000.0)+ \
            #                         2.75E-16 * np.exp(-heights.to('meter').magnitude / 1500.0)

        elif turbulence_model == "HVB_57":
            r0 = 0.05
            self.angle_iso = 7.0E-6
            A = 1.29E-12 * r0 ** (-5 / 3) * wavelength ** 2 - 1.61E-13 * self.angle_iso ** (-5 / 3) * wavelength ** 2 + 3.89E-15
            self.Cn = 5.94*10**(-53) * (self.windspeed_rms/27)**2 * self.heights**10 * np.exp(-self.heights/ 1000.0)+ \
                      2.7E-16 * np.exp(-self.heights / 1500.0) + \
                      A * np.exp(-self.heights / 100)
        # return self.Cn

    def windspeed(self, slew, Vg, wind_model_type = "bufton"):
        if wind_model_type == "bufton":

            # self.windspeed = np.zeros(len(self.heights[self.heights < self.h_limit]))
            self.windspeed = abs(slew) * self.heights + \
                         np.ones(np.shape(self.heights)) * Vg + \
                             30 * np.exp(-((self.heights - 9400.0 / 4800.0) ** 2))

            self.windspeed_rms = np.sqrt(1 / (self.h_limit - self.h_cruise) *
                                            np.trapz(self.windspeed ** 2, x=self.heights))
            # return self.windspeed_rms

    def r0_func(self, zenith_angles):
        self.r0 = (0.423 * k_number ** 2 / abs(np.cos(zenith_angles)) *
                          np.trapz(self.Cn, x=self.heights)) ** (-3 / 5)
        return self.r0

    def stehl_ratio(self, D):
        return (1 + (D / self.r0)**(5/3))**(-6/5)


    # ------------------------------------------------------------------------
    # -------------------------------VARIANCES--------------------------------
    # ------------------------------------------------------------------------
    def var_rytov_func(self, zenith_angles):
        self.var_rytov = 2.25 * k_number ** (7 / 6) * (1 / np.cos(zenith_angles)) ** (11 / 6) * \
                             np.trapz(self.Cn * (self.heights - self.h_cruise) ** (5 / 6), x=self.heights)

    def var_scint(self, zenith_angles: np.array, PDF_type="lognormal"):
        if PDF_type == "lognormal":
            self.var_scint = self.var_rytov

        elif PDF_type == "gamma-gamma":
            # self.var_scint = np.exp(
            #     0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6) +
            #     0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1

            self.alpha = (np.exp(0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6)) - 1) ** -1
            self.beta  = (np.exp(0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1) ** -1
            self.var_scint = 1 / self.alpha + 1 / self.beta + 1 / (self.alpha * self.beta)

        # Aperture averaging effect
        # REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, EQ.2.96-2.98
        h0 = ( np.trapz(self.Cn**2 * self.heights**2    , x = self.heights) /
               np.trapz(self.Cn**2 * self.heights**(5/6), x = self.heights) )**(6/7)
        A = (1 + 1.1 * (self.D_r ** 2 / (wavelength * h0 * 1 / np.cos(zenith_angles))) ** (7 / 6)) ** -1
        self.var_scint = self.var_scint * A

    def var_bw(self, D_t, zenith_angles):
        #REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.50
        # This reference computes the beam wander variance as a displacement at the receiver in meters (1 rms)
        w0 = D_t
        self.var_bw = 0.54 * (h_SC - h_AC)**2 * 1/np.cos(zenith_angles)**2 * (wavelength / (2*w0))**2 * (2*w0 / self.r0)**(5/3)


    def var_AoA(self, D_r):
        self.beta = 0.182 * (D_r/self.r0)**5/3 * (wavelength/D_r)**2

    # ------------------------------------------------------------------------
    # -------------------------------PDF-MODELS-------------------------------
    # ------------------------------------------------------------------------
    def PDF(self, ranges, w_r, zenith_angles = 0.0, steps=1000, effect = "scintillation"):
        # VERIFICATION REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, FIG.5.1

        if effect == "scintillation":
            self.h_scint = np.zeros((len(zenith_angles), steps))
            self.x_scint = np.zeros((len(zenith_angles), steps))
            self.pdf_scint = np.zeros((len(zenith_angles), steps))

            if PDF_scintillation == "lognormal":
                #REF:
                var = np.log(self.var_scint + 1)
                bias = -1/2 * var

                for i in range(len(zenith_angles)):
                    self.x_scint[i], self.pdf_scint[i] = dist.lognorm_pdf(var=var[i], bias=bias[i], steps=steps)
                    self.h_scint[i] = dist.lognorm_rvs(var=var[i], bias=bias[i], steps=steps)

            elif PDF_scintillation == "gamma-gamma":

                self.cdf_scint = np.zeros((len(zenith_angles), steps))

                for i in range(len(zenith_angles)):

                    self.x_scint[i], self.pdf_scint[i] = dist.gg_pdf(alpha=self.alpha[i], beta=self.beta[i], steps=steps)
                    self.cdf_scint[i, 0] = self.pdf_scint[i, 0]
                    for j in range(1, len(self.pdf_scint[i])):
                        self.cdf_scint[i, j] = np.trapz(self.pdf_scint[i, 1:j], x=self.x_scint[i, 1:j])
                    self.h_scint[i] = dist.gg_rvs(self.pdf_scint[i], steps)

            return self.h_scint

        elif effect == "beam wander":

            self.r_bw = np.zeros((len(zenith_angles), steps))
            self.h_bw = np.zeros((len(zenith_angles), steps))
            self.x_bw = np.zeros((len(zenith_angles), steps))
            self.pdf_bw = np.zeros((len(zenith_angles), steps))

            if PDF_beam_wander == "rayleigh":
                for i in range(len(zenith_angles)):
                    self.x_bw[i] = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), steps)
                    self.pdf_bw[i] = rayleigh.pdf(self.x_bw[i], scale=self.var_bw[i])
                    self.r_bw[i] = rayleigh.rvs(size=steps, scale=self.var_bw[i])

            elif PDF_beam_wander == "gaussian":
                for i in range(len(zenith_angles)):
                    self.x_bw[i], self.pdf_bw[i] = dist.norm_pdf(var=self.var_bw[i], steps=steps)
                    self.r_bw[i] = dist.norm_rvs(var=self.var_bw[i], steps=steps)
                    self.h_bw[i] = np.exp(-self.r_bw[i] ** 2 / w_r[i] ** 2)

            return self.r_bw, self.h_bw

    def plot(self, t, plot = "scint pdf", elevation=0.0, P_r = 0.0):

        fig_turb, ax1 = plt.subplots(1, 1, dpi=125)

        if plot == "scint pdf":
            fig_scint, axs = plt.subplots(3, 1, dpi=125)
            axs[0].plot(self.x_scint[0], self.pdf_scint[0], label="\epsilon: "+str(elevation[0])+", variance: "+str(self.var_scint[0]))
            axs[0].set_title(f'Intensity with scintillation - Lognorm distribution')
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Normalized intensity [-]')
            axs[0].legend()

            axs[1].plot(t, self.h_scint[0], label="variance: " + str(self.var_scint[0]))
            axs[1].plot(np.array((t[0], t[-1])), np.ones(2) * np.mean(self.h_scint[0]), label="h_scint, mean")
            axs[1].set_ylabel('Intensity [-]')
            axs[1].set_xlabel('Time [s]')
            axs[1].legend()

            # axs[1].plot(self.x_scint[2], self.cdf_scint[2])

            axs[2].plot(t, P_r[0], label="variance: " + str(self.var_scint[0]))
            axs[2].set_ylabel('Intensity [W/m^2]')
            axs[2].set_xlabel('Time [s]')
            axs[2].legend()

        if plot == "bw pdf":
            fig_scint, axs = plt.subplots(2, 1, dpi=125)
            axs[0].plot(self.x_bw[0], self.pdf_bw[0], label="variance: "+str(self.var_bw[0]))
            axs[0].set_title(f'Intensity with beam wander - rice distribution')
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Normalized intensity [-]')
            axs[0].legend()

            axs[1].plot(t, self.r_bw[0])
            axs[1].set_ylabel('Displacement [m]')
            axs[1].set_xlabel('Time [s]')
            axs[1].legend()

        # Plot Cn vs. Height
        elif plot == "Cn":
            fig_cn, axs = plt.subplots(1, 1, dpi=125)
            axs.set_title(f'Cn  vs  heights')
            axs.plot(self.heights, self.Cn)
            axs.set_yscale('log')
            axs.set_xscale('log')

        # Plot Cn vs. Height
        elif plot == "wind":
            fig_wind, axs = plt.subplots(1, 1, dpi=125)
            axs.set_title(f'Wind speed  vs  heights')
            axs.plot(self.heights[self.heights < self.h_limit], self.windspeed)
            axs.set_xlabel('heights (m)')
            axs.set_ylabel('wind speed (m/s)')

        elif plot == "scint vs. elevation":
            fig_scint_zen, axs = plt.subplots(1, 1, dpi=125)
            axs.plot(np.rad2deg(elevation), self.var_rytov)
            axs.set_title(f'Scintillation variance VS. Zenith angle')
            axs.set_ylabel('Rytov variance [-]')
            axs.set_xlabel('Zenith angle [deg]')

            axs.plot(np.rad2deg(elevation), self.var_scint, linestyle= "--")
            axs.set_ylabel('Scintillation index [-]')
            axs.set_xlabel('Zenith angle [deg]')

    def print(self):
        print('------------------------------------------------')
        print('------------TURBULENCE-ANALYSIS-----------------')
        print('------------------------------------------------')

        print('Windspeed rms [m/s] : ', self.windspeed_rms)
        print('Cn^2       [m^2/3]  : ', self.Cn[0])
        print('r0             [m]  : ', self.r0[0])
        print('Rytov variance [-]  : ', self.var_rytov[0])
        print('Scintillation variance [-] : ', self.var_scint[0])
        print('Beam wander variance   [-] : ', self.var_bw[0])
        # print('Rec. power with turbulence (mean)     [dBW]   : ', W2dB(np.mean(P_turb[0].to_base_units().magnitude)))
        # print('Rec. intensity with turbulence (mean) [W/m^2] : ', np.mean(self.I_turb))
        # print('Wave front error loss                 [dBW]   : ', W2dB(self.T_WFE[0]))




class attenuation:
    def __init__(self,
                 range_link,
                 att_coeff = 0.0025,         # Clear atmosphere: 0.0025 (1550nm, 690 nm), 0.1 (850nm), 0.13 (550nm)
                 refraction = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.05, 1.3))
                 ):

        # Range and height
        self.range_link = range_link
        self.h0 = h_AC
        self.h1 = h_SC
        self.h_limit = 100.0E3 #m
        self.H_scale = 6600.0

        # Absorption & Scattering coefficients
        # self.b_mol_0 = 5.0E-3
        self.b_sca_aer = 1.0
        self.b_abs_aer = 1.0
        self.b_sca_mol = 1.0
        self.b_abs_mol = 1.0
        self.b_v = att_coeff

    def T_ext(self, zenith_angles, method="standard_atmosphere", steps = 1000.0):
        if method == "standard_atmosphere":
            self.T_ext = np.zeros((len(zenith_angles), steps))

            for i in range(len(zenith_angles)):
                T_ext = np.exp( -self.b_v * self.H_scale * abs(1/np.cos(zenith_angles[i])) / 1000 *
                                     (1 - np.exp(-self.range_link[i] / (self.H_scale * abs(1/np.cos(zenith_angles[i])))  )) )

                self.T_ext[i, :] = T_ext
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

        print('Extinction loss [dBW] : ', dB2W(self.T_ext))







