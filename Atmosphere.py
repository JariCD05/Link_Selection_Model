from input import *
from helper_functions import *
from PDF import dist

import numpy as np
from scipy.stats import rice, rayleigh

from matplotlib import pyplot as plt


class turbulence:
    def __init__(self,
                 ranges,
                 link = 'up',
                 ):

        # Range and height
        self.ranges = np.array(ranges)
        self.h_cruise = h_AC
        self.h_limit = 20.0E3 #* ureg.meter  # m
        self.h_sc = h_SC

        # Create a height profile for computation of r0, Cn and wind speed
        self.heights = np.linspace(self.h_cruise, self.h_sc, 1000)
        # self.heights = self.heights[self.heights<self.h_limit]
        if link == "down":
            np.flip(self.heights)

        # Beam parameters
        w_r, z_r = beam_spread(w0, self.ranges)  # beam radius at receiver (without turbulence) (in m)
        self.Lambda0 = 2 * self.ranges / (k_number * w0 ** 2)
        # REF: Andrews, Laser beam propagation through random media, EQ.12.9
        self.Lambda = 2 * self.ranges / (k_number * w_r ** 2)
        self.Theta0 = self.Lambda0 / self.Lambda**2 - self.Lambda0**2
        self.Theta = self.Theta0 / (self.Theta0**2 + self.Lambda0**2)
        self.Theta_bar = 1 - self.Theta
        self.kappa0 = 30.0

    # ------------------------------------------------------------------------
    # --------------------------TURUBLENCE-ENVIRONMENT------------------------
    # --------------------------Cn^2--windspeed--r0--Stehl-Ratio--------------

    def Cn_func(self, turbulence_model = "HVB", dim = 2):

        if turbulence_model == "HVB":
            self.A = 0

        elif turbulence_model == "HVB_57":
            r0 = 0.05
            self.angle_iso = 7.0E-6
            self.A = 1.29E-12 * r0 ** (-5 / 3) * wavelength ** 2 - 1.61E-13 * self.angle_iso ** (-5 / 3) * wavelength ** 2 + 3.89E-15

        shift = 0.0
        self.Cn = 5.94E-53 * (self.windspeed_rms / 27) ** 2 * (self.heights-shift) ** 10 * np.exp(-(self.heights-shift) / 1000.0) + \
                  2.75E-16 * np.exp(-self.heights / 1500.0) + \
                  self.A * np.exp(-self.heights / 100)
        # return self.Cn

    def windspeed_func(self, slew, Vg, wind_model_type = "bufton"):
        if wind_model_type == "bufton":
            self.windspeed = abs(slew) * self.heights + np.ones(np.shape(self.heights)) * Vg + \
                             30 * np.exp(-(((self.heights - 9400.0) / 4800.0) ** 2))
            self.windspeed_rms = np.sqrt(1 / (self.h_limit - self.h_cruise) *
                                            np.trapz(self.windspeed[self.heights<self.h_limit] ** 2, x=self.heights[self.heights<self.h_limit]))
            # return self.windspeed_rms

    def r0_func(self, zenith_angles):

        # self.r0 = (0.423 * k_number ** 2 / abs(np.cos(zenith_angles)) * np.trapz(self.Cn, x=self.heights)) ** (-3 / 5)

        self.r0 = (0.423 * k_number ** 2 / abs(np.cos(zenith_angles)) *
                   np.trapz(self.Cn * ((self.heights[-1] - self.heights) / self.heights[-1]) ** (5/3), x=self.heights)) ** (-3 / 5)

        return self.r0

    # ------------------------------------------------------------------------
    # -------------------------STATIC-TURBULENCE-LOSSES-----------------------
    # -----------------------BEAM-SPREAD-&-WAVE-FRONT-ERROR-------------------
    # ------------------------------------------------------------------------
    def beam_spread(self):
        if self.r0.ndim != 1:
            raise ValueError('Incorrect argument dimension of r0')
        if len(self.r0) != len(self.ranges):
            raise ValueError('Incorrect argument shape of r0')

        w_r, z_r = beam_spread(w0, self.ranges)  # beam radius at receiver (without turbulence) (in m)
        # Only during uplink is the turbulence beamspread considered significant.
        # This is ignored for downlink
        if link == 'up':
            self.w_LT = beam_spread_turbulence_LT(self.r0, w_r)
            self.w_ST = beam_spread_turbulence_ST(self.Lambda0, self.Lambda, self.var_rytov, w_r)
            #REF:
            # self.w_ST1 = w0**2 * (1 + (self.ranges / z_r)**2) + 2 * (4.2 * self.ranges / (k_number * self.r0) * (1 - 0.26 * (self.r0/w0)**(1/3)))**2
            self.T_beamspread = (w_r / self.w_LT) ** 2

        elif link == 'down':
            self.T_beamspread = 1.0

    def Strehl_ratio_func(self, tip_tilt="NO"):
        # REF: R. SAATHOF, SLIDES LECTURE 3, 2021
        # REF: R. PARENTI, 2006, EQ.1-3
        D = 2 ** (3 / 2) * w0
        if tip_tilt == "YES":
            var_WFE = 1.03 * (D_t / self.r0) ** (5 / 3)
        elif tip_tilt == "NO":
            var_WFE = 0.134 * (D_t / self.r0) ** (5 / 3)

        self.Strehl_ratio = np.exp(-var_WFE)


    # ------------------------------------------------------------------------
    # -------------------------------VARIANCES--------------------------------
    # ------------------------------------------------------------------------
    def var_rytov_func(self, zenith_angles):
        self.var_rytov = 2.25 * k_number ** (7 / 6) * (1 / np.cos(zenith_angles)) ** (11 / 6) * \
                         np.trapz(self.Cn *
                                  (self.heights - self.h_cruise) ** (5/6), x=self.heights)
        # *
        #                           ((self.heights[-1] - self.heights) / self.heights[-1]) ** (5/6), x=self.heights)
        self.std_rytov = np.sqrt(self.var_rytov)

    def var_scint_func(self, zenith_angles: np.array):
        if PDF_scintillation == "lognormal":
            # REF: A survey of free space optical networks, I. SON, EQ.2
            self.var_scint = self.var_rytov
            # Aperture averaging effect
            # REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, EQ.2.96-2.98
            h0 = (np.trapz(self.Cn * self.heights ** 2, x=self.heights) /
                  np.trapz(self.Cn * self.heights ** (5 / 6), x=self.heights)) ** (6 / 7)
            self.Af = (1 + 1.1 * (D_r ** 2 / (wavelength * h0 / np.cos(zenith_angles))) ** (7 / 6)) ** (-1)
            self.var_scint = self.var_scint * self.Af

            # # Create lognormal parameters
            self.mean_scint = -0.5 * np.log(self.var_scint + 1)
            self.std_scint  = np.sqrt(np.log(self.var_scint + 1))

        elif PDF_scintillation == "gamma-gamma":
            # self.var_scint = np.exp(
            #     0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6) +
            #     0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1

            self.alpha = (np.exp(0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6)) - 1) ** -1
            self.beta  = (np.exp(0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1) ** -1
            self.var_scint = 1 / self.alpha + 1 / self.beta + 1 / (self.alpha * self.beta)


    def var_bw(self, zenith_angles):
        #REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.50
        # This reference computes the beam wander variance as a displacement at the receiver in meters (1 rms)
        # Beam wander is typically 10-100 urad.
        self.var_bw_r = 0.54 * (h_SC - h_AC)**2 * 1/np.cos(zenith_angles)**2 * (wavelength / (2*w0))**2 * (2*w0 / self.r0)**(5/3) * \
                      (1 - ((self.kappa0**2 * w0**2) / (1 + self.kappa0**2 * w0**2))**(1/6))
        self.std_bw_r = np.sqrt(self.var_bw_r)
        self.std_bw   = self.std_bw_r / self.ranges
        self.mean_bw  = np.zeros(np.shape(self.std_bw))

        # if PDF_beam_wander == "rayleigh":
        #     # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
        #     print('BW normal std: ', self.std_bw * 1.0E6, ' urad')
        #     self.std_bw = np.sqrt(2 / (4 - np.pi) * self.std_bw**2)
        #     print('BW rayleigh std: ', self.std_bw * 1.0E6, ' urad')
        # REF: Scintillation and Beam-Wander Analysis in an Optical Ground Station-Satellite Uplink, EQ.10-11
        # self.var_bw = 1 / 2 * 2.07 * 1 / np.cos(zenith_angles) * np.trapz(self.Cn * (self.heights[-1] - self.heights) ** 2 * (1 / self.w_ST[:, None]) ** (1 / 3), x=self.heights)

        # REF: MODELING THE PDF FOR THE IRRADIANCE OF AN UPLINK BEAM IN THE PRESENCE OF BEAM WANDER, R. PARENTI 2006, EQ.10
        # self.var_bw = 5.09 / (k_number**2 * self.r0**(5/3) * w0**(1/3))

    def var_AoA(self, zenith_angles):
        # REF: OPTICAL COMMUNICATION IN SPACE: CHALLENGES AND MITIGATION TECHNIQUES, H. KAUSHAL, EQ.23
        # self.var_aoa = 2.914 * np.trapz(self.Cn, x=self.heights) * D_r**(-1/3) / np.cos(zenith_angles)

        # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.28
        mu_0 = np.trapz(self.Cn, x=self.heights)
        mu_1u = np.trapz(self.Cn * (self.Theta[:, None] + self.Theta_bar[:, None]  *
                                   (self.heights - self.heights[0]) / (self.heights[-1] - self.heights[0]))**(5/3), x=self.heights)

        mu_2u = np.trapz(self.Cn * (1 - (self.heights - self.heights[0]) / (self.heights[-1] - self.heights[0]))**(5/3), x=self.heights)
        if link == 'up':
            self.var_aoa = 2.91 * (mu_1u + 0.62 * mu_2u * self.Lambda**(11/6)) / np.cos(zenith_angles) * (2 * D_r)**(1/3)
        elif link == 'down':
            self.var_aoa = 2.91 * mu_0 / np.cos(zenith_angles) * (2 * D_r)**(1/3)

        self.std_aoa = np.sqrt(self.var_aoa)
        self.mean_aoa = np.zeros(np.shape(self.std_aoa))

        if PDF_beam_wander == "rayleigh":
            # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
            self.std_aoa_r = np.sqrt(2 / (4 - np.pi) * self.var_bw_r)

    # ------------------------------------------------------------------------
    # -------------------------------PDF-MODELS-------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def PDF(self, data, iterable = 0.0, steps=1000, effect = "scintillation"):
        # VERIFICATION REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, FIG.5.1
        if effect == "scintillation":
            if PDF_scintillation == "lognormal":
                self.x_scint, self.pdf_scint = dist.lognorm_pdf(sigma=self.std_scint[:, None], mean=self.mean_scint[:, None], steps=steps)
                self.h_scint = dist.lognorm_rvs(data, mean=self.mean_scint[:, None], sigma=self.std_scint[:, None])
            elif PDF_scintillation == "gamma-gamma":
                self.h_scint = np.zeros((len(iterable), steps))
                self.x_scint = np.zeros((len(iterable), steps))
                self.pdf_scint = np.zeros((len(iterable), steps))

                self.cdf_scint = np.zeros((len(iterable), steps))
                for i in range(len(iterable)):
                    self.x_scint[i], self.pdf_scint[i] = dist.gg_pdf(alpha=self.alpha[i], beta=self.beta[i], steps=steps)
                    self.cdf_scint[i, 0] = self.pdf_scint[i, 0]
                    for j in range(1, len(self.pdf_scint[i])):
                        self.cdf_scint[i, j] = np.trapz(self.pdf_scint[i, 1:j], x=self.x_scint[i, 1:j])
                    self.h_scint[i] = dist.gg_rvs(self.pdf_scint[i], steps)
            return self.h_scint

        elif effect == "beam wander":
            if PDF_beam_wander == "rayleigh":
                self.x_bw, self.pdf_bw = dist.rayleigh_pdf(sigma=self.std_bw[:, None], steps=steps)
                self.angle_bw_R = dist.rayleigh_rvs(data=data, sigma=self.std_bw[:, None])
                self.angle_bw_X = data[0]
                self.angle_bw_Y = data[1]
                return self.angle_bw_R

            elif PDF_beam_wander == "rice":
                self.x_bw, self.pdf_bw = dist.norm_pdf(effect=effect, sigma=self.std_bw[:, None], steps=steps)
                self.angle_bw_X = dist.norm_rvs(data=data[0], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                self.angle_bw_Y = dist.norm_rvs(data=data[1], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                self.angle_bw_R = np.sqrt(self.angle_bw_X ** 2 + self.angle_bw_Y ** 2)
                return self.angle_bw_R

        elif effect == "angle of arrival":
            if PDF_AoA == "rayleigh":
                self.x_aoa, self.pdf_aoa = dist.rayleigh_pdf(sigma=self.std_aoa[:, None], steps=steps)
                self.angle_aoa_R = dist.rayleigh_rvs(data=data, sigma=self.std_aoa[:, None])
                return self.angle_aoa_R

            if PDF_AoA == "rice":
                self.x_aoa, self.pdf_aoa = dist.norm_pdf(effect=effect, sigma=self.std_aoa[:, None], steps=steps)
                self.angle_aoa_X = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                self.angle_aoa_Y = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                self.angle_aoa_R = np.sqrt(self.angle_aoa_X ** 2 + self.angle_aoa_Y ** 2)
            return self.angle_aoa_R

    def test_PDF(self, effect = "scintillation", index = 20):
        if effect == "scintillation":
            fig_test_pdf_scint, ax_test_pdf_scint = plt.subplots(3, 1)
            dist.plot(ax=ax_test_pdf_scint,
                      sigma=self.std_scint[:, None],
                      input=self.var_scint,
                      x=self.x_scint,
                      pdf=self.pdf_scint,
                      data=self.h_scint,
                      index=index,
                      effect=effect,
                      name=PDF_scintillation)

        elif effect == "beam wander":
            fig_test_pdf_bw, ax_test_pdf_bw = plt.subplots(3, 1)
            dist.plot(ax=ax_test_pdf_bw,
                      sigma=self.std_bw[:, None],
                      input=self.var_bw_r,
                      x=self.x_bw,
                      pdf=self.pdf_bw,
                      data=self.angle_bw_R,
                      # data=[self.angle_bw_X, self.angle_bw_Y],
                      index=index,
                      effect=effect,
                      name=PDF_beam_wander)

        elif effect == "angle of arrival":
            fig_test_pdf_aoa, ax_test_pdf_aoa = plt.subplots(3, 1)
            dist.plot(ax=ax_test_pdf_aoa,
                      sigma=self.std_aoa[:, None],
                      input=self.var_aoa,
                      x=self.x_aoa,
                      pdf=self.pdf_aoa,
                      data=self.angle_aoa_R,
                      index=index,
                      effect=effect,
                      name=PDF_AoA)



    def plot(self, t, plot = "scint pdf", index = 20, elevation=0.0, zenith=0.0, P_r = 0.0):

        fig_turb, ax1 = plt.subplots(1, 1, dpi=125)

        if plot == "scint pdf":
            fig_scint, axs = plt.subplots(3, 1, dpi=125)
            axs[0].plot(self.x_scint[index], self.pdf_scint[index], label="$\epsilon$: "+str(np.rad2deg(elevation[index]))+", variance: "+str(self.var_scint[index]))
            axs[0].set_title(f'Intensity scintillation PDF: '+str(PDF_scintillation))
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Normalized intensity [-]')
            axs[0].legend()

            axs[1].plot(t, self.h_scint[index], label="variance: " + str(self.var_scint[index]))
            axs[1].plot(np.array((t[0], t[-1])), np.ones(2) * np.mean(self.h_scint[index]), label="h_scint, mean")
            axs[1].set_ylabel('Intensity [-]')
            axs[1].set_xlabel('Time [s]')
            axs[1].legend()

            # axs[1].plot(self.x_scint[2], self.cdf_scint[2])

            axs[2].plot(t, P_r[index], label="variance: " + str(self.var_scint[index]))
            axs[2].set_ylabel('Intensity [W/m^2]')
            axs[2].set_xlabel('Time [s]')
            axs[2].legend()

        if plot == "bw pdf":
            fig_scint, axs = plt.subplots(2, 1, dpi=125)
            axs[0].plot(self.x_bw[index], self.pdf_bw[index], label="variance: "+str(self.var_bw[index]))
            axs[0].set_title(f'Beam wander displacement PDF: '+str(PDF_beam_wander))
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Normalized intensity [-]')
            axs[0].legend()

            axs[1].plot(t, self.r_bw[index])
            axs[1].set_ylabel('Displacement [m]')
            axs[1].set_xlabel('Time [s]')
            axs[1].legend()

        # Plot Cn vs. Height
        elif plot == "Cn":
            fig_cn, axs = plt.subplots(1, 1, dpi=125)
            axs.set_title(f'Cn  vs  heights')
            axs.plot(self.heights, self.Cn, label='V wind (rms) = '+str(self.windspeed_rms))

            axs.set_yscale('log')
            # axs.invert_yaxis()
            axs.set_xscale('log')
            axs.legend()

        # Plot Cn vs. Height
        elif plot == "wind":
            fig_wind, axs = plt.subplots(1, 1, dpi=125)
            axs.set_title(f'Wind speed  vs  heights')
            axs.plot(self.heights, self.windspeed)
            axs.plot(self.heights[self.heights < self.h_limit], self.windspeed[self.heights < self.h_limit])
            axs.set_xlabel('heights (m)')
            axs.set_ylabel('wind speed (m/s)')

        elif plot == "scint vs. elevation":
            fig_scint_zen, axs = plt.subplots(2, 1, dpi=125)
            axs[0].plot(np.rad2deg(zenith), self.var_rytov, label='Rytov')
            axs[0].set_title(f'Scintillation variance VS. Zenith angle')
            axs[0].set_ylabel('Rytov variance [-]')

            axs[0].plot(np.rad2deg(zenith), self.var_scint, linestyle= "--", label='Scint index')
            axs[0].set_ylabel('Scintillation index [-]')
            axs[0].legend()

            axs[1].plot(np.rad2deg(zenith), self.Af, linestyle="--", label='Dr = '+str(D_r))
            axs[1].set_ylabel('Aperture averaging factor [-]')
            axs[1].set_xlabel('Zenith angle [deg]')
            axs[1].legend()


    def print(self, index, elevation):
        print('------------------------------------------------')
        print('------------TURBULENCE-ANALYSIS-----------------')
        print('------------------------------------------------')

        print('Elevation     [deg]   : ', elevation[index])
        print('Windspeed rms [m/s]   : ', self.windspeed_rms)
        print('Cn^2 at h(0)  [m^2/3] : ', self.Cn[0])
        print('r0             [m]    : ', self.r0[index])
        print('Rytov std              (1 rms) [-] : ', self.std_rytov[index])
        print('Scintillation std      (1 rms)     : ', self.std_scint[index])
        print('Beam wander std        (1 rms) [urad] : ', self.std_bw[index])
        print('Beam wander std        (1 rms) [m]    : ', self.std_bw_r[index])
        print('Angle of arrival std (1 rms) [rad]    : ', self.std_aoa[index])
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

    def T_ext(self, zenith_angles, method="standard_atmosphere"):
        if method == "standard_atmosphere":
            self.T_clouds = 0.8

            T_ext = np.exp( -self.b_v * self.H_scale * abs(1/np.cos(zenith_angles) / 1000 *
                            (1 - np.exp(-self.range_link / (self.H_scale * abs(1/np.cos(zenith_angles)))  ))))


            self.T_ext = T_ext * self.T_clouds
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




