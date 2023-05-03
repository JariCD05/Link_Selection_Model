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
        # if link == "down":
        #     np.flip(self.heights)

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
        Lambda0 = 2 * self.ranges / (k_number * w0 ** 2)
        Lambda = 2 * self.ranges / (k_number * w_r ** 2)
        # Only during uplink is the turbulence beamspread considered significant.
        # This is ignored for downlink

        if link == 'up':
            self.w_LT = beam_spread_turbulence_LT(self.r0, w_r)
            self.w_ST = beam_spread_turbulence_ST(Lambda0, Lambda, self.var_rytov, w_r)
            #REF:
            # self.w_ST1 = w0**2 * (1 + (self.ranges / z_r)**2) + 2 * (4.2 * self.ranges / (k_number * self.r0) * (1 - 0.26 * (self.r0/w0)**(1/3)))**2
            self.h_beamspread = (w_r / self.w_LT) ** 2

        elif link == 'down':
            self.h_beamspread = 1.0

    def Strehl_ratio_func(self, tip_tilt="YES"):
        # REF: R. SAATHOF, SLIDES LECTURE 3, 2021
        # REF: R. PARENTI, 2006, EQ.1-3
        D = 2 ** (3 / 2) * w0
        if tip_tilt == "YES":
            var_WFE = 1.03 * (D_t / self.r0) ** (5 / 3)
        elif tip_tilt == "NO":
            var_WFE = 0.134 * (D_t / self.r0) ** (5 / 3)
        self.h_strehl = np.exp(-var_WFE)


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
            h0 = (np.trapz(self.Cn * self.heights ** 2,       x=self.heights) /
                  np.trapz(self.Cn * self.heights ** (5 / 6), x=self.heights)) ** (6 / 7)
            self.Af = (1 + 1.1 * (D_r ** 2 / (wavelength * h0 / np.cos(zenith_angles))) ** (7 / 6)) ** (-1)
            self.var_scint = self.var_scint * self.Af
            self.std_scint = np.sqrt(self.var_scint)
            # # Create lognormal parameters
            self.mean_scint_X = -0.5 * np.log(self.var_scint + 1)
            # self.std_scint_X  = np.sqrt(np.log(self.var_scint + 1)) # Giggenbach
            self.std_scint_X = np.sqrt(1/4 * np.log(self.var_scint + 1))  # Andrews, Random Media, Ch.5, eq.95


        # elif PDF_scintillation == "gamma-gamma":
            self.alpha = (np.exp(0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6)) - 1) ** -1
            self.beta  = (np.exp(0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1) ** -1
            self.var_scint_gg = 1 / self.alpha + 1 / self.beta + 1 / (self.alpha * self.beta)


    def var_bw_func(self, zenith_angles):
        #REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.50
        # This reference computes the beam wander variance as a displacement at the receiver in meters (1 rms)
        # Beam wander is typically 10-100 urad.
        self.var_bw_r = 0.54 * (h_SC - h_AC)**2 * 1/np.cos(zenith_angles)**2 * (wavelength / (2*w0))**2 * (2*w0 / self.r0)**(5/3) * \
                      (1 - ((self.kappa0**2 * w0**2) / (1 + self.kappa0**2 * w0**2))**(1/6))
        self.std_bw_r = np.sqrt(self.var_bw_r)

        self.std_bw   = self.std_bw_r / self.ranges
        self.var_bw   = self.std_bw ** 2
        self.mean_bw  = np.zeros(np.shape(self.std_bw))
        if PDF_beam_wander == "rayleigh":
            # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
            self.std_bw  = np.sqrt(2 / (4 - np.pi) * self.var_bw)
            self.mean_bw = np.sqrt(np.pi / 2) * self.std_bw



        # REF: Scintillation and Beam-Wander Analysis in an Optical Ground Station-Satellite Uplink, EQ.10-11
        # self.var_bw = 1 / 2 * 2.07 * 1 / np.cos(zenith_angles) * np.trapz(self.Cn * (self.heights[-1] - self.heights) ** 2 * (1 / self.w_ST[:, None]) ** (1 / 3), x=self.heights)
        # REF: MODELING THE PDF FOR THE IRRADIANCE OF AN UPLINK BEAM IN THE PRESENCE OF BEAM WANDER, R. PARENTI 2006, EQ.10
        # self.var_bw = 5.09 / (k_number**2 * self.r0**(5/3) * w0**(1/3))

    def var_aoa_func(self, zenith_angles):
        # REF: Andrews, Laser beam propagation through random media, EQ.12.9
        w_r, z_r = beam_spread(w0, self.ranges)  # beam radius at receiver (without turbulence) (in m)
        Lambda0 = 2 * self.ranges / (k_number * w0 ** 2)
        Lambda = 2 * self.ranges / (k_number * w_r ** 2)
        Theta0 = Lambda0 / Lambda ** 2 - Lambda0 ** 2
        Theta = Theta0 / (Theta0 ** 2 + Lambda0 ** 2)
        Theta_bar = 1 - Theta
        # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.28
        mu_0 = np.trapz(self.Cn, x=self.heights)
        mu_1u = np.trapz(self.Cn * (Theta[:, None] + Theta_bar[:, None]  *
                                   (self.heights - self.heights[0]) / (self.heights[-1] - self.heights[0]))**(5/3), x=self.heights)

        mu_2u = np.trapz(self.Cn * (1 - (self.heights - self.heights[0]) / (self.heights[-1] - self.heights[0]))**(5/3), x=self.heights)
        if link == 'up':
            self.var_aoa = 2.91 * (mu_1u + 0.62 * mu_2u * Lambda**(11/6)) / np.cos(zenith_angles) * (2 * D_r)**(1/3)
        elif link == 'down':
            self.var_aoa = 2.91 * mu_0 / np.cos(zenith_angles) * (2 * D_r)**(1/3)

        self.std_aoa = np.sqrt(self.var_aoa)
        self.mean_aoa = np.zeros(np.shape(self.std_aoa))

        if PDF_beam_wander == "rayleigh":
            # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
            self.std_aoa_r  = np.sqrt(2 / (4 - np.pi) * self.var_aoa)
            self.mean_aoa_r = np.sqrt(np.pi / 2) * self.std_aoa

    # ------------------------------------------------------------------------
    # -------------------------------PDF-MODELS-------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def create_turb_distributions(self, data, steps=1000, effect = "scintillation"):
        # VERIFICATION REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, FIG.5.1
        if effect == "scintillation":
            if PDF_scintillation == "lognormal":
                x_scint, pdf_scint = dist.lognorm_pdf(mean=self.mean_scint_X[:, None], sigma=self.std_scint_X[:, None], steps=steps)
                h_scint = dist.lognorm_rvs(data, mean=self.mean_scint_X[:, None], sigma=self.std_scint_X[:, None])

            elif PDF_scintillation == "gamma-gamma":
                h_scint = np.zeros((len(self.ranges), steps))
                x_scint = np.zeros((len(self.ranges), steps))
                pdf_scint = np.zeros((len(self.ranges), steps))
                cdf_scint = np.zeros((len(self.ranges), steps))

                for i in range(len(self.ranges)):
                    x_scint[i], pdf_scint[i] = dist.gg_pdf(alpha=self.alpha[i], beta=self.beta[i], steps=steps)
                    cdf_scint[i, 0] = pdf_scint[i, 0]
                    for j in range(1, len(pdf_scint[i])):
                        cdf_scint[i, j] = np.trapz(pdf_scint[i, 1:j], x=x_scint[i, 1:j])
                    h_scint[i] = dist.gg_rvs(pdf_scint[i], steps)
            return h_scint

        elif effect == "beam wander":
            if PDF_beam_wander == "rayleigh":
                x_bw, pdf_bw = dist.rayleigh_pdf(sigma=self.std_bw[:, None], steps=steps)
                angle_bw_X = dist.norm_rvs(data=data[0], sigma=self.std_bw[:, None], mean=0)
                angle_bw_Y = dist.norm_rvs(data=data[1], sigma=self.std_bw[:, None], mean=0)
                angle_bw_R = dist.rayleigh_rvs(data=data, sigma=self.std_bw[:, None])
                return angle_bw_R

            elif PDF_beam_wander == "rice":
                x_bw, pdf_bw = dist.norm_pdf(effect=effect, sigma=self.std_bw[:, None], steps=steps)
                angle_bw_X = dist.norm_rvs(data=data[0], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                angle_bw_Y = dist.norm_rvs(data=data[1], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                angle_bw_R = np.sqrt(angle_bw_X ** 2 + angle_bw_Y ** 2)
                return angle_bw_R

        elif effect == "angle of arrival":
            if PDF_AoA == "rayleigh":
                x_aoa, pdf_aoa = dist.rayleigh_pdf(sigma=self.std_aoa[:, None], steps=steps)
                angle_aoa_X = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=0)
                angle_aoa_Y = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=0)
                angle_aoa_R = dist.rayleigh_rvs(data=data, sigma=self.std_aoa[:, None])
                return angle_aoa_R

            if PDF_AoA == "rice":
                x_aoa, pdf_aoa = dist.norm_pdf(effect=effect, sigma=self.std_aoa[:, None], steps=steps)
                angle_aoa_X = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                angle_aoa_Y = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                angle_aoa_R = np.sqrt(angle_aoa_X ** 2 + angle_aoa_Y ** 2)
            return angle_aoa_R

    # def test_PDF(self, effect = "scintillation", index = 20):
    #     if effect == "scintillation":
    #         fig_test_pdf_scint, ax_test_pdf_scint = plt.subplots(3, 1)
    #         dist.plot(ax=ax_test_pdf_scint,
    #                   sigma=self.std_scint_X[:, None],
    #                   mean=self.mean_scint_X[:, None],
    #                   x=self.x_scint,
    #                   pdf=self.pdf_scint,
    #                   data=self.h_scint,
    #                   index=index,
    #                   effect=effect,
    #                   name=PDF_scintillation)
    #
    #     elif effect == "beam wander":
    #         fig_test_pdf_bw, ax_test_pdf_bw = plt.subplots(3, 1)
    #         dist.plot(ax=ax_test_pdf_bw,
    #                   sigma=self.std_bw[:, None],
    #                   mean=self.mean_bw[:, None],
    #                   x=self.x_bw,
    #                   pdf=self.pdf_bw,
    #                   data=self.angle_bw_R,
    #                   index=index,
    #                   effect=effect,
    #                   name=PDF_beam_wander)
    #
    #     elif effect == "angle of arrival":
    #         fig_test_pdf_aoa, ax_test_pdf_aoa = plt.subplots(3, 1)
    #         dist.plot(ax=ax_test_pdf_aoa,
    #                   sigma=self.std_aoa[:, None],
    #                   mean=self.mean_aoa[:, None],
    #                   x=self.x_aoa,
    #                   pdf=self.pdf_aoa,
    #                   data=self.angle_aoa_R,
    #                   index=index,
    #                   effect=effect,
    #                   name=PDF_AoA)


    def print(self, index, elevation, ranges):
        print('------------------------------------------------')
        print('------------TURBULENCE-ANALYSIS-----------------')
        print('------------------------------------------------')

        print('Range         [m]     : ', ranges[index])
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
        print('Wave front error loss         [dBW]   : ', W2dB(self.h_strehl[index]))
        print('Beam spread loss              [dBW]   : ', W2dB(self.h_beamspread[index]))
        print('Beam spread                     [m]   : ', W2dB(self.w_ST[index]))




class attenuation:
    def __init__(self,
                 att_coeff = 0.0025,         # Clear atmosphere: 0.0025 (1550nm, 690 nm), 0.1 (850nm), 0.13 (550nm)
                 refraction = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.05, 1.3))
                 ):

        # Range and height
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

    def h_ext_func(self, range_link, zenith_angles, method="standard_atmosphere"):
        if method == "standard_atmosphere":
            self.h_ext = np.exp( -self.b_v * self.H_scale * abs(1/np.cos(zenith_angles) / 1000 *
                            (1 - np.exp(-range_link / (self.H_scale * abs(1/np.cos(zenith_angles)))  ))))

    def h_clouds_func(self, data):
        mean = 1.44 #dB
        # mean = dB2W(mean)
        var = 2.31
        sigma = np.sqrt(var)
        self.h_clouds = dB2W(-dist.lognorm_rvs(data=data, sigma=sigma, mean=mean))

    def plot(self):

        fig_ext = plt.figure(figsize=(6, 6), dpi=125)
        ax_ext = fig_ext.add_subplot(111)
        ax_ext.set_title('Transmission due to Atmospheric attenuation')
        ax_ext.plot(np.rad2deg(self.zenith_angles), self.h_ext, linestyle='-')
        ax_ext.set_xlabel('Zenith angle (deg')
        ax_ext.set_ylabel('Transmission')

    def print(self):

        print('------------------------------------------------')
        print('ATTENUATION/EXTINCTION ANALYSIS')
        print('------------------------------------------------')

        print('Extinction loss [dBW] : ', dB2W(self.h_ext))




