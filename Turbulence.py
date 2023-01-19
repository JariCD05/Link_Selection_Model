# Load standard modules
import numpy as np
from scipy.stats import lognorm, gamma
from scipy.special import kv, kn as kv, kn
from scipy.special import gamma as gamma_function

import constants as cons
from matplotlib import pyplot as plt


class turbulence:
    def __init__(self,
                 range,
                 heights,
                 h_cruise = 10.0E3,
                 wavelength = 1550E-9,
                 D_r = 0.15,
                 D_t = 0.15,
                 angle_pe = 5.0E-6,
                 r0 = 0.05,                 # HVB 5/7 model
                 angle_iso = 7.0E-6,        # HVB 5/7 model
                 att_coeff = 0.0025,         # Clear atmosphere: 0.0025 (1550nm, 690 nm), 0.1 (850nm), 0.13 (550nm)
                 P_thres = 1.0E-6,
                 link = 'up',
                 h_sc = 1000.0E3,
                 eff_coupling = 0.8
                 ):

        # Range and height
        self.range = range
        self.heights = heights
        self.h0 = heights[0]
        self.h_cruise= h_cruise
        self.h_limit = 30.0E3 #m
        self.h_sc = h_sc

        # Terminal properties
        self.D_t = D_t
        self.D_r = D_r
        self.angle_pe = angle_pe
        self.wavelength = wavelength
        self.k = 2*np.pi/self.wavelength
        self.eff_coupling = eff_coupling

        self.P_thres = P_thres

        # Turbulence parameters
        self.angle_iso = angle_iso


    # ------------------------------------------------------------------------
    # --------------------------TURUBLENCE-ENVIRONMENT------------------------
    # --------------------------Cn^2--windspeed--r0--Stehl-Ratio--------------
    def Cn(self, turbulence_model = "HVB", dim = 2):

        if turbulence_model == "HVB":
            if dim == 1:
                self.Cn = 5.94E-53 * (self.windspeed_rms / 27) ** 2 * self.heights ** 10 * \
                          np.exp(-self.heights / 1000.0) + \
                          2.75E-16 * np.exp(-self.heights / 1500.0)

            elif dim == 2:
                self.Cn = np.zeros(np.shape(self.heights))
                for i in range(len(self.windspeed_rms)):
                    windspeed_rms = self.windspeed_rms[i]
                    heights = self.heights[i]
                    self.Cn[i] = 5.94E-53 * (windspeed_rms / 27) ** 2 * heights ** 10 * \
                                    np.exp(-heights / 1000.0) + \
                                    2.75E-16 * np.exp(-heights / 1500.0)

        elif turbulence_model == "HVB_57":
            r0 = 0.05
            self.angle_iso = 7.0E-6
            A = 1.29E-12 * r0 ** (-5 / 3) * self.wavelength ** 2 - 1.61E-13 * self.angle_iso ** (-5 / 3) * self.wavelength ** 2 + 3.89E-15
            self.Cn = 5.94*10**(-53) * (self.windspeed_rms/27)**2 * self.heights**10 * np.exp(-self.heights/ 1000.0)+ \
                      2.7E-16 * np.exp(-self.heights / 1500.0) + \
                      A * np.exp(-self.heights / 100)
        return self.Cn

    def windspeed(self, slew, Vg, wind_model_type = "Bufton", dim = 2):
        if wind_model_type == "Bufton":

            if dim == 1:
                self.windspeed = np.zeros(len(self.heights[self.heights < self.h_limit]))
                self.windspeed = abs(slew) * self.heights[self.heights < self.h_limit] + np.ones(
                    np.shape(self.heights[self.heights < self.h_limit])) * Vg + \
                                    30 * np.exp(-((self.heights[self.heights < self.h_limit] - 9400.0) / 4800.0) ** 2)

                self.windspeed_rms = np.sqrt(1 / (self.h_limit - self.h_cruise) *
                                                np.trapz(self.windspeed ** 2, x=self.heights[self.heights < self.h_limit]))

            elif dim == 2:
                self.windspeed     = np.zeros(len(self.heights[self.heights < self.h_limit])).reshape((len(slew),-1))
                self.windspeed_rms = np.zeros(len(slew))

                for i in range(len(slew)):
                    heights = self.heights[i]
                    self.windspeed[i] = abs(slew[i]) * heights[heights < self.h_limit] + np.ones(
                        np.shape(heights[heights < self.h_limit])) * Vg + \
                                        30 * np.exp(-((heights[heights < self.h_limit] - 9400.0) / 4800.0) ** 2)

                    self.windspeed_rms[i] = np.sqrt(1 / (self.h_limit - self.h_cruise[i]) *
                                                    np.trapz(self.windspeed[i] ** 2, x=heights[heights < self.h_limit]))
            return self.windspeed_rms

    def r0(self, zenith_angles, dim = 2):
        self.r0 = np.zeros(len(zenith_angles))

        if dim == 1:
            for i in range(len(zenith_angles)):
                self.r0[i] = (0.423 * self.k ** 2 * 1 / abs(np.cos(zenith_angles[i])) *
                              np.trapz(self.Cn[self.heights<self.h_limit], x=self.heights[self.heights<self.h_limit])) ** (-3 / 5)

        if dim == 2:
            for i in range(len(zenith_angles)):
                heights = self.heights[i]
                Cn = self.Cn[i]
                self.r0[i] = (0.423 * self.k ** 2 * 1 / abs(np.cos(zenith_angles[i])) * np.trapz(Cn[heights<self.h_limit], x=heights[heights<self.h_limit])) ** (-3 / 5)
        return self.r0

    def stehl_ratio(self, D):
        return (1 + (D / self.r0)**(5/3))**(-6/5)

    # ------------------------------------------------------------------------
    # ---------------------------------LOSSES---------------------------------
    # ------------------------------------------------------------------------

    def WFE(self, tip_tilt = "NO"):
        if tip_tilt == "YES":
            WFE = 1.03 * (self.D_r/self.r0)**(5/3)
        elif tip_tilt == "NO":
            WFE = 0.134 * (self.D_r/self.r0)**(5/3)

        self.T_WFE = self.eff_coupling * np.exp(-WFE)
        return self.T_WFE

    # ------------------------------------------------------------------------
    # -------------------------------VARIANCES--------------------------------
    # ------------------------------------------------------------------------
    def var_rytov(self, zenith_angles):
        self.var_rytov = np.zeros(len(zenith_angles))
        for i in range(len(zenith_angles)):
            self.var_rytov[i] = 2.25 * self.k ** (7 / 6) * (1 / np.cos(zenith_angles[i])) ** (11 / 6) * \
                             np.trapz(self.Cn * (self.heights - self.h_cruise) ** (5 / 6), x=self.heights)
        return self.var_rytov

    def var_scint(self, PDF_type="lognormal"):
        if PDF_type == "lognormal":
            self.var_scint = np.log(4*self.var_rytov) - 1

        elif PDF_type == "gamma-gamma":
            self.var_scint = np.exp(
                0.49 * self.var_rytov / (1 + 1.11 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (7 / 6) +
                0.51 * self.var_rytov / (1 + 0.69 * np.sqrt(self.var_rytov) ** (12 / 5)) ** (5 / 6)) - 1

        return self.var_scint

    def var_bw(self, w0, zenith_angles):
        self.var_bw = 0.54 * (self.h_sc - self.h0)**2 * 1/np.cos(zenith_angles) * (self.wavelength / (2*w0))**2 * (2*w0 / self.r0)**(5/3)
        return self.var_bw

    def var_AoA(self, D_r, wavelength):
        self.beta = 0.182 * (D_r/self.r0)**5/3 * (wavelength/D_r)**2
        return self.beta

    def Aperture_averaging(self):
        A = (1 + 1.1*(self.D_r**2 / (self.wavelength*self.h0*np.cos(self.angle_pe)))**(7/6))**-1
        var_scint = self.var_scint0 / A
        return var_scint

    # ------------------------------------------------------------------------
    # -------------------------------PDF-MODELS-------------------------------
    # ------------------------------------------------------------------------
    def PDF(self, zenith_angles = 0.0, PDF_type="lognormal", steps=1000):
        self.I_turb = np.zeros((len(zenith_angles), steps))
        self.P_turb = np.zeros((len(zenith_angles), steps))
        self.r_turb = np.zeros((len(zenith_angles), steps))
        self.x_turb = np.zeros((len(zenith_angles), steps))
        self.pdf_turb = np.zeros((len(zenith_angles), steps))

        if PDF_type == "lognormal":
            # var = np.log(self.var_rytov + 1)
            # mean = -1/2 * var
            # sigma = np.sqrt(var)
            sigma = np.sqrt(4* self.var_rytov)

            for i in range(len(zenith_angles)):
                self.x_turb[i] = np.linspace(0.0, lognorm.ppf(0.99, sigma[i]), steps)
                self.pdf_turb[i] = lognorm.pdf(self.x_turb[i], sigma[i])
                self.r_turb[i] = lognorm.rvs(size=steps, s=sigma[i])
                # self.I_turb[i] = I0 * self.r_turb[i]
                # self.P_turb[i] = self.I_turb[i] * np.pi/4 * self.D_r**2

            # return self.I_turb, self.P_turb
            return self.r_turb


        elif PDF_type == "gamma-gamma":
            random_seed = 42
            np.random.seed(random_seed)
            print('\n Random Seed :', random_seed, '\n')

            alpha = ( np.exp(0.49*self.var_rytov**2 / ( 1+1.11*np.sqrt(self.var_rytov)**(12/5))**(7/6)) - 1)**-1
            beta  = ( np.exp(0.51*self.var_rytov**2 / ( 1+0.69*np.sqrt(self.var_rytov)**(12/5))**(5/6)) - 1)**-1
            sigma = 1/alpha + 1/beta + 1/(alpha*beta)

            for i in range(len(zenith_angles)):

                x_alpha = np.linspace(gamma.ppf(0.01, alpha[i]), gamma.ppf(0.99, alpha[i]), steps)
                x_beta  = np.linspace(gamma.ppf(0.01, beta[i]), gamma.ppf(0.99, beta[i]), steps)
                f_alpha = gamma.pdf(x_alpha, alpha[i])
                f_beta  = gamma.pdf(x_beta, beta[i])

                self.x_turb[i] = np.linspace(0, 5, steps)
                Kv = kv(x_alpha[i]-x_beta[i], 2 * np.sqrt(x_alpha[i] * x_beta[i] * self.x_turb[i]))

                self.pdf_turb[i] = 2*(x_alpha[i]*x_beta[i])**((x_alpha[i]+x_beta[i])/2) / \
                                   (gamma_function(x_alpha[i])*gamma_function(x_beta[i])) * \
                                   self.x_turb[i]**((x_alpha[i]+x_beta[i])/2-1) * Kv
            # return pdf_gamma, I, sigma

    def plot(self, t, plot = "scint pdf", zenith=0.0):

        fig_turb, ax1 = plt.subplots(1, 1, dpi=125)

        if plot == "scint pdf":
            fig_scint, axs = plt.subplots(2, 1, dpi=125)
            axs[0].plot(self.x_turb[1], self.pdf_turb[1], label="variance: "+str(self.var_rytov[1]))
            axs[0].set_title(f'Intensity with scintillation - Lognorm distribution')
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Normalized intensity [-]')
            axs[0].legend()

            axs[1].plot(t, self.r_turb[1])
            # axs[1].plot(np.array((t[0], t[-1])), np.ones(2) * I0, label="I0")
            axs[1].plot(np.array((t[0], t[-1])), np.ones(2) * np.mean(self.r_turb[1]), label="I_turb, mean")
            axs[1].set_ylabel('Intensity [W/m^2]')
            axs[1].set_xlabel('Time [s]')
            axs[1].legend()

        # Plot Cn vs. Height
        elif plot == "Cn":
            fig_cn, axs = plt.subplots(1, 1, dpi=125)
            axs.set_title(f'Cn  vs  heights')
            axs.plot(self.heights[:20], self.Cn[:20])
            axs.set_yscale('log')
            axs.set_xscale('log')

        elif plot == "scint vs. zenith":
            fig_scint_zen, axs = plt.subplots(1, 1, dpi=125)
            axs.plot(np.rad2deg(zenith), self.var_rytov)
            axs.set_title(f'Scintillation variance VS. Zenith angle')
            axs.set_ylabel('Rytov variance [-]')
            axs.set_xlabel('Zenith angle [deg]')

            axs.plot(np.rad2deg(zenith), self.var_scint, linestyle= "--")
            axs.set_ylabel('Scintillation index [-]')
            axs.set_xlabel('Zenith angle [deg]')

    def print(self, P_r = 1.0E-5):
        P_turb = P_r * self.r_turb
        print('------------------------------------------------')
        print('------------TURBULENCE-ANALYSIS-----------------')
        print('------------------------------------------------')

        print('Windspeed rms [m/s] : ', self.windspeed_rms)
        print('Cn^2       [m^2/3]  : ', self.Cn[0])
        print('r0             [m]  : ', self.r0[0])
        print('Rytov variance [-]  : ', self.var_rytov[0])
        print('Scintillation index (moderate/strong) [-]     : ', self.var_scint[0])
        print('Rec. power with turbulence (mean)     [dBW]   : ', cons.W2dB(np.mean(P_turb[0])))
        # print('Rec. intensity with turbulence (mean) [W/m^2] : ', np.mean(self.I_turb))
        print('Wave front error loss                 [dBW]   : ', cons.W2dB(self.T_WFE[0]))

