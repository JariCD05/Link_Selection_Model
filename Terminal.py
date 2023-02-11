
from constants import *
from helper_functions import *
from PDF import dist
# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.special import erfc, erf, erfinv, erfcinv
from scipy.stats import rice, rayleigh, norm


class terminal_properties:
    def __init__(self,
                 D = 0.1,                   # 10 cm aperture
                 eff_quantum = 1.0,         # Responsivity (quantum efficiency)
                 BW = 1.0E9,                # 1 MHz Bandwidth
                 modulator = "OOK-NRZ",     # OOK modulation technique OOK/2-PPM/L-PPM/BPSK/DPSK
                 detection = "ADP",         # Detection technique PIN/ADP/Preamp/Coherent
                 R_L = 50,                  # Load resistor
                 T_s = 300,                 # Temperature
                 FOV = 1.0E-8,              # Field of view of the terminal (solid angle)
                 delta_wavelength = 5.0,    # Optical bandpass filter
                 M = 1000.0,                # Gain of receiver amplifier (ADP or OA)
                 F = 4.0,                   # Amplifier noise factor
                 ):

        self.BER_thres = BER_thres
        self.eff_quantum = eff_quantum
        self.D = D
        self.FOV = FOV
        self.BW = BW
        self.delta_wavelength = delta_wavelength
        self.modulator = modulator
        self.detection = detection
        self.R = eff_quantum * q / (h * v)
        self.R_L = R_L
        self.T_s = T_s
        self.M = M
        self.F = F

        # if modulator == "OOK-NRZ":
        #     self.Ts = 1/self.BW
        #     self.Tb = self.Ts/N_symb
        #     self.N_symb = N_symb

    # ------------------------------------------------------------------------
    # ----------------------------INCOMING PHOTONS----------------------------
    # ------------------------------------------------------------------------
    def Np_func(self, P_r):
        # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
        self.Np = Np_func(P_r, data_rate, self.eff_quantum)
        return self.Np

    # ------------------------------------------------------------------------
    # ----------------------------------NOISE---------------------------------
    # ------------------------------------------------------------------------
    # Incoming noise, 5 noise types: Shot, termal, background and beat (mixing)
    def noise(self,
              noise_type = "shot",
              P_r = 0.0,
              I_sun = 0.02):

        if noise_type == "shot":
            self.noise_sh = 2 * q * self.R * P_r * self.BW
            return self.noise_sh

        elif noise_type == "thermal":
            self.noise_th = (4 * k * self.T_s * self.BW / self.R_L)
            return self.noise_th

        elif noise_type == "background":
            A = 1 / 4 * np.pi * self.D ** 2
            # self.noise_bg = 2 * self.BW * self.R * P_bg * v * cons.h
            P_bg = I_sun * A * self.delta_wavelength*10**9 * self.FOV
            self.noise_bg = 2 * self.BW * self.R * P_bg * q
            return self.noise_bg

        elif noise_type == "beat":
            self.noise_beat = 0.0
            return self.noise_beat

        elif noise_type == "total":
            return self.M**2*self.F*(self.noise_sh+self.noise_bg) + self.noise_th

    # ------------------------------------------------------------------------
    # --------------------------------THRESHOLD-------------------------------
    # ------------------------------------------------------------------------
    # Number of received photons per bit in receiver terminal
    # def Np_thres(self):
    #     self.Np_thres = (self.P_r_thres / (Ep * data_rate) / self.eff_quantum).to_base_units()
    #     return self.Np_thres

    def threshold(self):

        # First compute SNR threshold. This depends on the modulation type (BER --> SNR)
        if self.modulator == "OOK-NRZ":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*BER_thres) )**2
            self.Q_thres   = ( np.sqrt(2) * erfcinv(2*BER_thres) )
        elif self.modulator == "2-PPM" or self.modulator == "2PolSK":
            self.SNR_thres = -2 * np.log(2*self.BER_thres)
            self.Q_thres = 0.0

        elif self.modulator == "M-PPM":
            self.SNR_thres = 2 / np.sqrt(self.M*np.log2(self.M)) * erfcinv(4*BER_thres / self.M)
            self.Q_thres = 0.0

        elif self.modulator == "DPSK":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*BER_thres) )**2
            self.Q_thres = 0.0

        elif self.modulator == "BPSK":
            self.SNR_thres = 1/2 * erfcinv(2*BER_thres)**2
            self.Q_thres = 0.0

        # Second compute Pr threshold. This depends on the detection type and noise (SNR --> Pr)
        if self.detection == "PIN":
            self.P_r_thres = np.sqrt(self.SNR_thres * self.noise_th) / self.R
        elif self.detection == "ADP":
            self.P_r_thres = np.sqrt(self.SNR_thres**2 * (self.M**2*self.F*(self.noise_sh+self.noise_bg) + self.noise_th)) / (self.M * self.R)

        # if self.detection == "PIN" and self.modulator == "OOK-NRZ":
        #     self.P_r_thres = self.Q_thres * self.noise_th / self.R
        #
        # elif self.detection == "ADP" and self.modulator == "OOK-NRZ":
        #     self.P_r_thres = self.Q_thres / self.R * ( 2 * q * self.F * self.Q_thres * self.BW
        #                                                + self.noise_th/self.M)
        #
        # elif self.detection == "PIN" and self.modulator == "BPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "ADP" and self.modulator == "BPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "PIN" and self.modulator == "DPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "ADP" and self.modulator == "DPSK":
        #     self.P_r_thres = 0.0

        self.Np_shot_limit = self.SNR_thres / self.eff_quantum

        # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
        self.Np_thres = Np_func(self.P_r_thres, data_rate, self.eff_quantum)

        return self.SNR_thres, self.P_r_thres, self.Np_thres

    # def P_r_thres(self):
        # if self.detection == "PIN" and self.modulator == "OOK-NRZ":
        #     self.P_r_thres = self.Q_thres * self.noise_th / self.R
        #
        # elif self.detection == "ADP" and self.modulator == "OOK-NRZ":
        #     self.P_r_thres = self.Q_thres / self.R * ( 2 * q * self.F * self.Q_thres * self.BW
        #                                                + self.noise_th/self.M)
        #
        # elif self.detection == "PIN" and self.modulator == "BPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "ADP" and self.modulator == "BPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "PIN" and self.modulator == "DPSK":
        #     self.P_r_thres = 0.0
        #
        # elif self.detection == "ADP" and self.modulator == "DPSK":
        #     self.P_r_thres = 0.0


        # if self.detection == "PIN":
        #     self.P_r_thres = np.sqrt(self.SNR_thres * self.noise_th) / self.R
        # elif self.detection == "ADP":
        #     self.P_r_thres = np.sqrt(self.SNR_thres * (self.M*self.F*(self.noise_sh+self.noise_bg) + self.noise_th)) / (self.M * self.R)

        # elif self.detection == "Preamp":
        #     self.SNR = (M * P_r * self.R)**2 / (noise_sh+noise_bg + noise_th + noise_beat)
        # elif self.detection == "quantum-limit":
        #     self.SNR = (P_r * self.R)**2 / noise_sh
        # self.P_r_thres = self.P_r_thres.to_base_units()
        # return self.P_r_thres

    # ------------------------------------------------------------------------
    # --------------------------------SNR-&-BER-------------------------------
    # ------------------------------------------------------------------------

    def SNR_func(self, P_r):

        if self.detection == "PIN":
            self.SNR = np.sqrt((P_r * self.R)**2 / self.noise_th)
        elif self.detection == "ADP":
            self.SNR = np.sqrt((self.M * P_r * self.R)**2 / ( self.M**2*self.F*(self.noise_sh+self.noise_bg) + self.noise_th))
        elif self.detection == "Preamp":
            self.SNR = np.sqrt((self.M * P_r * self.R)**2 / (self.M**2*(self.noise_sh+self.noise_bg) + self.noise_th + self.noise_beat))
        elif self.detection == "quantum-limit":
            self.SNR = np.sqrt((P_r * self.R)**2 / self.noise_sh)
        self.SNR = self.SNR
        return self.SNR

    def BER_func(self):
        if self.modulator == "OOK-NRZ":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.8
            self.BER = 1/2 * erfc( np.sqrt(self.SNR)/np.sqrt(2) )

        elif self.modulator == "2-PPM" or self.modulator == "2PolSK":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 FIG.7
            self.BER = 1/2 * np.exp(-1/2 * self.SNR)

        elif self.modulator == "M-PPM":
            self.BER = self.M/4 * erfc( 1/2 * self.SNR * np.sqrt(self.M*np.log2(self.M)) )

        elif self.modulator == "DPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.48
            self.BER = 1/2 * erfc( np.sqrt( self.SNR / 2) )

        elif self.modulator == "BPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.45
            self.BER = 1/2 * erfc( np.sqrt(self.SNR))

        elif self.modulator == "QPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.52
            self.BER = erfc(np.sqrt(self.SNR))

        return self.BER

    # ------------------------------------------------------------------------
    # --------------------------------POINTING--------------------------------
    # ------------------------------------------------------------------------
    def pointing(self, zenith_angles,
                 ranges,
                 pointing_error=0.0,
                 dist_type="rayleigh",
                 jitter_variance=0.5,
                 steps = 1000,
                 w0 = 0.15,
                 angle_divergence = angle_div_ac,
                 r0 = 0.0):

        w_r = beam_spread(w0, ranges)
        w_LT = beam_spread_turbulence(r0, w0, w_r)

        self.r_pe = np.zeros((len(zenith_angles), steps))
        self.h_pe = np.zeros((len(zenith_angles), steps))

        if dist_type == "rayleigh":
            self.x_pe = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), steps)
            self.pdf_pe = rayleigh.pdf(self.x_pe, scale=jitter_variance)
            self.angle_pe_r = rayleigh.rvs(size=steps, scale=jitter_variance)

        elif dist_type == "gaussian":
            # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
            self.x_pe, self.pdf_pe = dist.norm_pdf(var=jitter_variance, steps=steps)
            self.angle_pe_x = dist.norm_rvs(var=jitter_variance, steps=steps) + pointing_error
            self.angle_pe_y = dist.norm_rvs(var=jitter_variance, steps=steps) + pointing_error

            self.angle_pe_r = np.sqrt(self.angle_pe_x ** 2 + self.angle_pe_y ** 2)

        for i in range(len(zenith_angles)):
            self.r_pe[i] = self.angle_pe_r * ranges[i]
            self.h_pe[i] = np.exp(-self.r_pe[i] ** 2 / w_LT[i] ** 2)

        return self.r_pe, self.h_pe




    def plot(self, t, data: np.array, plot = "BER & SNR"):

        if plot == "BER & SNR vs. time":
            fig_ber, axs = plt.subplots(3, 1)
            axs[0].plot(t, self.BER[0])
            axs[0].set_title(f'BER (OOK & PIN) with scintillation')
            axs[0].set_ylabel('BER')
            axs[0].set_ylim(1.0E-9, 1.0)
            axs[0].set_yscale('log')
            axs[0].invert_yaxis()

            axs[1].plot(t, self.SNR[0])
            axs[1].set_title(f'SNR (OOK & PIN) with scintillation')
            axs[1].set_ylabel('SNR')
            axs[1].set_xlabel('Time (s)')

            axs[2].plot(self.SNR[0], self.BER[0])
            axs[2].set_title(f'SNR vs SNR')
            axs[2].set_ylabel('BER')
            axs[2].set_yscale('log')
            axs[2].set_xlabel('SNR')
            axs[2].set_xscale('log')

        if plot == "data vs elevation":
            fig_data_vs_elevation, axs = plt.subplots(3, 1)
            axs[0].plot(data[:, 0], data[:, 6])
            axs[0].set_title(f'BER vs elevation')
            axs[0].set_ylabel('BER')
            axs[0].set_ylim(1.0E-9, 1.0)
            axs[0].set_yscale('log')
            axs[0].invert_yaxis()

            axs[1].plot(data[:, 0], data[:, 5])
            axs[1].set_title(f'SNR vs elevation')
            axs[1].set_ylabel('SNR')

            axs[2].plot(data[:, 0], data[:, 1])
            axs[2].set_title(f'Pr vs elevation')
            axs[2].set_ylabel('Pr')


        elif plot == "pointing":
            fig_point_1, axs = plt.subplots(3, 1)
            fig_point_2, axs1 = plt.subplots(1, 1)
            axs[0].plot(self.x_pe, self.pdf_pe)
            axs[0].set_title(f'Pointing error displacement [x-axis] - gaussian distribution')
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Angle error [microrad]')

            axs[1].plot(t, self.angle_pe_r)
            axs[1].set_ylabel('I due to radial error [-]')
            axs[1].set_xlabel('Time (s)')
            axs[2].plot(t, self.r_pe[2])
            axs[2].set_ylabel('Radial displacement error [m]')
            axs[2].set_xlabel('Time (s)')

            axs1.scatter(self.angle_pe_x, self.angle_pe_y)
            axs1.set_ylabel('Y-axis angle error [microrad]')
            axs1.set_xlabel('X-axis angle error [microrad]')


    def print(self, type="terminal", index = 0):

        if type == "terminal":

            print('------------------------------------------------')
            print('RECEIVER/DETECTOR CHARACTERISTICS')
            print('------------------------------------------------')

            print('Time          [-]   : ', index)
            print('Bandwidth    [-]    : ', self.BW)
            # print('Symbol time  [-]    : ', self.Ts)
            # print('Bit    time  [-]    : ', self.Tb)
            print('Responsivity [-]    : ', self.R)
            print('BER threshold [-]   : ', self.BER_thres)
            print('Detection receiver  : ', self.detection)
            print('SNR threshold [-]     : ', self.SNR_thres)
            print('Pr   threshold [dBW]  : ', W2dB(self.P_r_thres[index]))
            print('Np threshold  [PPB]   : ', self.Np_thres[index])
            print('Np shot limit [PPB]   : ', self.Np_shot_limit)
            print('Modulation receiver : ', self.modulator)
            # print('Photon count receiver [Np]: ', terminal_sc.Np(P_r=P_r))

            # print('Pointing variance [-] : ', var_pointing)
            # print('Pointing mean     [-] : ', mean_pointing)

        elif type == "noise":

            print('------------------------------------------------')
            print('--------------NOISE-BUDGET----------------------')
            print('------------------------------------------------')

            print('Shot (mean)       [dBW] : ', W2dB(self.noise_sh[index]), self.noise_sh[index])
            print('Thermal (mean)    [dBW] : ', W2dB(self.noise_th), self.noise_th)
            print('Background (mean) [dBW] : ', W2dB(self.noise_bg), self.noise_bg)

        elif type == "BER/SNR":

            print('------------------------------------------------')
            print('-----------------BER-&-SNR----------------------')
            print('------------------------------------------------')

            print('BER (mean)       [-]   : ', np.mean(self.BER))
            print('SNR (mean)       [-]   : ', np.mean(self.SNR))