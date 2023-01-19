
import constants as cons

# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.special import erfc, erf, erfinv, erfcinv
from scipy.stats import rice, rayleigh


class terminal_properties:
    def __init__(self,
                 D = 0.1,                   # 10 cm aperture
                 data_rate = 1.0E9,         # 1 GBit/s
                 eff_quantum = 1.0,         # Responsivity (quantum efficiency)
                 BW = 1.0E9,                # 1 MHz Bandwidth
                 wavelength = 1.55E-9,      # 1550 nm wavelength
                 modulator = "OOK-NRZ",     # OOK modulation technique OOK/2-PPM/L-PPM/BPSK/DPSK
                 N_symb = 4,                # Symbol size
                 detection = "ADP",         # Detection technique PIN/ADP/Preamp/Coherent
                 R_L = 50,                  # Load resistor
                 T_s = 300,                 # Temperature
                 FOV = 1.0E-8,              # Field of view of the terminal (solid angle)
                 delta_wavelength = 5.0,    # Optical bandpass filter
                 M = 1000.0,                # Gain of receiver amplifier (ADP or OA)
                 F = 4.0,                   # Amplifier noise factor
                 BER_thres = 1.0E-9         # BER threshold
                 ):

        self.BER_thres = BER_thres
        self.eff_quantum = eff_quantum
        self.D = D
        self.FOV = FOV
        self.speed_of_light = 3.0E8 #m/s
        self.data_rate = data_rate
        self.BW = BW
        self.delta_wavelength = delta_wavelength
        # self.BW = self.data_rate / 2
        self.wavelength = wavelength
        self.v = cons.speed_of_light / self.wavelength
        self.modulator = modulator
        self.detection = detection
        self.R = eff_quantum * cons.q / (cons.h * self.v)
        self.R_L = R_L
        self.T_s = T_s
        self.G = (np.pi * self.D / self.wavelength) ** 2
        self.M = M
        self.F = F

        if modulator == "OOK-NRZ":
            self.Ts = 1/self.BW
            self.Tb = self.Ts/N_symb
            self.N_symb = N_symb

    # ------------------------------------------------------------------------
    # ----------------------------INCOMING PHOTONS----------------------------
    # ------------------------------------------------------------------------
    def Np(self, P_r):
        E_b = cons.h * self.v
        self.Np = P_r / (E_b * self.data_rate) / self.eff_quantum
        return self.Np

    # ------------------------------------------------------------------------
    # ----------------------------------NOISE---------------------------------
    # ------------------------------------------------------------------------
    # Incoming noise, 5 noise types: Shot, termal, background and beat (mixing)
    def noise(self,
              noise_type = "shot",
              P_r = 0.0,
              I_sun = 0.02,
              delta_wavelength = 5):

        if noise_type == "shot":
            self.noise_sh = 2 * cons.q * self.R * P_r * self.BW
            return self.noise_sh

        elif noise_type == "thermal":
            self.noise_th = 4 * cons.k * self.T_s * self.BW / self.R_L
            return self.noise_th

        elif noise_type == "background":
            A = 1 / 4 * np.pi * self.D ** 2
            # self.noise_bg = 2 * self.BW * self.R * P_bg * v * cons.h
            P_bg = I_sun * A * delta_wavelength * self.FOV
            self.noise_bg = 2 * self.BW * self.R * P_bg * cons.q
            print('test: ')
            print(self.noise_bg, self.FOV, I_sun, P_bg)
            return self.noise_bg

        elif noise_type == "beat":
            self.noise_beat = 0.0
            return self.noise_beat

    # ------------------------------------------------------------------------
    # --------------------------------THRESHOLD-------------------------------
    # ------------------------------------------------------------------------
    # Number of received photons per bit in receiver terminal
    def Np_thres(self):
        E_b = cons.h * self.v
        self.Np_thres = self.P_r_thres / (E_b * self.data_rate) / self.eff_quantum
        return self.Np_thres

    def threshold(self,
              M = 0
              ):

        if self.modulator == "OOK-NRZ":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*self.BER_thres) )**2
            self.Q_thres   = ( np.sqrt(2) * erfcinv(2*self.BER_thres) )
            self.Np_quantum_limit = -2 * np.log( self.BER_thres )

        elif self.modulator == "2-PPM" or self.modulator == "2PolSK":
            self.SNR_thres = -2 * np.log(2*self.BER_thres)
            self.Q_thres = 0.0

        elif self.modulator == "M-PPM":
            self.SNR_thres = 2 / np.sqrt(M*np.log2(M)) * erfcinv(4*self.BER_thres / M)
            self.Q_thres = 0.0

        elif self.modulator == "DPSK":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*self.BER_thres) )**2
            self.Q_thres = 0.0

        elif self.modulator == "BPSK":
            self.SNR_thres = 1/2 * erfcinv(2*self.BER_thres)**2
            self.Q_thres = 0.0

        self.Np_shot_limit = self.SNR_thres / self.eff_quantum
        return self.SNR_thres, self.Q_thres, self.Np_shot_limit

    def P_r_thres(self):
        if self.detection == "PIN" and self.modulator == "OOK-NRZ":
            self.P_r_thres = self.Q_thres * self.noise_th / self.R

        elif self.detection == "ADP" and self.modulator == "OOK-NRZ":
            self.P_r_thres = self.Q_thres / self.R * ( 2 * cons.q * self.F * self.Q_thres * self.BW
                                                       + self.noise_th/self.M)

        elif self.detection == "PIN" and self.modulator == "BPSK":
            self.P_r_thres = 0.0

        elif self.detection == "ADP" and self.modulator == "BPSK":
            self.P_r_thres = 0.0

        elif self.detection == "PIN" and self.modulator == "DPSK":
            self.P_r_thres = 0.0

        elif self.detection == "ADP" and self.modulator == "DPSK":
            self.P_r_thres = 0.0

        return self.P_r_thres

    # ------------------------------------------------------------------------
    # --------------------------------SNR-&-BER-------------------------------
    # ------------------------------------------------------------------------

    def SNR(self,
            P_r,
            noise_sh = 0.0,
            noise_th = 0.0,
            noise_bg = 0.0,
            noise_beat = 0.0,
            M = 0.0,
            F = 0.0):

        if self.detection == "PIN":
            self.SNR = (P_r * self.R)**2 / noise_th
        elif self.detection == "ADP":
            self.SNR = (M * P_r * self.R)**2 / ( M*F*(noise_sh+noise_bg) + noise_th)
        elif self.detection == "Preamp":
            self.SNR = (M * P_r * self.R)**2 / (noise_sh+noise_bg + noise_th + noise_beat)
        elif self.detection == "quantum-limit":
            self.SNR = (P_r * self.R)**2 / noise_sh

        return self.SNR

    def BER(self, M = 0):
        if self.modulator == "OOK-NRZ":
            self.BER = 1/2 * erfc( self.SNR/np.sqrt(2) )

        elif self.modulator == "2-PPM" or self.modulator == "2PolSK":
            self.BER = 1/2 * np.exp(-1/2 * self.SNR)

        elif self.modulator == "M-PPM":
            self.BER = M/4 * erfc( 1/2 * self.SNR * np.sqrt(M*np.log2(M)) )
            self.Q_thres = 0.0

        elif self.modulator == "DPSK":
            self.BER = 1/2 * erfc( np.sqrt( self.SNR / 2) )

        elif self.modulator == "BPSK":
            self.BER = 1/2 * erfc( 2 * self.SNR )

        return self.BER

    # ------------------------------------------------------------------------
    # --------------------------------POINTING--------------------------------
    # ------------------------------------------------------------------------
    def pointing(self, boresight_error=0.0, dist_type="rayleigh", var_pointing=0.5, steps = 1000, divergence = 1.0E-5):

        self.var_pointing = var_pointing

        if dist_type == "rayleigh":
            self.x_pe = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), steps)
            self.r_pe = rayleigh.rvs(size=steps, scale=var_pointing)
            self.pdf_pe = rayleigh.pdf(self.x_pe, scale=var_pointing)

            beta = divergence**2 / (4*var_pointing)
            I = np.linspace(0,1,steps)
            self.I_pe = beta * I**beta
            self.T_pe = beta / (beta + 1)
            return self.r_pe, self.T_pe

        elif dist_type == "rice":
            self.x_pe = np.linspace(rice.ppf(0.01, boresight_error), rice.ppf(0.99, boresight_error), steps)
            self.r_pe = rice.rvs(size=steps, scale=var_pointing, b=boresight_error)
            self.pdf_pe = rice.pdf(self.x_pe, b=boresight_error, scale=var_pointing)




    def plot(self, t, plot = "BER & SNR"):

        if plot == "BER & SNR":
            fig_ber, axs = plt.subplots(2, 1, dpi=125)
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

        elif plot == "pointing":
            fig_point, axs = plt.subplots(2, 1, dpi=125)
            axs[0].plot(self.x_pe, self.pdf_pe)
            axs[0].set_title(f'Pointing error displacement - Rice distribution')
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Radial displacement [m]')

            axs[1].plot(t, self.r_pe)
            axs[1].set_ylabel('Radial displacement [m]')
            axs[1].set_xlabel('Time (s)')


    def print(self, type="terminal"):

        if type == "terminal":

            print('------------------------------------------------')
            print('RECEIVER/DETECTOR CHARACTERISTICS')
            print('------------------------------------------------')

            print('Bandwidth    [-]    : ', self.BW)
            print('Symbol time  [-]    : ', self.Ts)
            print('Bit    time  [-]    : ', self.Tb)

            print('Responsivity [-]    : ', self.R)
            print('BER threshold [-]   : ', self.BER_thres)
            print('Detection receiver  : ', self.detection)
            print('SNR threshold [-]   : ', self.SNR_thres)
            print('Q   threshold [-]   : ', self.Q_thres)
            print('Np shot limit [-]   : ', self.Np_shot_limit)
            print('Np quantum limit [-]: ', self.Np_quantum_limit)
            print('Modulation receiver : ', self.modulator)
            # print('Photon count receiver [Np]: ', terminal_sc.Np(P_r=P_r))

            # print('Pointing variance [-] : ', var_pointing)
            # print('Pointing mean     [-] : ', mean_pointing)

        elif type == "noise":

            print('------------------------------------------------')
            print('--------------NOISE-BUDGET----------------------')
            print('------------------------------------------------')

            print('Shot (mean)       [dBW] : ', cons.W2dB(np.mean(self.noise_sh)))
            print('Thermal (mean)    [dBW] : ', cons.W2dB(np.mean(self.noise_th)))
            print('Background (mean) [dBW] : ', cons.W2dB(np.mean(self.noise_bg)))

        elif type == "BER/SNR":

            print('------------------------------------------------')
            print('-----------------BER-&-SNR----------------------')
            print('------------------------------------------------')

            print('BER (mean)       [-]   : ', np.mean(self.BER))
            print('SNR (mean)       [-]   : ', np.mean(self.SNR))