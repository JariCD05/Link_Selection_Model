# Load other modules
from input import *
from helper_functions import *
from PDF import dist


# Load packages
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.special import erfc, erf, erfinv, erfcinv, j1, laguerre
from scipy.stats import rice, rayleigh, norm
from scipy.special import binom


class terminal_properties:
    def __init__(self):
        self.D = w0 * clipping_ratio * 2
        self.R = eff_quantum * q / (h * v)
        self.Be = BW / 2
        self.m = 2
        self.modulation = modulation
        self.detection = detection
        self.data_rate = data_rate
        self.Sn = noise_factor * (M - 1) * (h * v / 2) + (h * v / 2)

        # if modulator == "OOK-NRZ":
        #     self.Ts = 1/self.BW
        #     self.Tb = self.Ts/N_symb
        #     self.N_symb = N_symb

    # ------------------------------------------------------------------------
    # ----------------------------------NOISE---------------------------------
    # ------------------------------------------------------------------------
    # Incoming noise, 5 noise types: Shot, termal, background and beat (mixing)
    def noise(self,
              noise_type = "shot",
              P_r = 0.0,
              I_sun = 0.02,
              M = 50,
              noise_factor = 2):

        # This noise type is defined as the variance of incoming photons, arriving at the RX detector, hence it is directly dependant on the amount of incoming photons.
        # For a low number of incoming photons, the shot noise distribution is POISSON.
        # But for sufficient amount of incoming light, this is accurately approximated by a Gaussian distribution.
        # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.55
        noise_sh = 4 * self.Sn * self.R**2 * P_r * self.Be / eff_quantum

        # This noise type is defined as a static noise, independent of incoming photons. Hence it can be expressed as AWGN.
        # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.100
        noise_th = (4 * k * T_s * self.Be / R_L)

        # This noise types defines the solar background noise, which is simplified to direct incoming sunlight.
        # Solar- and atmospheric irradiance are defined in input.py, atmospheric irradiance is neglected by default and can be added as an extra contribution.
        A_r = 1 / 4 * np.pi * D_r ** 2
        P_bg = (I_sun + I_sky) * A_r * delta_wavelength*10**9 * FOV_r
        noise_bg = 4 * self.Sn * self.R**2 * P_bg * self.Be

        # This noise types defines the noise-against-noise optical beating and is the result of the squared response of the optical detector.
        #REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.98
        noise_beat = 2 * self.m * self.R**2 * self.Sn**2 * (BW - self.Be/2) * self.Be

        return noise_sh, noise_th, noise_bg, noise_beat

    # ------------------------------------------------------------------------
    # --------------------------------THRESHOLD-------------------------------
    # ------------------------------------------------------------------------
    # Number of received photons per bit in receiver terminal
    # def Np_thres(self):
    #     self.Np_thres = (self.P_r_thres / (Ep * data_rate) / self.eff_quantum).to_base_units()
    #     return self.Np_thres

    def threshold(self,
                  BER_thres = 1.0E-9,
                  M_PPM = 32,
                  modulation = "OOK-NRZ",
                  detection = "APD",
                  noise_th = False):
        BER_thres = np.array(BER_thres)
        # Firstly, compute SNR threshold. This depends on the modulation type (BER --> SNR)
        if modulation == "OOK-NRZ":
            # self.SNR_thres = ( np.sqrt(2) * erfcinv(2*BER_thres) )**2
            self.Q_thres = np.sqrt(2) * erfcinv(2*BER_thres)

        # CHECK THESE!!!
        elif modulation == "2-PPM":
            self.SNR_thres = 2 / np.sqrt(2 * np.log2(2)) * erfcinv(2 * BER_thres)
        elif modulation == "M-PPM":
            self.SNR_thres = 2 / np.sqrt(M_PPM*np.log2(M_PPM)) * erfcinv(4/M_PPM*BER_thres)

        elif modulation == "DPSK":
            self.Q_thres = np.sqrt(2) * erfcinv(2*BER_thres)
        elif modulation == "BPSK":
            self.Q_thres = erfcinv(2*self.BER_thres)
        # self.Q_thres = np.sqrt(self.SNR_thres)

        # Secondly, compute Pr threshold. This depends on the detection type and noise (SNR --> Pr)
        if detection == "PIN":
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.123
            self.P_r_thres = self.Q_thres * 2 * np.sqrt(4 * k * T_s * self.Be / R_L) / self.R
        elif detection == "APD" or detection == "Preamp":
            # Sn = noise_factor * M * h * v / 2
            # Sn = noise_factor * (M - 1) * (h * v / 2) + (h * v / 2)
            # self.P_r_thres = np.sqrt(self.SNR_thres * M**2 * noise_factor * (self.noise_sh) + self.noise_th + self.noise_beat) / (M * self.R)
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.125
            self.P_r_thres = 2 * self.Q_thres * self.Sn * 2 * self.Be / M * (self.Q_thres + np.sqrt(self.m/2 * (2*BW/(2*self.Be) - 1/2) + 2 * k * T_s / (R_L * 4 * self.Be * self.R**2 * self.Sn**2)))
        elif detection == "quantum limit":
            # self.P_r_thres = np.sqrt( self.SNR * self.noise_sh ) / self.R
            self.P_r_thres = self.Q_thres**2 * h * v * 2 * self.Be / eff_quantum

        # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
        self.PPB_thres = PPB_func(self.P_r_thres, data_rate)
        # self.PPB_thres_check = PPB_func(self.P_r_thres_check, data_rate)

        # return self.SNR_thres, self.P_r_thres, self.PPB_thres

    # ------------------------------------------------------------------------
    # --------------------------------SNR-&-BER-------------------------------
    # ------------------------------------------------------------------------

    def SNR_func(self,
                 P_r,
                 noise_sh,
                 noise_bg,
                 noise_beat,
                 noise_th,
                 detection = "APD",):

        if detection == "PIN":
            noise = noise_th
            signal = (P_r * self.R)**2
            SNR = signal / noise
            Q = np.sqrt(signal) / (2 * np.sqrt(noise))
            self.SNR_check = R_L * self.R**2 * P_r**2 / (4 * k * T_s * self.Be)
            # self.Q_check = P_r * self.R / (2 * self.noise_th)
        elif detection == "APD" or detection == "Preamp":
            SNR = (M * P_r * self.R)**2 / (M * (noise_sh + noise_bg) + noise_beat + noise_th)
            # Q = M * P_r * self.R / ( np.sqrt(M * (noise_sh + noise_bg) + noise_beat + noise_th) + np.sqrt(noise_beat + noise_th))
            Q = M * P_r * self.R / (np.sqrt(M * noise_sh)  + 2*np.sqrt(noise_beat) + 2*np.sqrt(noise_th))
            Sn = noise_factor * M * h*v/2
            self.SNR_check = (M * P_r)**2 / (2 * ((self.m * Sn * (BW - self.Be/2) + 2*M*P_r) * Sn + 2*k*T_s/(self.R**2*R_L))*self.Be)
            # self.Q_check = M * P_r * self.R / (2 * ((self.m * Sn * (BW - self.Be/2) + 2*M*P_r) * Sn + 4*k*T_s/(self.R**2*R_L))*self.Be)
        elif detection == "quantum limit":
            SNR = (P_r * self.R)**2 / noise_sh
            Q = P_r * self.R / np.sqrt(noise_sh)
            self.SNR_check = eff_quantum * P_r / (2 * h * v * self.Be)
            # self.Q_check = P_r * self.R / np.sqrt(2 * q * self.R * P_r * self.Be)

        return SNR, Q

    def BER_func(self,
                 Q: np.array,
                 M_PPM = 32,
                 modulation = "OOK-NRZ"):
        if modulation == "OOK-NRZ":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.8
            BER = 1/2 * erfc( Q / np.sqrt(2) )

        elif modulation == "2-PPM" or modulation == "2PolSK":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 FIG.7
            BER = 1/2 * np.exp(-1/2 * Q**2)

        elif modulation == "M-PPM":
            BER = M_PPM/4 * erfc( 1/2 * Q**2 * np.sqrt(M_PPM*np.log2(M_PPM)) )

        elif modulation == "DPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.48
            BER = 1/2 * erfc( Q / np.sqrt(2) )

        elif modulation == "BPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.45
            BER = 1/2 * erfc( Q )

        elif modulation == "QPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.52
            BER = erfc( Q )
        return BER

    def coding(self, K, N, BER):
        # REF: CCSDS Historical Document, 2006, CH.5.5, EQ.3-4
        self.parity_bits = int((N - K) / 2)
        SER = 1 - (1 - BER) ** symbol_length
        SER_coded = np.zeros_like(SER)
        k_values = np.arange(self.parity_bits, N - 1)

        if BER.ndim > 1:
            for i in range(SER.shape[0]):
                SER_coded[i, :] = \
                    SER[i, :] * np.sum(binom(N - 1, k_values) * np.power.outer(SER[i, :], k_values) * np.power.outer(1 - SER[i, :], N - k_values - 1), axis=1)
        self.BER_coded = 2 ** (symbol_length - 1) / N * SER_coded

        return self.BER_coded

    def interleaving(self, errors, verification='no'):
        spread = int(np.round(latency_interleaving / step_size_channel_level,0) + 1)
        errors_per_sample = errors / spread
        errors_interleaved = np.zeros(np.shape(errors))

        for i in range(0, spread):
            errors_interleaved += np.roll(errors_per_sample, i, axis=1)
        return errors_interleaved


    # ------------------------------------------------------------------------
    # -----------------------------BIT-SIMULATION-----------------------------
    # ------------------------------------------------------------------------

    # This method is still to be added
    # First, it generates a frame of bits X, encodes & interleaves this, then modulates into a wave signal
    # Second, it detects, demodulates into bits and decodes & interleaves the bits into bit frame Y
    # Thirdly, bit frame is X and Y are compared and BER is computed

    def bit_generation(self):
        return

    def modulation(self):
        return

    # def coding(self):
    #     return

    # def interleaving(self):
    #     return


    # ------------------------------------------------------------------------
    # --------------------------------POINTING--------------------------------
    # ------------------------------------------------------------------------
    def create_pointing_distributions(self,
                                     data,
                                     steps = 1000,
                                     effect = "TX jitter"):

        if effect == 'TX jitter':

            if PDF_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_t = np.sqrt(2 / (4 - np.pi) * std_pj_t ** 2)
                self.mean_pj_t = np.sqrt(np.pi / 2) * self.std_pj_t
                self.x_pe_t, self.pdf_pe_t = dist.rayleigh_pdf(sigma=self.std_pj_t, steps=steps)
                self.angle_pe_t_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_t, mean=0)
                self.angle_pe_t_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_t, mean=0)
                self.angle_pe_t_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_t)
            elif PDF_pointing == "rice":
                self.std_pj_t = std_pj_t
                self.var_pj_t = std_pj_t ** 2
                self.mean_pj_t = np.sqrt( angle_pe_t**2 + angle_pe_t **2)

                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_t, self.pdf_pe_t = dist.rice_pdf(sigma=self.std_pj_t, mean=self.mean_pj_t, steps=steps)
                self.angle_pe_t_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_t, mean=0)
                self.angle_pe_t_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_t, mean=0)
                self.angle_pe_t_R = np.sqrt(self.angle_pe_t_X ** 2 + self.angle_pe_t_Y ** 2)
            return self.angle_pe_t_R

        elif effect == 'RX jitter':

            if PDF_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_r = np.sqrt(2 / (4 - np.pi) * std_pj_r**2)
                self.mean_pj_r = np.sqrt(np.pi / 2) * self.std_pj_r
                self.x_pe_r, self.pdf_pe_r = dist.rayleigh_pdf(sigma=self.std_pj_r, steps=steps)
                self.angle_pe_r_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_r)

            elif PDF_pointing == "rice":
                self.std_pj_r = std_pj_r
                self.var_pj_r = std_pj_r ** 2
                self.mean_pj_r = np.sqrt( angle_pe_r**2 + angle_pe_r **2)
                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_r, self.pdf_pe_r = dist.rice_pdf(sigma=self.std_pj_r, mean=self.mean_pj_r, steps=steps)
                self.angle_pe_r_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_R = np.sqrt(self.angle_pe_r_X ** 2 + self.angle_pe_r_Y ** 2)
            return self.angle_pe_r_R


    def test_PDF(self, effect = "TX jitter", index = 20):
        fig_test_pdf_LCT, ax_test_pdf_LCT = plt.subplots(1, 1)
        if effect == "TX jitter":
            dist.plot(ax=ax_test_pdf_LCT,
                      sigma=self.std_pj_t,
                      mean=self.mean_pj_t,
                      x=self.x_pe_t,
                      pdf=self.pdf_pe_t,
                      data=self.angle_pe_t_R,
                      index=index,
                      effect=effect,
                      name=PDF_pointing)

        elif effect == "RX jitter":
            dist.plot(ax=ax_test_pdf_LCT,
                      sigma=self.std_pj_r,
                      mean=self.mean_pj_r,
                      x=self.x_pe_r,
                      pdf=self.pdf_pe_r,
                      data=self.angle_pe_r_R,
                      index=index,
                      effect=effect,
                      name=PDF_pointing)