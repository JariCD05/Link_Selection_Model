
from input import *
from helper_functions import *
from PDF import dist
# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.special import erfc, erf, erfinv, erfcinv, j1
from scipy.stats import rice, rayleigh, norm
from scipy.special import binom


class terminal_properties:
    def __init__(self):
        self.D = w0 * clipping_ratio * 2
        self.R = eff_quantum * q / (h * v)
        self.Be = BW / 2
        self.m = 2

        # if modulator == "OOK-NRZ":
        #     self.Ts = 1/self.BW
        #     self.Tb = self.Ts/N_symb
        #     self.N_symb = N_symb

    # ------------------------------------------------------------------------
    # ----------------------------INCOMING PHOTONS----------------------------
    # ------------------------------------------------------------------------
    def PPB_func(self, P_r, data_rate):
        # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
        self.PPB = PPB_func(P_r, data_rate)
        return self.PPB
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

        self.Sn = noise_factor * (M - 1) * (h * v / 2) + (h * v / 2)
        if noise_type == "shot":
            # This noise type is defined as the variance of incoming photons, arriving at the RX detector, hence it is directly dependant on the amount of incoming photons.
            # For a low number of incoming photons, the shot noise distribution is POISSON.
            # But for sufficient amount of incoming light, this is accurately approximated by a Gaussian distribution.
            # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.55
            self.noise_sh = 4 * self.Sn * self.R**2 * P_r * self.Be / eff_quantum
            return self.noise_sh

        elif noise_type == "thermal":
            # This noise type is defined as a static noise, independent of incoming photons. Hence it can be expressed as AWGN.
            # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.100
            self.noise_th = (4 * k * T_s * self.Be / R_L)
            return self.noise_th

        elif noise_type == "background":
            # This noise types defines the solar background noise, which is simplified to direct incoming sunlight.
            # Solar- and atmospheric irradiance are defined in input.py, atmospheric irradiance is neglected by default and can be added as an extra contribution.
            A_r = 1 / 4 * np.pi * D_r ** 2
            P_bg = (I_sun + I_sky) * A_r * delta_wavelength*10**9 * FOV_r
            self.noise_bg = 4 * self.Sn * self.R**2 * P_bg * self.Be
            return self.noise_bg

        elif noise_type == "beat":
            # This noise types defines the noise-against-noise optical beating and is the result of the squared response of the optical detector.
            #REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.98
            self.noise_beat = 2 * self.m * self.R**2 * self.Sn**2 * (BW - self.Be/2) * self.Be
            return self.noise_beat

        elif noise_type == "total":
            return M**2 * noise_factor * (self.noise_sh+self.noise_bg) + self.noise_th + self.noise_beat

    # ------------------------------------------------------------------------
    # --------------------------------THRESHOLD-------------------------------
    # ------------------------------------------------------------------------
    # Number of received photons per bit in receiver terminal
    # def Np_thres(self):
    #     self.Np_thres = (self.P_r_thres / (Ep * data_rate) / self.eff_quantum).to_base_units()
    #     return self.Np_thres

    def threshold(self,
                  BER_thres = 1.0E-9,
                  M_PPM = 2,
                  noise_factor = 2,
                  M = 50,
                  modulation = "OOK-NRZ",
                  detection = "APD",
                  data_rate = 10.0E9):

        # First compute SNR threshold. This depends on the modulation type (BER --> SNR)
        if modulation == "OOK-NRZ":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*BER_thres) )**2
        elif modulation == "2-PPM":
            self.SNR_thres = 2 / np.sqrt(2 * np.log2(2)) * erfcinv(4 * BER_thres / 2)
        elif modulation == "M-PPM":
            self.SNR_thres = 2 / np.sqrt(M_PPM*np.log2(M_PPM)) * erfcinv(4*BER_thres / M_PPM)
        elif modulation == "DPSK":
            self.SNR_thres = ( np.sqrt(2) * erfcinv(2*BER_thres) )**2
        elif modulation == "BPSK":
            self.SNR_thres = 1/2 * erfcinv(2*BER_thres)**2
        # self.SNR_thres = self.SNR_thres[:(len(BER_thres))]
        self.Q_thres = np.sqrt(self.SNR_thres)

        # Second compute Pr threshold. This depends on the detection type and noise (SNR --> Pr)
        if detection == "PIN":
            self.P_r_thres = np.sqrt(self.SNR_thres * self.noise_th) / self.R
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.123
            self.P_r_thres_check = self.Q_thres / self.R * np.sqrt(8 * k * T_s * 2 * self.Be / R_L)
        elif detection == "APD" or detection == "Preamp":
            Sn = noise_factor * M * h * v / 2
            # self.P_r_thres = np.sqrt(self.SNR_thres * M**2 * noise_factor * (self.noise_sh) + self.noise_th + self.noise_beat) / (M * self.R)
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.125
            self.P_r_thres = self.Q_thres * Sn * 2 * self.Be / M * (self.Q_thres + np.sqrt(self.m/2 * (2*BW/(2*self.Be) - 1/2) + 2 * k * T_s / (R_L * 4 * self.Be * self.R**2 * Sn**2)))
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

    def SNR_func(self, P_r,
                 detection = "APD",):

        if detection == "PIN":
            self.SNR = (P_r * self.R)**2 / self.noise_th
            self.Q   = P_r * self.R / (2 * self.noise_th)
            self.SNR_check = R_L * self.R**2 * P_r**2 / (4 * k * T_s * self.Be)
            self.Q_check = P_r * self.R / (2 * self.noise_th)
        elif detection == "APD" or detection == "Preamp":
            self.SNR = (M * P_r * self.R)**2 / ( M * (self.noise_sh + self.noise_bg) + self.noise_beat + self.noise_th)
            self.Q = M * P_r * self.R / np.sqrt( M * (self.noise_sh + self.noise_bg) + self.noise_beat + 2 * self.noise_th)
            Sn = noise_factor * M * h*v/2
            self.SNR_check = (M * P_r)**2 / (2 * ((self.m * Sn * (BW - self.Be/2) + 2*M*P_r) * Sn + 2*k*T_s/(self.R**2*R_L))*self.Be)
            self.Q_check = M * P_r * self.R / (2 * ((self.m * Sn * (BW - self.Be/2) + 2*M*P_r) * Sn + 4*k*T_s/(self.R**2*R_L))*self.Be)
        elif detection == "quantum limit":
            self.SNR = (P_r * self.R)**2 / self.noise_sh
            self.Q = P_r * self.R / np.sqrt(self.noise_sh)
            self.SNR_check = eff_quantum * P_r / (2 * h * v * self.Be)
            self.Q_check = P_r * self.R / np.sqrt(2 * q * self.R * P_r * self.Be)


        self.Q = np.sqrt(self.SNR)
        return self.SNR
            # , self.SNR_avg

    def BER_func(self,
                 M_PPM=2,
                 modulation = "OOK-NRZ"):
        if modulation == "OOK-NRZ":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.8
            self.BER = 1/2 * erfc( self.Q/np.sqrt(2) )
            # self.BER_avg = 1/2 * erfc( np.sqrt(self.SNR_avg)/np.sqrt(2) )

        elif modulation == "2-PPM" or modulation == "2PolSK":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 FIG.7
            self.BER = 1/2 * np.exp(-1/2 * self.Q**2)
            # self.BER_avg = 1/2 * np.exp(-1/2 * self.SNR_avg)

        elif modulation == "M-PPM":
            self.BER = M_PPM/4 * erfc( 1/2 * self.Q**2 * np.sqrt(M_PPM*np.log2(M_PPM)) )
            # self.BER_avg = M/4 * erfc( 1/2 * self.SNR_avg * np.sqrt(M*np.log2(M)) )

        elif modulation == "DPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.48
            self.BER = 1/2 * erfc( self.Q / np.sqrt(2) )
            # self.BER_avg = 1/2 * erfc( np.sqrt( self.SNR_avg / 2) )

        elif modulation == "BPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.45
            self.BER = 1/2 * erfc( self.Q )
            # self.BER_avg = 1/2 * erfc( np.sqrt(self.SNR_avg))

        elif modulation == "QPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.52
            self.BER = erfc( self.Q )
            # self.BER_avg =erfc(np.sqrt(self.SNR_avg))

        return self.BER
            # , self.BER_avg

    def coding(self, duration, K, N, samples):
        # REF: CCSDS Historical Document, 2006, CH.5.5, EQ.3-4
        dt = duration/samples
        E = int((N - K) / 2)

        number_of_bits = data_rate * duration
        number_of_bits_per_sample = data_rate * dt
        number_of_codewords = number_of_bits / (K * symbol_length)

        SER = 1 - (1 - self.BER) ** symbol_length
        SER_coded = np.zeros_like(SER)
        k_values = np.arange(E, N - 1)
        for i in range(SER.shape[0]):
            SER_coded[i, :] = SER[i, :] * np.sum(
                binom(N - 1, k_values) * np.power.outer(SER[i, :], k_values) * np.power.outer(1 - SER[i, :],
                                                                                              N - k_values - 1), axis=1)

        self.BER_coded = 2 ** (symbol_length - 1) / N * SER_coded

        for i in range(len(self.BER_coded)):
            for j in range(samples):
                if self.BER_coded[i, j] > self.BER[i, j]:
                    self.BER_coded[i, j] = self.BER[i, j]


        return self.BER_coded

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

    def interleaving(self):
        return


    # ------------------------------------------------------------------------
    # --------------------------------POINTING--------------------------------
    # ------------------------------------------------------------------------
    def pointing(self,
                 data,
                 steps = 1000,
                 effect = "TX jitter"):

        # self.std_pj_t  = std_pj_t
        # self.std_pj_r  = std_pj_r
        # self.mean_pj_t = angle_pe_t
        # self.mean_pj_r = angle_pe_r

        if effect == 'TX jitter':

            if PDF_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_t = np.sqrt(2 / (4 - np.pi) * std_pj_t ** 2)
                self.x_pe_t, self.pdf_pe_t = dist.rayleigh_pdf(sigma=self.std_pj_t, steps=steps)
                self.angle_pe_t_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_t)

            elif PDF_pointing == "rice":
                self.std_pj_t = std_pj_t
                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_t, self.pdf_pe_t = dist.rice_pdf(sigma=self.std_pj_t, mean=angle_pe_t, steps=steps)
                self.angle_pe_t_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_t, mean=angle_pe_t)
                self.angle_pe_t_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_t, mean=angle_pe_t)
                self.angle_pe_t_R = np.sqrt(self.angle_pe_t_X ** 2 + self.angle_pe_t_Y ** 2)
            return self.angle_pe_t_R

        elif effect == 'RX jitter':

            if PDF_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_r = np.sqrt(2 / (4 - np.pi) * std_pj_r**2)
                self.x_pe_r, self.pdf_pe_r = dist.rayleigh_pdf(sigma=self.std_pj_r, mean=angle_pe_t, steps=steps)
                self.angle_pe_r_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_r)

            elif PDF_pointing == "rice":
                self.std_pj_r = std_pj_r
                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_r, self.pdf_pe_r = dist.rice_pdf(sigma=self.std_pj_r, mean=angle_pe_r, steps=steps)
                self.angle_pe_r_X = dist.norm_rvs(data=data[0], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_Y = dist.norm_rvs(data=data[1], sigma=self.std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_R = np.sqrt(self.angle_pe_r_X ** 2 + self.angle_pe_r_Y ** 2)
            return self.angle_pe_r_R


    def test_PDF(self, effect = "TX jitter", index = 20):
        fig_test_pdf_LCT, ax_test_pdf_LCT = plt.subplots(1, 1)
        if effect == "TX jitter":
            dist.plot(ax=ax_test_pdf_LCT,
                      sigma=self.std_pj_t,
                      input=[std_pj_t**2, angle_pe_t],
                      x=self.x_pe_t,
                      pdf=self.pdf_pe_t,
                      data=self.angle_pe_t_R,
                      index=index,
                      effect=effect,
                      name=PDF_pointing)

        elif effect == "RX jitter":
            dist.plot(ax=ax_test_pdf_LCT,
                      sigma=self.std_pj_r,
                      input=[std_pj_r**2, angle_pe_r],
                      x=self.x_pe_r,
                      pdf=self.pdf_pe_r,
                      data=self.angle_pe_r_R,
                      index=index,
                      effect=effect,
                      name=PDF_pointing)


    def plot(self, t, data=0.0, plot = "BER & SNR", index = 20):

        # if plot == "Airy disk":
        #     fig_airy, ax_airy = plt.subplots(1, 1)
        #     ax_airy.plot(


        if plot == "BER & SNR":
            fig_ber, axs = plt.subplots(1, 1)
            snr_theoretical = np.linspace(0, 100, 100)
            ber_theoretical = 1/2 * erfc( np.sqrt(snr_theoretical))
            # axs.plot(W2dB(snr_theoretical), ber_theoretical, label='Theoretical', color='orange')
            axs.scatter(W2dB(self.SNR[index]), self.BER[index], label='Monte Carlo results: Uncoded (Channel BER)')
            axs.scatter(W2dB(self.SNR[index]), self.BER_coded[index], label='Monte Carlo results: (255,223) RS coded')
            axs.set_title(f'BER vs SNR')
            axs.set_ylabel('Bit Error Rate (BER)')
            axs.set_yscale('log')
            axs.set_xlabel('SNR (dB)')
            axs.grid()

            N, K = 255, 223
            E = int((N - K) / 2)
            symbol_length = 8
            SNR = np.linspace(0, 50, 200)
            Vb = 1 / 2 * erfc(np.sqrt(SNR))
            Vs = 1 - (1 - Vb) ** symbol_length
            # Vs = np.flip(np.linspace(0.01, 0.07, 20))
            # Vb = 1 - (1 - Vs)**(1/symbol_length)
            SER_coded = np.zeros(np.shape(Vs))
            for i in range(len(Vs)):
                SER_coded[i] = Vs[i] * sum(
                    binom(N - 1, k) * Vs[i] ** k * (1 - Vs[i]) ** (N - k - 1) for k in range(E, N - 1))
            BER_coded = 2 ** (symbol_length - 1) / N * SER_coded
            axs.plot(W2dB(SNR), Vb, label='Theory: Uncoded (Channel BER)', color='orange')
            axs.plot(W2dB(SNR), BER_coded, label='Theory: (255,223) RS coded', color='green')
            axs.legend()

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
            axs[0].plot(self.x_pe_t, self.pdf_pe_t)
            axs[0].set_title(f'Pointing jitter angle [x-axis]: '+str(PDF_pointing))
            axs[0].set_ylabel('Probability [-]')
            axs[0].set_xlabel('Angle jitter [rad]')

            axs[1].plot(t, self.angle_pe_t_X, label="X-axis")
            axs[1].plot(t, self.angle_pe_t_Y, label="Y-axis")
            axs[1].set_ylabel('I due to radial error [-]')
            axs[1].set_xlabel('Time (s)')

            axs1.scatter(self.angle_pe_t_X*1.0E6, self.angle_pe_t_Y*1.0E6)
            axs1.set_ylabel('Y-axis angle error [urad]')
            axs1.set_xlabel('X-axis angle error [urad]')


    def print(self, type="terminal", index = 0):

        if type == "terminal":

            print('------------------------------------------------')
            print('RECEIVER/DETECTOR CHARACTERISTICS')
            print('------------------------------------------------')

            print('Time          [-]   : ', index)
            print('Bandwidth    [-]    : ', BW)
            # print('Symbol time  [-]    : ', self.Ts)
            # print('Bit    time  [-]    : ', self.Tb)
            print('Responsivity [-]    : ', self.R)
            print('BER threshold [-]   : ', BER_thres)
            print('Detection receiver  : ', detection)
            print('SNR threshold [-]     : ', self.SNR_thres)
            print('Pr   threshold [dBW]  : ', W2dB(self.P_r_thres[index]))
            print('Np threshold  [PPB]   : ', self.PPB_thres[index])
            print('Modulation receiver : ', modulation)
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