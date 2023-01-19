# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
import constants as cons


class link_budget:
    def __init__(self,
                 range,
                 heights: np.array,
                 P_t = 0.1,                 # 100 mW transmit power
                 eff_t = 0.5,               # 0.7 transmitter efficiency
                 eff_r = 0.5,               # 0.7 receiver efficiency
                 eff_coupling = 0.8,        # Coupling efficiency receiver
                 D_t = 0.1,                 # 10 cm transmitter aperture
                 D_r = 0.1,                 # 10 cm receiver aperture
                 wavelength = 1550E-9,      # 1550 nm wavelength
                 angle_jit_t = 5E-6,        # 5 micro rad jitter error transmitter
                 angle_jit_r = 5E-6,        # 5 micro rad jitter error receiver
                 att_coeff = 0.025,         # Clear sky, 1550nm wavelength
                 P_r_threshold = 10.0E-6,   # Receiver threshold
                 r0 = 0.05,                 # Fried's number
                 T_att = 1.0
                 ):

        # Range
        self.range = range
        self.heights = heights
        self.h0 = heights[0]

        # Receiver & Transmitter apertures
        self.D_t = D_t  # diameter in m
        self.D_r = D_r # diameter in m

        # Design variables
        self.wavelength = wavelength #m
        self.diff_limit = self.wavelength / self.D_r
        self.angle_div = 2.44 * self.diff_limit #rad
        self.angle_jit_t = angle_jit_t #rad
        self.angle_jit_r = angle_jit_r  # rad
        self.angle_point_t = angle_jit_t
        self.angle_point_r = angle_jit_r

        # Transmitter power
        self.P_t   = P_t #W
        self.I_t_0 = self.P_t / (np.pi/4 * self.D_t**2) #W/m^2

        # Gains
        self.G_t = (np.pi*self.D_t / self.wavelength)**2
        # self.G_t = 8/self.angle_div**2
        self.G_r = (np.pi*self.D_r / self.wavelength)**2
        self.G_c = cons.dB2W(4.0)

        # Losses
        self.eff_t = eff_t
        self.eff_r = eff_r
        self.eff_coupling = eff_coupling
        self.T_fs = (self.wavelength / (4 * np.pi * self.range)) ** 2
        self.T_atm_att = T_att

        # # Need attenuation coefficient profiles
        # self.T_atm_turb_scint = 1.0
        # # Transmitter pointing error loss
        self.T_point_t = np.exp(-self.G_t * self.angle_point_t ** 2)
        self.T_point_r = np.exp(-self.G_r * self.angle_point_r ** 2)
        # # self.T_point = np.exp(-self.angle_jit**2 / self.angle_div**2)
        self.T_div = self.D_r**2 / (self.D_t + self.angle_div*range)**2


    # ------------------------------------------------------------------------
    # -----------------------RECEIVED-POWER-&-INTENSITY-----------------------
    # ------------------------------------------------------------------------

    def P_r(self):
        P_r = self.P_t * \
              self.G_t * self.G_r * self.G_c * self.eff_t * self.eff_r * self.T_fs * self.T_atm_att\
              *  self.T_point_t * self.T_point_r
              # * self.T_atm_turb_scint
        self.P_r = P_r
        return P_r

    def I_t(self, r):
        I_t = self.I_t_0 * np.exp( -2 * r**2 / self.w0**2)
        self.I_t = I_t
        return I_t

    def I_r(self, r, beam_spread="YES", T_WFE=1.0, T_att=1.0):
        if beam_spread == "YES":
            w_r = self.w_LT
        else:
            w_r = self.w_r

        self.T_att = T_att
        self.T_WFE = T_WFE

        self.I_r_0 = (self.w0 / w_r)**2 * self.I_t_0 * \
                      self.G_t * self.G_r * self.G_c * \
                      self.eff_t * self.eff_r * self.T_fs * T_WFE * T_att * self.T_point_t * self.T_point_r
                      # * self.T_atm_turb_scint

        I_r = self.I_r_0 * np.exp( -2 * r**2 / w_r**2)
        self.I_r = I_r
        return I_r

    def link_margin(self, P_r_thres, P_turb):
        self.P_turb = P_turb
        self.LM = self.P_turb / P_r_thres
        return link_budget.W2dB(self, self.LM)

    # ------------------------------------------------------------------------
    # -------------------------------BEAM-SPREAD------------------------------
    # ------------------------------------------------------------------------

    def beam_spread(self, r0):
        self.w_LT = np.zeros(len(r0))
        print('TEST')

        for i in range(len(r0)):
            if self.D_r/r0[i] < 1.0:
                self.w_LT[i] = self.w_r[i] * np.sqrt(1 + self.D_r / r0[i])**(5/3)
            elif self.D_r/r0[i] > 1.0:
                self.w_LT[i] = self.w_r[i] * np.sqrt(1 + (self.D_r / r0[i])**(5/3))**(3/5)
        print(1/0)
    def FWHM(self, z):
        return np.sqrt(2*np.log(2)) * link_budget.w(z)



    def plot(self, r_t=0.0, r_r=0.0, t = 0.0, ranges=0.0, type= "time-series"):

        if type == "gaussian beam profile":
            # fig_I = plt.figure(figsize=(6,6), dpi=125)
            fig_I, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.set_title('Normalized Gaussian beam intensity at transmitter, w0: ' + str(self.w0))
            ax2.set_title('Normalized Gaussian beam intensity at receiver, wr: ' + str(self.w_r))
            ax1.plot(r_t, self.I_t, linestyle='-', label="Pt: " + str(self.P_t) + " W, " + str(cons.W2dB(self.P_t) + 30) + " dBm")
            ax1.plot(np.ones(2) * -self.D_t, np.array((0, 1)), color='orange', label='Telescope obstruction')
            ax1.plot(np.ones(2) * self.D_t, np.array((0, 1)), color='orange')

            ax2.plot(r_r, self.I_r, linestyle='-',
                     label="Pr: " + str(np.format_float_scientific(np.float32(self.P_r), exp_digits=2)) + " W, " + str(
                         np.round(cons.W2dB(self.P_r) + 30, 2)) + " dBm")
            ax2.plot(np.ones(2) * -3*self.D_r, np.array((0, self.I_r_0)), color='orange', label='Telescope obstruction')
            ax2.plot(np.ones(2) * 3*self.D_r, np.array((0, self.I_r_0)), color='orange')

            ax2.set_xlabel('Radial position from beam center')
            ax1.set_ylabel('Normalized intensity')
            ax2.set_ylabel('Normalized intensity')
            ax1.legend()
            ax2.legend()

        elif type == "time-series":
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title('Link budget time series')
            ax1.plot(t/3600.0, self.P_r)
            # ax2.plot(t, self.link_margin)
            ax2.plot(t/3600.0, ranges)

            ax2.set_xlabel('Time')
            ax1.set_ylabel('Power (dBW)')
            # ax2.set_ylabel('Link margin (dBW)')
            ax2.set_ylabel('Range (m)')

    def print(self,
              index = 0,
              type="link budget",
              L_scint = 0.0,
              P_turb = 0.0,
              Np_turb = 0.0,
              P_r_thres = 0.0,
              Np_thres = 0.0,
              data_rate = 1.0E9,
              elevation = 0.0,
              P_r = 1.0E-5
              ):

        if type == "link budget":
            print('------------------------------------------------')
            print('LINK-BUDGET ANALYSIS')
            print('------------------------------------------------')
            print('LASER LINK')
            print('Height SC                 (km): ', self.heights[index][-1]/1000)
            print('Height AC                 (km): ', self.heights[index][0]/1000)
            print('Range                     (km): ', self.range[index]/1000)
            print('Elevation                (deg): ', np.rad2deg(elevation[index]))
            print('________________________')
            print('TRANSMITTER')
            print('Divergence angle   (micro rad): ', self.angle_div / 1.0E-6)
            print('Jitter angle trans (micro rad): ', self.angle_jit_t / 1.0E-6)
            print('Jitter angle rec   (micro rad): ', self.angle_jit_r / 1.0E-6)
            print('Power transmitter        (dB) : ', cons.W2dBm(self.P_t))
            print('Wavelength         (micro rad): ', self.wavelength/1.0E-6)
            print('Data rate             (Gbit/s): ', data_rate/1.0E9)
            print('________________________')
            print('GAINS')
            print('Transmitter gain   (dB) : ', cons.W2dB(self.G_t))
            print('Receiver gain      (dB) : ', cons.W2dB(self.G_r))
            print('Coding gain        (dB) : ', cons.W2dB(self.G_c))
            print('________________________')
            print('LOSSES')
            print('Free space loss    (dBW) : ', cons.W2dB(self.T_fs[index]))
            print('Transmitter loss   (dBW) : ', cons.W2dB(self.eff_t))
            print('Receiver loss      (dBW) : ', cons.W2dB(self.eff_r))
            print('Divergence loss CHECK    : ', cons.W2dB(self.G_t) + cons.W2dB(self.G_r) + cons.W2dB(self.T_fs[index]))
            # print('Scintillation loss (dB) : ', W2dB(LB_comm_up.T_atm_turb_scint))
            print('Divergence loss     (dB) : ', cons.W2dB(self.T_div))
            print('Pointing loss t    (dBW) : ', cons.W2dB(self.T_point_t))
            print('Pointing loss r    (dBW) : ', cons.W2dB(self.T_point_r))
            print('Attenuation loss   (dBW) : ', cons.W2dB(self.T_att[index]))
            # print('WFE loss           (dBW) : ', cons.W2dB(self.T_WFE[index]))
            print('________________________')
            print('RECEIVER')
            print('Power                (dB): ', cons.W2dBm(self.P_r[index]))
            print('Beam width w/o turb  (m) : ', self.w_r[index])
            print('Beam width with turb (m) :' , self.w_LT[index])
            print('________________________')

        elif type == "link margin":

            print('------------------------------------------------')
            print('-------------LINK-MARGIN------------------------')
            print('------------------------------------------------')
            # print('Scintillation loss [dBW] : ', cons.W2dB(L_scint[index]))
            print('Power received w/o turb       [dBW] : ', cons.W2dB(P_r))
            print('Power received w turb (mean)  [dBW] : ', cons.W2dB(np.mean(P_turb[index])))
            print('Np received (mean)            [PPB] : ', np.mean(Np_turb[index]))
            print('________________________')
            print('Power threshold  [dBW] : ', cons.W2dB(P_r_thres))
            print('Np threshold     [PPB] : ', Np_thres)
            print('Link margin      [dBW] : ', np.mean(self.LM))
            print('________________________')







