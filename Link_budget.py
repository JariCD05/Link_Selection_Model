# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from constants import *
from helper_functions import *


class link_budget:
    def __init__(self,
                 range,
                 P_t = 0.1,                 # 100 mW transmit power
                 eff_t = 0.5,               # 0.7 transmitter efficiency
                 eff_r = 0.5,               # 0.7 receiver efficiency
                 eff_coupling = 0.8,        # Coupling efficiency receiver
                 D_t = 0.1,                 # 10 cm transmitter aperture
                 D_r = 0.1,                 # 10 cm receiver aperture,
                 w0 = 0.1,
                 angle_pe_t = 5E-6,        # 5 micro rad jitter error transmitter
                 angle_pe_r = 5E-6,        # 5 micro rad jitter error receiver
                 var_pj_t = 0.5,
                 var_pj_r = 0.5,
                 T_att = 1.0,
                 n = 1.002,                  # Refraction index air
                 coding= "yes"
                 ):

        self.range = range

        self.D_t = D_t  # diameter in m
        self.D_r = D_r # diameter in m

        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 13
        self.diff_limit = wavelength / self.D_r
        self.angle_div = 2.44 * self.diff_limit #rad


        self.var_pj_t = var_pj_t #rad
        self.var_pj_r = var_pj_r  # rad
        self.angle_pe_t = angle_pe_t
        self.angle_pe_r = angle_pe_r

        self.w0 = w0
        self.w_r = beam_spread(self.w0, self.range)

        # Transmitter power
        self.P_t = P_t #W
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 19
        # self.I_t_0 = self.P_t / (np.pi/4 * self.D_t**2) #W/m^2
        self.I_t_0 = P_to_I(self.P_t, np.linspace(0, self.D_t/2, 1000), self.w0)

        # Gains
        # REF: PERFORMANCE LIMITATOIN OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.14
        # self.G_t = (np.pi*self.D_t / self.wavelength)**2
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.G_t = 8 / self.angle_div**2
        self.G_r = (np.pi*self.D_r / wavelength)**2

        # REF: OPTICAL ON-OFF KEYING FOR LOW EARTH ORBIT DOWNLINK APPLICATIONS, D.GIGGENBACH, TABLE 11.5
        self.G_c = dB2W(4.0)

        # Losses
        self.eff_t = eff_t
        self.eff_r = eff_r
        self.eff_coupling = eff_coupling
        self.T_tracking = h_tracking
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.T_fs = (wavelength / (4 * np.pi * self.range)) ** 2
        self.T_atm_att = T_att

        # # Transmitter pointing error loss
        # REF: PERFORMANCE LIMITATOIN OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.16
        # self.T_point_t = np.exp(-self.G_t * self.angle_pe_t ** 2)
        # self.T_point_r = np.exp(-self.G_r * self.angle_pe_r ** 2)

        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 40
        self.T_point_t = np.exp(-self.angle_pe_t**2 / self.angle_div**2)
        self.T_point_r = np.exp(-self.angle_pe_r ** 2 / self.angle_div ** 2)
        # self.T_jit_t = self.angle_div**2 / (self.angle_div**2 + 4 * self.var_pj_t)
        # self.T_jit_r = self.angle_div**2 / (self.angle_div**2 + 4 * self.var_pj_r)
        self.T_div = self.D_r**2 / (self.D_t + self.angle_div*range)**2


    # ------------------------------------------------------------------------
    # -----------------------RECEIVED-POWER-&-INTENSITY-----------------------
    # ------------------------------------------------------------------------

    def P_r_0_func(self, beam_spread="YES", T_WFE=1.0):
        self.T_WFE = T_WFE
        self.P_r_0 = self.P_t * \
              self.G_t * self.G_r * self.G_c * self.eff_t * self.eff_r * self.eff_coupling * self.T_tracking *\
              self.T_fs * self.T_point_t * self.T_point_r #\
                     # * self.T_beamspread * self.T_WFE

        # print(self.G_t * self.G_r * self.G_c * self.eff_t * self.eff_r * self.T_fs * self.T_atm_att)s
        return self.P_r_0

    # def I_t(self, r):
    #     # REF: OPTICAL SCINTILLATIONS AND FADE STATISTICS..., L.ANDREWS, EQ.12
    #     # self.I_t_0 = self.P_t / np.trapz(np.exp(-2*r**2/self.w0**2), x=r)
    #     I_t = self.I_t_0 * np.exp( -2 * r**2 / self.w0**2)
    #     self.I_t = I_t
    #     return I_t

    def I_r_0_func(self, T_WFE=1.0):
        self.T_WFE = T_WFE

        # REF: LONG TERM IRRADIANCE STATISTICS FOR OPTICAL GEO..., T.KAPSIS, 2019, EQ.1
        self.I_r_0 = (self.w0 / self.w_r)**2 * self.I_t_0 * \
                      self.G_t * self.G_r * self.G_c * \
                      self.eff_t * self.eff_r * self.T_fs * self.T_WFE * self.T_point_t * self.T_point_r
                      # * self.T_atm_att * self.T_atm_turb_scint

        return self.I_r_0

    def link_margin(self, P_r_thres, P_turb):
        self.P_turb = P_turb
        self.LM = self.P_turb / P_r_thres
        return self.LM




    # ------------------------------------------------------------------------
    # -------------------------------BEAM-SPREAD------------------------------
    # ------------------------------------------------------------------------

    def beam_spread(self, r0):
        self.w_LT = beam_spread_turbulence(r0, self.D_r, self.w_r)
        self.T_beamspread = (self.w_r / self.w_LT)**2
        self.w_r = self.w_LT
        return self.w_r, self.T_beamspread

        # # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.12.48
        # self.w_LT = np.zeros(len(r0))
        # for i in range(len(r0)):
        #     if self.D_r/r0[i] < 1.0:
        #         self.w_LT[i] = (self.w_r[i] * np.sqrt(1 + self.D_r / r0[i])**(5/3))
        #     elif self.D_r/r0[i] > 1.0:
        #         self.w_LT[i] = (self.w_r[i] * np.sqrt(1 + (self.D_r / r0[i])**(5/3))**(3/5))

    def WFE(self, tip_tilt="NO"):
        D = 2**(3/2) * self.w0
        if tip_tilt == "YES":
            WFE = 1.03 * (D / self.r0) ** (5 / 3)
        elif tip_tilt == "NO":
            WFE = 0.134 * (D / self.r0) ** (5 / 3)

        self.T_WFE = self.eff_coupling * np.exp(-WFE)
        return self.T_WFE


    def plot(self, r_t=0.0, r_r=0.0, t = 0.0, ranges=0.0, type= "time-series"):

        if type == "gaussian beam profile":
            # fig_I = plt.figure(figsize=(6,6), dpi=125)
            fig_I, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.set_title('Normalized Gaussian beam intensity at transmitter, w0: ' + str(self.w0))
            ax2.set_title('Normalized Gaussian beam intensity at receiver, wr: ' + str(self.w_r))
            ax1.plot(r_t, self.I_t, linestyle='-', label="Pt: " + str(self.P_t) + " W, " + str(W2dB(self.P_t) + 30) + " dBm")
            ax1.plot(np.ones(2) * -self.D_t, np.array((0, 1)), color='orange', label='Telescope obstruction')
            ax1.plot(np.ones(2) * self.D_t, np.array((0, 1)), color='orange')

            ax2.plot(r_r, self.I_r, linestyle='-',
                     label="Pr: " + str(np.format_float_scientific(np.float32(self.P_r), exp_digits=2)) + " W, " + str(
                         np.round(W2dB(self.P_r) + 30, 2)) + " dBm")
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
              t = 0.0,
              index = 0,
              type="link budget",
              L_scint = 0.0,
              P_turb = 0.0,
              Np_turb = 0.0,
              P_r_thres = 0.0,
              Np_thres = 0.0,
              elevation = 0.0,
              P_r = 1.0E-5
              ):

        if type == "link budget":
            print('------------------------------------------------')
            print('LINK-BUDGET ANALYSIS')
            print('------------------------------------------------')
            print('LASER LINK')
            # print('Time                      (s) : ', t[index])
            print('Range                     (km): ', self.range[index])
            print('Elevation                (deg): ', np.rad2deg(elevation[index]))
            print('________________________')
            print('TRANSMITTER')
            print('Divergence angle   (micro rad): ', self.angle_div)
            print('Pointing error T   (micro rad): ', self.angle_pe_t )
            print('Pointing error R   (micro rad): ', self.angle_pe_r )
            print('Power transmitter        (dB) : ', W2dBm(self.P_t))
            print('Wavelength         (micro rad): ', wavelength)
            print('Data rate             (Gbit/s): ', data_rate)
            print('________________________')
            print('GAINS')
            print('Transmitter gain   (dB) : ', W2dB(self.G_t))
            print('Receiver gain      (dB) : ', W2dB(self.G_r))
            print('Coding gain        (dB) : ', W2dB(self.G_c))
            print('________________________')
            print('LOSSES')
            print('Free space loss    (dBW) : ', W2dB(self.T_fs[index]))
            print('Transmitter loss   (dBW) : ', W2dB(self.eff_t))
            print('Receiver (optical)  loss (dBW) : ', W2dB(self.eff_r))
            print('Receiver (coupling) loss (dBW) : ', W2dB(self.eff_coupling))
            print('Receiver (tracking) loss (dBW) : ', W2dB(self.T_tracking))
            print('Divergence loss CHECK    : ', W2dB(self.G_t) + W2dB(self.G_r) + W2dB(self.T_fs[index]))
            # print('Scintillation loss (dB) : ', W2dB(LB_comm_up.T_atm_turb_scint))
            # print('Divergence loss     (dB) : ', W2dB(self.T_div[index]))
            print('Pointing loss t    (dBW) : ', W2dB(self.T_point_t))
            print('Pointing loss r    (dBW) : ', W2dB(self.T_point_r))
            # print('Attenuation loss   (dBW) : ', W2dB(self.T_atm_att[index]))
            # print('WFE loss           (dBW) : ', cons.W2dB(self.T_WFE[index]))
            print('________________________')
            print('RECEIVER')
            print('Received signal      (dBW): ', W2dB(self.P_r_0[index]))
            print('Received signal      (PPB): ', Np_func(self.P_r_0[index], BW_sc, eff_quantum_sc))
            print('Beam width w/o turb  (m) : ', self.w_r[index])
            print('Beam width with turb (m) :' , self.w_LT[index])
            print('________________________')

        elif type == "link margin":

            print('------------------------------------------------')
            print('-------------LINK-MARGIN------------------------')
            print('------------------------------------------------')
            # print('Scintillation loss [dBW] : ', cons.W2dB(L_scint[index]))
            print('Power received w/o turb       [dBW] : ', W2dB(P_r))
            print('Power received w turb (mean)  [dBW] : ', W2dB(np.mean(P_turb[index])))
            print('Np received (mean)            [PPB] : ', np.mean(Np_turb[index]))
            print('________________________')
            print('Power threshold  [dBW] : ', W2dB(np.mean(P_r_thres[index])))
            print('Np threshold     [PPB] : ', np.mean(Np_thres[index]))
            print('Link margin      [dBW] : ', W2dB(np.mean(self.LM)))
            print('________________________')







