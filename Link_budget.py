# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from input import *
from helper_functions import *


class link_budget:
    def __init__(self, ranges: np.array):

        # Range
        # ----------------------------------------------------------------------------
        self.ranges = ranges

        # Power / intensity
        # ----------------------------------------------------------------------------
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 19
        # self.I_t_0 = self.P_t / (np.pi/4 * self.D_t**2) #W/m^2
        self.I_t_0 = P_to_I(P_t, np.linspace(0, D_t/2, 1000), w0)

        # Gains
        #----------------------------------------------------------------------------
        # REF: PERFORMANCE LIMITATION OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.14
        # self.G_t = (np.pi*self.D_t / self.wavelength)**2
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.G_t = 8 / angle_div**2
        self.G_r = (np.pi*D_r / wavelength)**2
        # REF: OPTICAL ON-OFF KEYING FOR LOW EARTH ORBIT DOWNLINK APPLICATIONS, D.GIGGENBACH, TABLE 11.5
        self.G_c = dB2W(4.0)

        # Losses
        # ----------------------------------------------------------------------------
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.T_fs = (wavelength / (4 * np.pi * self.ranges)) ** 2
        print(np.shape(self.T_fs))
        for i in range(len(self.T_fs)):
            if self.ranges[i] == 0.0:
                self.T_fs[i] = 0.0


        # # Transmitter pointing error loss
        # REF: PERFORMANCE LIMITATION OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.16
        # self.T_point_t = np.exp(-self.G_t * self.angle_pe_t ** 2)
        # self.T_point_r = np.exp(-self.G_r * self.angle_pe_r ** 2)

        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 40
        self.T_point_t = np.exp(-angle_pe_t**2 / angle_div**2)
        self.T_point_r = np.exp(-angle_pe_r ** 2 / angle_div ** 2)
        self.T_div = D_r**2 / (D_t + angle_div*ranges)**2

        # EXCEPTIONS
        # ----------------------------------------------------------------------------
        # Check for the correct type of the input variables

        if type(self.ranges) is not np.ndarray or type(self.T_fs) is not np.ndarray or type(self.T_div) is not np.ndarray:
            raise TypeError('input of the range must be of type ARRAY')

        if type(self.G_t) is not float or type(self.G_r) is not float or type(self.G_c) is not float:
            raise TypeError('input of the range must be of type FLOAT')

        if type(self.T_point_r) is not np.float64 or type(self.T_point_r) is not np.float64 or \
           type(self.I_t_0) is not np.float64:
            raise TypeError('input of the range must be of type NP.FLOAT64')

    # ------------------------------------------------------------------------
    # -----------------------RECEIVED-POWER-&-INTENSITY-----------------------
    # ------------------------------------------------------------------------

    # This function computes the static power at the receiver without any turbulence losses
    def P_r_0_func(self, beam_spread="YES", T_WFE=1.0):
        self.T_WFE = T_WFE
        self.P_r_0 = P_t * \
                     self.G_t * self.G_r * self.G_c * eff_t * eff_r * eff_coupling * h_tracking *\
                     self.T_fs * self.T_point_t * self.T_point_r * self.T_beamspread #* self.T_WFE
        self.Np_r_0 = Np_func(self.P_r_0, data_rate, eff_quantum)
        return self.P_r_0

    # This function computes the static intensity at the receiver without any turbulence losses
    def I_r_0_func(self, T_WFE=1.0):
        self.T_WFE = T_WFE

        # REF: LONG TERM IRRADIANCE STATISTICS FOR OPTICAL GEO..., T.KAPSIS, 2019, EQ.1
        self.I_r_0 = (w0 / self.w_r)**2 * self.I_t_0 * \
                      self.G_t * self.G_r * self.G_c * eff_t * eff_r * eff_coupling * h_tracking *\
                      self.T_fs * self.T_point_t * self.T_point_r #* self.T_WFE
        return self.I_r_0

    # Firstly, the static link budget is computed with 'P_r_0_func' and 'I_r_0_func'.
    # Then, this function implements all dynamic contributions of the link budget that are imported from the database,
    # which are scintillation loss (T_scint), pointing jitter loss (T_pj), beam wander loss (T_bw) and the resulting received power (P_r)
    def dynamic_contributions(self, h_scint, h_pj, h_bw, P_r, P_r_threshold, Np_r, Np_r_threshold):
        self.T_scint = h_scint
        self.T_pj = h_pj
        # self.T_pj_r = h_pj_r
        self.T_bw = h_bw
        self.P_r = P_r
        self.P_r_threshold = P_r_threshold
        self.Np_r = Np_r
        self.Np_r_threshold = Np_r_threshold

    # This function computes the link margin
    def link_margin(self):
        self.LM_communication = self.P_r / self.P_r_threshold
        self.LM_acquisition = self.P_r / sensitivity_acquisition


    # ------------------------------------------------------------------------
    # -------------------------STATIC-TURBULENCE-LOSSES-----------------------
    # -----------------------BEAM-SPREAD-&-WAVE-FRONT-ERROR-------------------
    # ------------------------------------------------------------------------

    def beam_spread(self, r0):
        self.w_r = beam_spread(w0, self.ranges)  # beam radius at receiver (without turbulence) (in m)

        # Only during uplink is the turbulence beamspread considered significant.
        # This is ignored for downlink
        if link == 'up':
            self.w_LT = beam_spread_turbulence(r0, D_r, self.w_r)
            self.T_beamspread = (self.w_r / self.w_LT)**2
            self.w_r = self.w_LT

        elif link == 'down':
            self.T_beamspread = 1.0

        return self.w_r

    def WFE(self, r0, tip_tilt="NO"):
        D = 2**(3/2) * w0
        if tip_tilt == "YES":
            WFE = 1.03 * (D / r0) ** (5 / 3)
        elif tip_tilt == "NO":
            WFE = 0.134 * (D / r0) ** (5 / 3)

        self.T_WFE = eff_coupling * np.exp(-WFE)


    def plot(self, r_t=0.0, r_r=0.0, t = 0.0, ranges=0.0, type= "time-series"):

        if type == "gaussian beam profile":
            # fig_I = plt.figure(figsize=(6,6), dpi=125)
            fig_I, ax1 = plt.subplots(3, 1)
            ax1.set_title('Normalized Gaussian beam intensity at receiver, wr: ' + str(self.w_r))
            ax1.plot(r_t, self.I_r_0, linestyle='-', label="Pt: " + str(P_t) + " W, " + str(W2dB(P_t) + 30) + " dBm")
            ax1.plot(np.ones(2) * -D_t, np.array((0, 1)), color='orange', label='Telescope obstruction')
            ax1.plot(np.ones(2) * D_t, np.array((0, 1)), color='orange')

            ax1.set_xlabel('Radial position from beam center')
            ax1.set_ylabel('Normalized intensity')
            ax1.legend()

        elif type == "time-series":
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title('Link budget time series')
            ax1.plot(t/3600.0, self.P_r_0)
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
              elevation = 0.0,
              ):

        if type == "link budget":
            print('------------------------------------------------')
            print('LINK-BUDGET ANALYSIS')
            print('------------------------------------------------')
            print('LASER LINK')
            print('Time                       (s) : ', t[index])
            print('Range                     (km) : ', self.ranges[index])
            print('Elevation                (deg) : ', np.rad2deg(elevation[index]))
            print('________________________')
            print('TRANSMITTER')
            print('Divergence angle   (micro rad) : ', angle_div)
            print('Pointing error T   (micro rad) : ', angle_pe_t )
            print('Pointing error R   (micro rad) : ', angle_pe_r )
            print('Power transmitter        (dB)  : ', W2dBm(P_t))
            print('Wavelength         (micro rad) : ', wavelength)
            print('Data rate             (Gbit/s) : ', data_rate)
            print('________________________')
            print('GAINS')
            print('Transmitter gain          (dB) : ', W2dB(self.G_t))
            print('Receiver gain             (dB) : ', W2dB(self.G_r))
            print('Coding gain               (dB) : ', W2dB(self.G_c))
            print('________________________')
            print('LOSSES')
            print('Free space loss          (dBW) : ', W2dB(self.T_fs[index]))
            print('Transmitter loss         (dBW) : ', W2dB(eff_t))
            print('Receiver (optical)  loss (dBW) : ', W2dB(eff_r))
            print('Receiver (coupling) loss (dBW) : ', W2dB(eff_coupling))
            print('Receiver (tracking) loss (dBW) : ', W2dB(h_tracking))
            print('Divergence loss          (dBW) : ', W2dB(self.G_t) + W2dB(self.G_r) + W2dB(self.T_fs[index]))
            print('Divergence loss CHECK    (dBW) : ', W2dB(self.T_div[index]))
            print('Static pointing loss t   (dBW) : ', W2dB(self.T_point_t))
            print('Static pointing loss r   (dBW) : ', W2dB(self.T_point_r))
            print('Pointing jitter loss t   (dBW) : ', W2dB(self.T_pj[index]))
            # print('Pointing jitter loss r   (dBW) : ', W2dB(self.T_pj_r[index]))
            print('Beam wander loss         (dBW) : ', W2dB(self.T_bw[index]))
            print('Scintillation loss       (dBW) : ', W2dB(self.T_scint[index]))
            print('Beam spread loss         (dBW) : ', W2dB(self.T_beamspread[index]))
            # print('WFE loss               (dBW) : ', cons.W2dB(self.T_WFE[index]))
            # print('Attenuation loss       (dBW) : ', W2dB(self.T_atm_att[index])
            print('________________________')
            print('RECEIVER')
            print('Received static power   (dBW) : ', W2dB(self.P_r_0[index]))
            print('Received static photons (PPB) : ', W2dB(self.Np_r_0[index]))
            print('Received dynamic signal (dBW) : ', W2dB(self.P_r[index]))
            print('Received dynamic signal (PPB) : ', W2dB(self.Np_r[index]))
            print('Beam width at receiver  (m)   : ', self.w_r[index])
            print('________________________')
            print('LINK MARGIN')
            print('Power threshold communication (dBW) : ', W2dB(self.P_r_threshold[index]))
            print('Photon threshold communication (PPB): ', W2dB(self.Np_r_threshold[index]))
            print('Power threshold acquisition   (dBW) : ', W2dB(sensitivity_acquisition))
            print('Photon threshold acquisition  (PPB) : ', W2dB(Np_func(self.P_r_threshold[index],
                                                                 data_rate, eff_quantum)))
            print('Link margin communication     (dBW) : ', W2dB(self.LM_communication[index]))
            print('Link margin acquisition       (dBW) : ', W2dB(self.LM_acquisition[index]))







