# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from blume.table import table
import pandas as pd
from IPython.display import display

from input import *
from helper_functions import *


class link_budget:
    def __init__(self,
                 ranges: np.array,
                 h_strehl: float,
                 w_ST: np.array,
                 h_beamspread: np.array,
                 h_ext: float):

        # Range
        # ----------------------------------------------------------------------------
        self.ranges = np.array(ranges)

        # Laser profile
        # ----------------------------------------------------------------------------
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 19
        # self.I_t_0 = self.P_t / (np.pi/4 * self.D_t**2) #W/m^2
        self.P_t = P_t
        # REF: Wikipedia: Gaussian Beam
        self.I_t_0 = 2 * self.P_t / (np.pi * w0**2)
        self.w_r = np.array(w_ST)

        # Gains
        #----------------------------------------------------------------------------
        # REF: PERFORMANCE LIMITATION OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.14
        # self.G_t = (np.pi*self.D_t / self.wavelength)**2
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.G_t = 8 / angle_div**2
        self.G_r = (np.pi*D_r / wavelength)**2
        # REF: OPTICAL ON-OFF KEYING FOR LOW EARTH ORBIT DOWNLINK APPLICATIONS, D.GIGGENBACH, TABLE 11.5
        # self.G_c = 1.0 #dB2W(4.0)

        # Losses
        # ----------------------------------------------------------------------------
        self.h_ext = h_ext
        self.h_strehl = np.array(h_strehl)
        self.h_beamspread = np.array(h_beamspread)
        self.T_clipping = np.exp(-2 * clipping_ratio ** 2 * obscuration_ratio ** 2) - np.exp(-2 * clipping_ratio_ac ** 2)
        self.G_t = self.G_t * self.T_clipping
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.h_fs = (wavelength / (4 * np.pi * self.ranges)) ** 2
        self.h_l = self.G_t * self.G_r * eff_t * eff_r * eff_coupling * h_splitting * self.h_ext * self.h_fs

        # Divergence angle
        # ----------------------------------------------------------------------------
        self.angle_div = np.sqrt(8 / self.G_t)


        # # Transmitter pointing error loss
        # REF: PERFORMANCE LIMITATION OF LASER SAT COMMUNICATION..., S.ARNON, 2003, EQ.16
        # self.T_point_t = np.exp(-self.G_t * self.angle_pe_t ** 2)
        # self.T_point_r = np.exp(-self.G_r * self.angle_pe_r ** 2)

        # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021, SLIDE 40
        # self.T_point_t = np.exp(-angle_pe_t**2 / angle_div**2)
        # self.T_point_r = np.exp(-angle_pe_r ** 2 / angle_div ** 2)
        self.T_div = D_r**2 / (D_t + angle_div*ranges)**2

    # ------------------------------------------------------------------------
    # -----------------------RECEIVED-POWER-&-INTENSITY-----------------------
    # ------------------------------------------------------------------------

    # This function computes the static power at the receiver without any turbulence losses
    def P_r_0_func(self):
        self.P_r_0 = self.P_t * self.h_l * self.h_beamspread * self.h_strehl
        return self.P_r_0

    # This function computes the static intensity at the receiver without any turbulence losses
    def I_r_0_func(self):
        # REF: LONG TERM IRRADIANCE STATISTICS FOR OPTICAL GEO..., T.KAPSIS, 2019, EQ.1
        self.I_r_0 =  (w0 / self.w_r)**2 * self.I_t_0 * \
                      self.G_t * self.G_r * eff_t * eff_r * eff_coupling * h_splitting *\
                      self.h_fs * self.h_strehl * self.h_ext
        return self.I_r_0

    # Firstly, the static link budget is computed with 'P_r_0_func' and 'I_r_0_func'.
    # Then, this function implements all dynamic contributions of the link budget that are imported from the database,
    # which are scintillation loss (T_scint), pointing jitter loss (T_pj), beam wander loss (T_bw) and the resulting received power (P_r)
    def dynamic_contributions(self, P_r, PPB, T_dyn_tot, T_scint, T_TX, T_RX):
        # print(performance_output.type())
        # if performance_output[0].ndim != 2:
        #     raise ValueError('Incorrect argument dimension of performance_output')
        # # if performance_output.astype() !=
        # if len(performance_output[:, 0]) != len(self.ranges)+1:
        #     print(len(performance_output[:, 0]), len(self.ranges)+1)
        #     raise ValueError('Incorrect argument shape of performance_output')

        self.P_r = P_r
        self.PPB = PPB
        self.T_dyn_tot = T_dyn_tot
        self.T_scint = T_scint
        self.T_TX = T_TX
        self.T_RX = T_RX

    def P_r_tracking_func(self):
        self.P_r_tracking = (1 - h_splitting) * self.P_r


    # This function computes the link margin
    def link_margin(self, P_r_thres, PPB_thres):
        self.P_r_thres_BER9 = P_r_thres[0]
        self.P_r_thres_BER6 = P_r_thres[1]
        self.P_r_thres_BER3 = P_r_thres[2]
        self.PPB_thres_BER9 = PPB_thres[0]
        self.PPB_thres_BER6 = PPB_thres[1]
        self.PPB_thres_BER3 = PPB_thres[2]

        self.LM_tracking = self.P_r_tracking / sensitivity_acquisition
        self.LM_acquisition = self.P_r / sensitivity_acquisition

        self.LM_comm_BER9 = self.P_r / self.P_r_thres_BER9
        self.LM_comm_BER6 = self.P_r / self.P_r_thres_BER6
        self.LM_comm_BER3 = self.P_r / self.P_r_thres_BER3
        return self.LM_comm_BER9, self.LM_comm_BER6, self.LM_comm_BER3

    # def save_data(self, time):
    #     self.gains = np.array((self.G_t, self.G_r))
    #
    #     losses_metrics = ['h_fs', 'eff_t', 'pointing_jitter_t', 'eff_r', 'pointing_jitter_r', 'coupling_r', 'splitting_r', 'h_bw', 'h_scint', 'h_beamspread', 'h_ext']
    #     number_of_losses = len(losses_metrics)
    #     self.losses = np.zeros((number_of_losses, len(time)))
    #
    #     self.losses[0] = self.T_fs[:, None]
    #     self.losses[1] = self.T_att
    #     self.losses[2] = self.Strehl_ratio
    #     self.losses[3] = self.T_beamspread
    #     self.losses[4] = eff_t
    #     self.losses[5] = eff_r
    #     self.losses[6] = h_splitting
    #     self.losses[7] = eff_coupling
    #
    #     self.losses[8] = self.T_TX[i]
    #     self.losses[9] = self.T_RX[i]
    #     self.losses[10] = self.T_scint[i]

    def plot(self, time:np.array, indices:list, elevation: np.array, type= "time-series"):
        time_hrs = time / 3600
        if type == "gaussian beam profile":
            r_t = np.linspace(-w0 * 1.5, w0 * 1.5, 1000)

            I_t = self.I_t_0 * np.exp(-r_t**2 / w0**2)

            fig_I, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title('Gaussian beam intensity at transmitter , w0: ' + str(np.round(w0,2))+' m')
            ax1.set_ylabel('Intensity (W/$m^2$)')
            ax2.set_title('Normalized Gaussian beam intensity at receiver')
            ax2.set_ylabel('Intensity (W/$m^2$)')
            ax2.set_xlabel('Radial position from beam center (m)')
            ax1.plot(np.ones(2) * -D_t, np.array((0, self.I_t_0)), color='orange', label='Telescope obstruction')
            ax1.plot(np.ones(2) *  D_t, np.array((0, self.I_t_0)), color='orange', )
            ax1.plot(np.ones(2) * -w0, np.array((0, self.I_t_0)), color='green', label='Beam waist $w0$ (1/$e^2$)')
            ax1.plot(np.ones(2) * w0, np.array((0, self.I_t_0)), color='green', )
            ax1.plot(r_t, I_t, linestyle='-', label="Pt: " + str(W2dBm(P_t)) + " dBm")

            ax2.plot(np.ones(2) * -D_r, np.array((0, 1)), color='orange', label='Telescope obstruction')
            ax2.plot(np.ones(2) * D_r, np.array((0, 1)), color='orange', )

            for i in indices:
                r_r = np.linspace(-self.w_r[-1] * 1.5, self.w_r[-1] * 1.5, 1000)
                I_r = self.I_r_0 * np.exp(-r_r ** 2 / self.w_r[i] ** 2)
                ax2.plot(r_r, I_r, linestyle='-', label="$W_r$="+str(np.round(self.w_r[i],2))+
                                                        ' m (1/$e^2$), Pr=' + str(np.round(W2dBm(np.mean(self.P_r[i])),2)) +
                                                        " dBm, $\epsilon$="+str(np.round(np.rad2deg(elevation[i]),2))+'$\degree$')

            ax2.set_ylim(0.0, self.I_r_0.max()*1.2)
            ax1.legend()
            ax2.legend()

        elif type == "table":
            parameters = ['GEOMETRY',
                          'Time', 'Range', 'elevation', 'Link', ' ',
                          'LCT',
                          'divergence angle', 'Mech. pointing error TX', 'Mech. pointing jitter TX', 'Mech. pointing error RX', 'Mech. pointing jitter RX',
                          'Power TX', 'Wavelength', 'data rate', ' ',
                          'GAINS',
                          'Gain TX', 'Gain RX',  ' ',
                          'STATIC LOSSES',
                          'Free space loss', 'TX system efficiency', 'RX system efficiency', 'RX coupling efficiency', 'RX splitting efficiency',
                          'Beam spread loss(ST)', 'WFE loss (Strehl ratio)', 'Attenuation loss', ' ',
                          'DYNAMIC LOSSES ',
                          'TX loss (mech. jitter and BW)', 'RX loss (mech. jitter and AoA)', 'Scintillation loss', 'Total dynamic loss', ' ',
                          'RX signals',
                          'Static power RX', 'Dynamic power RX', 'Tracking signal RX', 'Beam width at RX', ' ',
                          'LINK MARGIN',
                          'Power threshold communication  (1.0E-9)', 'Power threshold communication  (1.0E-6)', 'Power threshold communication  (1.0E-3)',
                          'Power threshold acquisition', 'Link margin communication (1.0E-9)', 'Link margin communication (1.0E-6)', 'Link margin communication (1.0E-3)',
                          'Link margin tracking', 'Link margin acquisition'
                          ]
            units = ['',
                      'hrs', 'm', 'degree', '-',
                      '', ' ',
                      'rad', 'rad', 'rad', 'rad', 'rad', 'dBm', 'nm', 'Gbit/s',
                      '', ' ',
                      'dB', 'dB',
                      '', ' ',
                      'dB', 'dB', 'dB', 'dB', 'dB', 'dB', 'dB', 'dB',
                      '', ' ',
                      'dB', 'dB', 'dB', 'dB',
                      '', ' ',
                      'dBm', 'dBm', 'dBm', 'm',
                      '', ' ',
                      'dBm', 'dBm', 'dBm','dBm',
                      'dB', 'dB', 'dB', 'dB', 'dB'
                      ]

            for index in indices:
                values = [' ',
                          time_hrs[index], self.ranges[index],
                          np.rad2deg(elevation[index]),
                          link,
                          ' ', ' ',
                          angle_div, angle_pe_t, std_pj_t, angle_pe_r, std_pj_r, W2dBm(P_t), wavelength*1.0E9, data_rate/1.0E9,
                          ' ', ' ',
                          W2dB(self.G_t),W2dB(self.G_r),
                          ' ', ' ',
                          W2dB(self.h_fs[index]), W2dB(eff_t), W2dB(eff_r), W2dB(eff_coupling), W2dB(h_splitting),
                          W2dB(self.h_beamspread[index]), W2dB(self.h_strehl[index]), W2dB(self.h_ext[index]),
                          ' ', ' ',
                          W2dB(self.T_TX[index]), W2dB(self.T_RX[index]), W2dB(self.T_scint[index]), W2dB(self.T_dyn_tot[index]),
                          ' ', ' ',
                          W2dBm(self.P_r_0[index]), W2dBm(self.P_r[index]),  W2dBm(self.P_r_tracking[index]),  self.w_r[index],
                          ' ', ' ',
                          W2dBm(self.P_r_thres_BER9), W2dBm(self.P_r_thres_BER6), W2dBm(self.P_r_thres_BER3), W2dBm(sensitivity_acquisition),
                          W2dB(self.LM_comm_BER9[index]), W2dB(self.LM_comm_BER6[index]), W2dB(self.LM_comm_BER3[index]), W2dB(self.LM_tracking[index]), W2dB(self.LM_acquisition[index])
                          ]


                data = {'Parameter': parameters,
                        'Unit': units,
                        'Value': values}
                columns = ('Parameter',
                           'Unit',
                           'Value')

                df = pd.DataFrame(data, columns=columns)
                filename = 'link_budget_'+str(np.round(np.rad2deg(elevation[index]),2))+'.csv'
                # df.to_csv(filename)
                print(df.to_latex())

    def print(self,
              t = 0.0,
              index = 0,
              elevation = 0.0,
              ):

            print('------------------------------------------------')
            print('LINK-BUDGET ANALYSIS')
            print('------------------------------------------------')
            print('LASER LINK')
            print('Time                       (s) : ', t[index])
            print('Range                      (m) : ', self.ranges[index])
            print('Elevation                (deg) : ', np.rad2deg(elevation[index]))
            print('________________________')
            print('TRANSMITTER')
            print('Divergence angle         (rad) : ', angle_div)
            print('Pointing error T         (rad) : ', angle_pe_t)
            print('Jitter std T      (rad, 1 rms) : ', std_pj_t)
            print('Pointing error R         (rad) : ', angle_pe_r)
            print('Jitter std R      (rad, 1 rms) : ', std_pj_r)
            print('Power transmitter        (dBm) : ', W2dBm(P_t))
            print('Wavelength               (rad) : ', wavelength)
            print('Data rate              (bit/s) : ', data_rate)
            print('________________________')
            print('GAINS')
            print('Transmitter gain          (dB) : ', W2dB(self.G_t))
            print('Receiver gain             (dB) : ', W2dB(self.G_r))
            print('________________________')
            print('STATIC LOSSES')
            print('Free space loss          (dB) : ', W2dB(self.h_fs[index]))
            print('Transmitter loss         (dB) : ', W2dB(eff_t))
            print('Receiver (system)  loss  (dB) : ', W2dB(eff_r))
            print('Receiver (coupling) loss (dB) : ', W2dB(eff_coupling))
            print('Receiver (splitting) loss (dB) : ', W2dB(h_splitting))
            # print('Divergence loss CHECK    (dB) : ', W2dB(self.T_div[index]))
            # print('Divergence loss CHECK    (dB) : ', W2dB((w0/self.w_r[index])**2))
            print('Beam spread loss (ST)     (dB) : ', W2dB(self.h_beamspread[index]))
            print('WFE loss (Strehl ratio)   (dB) : ', W2dB(self.h_strehl[index]))
            print('Attenuation loss          (dB) : ', W2dB(self.h_ext[index]))
            print('________________________')
            print('DYNAMIC LOSSES')
            print('TX loss (mech. jit and BW) (dB) : ', W2dB(self.T_TX[index]))
            print('RX loss (mech. jit and AoA (dB) : ', W2dB(self.T_RX[index]))
            print('Scintillation loss         (dB) : ', W2dB(self.T_scint[index]))
            print('Total dynamic loss         (dB) : ', W2dB(self.T_dyn_tot[index]))
            print('Received dyn. power CHECK  (dBm): ', W2dBm(P_t * self.G_t * self.G_r *
                                                                eff_t * eff_r * eff_coupling * h_splitting *
                                                                self.h_fs[index] * self.h_beamspread[index] *
                                                                self.h_strehl[index] * self.h_ext[index] *
                                                              self.T_dyn_tot[index]))
            print('________________________')
            print('RECEIVER')
            print('Received static power   (dBm) : ', W2dBm(self.P_r_0[index]))
            print('Received dynamic signal (dBm) : ', W2dBm(self.P_r[index]))
            # print('Received dynamic signal (PPB) : ', self.PPB[index])
            print('Received tracking signal (dBm): ', W2dBm(self.P_r_tracking[index]))
            print('Beam width at receiver  (m)   : ', self.w_r[index])

            print('________________________')
            print('LINK MARGIN')
            print('Power threshold communication  (1.0E-9) (dBm) : ', W2dBm(self.P_r_thres_BER9))
            print('Power threshold communication  (1.0E-6) (dBm) : ', W2dBm(self.P_r_thres_BER6))
            print('Power threshold communication  (1.0E-3) (dBm) : ', W2dBm(self.P_r_thres_BER3))
            print('Photon threshold communication (1.0E-9) (PPB) : ', W2dB(self.PPB_thres_BER9))
            print('Power threshold acquisition             (dBm) : ', W2dBm(sensitivity_acquisition))
            print( '')
            print('Link margin communication (1.0E-9)      (dB)  : ', W2dB(self.LM_comm_BER9[index]))
            print('Link margin communication (1.0E-6)      (dB)  : ', W2dB(self.LM_comm_BER6[index]))
            print('Link margin communication (1.0E-3)      (dB)  : ', W2dB(self.LM_comm_BER3[index]))
            print('Link margin tracking           (dB) : ', W2dB(self.LM_tracking[index]))
            print('Link margin acquisition        (dB) : ', W2dB(self.LM_acquisition[index]))







