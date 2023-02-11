import numpy as np
import sqlite3
from tudatpy.kernel import constants as cons_tudat
from matplotlib import pyplot as plt
from constants import *


def gaussian_beam(I0, r, w0, w_z):
    return I0 * (w0 / w_z)**2 * np.exp(-2 * r ** 2 / w_z ** 2)

def I_to_P(I, r, w_z):
    return I * np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def P_to_I(P, r, w_z):
    return P / np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def acquisition():
    # Add latency due to acquisition process (a reference value of 50 seconds is taken)
    # A more detailed method can be added here for analysis of the acquisition phase
    acquisition_time = 50.0  # seconds
    acquisition_indices = int(acquisition_time/step_size_dim2)
    return acquisition_time, acquisition_indices


def beam_spread(w0, ranges):
    z_r = np.pi * w0 ** 2 * n_index / wavelength
    w_r = w0 * np.sqrt(1 + (ranges / z_r) ** 2)
    return w_r


def beam_spread_turbulence(r0, w_0, w_r):
    # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.12.48
    w_LT = np.zeros(len(r0))
    D_0 = 2 * w_0

    for i in range(len(r0)):
        if D_0/r0[i] < 1.0:
            w_LT[i] = w_r[i] * np.sqrt(1 + (D_0 / r0[i])**(5/3))
        elif D_0/r0[i] > 1.0:
            w_LT[i] = w_r[i] * np.sqrt(1 + (D_0 / r0[i])**(5/3))**(3/5)

    return w_LT

def Np_func(P_r, data_rate, eff_quantum):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    return P_r / (Ep * data_rate) / eff_quantum

def data_rate_func(P_r, N_p, eff_quantum):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    # data_rate = P_r / (Ep * N_p) / eff_quantum
    return P_r / (Ep * N_p) / eff_quantum



def hand_over_strategy(number_of_planes, number_sats_per_plane, index, elevation_angles, time):
    start_elev = []
    start_sat = []

    # The handover time is initiated with t=0, then the handover procedure starts
    t_handover = 0

    while len(start_elev) == 0:
        for plane in range(number_of_planes):
            for sat in range(number_sats_per_plane):

                elev_last = elevation_angles[plane][sat][index - 1]
                elev = elevation_angles[plane][sat][index]

                # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD, AND ALSO AN ELEVATION THAT IS STILL INCREASING (START OF WINDOW)
                if elev > elevation_min and elev > elev_last:
                    start_elev.append(elev)
                    start_sat.append([plane, sat])

        if len(start_elev) > 0:
            print('First satellites in LOS at t = ', time[index])
            # CHOOSE THE SATELLITE WITH THE SMALLEST ELEVATION ANGLE
            current_plane = start_sat[np.argmin(start_elev)][0]
            current_sat = start_sat[np.argmin(start_elev)][1]
            current_elevation = start_elev[np.argmin(start_elev)]

            print('------------------------------------------------')
            print('HAND OVER')
            print('------------------------------------------------')
            print('Chosen satellite to start: plane = ', current_plane, ', sat in plane = ', current_sat, ', elevation = ', current_elevation)
            print('Handover time: ', t_handover/60, 'minutes, Current time: ', time[index])

        elif time[index] > end_time:
            break
        else:
            t_handover += step_size_dim2
            index += 1


    return current_plane, current_sat, t_handover, index



def connect_dim1_dim2(terminal, P_r_0, I_r_0, zenith_angles_dim1, zenith_angles_dim2, dim_1_results):
    BER_avg_array = np.ones(len(zenith_angles_dim1))
    noise_array = np.ones(len(zenith_angles_dim1))
    h_avg_array = np.ones(len(zenith_angles_dim1))

    for i in range(len(zenith_angles_dim1)):
        current_zenith = zenith_angles_dim1[i]
        current_elev = np.pi / 2 - current_zenith
        index_dim1 = np.argmin(np.abs(current_zenith- zenith_angles_dim2))

        # Take
        current_P_r_0 = P_r_0[index_dim1]
        current_I_r_0 = I_r_0[index_dim1]
        # -----------------------------------------------------------------------
        # --------STATISTICAL-PARAMETERS-&-POWER/INTENSITY-FLUCTUATIONS----------
        # -----------------------------------------------------------------------
        # t = dim_1_results[i][0]
        h_tot = dim_1_results[i][1]
        r_tot = dim_1_results[i][2]
        P_r_fluct = current_P_r_0 * h_tot
        I_r_0_fluct = current_I_r_0 * h_tot

        h_avg = np.mean(h_tot)

        # -----------------------------------------------------------------------
        # -----------------------------NOISE-BUDGET-------------------------------
        # ------------------------------------------------------------------------

        noise_sh = terminal.noise(noise_type="shot", P_r=P_r_fluct)
        noise_th = terminal.noise(noise_type="thermal")
        noise_bg = terminal.noise(noise_type="background", P_r=P_r_fluct, I_sun=I_sun)

        # ------------------------------------------------------------------------
        # --------------------------THRESHOLD-AT-RECEIVER-------------------------
        # ------------------------------------------------------------------------
        SNR_thres_sc, P_r_thres_sc, Np_thres_sc = terminal.threshold()
        # ------------------------------------------------------------------------
        # -----------------------FLUCTUATING-SNR-&-BER----------------------------
        # ------------------------------------------------------------------------
        # SNR & BER at receiver
        SNR = terminal.SNR_func(P_r_fluct)
        BER = terminal.BER_func()

        BER_avg = np.mean(BER)
        # Average out BER values!

        # ------------------------------------------------------------------------
        # ----------------------------FADE-STATISTICS-----------------------------
        # ------------------------------------------------------------------------
        number_of_fades = np.count_nonzero(P_r_fluct < P_r_thres_sc)
        fade_time = number_of_fades * step_size_dim1

        # ------------------------------------------------------------------------
        # -----------------------ADD-OBTAINED-VALUES-TO-ARRAY---------------------
        # ------------------------------------------------------------------------
        BER_avg_array[i] = BER_avg
        noise_array[i]   = np.mean(noise_sh + noise_th + noise_bg)
        h_avg_array[i]   = h_avg

    return h_avg_array, noise_array, BER_avg_array, fade_time


def get_data(metric, index = 0):
    try:
        con = sqlite3.connect('link_data_16_param.db')
        cur = con.cursor()

        # cur.execute('SELECT * FROM performance_metrics')
        # data = cur.fetchall()

        if metric == "elevation":
            cur.execute('SELECT elevation FROM performance_metrics')
            data = cur.fetchall()
            data = np.array(data).flatten()
            print(np.shape(data))

        elif metric == "all":
            cur.execute('SELECT * FROM performance_metrics')
            data = cur.fetchall()
            print(len(data))
            print(len(index))
            print(np.shape(np.array(data).flatten().reshape(len(data),-1)))
            data = np.array(data).flatten().reshape(len(data), -1)[index, :]

        return data

    except sqlite3.Error as error:
        print("Failed to read data from table", error)
    finally:
        if con:
            con.close()