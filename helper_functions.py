import numpy as np
import sqlite3
from tudatpy.kernel import constants as cons_tudat
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from input import *


from scipy.special import j0, j1
from scipy.signal import butter, filtfilt, welch, bessel
from scipy.fft import rfft, rfftfreq
from scipy.special import erfc, erf, erfinv, erfcinv
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array
import csv

def W2dB(x):
    return 10 * np.log10(x)
def dB2W(x):
    return 10**(x/10)
def W2dBm(x):
    return 10 * np.log10(x) + 30
def dBm2W(x):
    return 10**((x-30)/10)

def interpolator(x,y,x_interpolate, interpolation_type='cubic spline'):
    if interpolation_type == 'cubic spline':
        interpolator_settings = interpolators.cubic_spline_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)
    elif interpolation_type == 'hermite spline':
        interpolator_settings = interpolators.hermite_spline_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)
    elif interpolation_type == 'linear':
        interpolator_settings = interpolators.linear_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)

    y_dict = dict(zip(x, zip(y)))
    interpolator = interpolators.create_one_dimensional_vector_interpolator(y_dict, interpolator_settings)
    y_interpolated = dict()
    for i in x_interpolate:
        y_interpolated[i] = interpolator.interpolate(i)
    return result2array(y_interpolated)[:,1]

def h_p_airy(angle, D_r, focal_length):
    # REF: Handbook of Image and Video Processing (Second Edition), 2005, EQ.1.6
    # x = k_number * D_r / focal_length * r
    # return (2 * j1(x) / x)**2

    # REF: Wikipedia Airy Disk
    I_norm = 2 * j1(k_number * D_r * np.sin(angle)) / (k_number * D_r * np.sin(angle))

    r = angle * focal_length
    omega_0 = 0.90 * wavelength * focal_length / D_r
    I_norm_gauss_approx = np.exp(-2*r / omega_0)
    return I_norm_gauss_approx

def h_p_gaussian(angles, L, w_z):
    r = angles * L[:, None]
    v = np.sqrt(np.pi)*r / (np.sqrt(2) * w_z[:, None])
    A0 = erf(v)**2
    w_z_eq = w_z[:, None]**2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v**2))
    h_p_power = A0 * np.exp(-2 * r**2 / w_z_eq**2)
    h_p_intensity = np.exp(-angles ** 2 / angle_div ** 2)
    return h_p_power, h_p_intensity

def I_to_P(I, r, w_z):
    return I * np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def P_to_I(P, r, w_z):
    return P / np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def acquisition():
    # Add latency due to acquisition process (a reference value of 50 seconds is taken)
    # A more detailed method can be added here for analysis of the acquisition phase
    acquisition_time = 50.0  # seconds
    acquisition_indices = int(acquisition_time/step_size_AC)
    return acquisition_time, acquisition_indices

def radius_of_curvature(ranges):
    z_r = np.pi * w0 ** 2 * n_index / wavelength
    R = ranges * (1 + (z_r / ranges)**2)
    return R

def beam_spread(w0, ranges):
    z_r = np.pi * w0 ** 2 * n_index / wavelength
    w_r = w0 * np.sqrt(1 + (ranges / z_r) ** 2)
    # w_r = angle_div * ranges
    return w_r, z_r

def beam_spread_turbulence_ST(Lambda0, Lambda, var, w_r):
    # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.6.101
    W_ST = w_r * np.sqrt(1 + 1.33 * var * Lambda**(5/6) * (1 - 0.66 * (Lambda0**2 / (1 + Lambda0**2))**(1/6)))
    return W_ST

def beam_spread_turbulence_LT(r0, w_r):
    # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.12.48
    w_LT = np.zeros(len(r0))
    D_0 = D_t
    # D_0 = 2**(3/2)
    for i in range(len(r0)):
        if D_0/r0[i] < 1.0:
            w_LT[i] = w_r[i] * (1 + (D_0 / r0[i])**(5/3))**(1/2)
        elif D_0/r0[i] > 1.0:
            w_LT[i] = w_r[i] * (1 + (D_0 / r0[i])**(5/3))**(3/5)
    return w_LT

def PPB_func(P_r, data_rate):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    return P_r / (h * v * data_rate)

def Np_func(P_r, BW):
    # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, 2016, PAR. 3.4.3.3
    return P_r / (h * v * BW)

def data_rate_func(P_r, PPB):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    # data_rate = P_r / (Ep * N_p) / eff_quantum
    return P_r / (Ep * PPB *eff_quantum)

def save_to_file(data):
    data_merge = (data[0]).copy()
    data_merge.update(data[1])

    filename = "output.csv"
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_merge.keys())
            writer.writeheader()
            writer.writerow(data_merge)

    except IOError:
        print("I/O error")

def save_to_database(data, data_metrics, iterable):
    # Convert data array to dictionary with all SLQT3 column names as keys
    data_array = np.empty((len(iterable), len(data_metrics)))
    for i in range(len(data_metrics)):
        data_array[:, i] = data[data_metrics[i]]

    # Add row to data array with zero values
    data = np.vstack((np.zeros(len(data_metrics)), data_array))

    # Save data to sqlite3 database, 7 metrics are saved for each elevation angle (8 columns and N rows)
    filename = 'link_data_8_metrics_constant_data_rate.db'
    con = sqlite3.connect(filename)
    cur = con.cursor()
    cur.execute("CREATE TABLE performance_metrics(elevation, P_r, PPB, h_tot, P_margin_req_BER3, P_margin_req_BER6, P_margin_req_BER9, total_errors, total_errors_coded)")
    cur.executemany("INSERT INTO performance_metrics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
    con.commit()
    cur.close()
    con.close()


def load_from_database(metric, index_lb=0.0, index_ub=0.0):
    try:
        filename = 'link_data_8_metrics_constant_data_rate.db'
        # filename = 'test.db'
        con = sqlite3.connect(filename)
        cur = con.cursor()

        cur.execute('SELECT * FROM performance_metrics')
        data = cur.fetchall()

        if metric == "elevation":
            cur.execute('SELECT elevation FROM performance_metrics')
            data = cur.fetchall()
            data = np.array(data).flatten()

        elif metric == "all":
            cur.execute('SELECT * FROM performance_metrics')
            desc = cur.description
            column_names = [col[0] for col in desc]

            data1 = np.array(data).flatten().reshape(len(data), -1)[index_lb, :]
            data2 = np.array(data).flatten().reshape(len(data), -1)[index_ub, :]
            data_array = (data1 + data2) / 2

            # data_array = np.array(data).flatten().reshape(len(data), -1)
            # data_array = data_array[index_lb, :]
            # data_array = np.array(data).flatten().reshape(len(data), -1)[index_lb, :]


            # Convert data array to dictionary with all SLQT3 column names as keys
            data = dict()
            for i in range(len(column_names)):
                column_name = column_names[i]
                data[column_name] = data_array[:,i]
        return data

    except sqlite3.Error as error:
        print("Failed to read data from table", error)
    # if data == None:
    #     raise LookupError(("No database file found with name: "+str(filename)))

    finally:
        if con:
            con.close()


def filtering(effect: str,                  # Effect is eiter turbulence (scintillation, beam wander, angle of arrival) or jitter (TX jitter, RX jitter)
              order,                        # Order of the filter
              data: np.ndarray,             # Input dataset u (In this model this will be scintillation, beam wander jitter etc.)
              filter_type: str,             # 'Low pass' is taken for turbulence sampling
              f_cutoff_low = False,         # 1 kHz is taken as the standard cut-off frequency for turbulence
              f_cutoff_band = False,        # [100, 300] is taken as the standard bandpass range, used for mechanical jitter
              f_cutoff_band1=False,         # [900, 1050] is taken as the standard bandpass range, used for mechanical jitter
              f_sampling=10E3,              # 10 kHz is taken as the standard sampling frequency for all temporal fluctuations
              plot='no',                  # Option to plot the frequency domain of the sampled data and input data
              ):
    order_test = 2 * order
    # Digital filter settings
    if filter_type == 'lowpass':
        # This filter is initiated when the option 'lowpass' is used.
        # This is the case for the turbulence filter.
        b,  a  = butter(N=order, Wn=f_cutoff_low, btype=filter_type, analog=False, fs=f_sampling)
        b_test, a_test  = butter(N=order_test, Wn=f_cutoff_low, btype=filter_type, analog=False, fs=f_sampling)

    elif filter_type == 'multi':
        # This filter is initiated when the option 'multi' is used. Here, both 'lowpass' and 'bandpass' filters are used.
        # This is the case for the mechanical jitter filter.
        b,  a  = butter(N=order, Wn=f_cutoff_low, btype='lowpass', analog=False, fs=f_sampling)
        b1, a1 = butter(N=order, Wn=f_cutoff_band, btype='bandpass', analog=False,fs=f_sampling)
        b_test, a_test = butter(N=order_test, Wn=f_cutoff_low, btype='lowpass', analog=False, fs=f_sampling)
        b1_test, a1_test = butter(N=order_test, Wn=f_cutoff_band, btype='bandpass', analog=False, fs=f_sampling)

        if f_cutoff_band1:
            # An extra bandpass filter is initiated when this options is chosen.
            b2, a2 = butter(N=order, Wn=f_cutoff_band1, btype='bandpass', analog=False, fs=f_sampling)
            b2_test, a2_test = butter(N=order_test, Wn=f_cutoff_band1, btype='bandpass', analog=False, fs=f_sampling)

    # Applying a lowpass filter in order to obtain the frequency response of the turbulence (~1000 Hz) and jitter (~ 1000 Hz)
    # For beam wander the displacement values (m) are filtered.
    # For angle of arrival and mechanical pointing jitter for TX and RX, the angle values (rad) are filtered.
    # if effect == 'scintillation' or effect == 'beam wander' or effect == 'angle of arrival':

    if effect == 'scintillation' or effect == 'beam wander' or effect == 'angle of arrival':
        data_filt = np.empty(np.shape(data))
        data_filt_test = np.empty(np.shape(data))
        for i in range(len(data)):
            data_filt[i] = filtfilt(b, a, data[i])
            data_filt_test[i] = filtfilt(b_test, a_test, data[i])

    elif effect == 'TX jitter' or effect == 'RX jitter':
        data_filt_low = filtfilt(b, a, data)
        data_filt_low_test = filtfilt(b_test,  a_test, data)

        if not f_cutoff_band1:
            data_filt      = filtfilt(b1, a1, data_filt_low)
            data_filt      = data_filt + data_filt_low
            data_filt_test = filtfilt(b1_test, a1_test, data_filt_low_test)
            data_filt_test = data_filt_test + data_filt_low_test
        else:
            data_filt1 = filtfilt(b1, a1, data_filt_low)
            data_filt2 = filtfilt(b2, a2, data_filt_low)
            data_filt  = data_filt1 + data_filt2 + data_filt_low

            data_filt1_test = filtfilt(b1_test, a1_test, data_filt_low_test)
            data_filt2_test = filtfilt(b2_test, a2_test, data_filt_low_test)
            data_filt_test  = data_filt1_test + data_filt2_test + data_filt_low_test

    elif effect == 'extinction':
        data_filt = filtfilt(b, a, data)
        data_filt_test = filtfilt(b_test, a_test, data)

    if plot == "yes":
        # Create PSD of the filtered signal with the defined sampling frequency
        f_0, psd_0 = welch(data, f_sampling, nperseg=1024)
        f, psd_data = welch(data_filt, f_sampling, nperseg=1024)
        f_test, psd_data_test = welch(data_filt_test, f_sampling, nperseg=1024)

        # Plot the frequency domain
        fig, ax = plt.subplots(2,1)
        ax[0].set_title(str(filter_type)+' (butter) filtered signal of '+str(effect), fontsize=15)
        print(data.ndim)
        if data.ndim > 1:
            uf = rfft(data[0])
            yf = rfft(data_filt[0])
            yf_test = rfft(data_filt_test[0])
            xf = rfftfreq(len(data[0]), 1 / f_sampling)
            # Plot Amplitude over frequency domain
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq='+str(f_cutoff_low)+' Hz, order= '+str(order))
            ax[0].plot(xf, abs(yf_test), label='Filtered: cut-off freq='+str(f_cutoff_low)+' Hz, order= '+str(order_test))
            ax[0].set_ylabel('Amplitude [rad]')

            # Plot PSF over frequency domain
            ax[1].semilogy(f_0, psd_0[0], label='unfiltered')
            ax[1].semilogy(f, psd_data[0], label='order='+str(order))
            ax[1].semilogy(f, psd_data_test[0], label='order='+str(order_test))
            ax[1].set_xscale('log')
            ax[1].set_xlabel('frequency [Hz]')
            ax[1].set_ylabel('PSD [rad**2/Hz]')


        elif data.ndim == 1:
            uf = rfft(data)
            yf = rfft(data_filt)
            yf_test = rfft(data_filt_test)
            xf = rfftfreq(len(data), 1 / f_sampling)
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq=(lowpass='+str(f_cutoff_low)+', bandpass='+str(f_cutoff_band)+', '+ str(f_cutoff_band1)+' Hz, order= '+str(order))
            ax[0].plot(xf, abs(yf_test), label='Filtered: cut-off freq=(lowpass='+str(f_cutoff_low)+', bandpass='+str(f_cutoff_band)+', '+ str(f_cutoff_band1)+' Hz, order= '+ str(order_test))
            # ax[0].set_xscale('log')
            ax[0].set_ylabel('Amplitude [rad]')
            ax[1].semilogy(f_0, psd_0, label='unfiltered')
            ax[1].semilogy(f, psd_data, label='order='+str(order))
            ax[1].semilogy(f, psd_data_test, label='order='+str(order_test))
            ax[1].set_xscale('log')
            if effect == 'extinction':
                ax[1].set_xlim(1.0E-3, 1.0E0)
            else:
                ax[1].set_xlim(1.0E0, 1.2E3)
            ax[1].set_xlabel('frequency [Hz]')
            ax[1].set_ylabel('PSD [rad**2/Hz]')

        ax[0].grid()
        ax[1].grid()
        ax[0].legend()
        ax[1].legend()
        plt.show()

    # Normalize data
    sums = data_filt.std(axis=data_filt.ndim-1)
    if data_filt.ndim > 1:
        data_filt = data_filt / sums[:, None]
    elif data_filt.ndim == 1:
        data_filt = data_filt / sums

    return data_filt

def sensitivity_dim1(elevation, P_r, errors, bits):
    errors = errors * step_size_AC / interval_dim1
    bits = bits / interval_dim1
    elevation_01 = np.rad2deg(elevation)
    elevation_02 = elevation_01[::2]
    elevation_04 = elevation_01[::4]
    elevation_06 = elevation_01[::6]
    elevation_08 = elevation_01[::8]
    elevation_10 = elevation_01[::10]

    d_elev_01 = abs(np.diff(elevation_01))
    d_elev_02 = abs(np.diff(elevation_02))
    d_elev_04 = abs(np.diff(elevation_04))
    d_elev_06 = abs(np.diff(elevation_06))
    d_elev_08 = abs(np.diff(elevation_08))
    d_elev_10 = abs(np.diff(elevation_10))

    Pr_01 = W2dBm(P_r.mean(axis=1))
    Pr_02 = Pr_01[::2]
    Pr_04 = Pr_01[::4]
    Pr_06 = Pr_01[::6]
    Pr_08 = Pr_01[::8]
    Pr_10 = Pr_01[::10]

    d_Pr_01 = abs(np.diff(Pr_01))
    d_Pr_02 = abs(np.diff(Pr_02))
    d_Pr_04 = abs(np.diff(Pr_04))
    d_Pr_06 = abs(np.diff(Pr_06))
    d_Pr_08 = abs(np.diff(Pr_08))
    d_Pr_10 = abs(np.diff(Pr_10))

    errors_01 = errors
    errors_02 = errors_01[::2]
    errors_04 = errors_01[::4]
    errors_06 = errors_01[::6]
    errors_08 = errors_01[::8]
    errors_10 = errors_01[::10]

    d_errors_01 = abs(np.diff(errors_01))
    d_errors_02 = abs(np.diff(errors_02))
    d_errors_04 = abs(np.diff(errors_04))
    d_errors_06 = abs(np.diff(errors_06))
    d_errors_08 = abs(np.diff(errors_08))
    d_errors_10 = abs(np.diff(errors_10))

    elevation = np.concatenate((elevation_01[1:], elevation_02[1:], elevation_04[1:], elevation_06[1:], elevation_08[1:], elevation_10[1:]))
    d_elevation = np.concatenate((d_elev_01,      d_elev_02,        d_elev_04,        d_elev_06,        d_elev_08,        d_elev_10))
    Pr_mean         = np.concatenate((Pr_01[1:],  Pr_02[1:],        Pr_04[1:],        Pr_06[1:],        Pr_08[1:],        Pr_10[1:]))
    d_Pr_mean       = np.concatenate((d_Pr_01,    d_Pr_02,          d_Pr_04,          d_Pr_06,          d_Pr_08,          d_Pr_10))
    errors   = np.concatenate((errors_01[1:],     errors_02[1:],    errors_04[1:],    errors_06[1:],    errors_08[1:],    errors_10[1:]))
    d_errors = np.concatenate((d_errors_01,       d_errors_02,      d_errors_04,      d_errors_06,      d_errors_08,      d_errors_10))

    fig, ax = plt.subplots(2, 2)
    ax[0,0].set_title('Elevation w.r.t. mean power at RX')
    ax[0,0].scatter(elevation, Pr_mean, s=10)
    # ax[0].scatter(np.linspace(0,100, len(elevation)), elevation)
    ax[0,0].set_ylabel('$P_{RX}$ (dBm)')

    # d_Pr_fit = np.polyfit(elevation_01[1:], d_Pr_01 / 1.0E6, 2)
    # d_Pr_fit = np.poly1d(d_Pr_fit)
    ax[1,0].scatter(elevation_01[1:], d_Pr_01, s=10, label='$\Delta \epsilon$= '+ str(np.round(d_elev_01[0], 2)) + ' ($\degree$)')
    ax[1,0].scatter(elevation_10[1:], d_Pr_10, s=10,
                     label='$\Delta \epsilon$= ' + str(np.round(d_elev_10[0], 2)) + ' ($\degree$)')
    # ax[1].plot(elevation_01[1:], d_Pr_fit(elevation_01[1:]))
    ax[1,0].set_ylabel('$\Delta$ mean $P_{RX}$ ($\Delta$ dBm) \n'
                     'for $\Delta \epsilon$='+str(np.round(d_elev_01[0],2))+' ($\degree$)')

    ax[0,1].set_title('Elevation w.r.t. error bits per second')
    ax[0,1].scatter(elevation, np.ones(len(elevation))*bits/1.0E6, s=10, label='Total Mbits/s='+str(bits/1E6))
    ax[0,1].scatter(elevation, errors/1.0E6, s=10, label='Error Mbits/s')
    # ax[0].scatter(np.linspace(0,100, len(elevation)), elevation)
    ax[0,1].set_ylabel('Errors per second (Gbits/s)')
    ax[0,1].set_yscale('log')
    # d_e_fit = np.polyfit(elevation_01[1:], d_errors_01/1.0E6, 1)
    # d_e_fit = np.poly1d(d_e_fit)
    ax[1,1].scatter(elevation_01[1:], d_errors_01/bits, s=10, label='$\Delta \epsilon$= '+ str(np.round(d_elev_01[0], 2)) + ' ($\degree$)')
    ax[1,1].scatter(elevation_10[1:], d_errors_10/bits, s=10, label='$\Delta \epsilon$= '+ str(np.round(d_elev_10[0], 2)) + ' ($\degree$)')
    ax[1,1].set_yscale('log')
    ax[1,1].set_ylabel('$\Delta$ errors / Total bits (per second)')

    ax[1,0].set_xlabel('Elevation $\epsilon$ ($\degree$)')
    ax[1,1].set_xlabel('Elevation $\epsilon$ ($\degree$)')

    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    ax[1,0].legend(fontsize=20)
    ax[0,1].legend(fontsize=20)
    ax[1,1].legend(fontsize=20)

    fig1 = plt.figure(figsize=(6, 6), dpi=125)
    ax = fig1.add_subplot(111, projection='3d')
    ax.set_title('Sensitivity analysis of elevation w.r.t. error bits received \n '
                 '(' + str(len(elevation)) + ' distributed samples of $\epsilon$)')
    x = elevation
    y = d_elevation
    z = np.log(d_errors) #d_Pr_mean

    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    ax.set_ylabel('$\Delta$ $\epsilon$ [$\Delta \degree$]')
    ax.set_zlabel('log($\Delta$ errors) [$\Delta$ bits]')
    ax.set_xlabel('Elevation $\epsilon$ [$\degree$]')
    fig.tight_layout()
    plt.show()
