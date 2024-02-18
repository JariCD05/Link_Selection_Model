import sqlite3

import numpy as np
import scipy.signal
from input import *

import random
from scipy.special import j0, j1, binom
from scipy.stats import rv_histogram, norm
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq
from scipy.special import erfc, erf, erfinv, erfcinv
from scipy.special import erfc, erfcinv
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
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
    elif interpolation_type == 'lagrange':
        interpolator_settings = interpolators.lagrange_interpolation(8)

    y_dict = dict(zip(x, zip(y)))
    interpolator = interpolators.create_one_dimensional_vector_interpolator(y_dict, interpolator_settings)
    y_interpolated = dict()
    for i in x_interpolate:
        y_interpolated[i] = interpolator.interpolate(i)
    return result2array(y_interpolated)[:,1]


def cross_section(elevation_cross_section, elevation, time_links):
    time_cross_section = []
    indices = []
    for e in elevation_cross_section:
        index = np.argmin(abs(elevation - np.deg2rad(e)))
        indices.append(index)
        t = time_links[index] / 3600
        time_cross_section.append(t)
    return indices, time_cross_section

def h_p_airy(angle, D_r, focal_length):
    # REF: Wikipedia Airy Disk
    # Fraunhofer diffraction pattern
    I_norm = (2 * j1(k_number * D_r/2 * np.sin(angle)) /
                    (k_number * D_r/2 * np.sin(angle)))**2

    P_norm = (j0(k_number * D_r/2 * np.sin(angle)) )**2 + (j1(k_number * D_r/2 * np.sin(angle)) )**2
    return P_norm

def h_p_gaussian(angles, angle_div):
    h_p_intensity = np.exp(-2*angles ** 2 / angle_div ** 2)
    return h_p_intensity

def I_to_P(I, r, w_z):
    return I * np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def P_to_I(P, r, w_z):
    return P / np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def acquisition(current_index, current_acquisition_time, step_size):
    # Add latency due to acquisition process (a reference value of 50 seconds is taken, found in input.py)
    # A more detailed method can be added here for analysis of the acquisition phase
    total_acquisition_time = current_acquisition_time + acquisition_time
    index = current_index + int(acquisition_time / step_size)
    return total_acquisition_time, index

def radius_of_curvature(ranges):
    z_r = np.pi * w0 ** 2 * n_index / wavelength
    R = ranges * (1 + (z_r / ranges)**2)
    return R

def beam_spread(angle_div, ranges):
    w_r = angle_div * ranges
    return w_r

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
    return P_r / (h * v * PPB)

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

    # Applying a lowpass filter in order to obtain the frequency response of the turbulence (~1000 Hz) and jitter (~ 1000 Hz)
    # For beam wander the displacement values (m) are filtered.
    # For angle of arrival and mechanical pointing jitter for TX and RX, the angle values (rad) are filtered.

    eps = 1.0E-9

    if effect == 'scintillation' or effect == 'beam wander' or effect == 'angle of arrival':
        data_filt = np.empty(np.shape(data))
        for i in range(len(data)):
            # Digital filter settings
            b, a = butter(N=order, Wn=f_cutoff_low[i], btype=filter_type, analog=False, fs=f_sampling)

            z, p, k = scipy.signal.tf2zpk(b, a)
            r = np.max(np.abs(p))
            approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
            data_filt[i] = filtfilt(b, a, data[i], method="gust", irlen=approx_impulse_len)


    elif effect == 'TX jitter' or effect == 'RX jitter':
        # Digital filter settings
        b, a   = butter(N=order, Wn=f_cutoff_low,   btype='lowpass',  analog=False, fs=f_sampling)
        b1, a1 = butter(N=order, Wn=f_cutoff_band,  btype='bandpass', analog=False, fs=f_sampling)
        b2, a2 = butter(N=order, Wn=f_cutoff_band1, btype='bandpass', analog=False, fs=f_sampling)

        z, p, k = scipy.signal.tf2zpk(b, a)
        r = np.max(np.abs(p))
        approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
        data_filt_low = filtfilt(b, a, data, method="gust", irlen=approx_impulse_len)

        if not f_cutoff_band1:
            data_filt      = filtfilt(b1, a1, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt      = data_filt + data_filt_low
        else:
            data_filt1 = filtfilt(b1, a1, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt2 = filtfilt(b2, a2, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt  = data_filt1 + data_filt2 + data_filt_low

    if plot == "yes":
        # Create PSD of the filtered signal with the defined sampling frequency
        f_0, psd_0 = welch(data, f_sampling, nperseg=1024)
        f, psd_data = welch(data_filt, f_sampling, nperseg=1024)

        # Plot the frequency domain
        fig, ax = plt.subplots(2,1)
        ax[0].set_title(str(filter_type)+' (butter) filtered signal of '+str(effect), fontsize=15)
        if data.ndim > 1:
            uf = rfft(data[0])
            yf = rfft(data_filt[0])
            xf = rfftfreq(len(data[0]), 1 / f_sampling)
            # Plot Amplitude over frequency domain
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq='+str(f_cutoff_low)+' Hz, order= '+str(order))
            ax[0].set_ylabel('Amplitude [rad]')

            # Plot PSF over frequency domain
            ax[1].semilogy(f_0, psd_0[0], label='unfiltered')
            ax[1].semilogy(f, psd_data[0], label='order='+str(order))
            ax[1].set_xscale('log')
            ax[1].set_xlabel('frequency [Hz]')
            ax[1].set_ylabel('PSD [W/Hz]')


        elif data.ndim == 1:
            uf = rfft(data)
            yf = rfft(data_filt)
            xf = rfftfreq(len(data), 1 / f_sampling)
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq=(lowpass='+str(f_cutoff_low)+', bandpass='+str(f_cutoff_band)+', '+ str(f_cutoff_band1)+' Hz, order= '+str(order))
            # ax[0].set_xscale('log')
            ax[0].set_ylabel('Amplitude [rad]')
            ax[1].semilogy(f_0, psd_0, label='unfiltered')
            ax[1].semilogy(f, psd_data, label='order='+str(order))
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


def conversion_ECEF_to_ECI(pos_ECEF, time):
    theta_earth = omega_earth * time
    pos_ECI = np.zeros((len(pos_ECEF), 3))
    for i in range(len(pos_ECEF)):
        ECEF_to_ECI = np.array(((np.cos(theta_earth[i]), -np.sin(theta_earth[i]), 0),
                                (np.sin(theta_earth[i]), np.cos(theta_earth[i]), 0),
                                (0, 0, 1)))

        pos_ECI[i] = np.matmul(pos_ECEF[i], ECEF_to_ECI)

    return pos_ECI.tolist()

def Strehl_ratio_func(D_t, r0, tip_tilt="YES"):
    # REF: R. PARENTI, 2006, EQ.1-3
    if tip_tilt == "NO":
        var_WFE = 1.03 * (D_t / r0) ** (5 / 3)
    elif tip_tilt == "YES":
        var_WFE = 0.134 * (D_t / r0) ** (5 / 3)

    return np.exp(-var_WFE)

def flatten(l):
    new_data = [item for sublist in l for item in sublist]
    return np.array(new_data)

def autocorr(x):
    x = np.array(x)
    result_tot = []
    for x_i in x:
        mean = x_i.mean()
        var = x_i.var()
        norm_x_i = x_i - mean

        auto_corr = np.correlate(norm_x_i, norm_x_i, mode='full')[len(norm_x_i)-1:]
        auto_corr = auto_corr / var / len(norm_x_i)
        result_tot.append(auto_corr)
    return np.array(result_tot)

def autocovariance(x, scale='micro'):
    x -= x.mean()
    auto_cor = scipy.signal.correlate(x, x)
    auto_cor = auto_cor / np.max(auto_cor)
    lags = scipy.signal.correlation_lags(len(x), len(x))
    if scale == 'micro':
        lags = lags * step_size_channel_level * 1000
    elif scale == 'macro':
        lags = lags * step_size_link / 60

    return auto_cor, lags

def data_to_time(data, data_list, time):
    time_list = []
    indices = []
    for d in data:
        index = np.argmin(abs(data_list - np.deg2rad(d)))
        indices.append(index)
        t = time[index] / 3600
        time_list.append(t)
    return time_list

def distribution_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=steps)
        dist = rv_histogram(hist, density=True)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        std  = dist.std()
        mean = dist.mean()

    else:
        pdf = np.empty((length, len(x)))
        cdf = np.empty((length, len(x)))
        std  = np.empty(length)
        mean = np.empty(length)
        for i in range(length):
            hist = np.histogram(data[i], bins=steps)
            dist = rv_histogram(hist, density=True)
            pdf[i] = dist.pdf(x)
            cdf[i] = dist.cdf(x)
            std[i]  = dist.std()
            mean[i] = dist.mean()

    return pdf, cdf, x, std, mean

def pdf_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=steps*1)
        rv = rv_histogram(hist, density=True)
        pdf = rv.pdf(x)
    else:
        pdf = np.empty((length, len(x)))
        for i in range(length):
            hist = np.histogram(data[i], bins=steps*1)
            rv = rv_histogram(hist, density=True)
            pdf[i] = rv.pdf(x)
    return pdf, x

def cdf_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=int(steps*1))
        dist = rv_histogram(hist, density=True)
        cdf = dist.cdf(x)
    else:
        cdf = np.empty((length, len(x)))
        for i in range(length):
            hist = np.histogram(data[i], bins=int(steps*1))
            dist = rv_histogram(hist, density=True)
            cdf[i] = dist.cdf(x)
    return cdf, x

def shot_noise(Sn, R, P, Be, eff_quantum):
    noise_sh = 4 * Sn * R ** 2 * P * Be / eff_quantum
    return noise_sh

def background_noise(Sn, R, I, D, delta_wavelength, FOV, Be):
    # This noise types defines the solar background noise, which is simplified to direct incoming sunlight.
    # Solar- and atmospheric irradiance are defined in input.py, atmospheric irradiance is neglected by default and can be added as an extra contribution.
    A_r = 1 / 4 * np.pi * D ** 2
    P_bg = I * A_r * delta_wavelength * 1E9 * FOV
    noise_bg = 4 * Sn * R ** 2 * P_bg * Be
    return noise_bg

def BER_avg_func(pdf_x, pdf_y, LCT, total=False):
    # Pr = P_r_0[:, None] * pdf_h_x
    Pr = dBm2W(pdf_x)
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=Pr, I_sun=I_sun)
    SNR, Q = LCT.SNR_func(P_r=Pr,
                          detection=detection,
                          noise_sh=noise_sh, noise_th=noise_th,
                          noise_bg=noise_bg,noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation)

    if total == False:
        BER_avg = np.trapz(pdf_y * BER, x=pdf_x, axis=1)
    else:
        BER_avg = np.trapz(pdf_y * BER, x=pdf_x, axis=0)

    return BER_avg

def penalty(P_r, desired_frac_fade_time):
    # This functions computes a power penalty, based on the method of Giggenbach.

    if P_r.ndim > 1:
        closest_P_min = np.empty(len(P_r))
        for i in range(len(P_r)):
            closest_frac_fade_time = np.inf
            # P_min_range = np.arange(W2dBm(P_r[i].min()), W2dBm(P_r[i].max()), 0.5)
            P_min_range = np.arange(-100.0, -10.0, 0.1)
            for P_min in P_min_range:
                P_min = dBm2W(P_min)
                frac_fade_time = np.count_nonzero(P_r[i] < P_min) / len(P_r[i])
                # Check if the current fractional fade time is closer to the desired value than the previous closest value
                if abs(frac_fade_time - desired_frac_fade_time) < abs(closest_frac_fade_time - desired_frac_fade_time):
                    closest_frac_fade_time = frac_fade_time
                    closest_P_min[i] = P_min
    else:
        closest_frac_fade_time = np.inf
        P_min_range = np.linspace((P_r).min(), (P_r).max(), 1000)
        for P_min in P_min_range:
            frac_fade_time = np.count_nonzero(P_r < P_min) / len(P_r)
            # Check if the current fractional fade time is closer to the desired value than the previous closest value
            if abs(frac_fade_time - desired_frac_fade_time) < abs(closest_frac_fade_time - desired_frac_fade_time):
                closest_frac_fade_time = frac_fade_time
                closest_P_min = P_min

    h_penalty = (closest_P_min / P_r.mean(axis=1)).clip(min=0.0, max=1.0)
    return h_penalty


def get_difference_wrt_kepler_orbit(
        state_history: dict,
        central_body_gravitational_parameter: float):

    """"
    This function takes a Cartesian state history (dict of time as key and state as value), and
    computes the difference of these Cartesian states w.r.t. an unperturbed orbit. The Keplerian
    elemenets of the unperturbed trajectory are taken from the first entry of the state_history input
    (converted to Keplerian elements)

    Parameters
    ----------
    state_history : Cartesian state history
    central_body_gravitational_parameter : Gravitational parameter that is to be used for Cartesian<->Keplerian
                                            conversion

    Return
    ------
    Dictionary (time as key, Cartesian state difference as value) of difference of unperturbed trajectory
    (semi-analytically propagated) w.r.t. state_history, at the epochs defined in the state_history.
    """

    # Obtain initial Keplerian elements abd epoch from input
    initial_keplerian_elements = element_conversion.cartesian_to_keplerian(
        list(state_history.values())[0], central_body_gravitational_parameter)
    initial_time = list(state_history.keys())[0]

    # Iterate over all epochs, and compute state difference
    keplerian_solution_difference = dict()
    for epoch in state_history.keys():

        # Semi-analytically propagated Keplerian state to current epoch
        propagated_kepler_state = two_body_dynamics.propagate_kepler_orbit(
            initial_keplerian_elements, epoch - initial_time, central_body_gravitational_parameter)

        # Converted propagated Keplerian state to Cartesian state
        propagated_cartesian_state = element_conversion.keplerian_to_cartesian(
            propagated_kepler_state, central_body_gravitational_parameter)

        # Compute difference w.r.t. Keplerian orbit
        keplerian_solution_difference[epoch] = propagated_cartesian_state - state_history[epoch]

    return keplerian_solution_difference

