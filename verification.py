import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim, lsim2, bessel, butter, filtfilt, lfilter, lfilter_zi
from scipy.fft import fft, rfft, ifft, fftfreq, rfftfreq
from scipy.special import binom, i0, i1, erfc, erfinv, erfcinv, gamma
from scipy.stats import norm, lognorm, rayleigh, rice, rv_histogram
from itertools import chain

from input import *
from helper_functions import *
from Constellation import constellation
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from PDF import dist

# -------------------------------------------------------------------------------
# Constellation modeling verification
# -------------------------------------------------------------------------------

def test_constellation(constellation):
    constellation = constellation('LEO_1',
                                  number_of_planes=1,
                                  number_sats_per_plane=1,
                                  height_init=h_SC,
                                  inc_init=inc_SC,
                                  RAAN_init=120.0,
                                  TA_init= 180.0,
                                  omega_init=0.0)

    constellation.propagate(start_time, end_time)
    constellation.verification(start_time,end_time)

# -------------------------------------------------------------------------------
# Filtering verification
# -------------------------------------------------------------------------------

def test_filter():
    duration = 100
    dt = 0.0001
    t = np.arange(0, duration, dt)
    samples = len(t)
    u = np.random.normal(loc=1, scale=0, size=samples)

    cut_off_frequency = 1000
    sampling_frequency = 1/dt
    nyquist = sampling_frequency / 2

    # b, a = bessel(N=5, Wn=cut_off_frequency, btype='lowpass', analog=False, fs=sampling_frequency)
    # y = filtfilt(b, a, u)
    # t_out, y_out, xout = lsim((b, a), U=u, T=t)

    order = 5
    order1 = 8
    order2 = 9

    cutoff_ub = cut_off_frequency*1.1
    cutoff_lb = cut_off_frequency*0.9
    b, a = bessel(N=order, Wn=[cutoff_lb/nyquist, cutoff_ub/nyquist], btype='bandpass')
    b1, a1 = bessel(N=order1, Wn=[cutoff_lb/nyquist, cutoff_ub/nyquist], btype='bandpass')
    b2, a2 = bessel(N=order2, Wn=[cutoff_lb/nyquist, cutoff_ub/nyquist], btype='bandpass')
    # b, a = butter(N=order, Wn=[cutoff_lb, cutoff_ub], btype='lowpass', analog=False, fs=sampling_frequency)
    # b1, a1 = butter(N=order1, Wn=cut_off_frequency, btype='lowpass', analog=False, fs=sampling_frequency)
    # b2, a2 = butter(N=order2, Wn=cut_off_frequency, btype='lowpass', analog=False, fs=sampling_frequency)

    y = filtfilt(b, a, u)
    yf = rfft(y)
    xf = rfftfreq(samples, 1 / sampling_frequency)
    uf = rfft(u)

    y1 = filtfilt(b1, a1, u)
    yf1 = rfft(y1)

    y2 = filtfilt(b2, a2, u)
    yf2 = rfft(y2)


    # Plot the time domain U
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Low-pass filtered signal of a Monte Carlo dataset (Gaussian distribution)')
    plt.plot(t, u)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')


    # Plot the magnitude spectrum U
    ax.plot(xf[10:], abs(uf[10:]), label='Gaussian ($\sigma$=1, $\mu$=0.5): sampling freq=1000 Hz')
    ax.plot(xf[10:], abs(yf[10:]), label='Filtered: cut-off freq=1000 Hz, order='+str(order))
    ax.plot(xf[10:], abs(yf1[10:]), label='Filtered: cut-off freq=1000 Hz, order='+str(order1))
    ax.plot(xf[10:], abs(yf2[10:]), label='Filtered: cut-off freq=1000 Hz, order='+str(order2))
    ax.set_xlabel('Frequency spectrum (Hz)')
    ax.set_ylabel('Signal magnitude (-)')
    ax.legend()


# -------------------------------------------------------------------------------
# Coding verification
# -------------------------------------------------------------------------------

def test_RS_coding(method):
    if method == 'simple':
        # Method 1
        # REF: CCSDS Historical Document, 2006, CH.5.5, EQ.3-4
        duration = 1
        dt = 1.0E-4
        t = np.arange(0.0, duration, dt)
        samples = len(t)
        N, K = 255, 223
        E = int((N - K)/2)
        symbol_length = 8
        number_of_bits = data_rate * duration
        number_of_bits_per_codeword = K * symbol_length
        number_of_codewords = number_of_bits / (number_of_bits_per_codeword)

        # OOK-NRZ modulation scheme
        SNR = np.linspace(0, 50, samples)
        Vb = 1/2 * erfc( np.sqrt(SNR) )
        Vs = 1 - (1 - Vb)**symbol_length

        k_values = np.arange(E, N - 1)
        binom_values = binom(N - 1, k_values)
        SER_coded = Vs * (binom_values * np.power.outer(Vs, k_values) * np.power.outer(1 - Vs, N - k_values - 1)).sum(axis=1)
        BER_coded = 2**(symbol_length-1)/N * SER_coded

        if method == 'complex':
            # Method 2
            Vb = np.ones(samples) * 1E-5
            Vs = 1 - (1 - Vb)**symbol_length

            step_size_codewords = K / data_rate
            mapping = list(np.arange(0, samples, step_size_codewords).astype(int))

            bits_per_codeword = number_of_bits / number_of_codewords
            Vs_per_codeword = Vs[0, mapping]
            Vb_per_codeword = Vb[0, mapping]

            codewords = t[mapping]


            SER_uncoded = Vs[mapping]
            BER_uncoded = Vb[mapping]
            SER_coded = np.zeros(samples)
            BER_coded = np.zeros(samples)

            for i in range(number_of_codewords):
                SER_coded[i] = Vs[i] * sum(binom(N - 1, k) * Vs[i] ** k * (1 - Vs[i]) ** (N - k - 1) for k in range(E, N - 1))

            BER_coded = Vb/Vs * SER_coded

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(W2dB(SNR), Vb, label='Uncoded (Channel BER)')
    ax.plot(W2dB(SNR), BER_coded, label='(255,223) RS coded')
    ax.invert_yaxis()
    ax.set_ylim(1.0E-9, 1.0E-1)
    ax.set_yscale('log')
    ax.grid()
    ax.set_title('Uncoded BER vs RS coded BER (255,223)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Probability of Error (# Error bits/total bits)')
    ax.legend()
    plt.show()

# -------------------------------------------------------------------------------
# Sensitivity verification
# -------------------------------------------------------------------------------

def test_sensitivity(method):
    BER_thres = 1.0E-9
    modulation = "OOK-NRZ"
    detection = "APD"
    data_rate = 10.0E9

    LCT = terminal_properties()

    P_r = np.linspace(dBm2W(-40), dBm2W(-30), 1000)
    N_p = Np_func(P_r, data_rate)
    N_p = W2dB(N_p)

    noise_sh = LCT.noise(noise_type="shot", P_r=P_r)
    noise_th = LCT.noise(noise_type="thermal")
    noise_bg = LCT.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
    noise_beat = LCT.noise(noise_type="beat")
    LCT.threshold(modulation=modulation, detection=detection)

    SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                          noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation)

    if method == "BER/PPB":
        plt.plot(W2dB(SNR), BER)
        plt.yscale('log')
        ax = plt.gca()
        # ax.invert_xaxis()
        ax.grid()
        ax.set_title('BER vs PPB')
        ax.set_xlabel('PPB (dB)')
        ax.set_ylabel('-$log_{10}$(BER)')
        plt.show()

    elif method == "compare_backward_threshold_function_with_forward_BER_function":
        BER_thres = 1.0E-6
        modulation = "OOK-NRZ"
        detection = "APD"
        data_rate = 10.0E9

        LCT = terminal_properties()

        P_r = np.linspace(dBm2W(-40), dBm2W(-20), 100)
        N_p = Np_func(P_r, data_rate)
        N_p = W2dB(N_p)

        noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
        LCT.threshold(modulation=modulation, detection=detection, BER_thres=BER_thres)

        SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        BER = LCT.BER_func(Q=Q, modulation=modulation)

        fig, ax = plt.subplots(1, 1)
        ax.plot(W2dBm(P_r), BER)
        ax.plot(W2dBm(LCT.P_r_thres * np.ones(2)), [BER.min(), BER.max()], label='Pr threshold')
        ax.plot([W2dBm(P_r.min()), W2dBm(P_r.max())], BER_thres * np.ones(2), label='BER threshold')

        ax.set_ylabel('BER')
        ax.set_yscale('log')
        ax.set_xlabel('Pr')
        ax.grid()
        ax.legend()
        plt.show()

    elif method == "Gallion":
        # Reproduction of P.GALLION, FIG.3.12
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Comparison with literature: Direct-detection (NF=4, G=50 '+str(detection)+', Modulation ('+str(modulation)+')')

        duration = 100
        samples = 1000
        t = np.linspace(0, duration, samples)
        N, K = 255, 223
        E = int((N - K) / 2)
        symbol_length = 8

        data_rate = np.linspace(0.01*BW, BW, 50)
        M = 5
        noise_factor = 2
        R = eff_quantum * q / h / v

        P_r = np.linspace(dBm2W(-60), dBm2W(-20), 10000)

        P_r = dB2W(-90)
        noise_sh = LCT.noise(noise_type="shot", P_r=P_r)
        noise_th = LCT.noise(noise_type="thermal")
        noise_bg = LCT.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
        noise_beat = LCT.noise(noise_type="beat")

        SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        BER = LCT.BER_func(Q=Q, modulation=modulation)

        # data_rates = [data_rate/100, data_rate/10, data_rate]
        detection = 'quantum limit'
        noise_factor = 1
        M = 1
        for i in [0,1]:
            noise_bg = LCT.noise(noise_type='background', I_sun=I_sun)
            noise_th = LCT.noise(noise_type="thermal")
            noise_beat = LCT.noise(noise_type="beat")
            noise_sh = LCT.noise(noise_type="shot", P_r=P_r, M=M, noise_factor=noise_factor)
            if i == 3:
                noise_bg = LCT.noise(noise_type='background', I_sun=I_sun)
                noise_th = LCT.noise(noise_type="thermal")
                noise_beat = LCT.noise(noise_type="beat")
                label = 'Shot, background, thermal, beat noise'
                print(W2dB(noise_bg), W2dB(noise_sh[-1]), W2dB(noise_th), W2dB(noise_beat))
            elif i == 0:
                LCT.noise_bg = 0.0
                label = 'W/O background noise'
            elif i == 1:
                LCT.noise_th = 0.0
                label = 'W/O thermal noise'
            elif i == 2:
                LCT.noise_beat = 0.0
                label = 'W/O beat noise'

            LCT.PPB_func(P_r=P_r, data_rate=data_rate)
            LCT.SNR_func(P_r=P_r, detection=detection)
            LCT.BER_func(modulation=modulation)
            LCT.coding(duration, K, N, [1], samples, test=True)
            ax.plot(W2dB(LCT.PPB), -np.log10(LCT.BER), label=label)
            ax.plot(W2dB(LCT.PPB), -np.log10(LCT.BER), label='Uncoded')
            ax.plot(W2dB(LCT.PPB), -np.log10(LCT.BER_coded), label='Coded RS('+str(N)+', '+str(K)+')')
            ax.plot(W2dB(eff_quantum * LCT.PPB), LCT.PPB, label='$\eta$PPB')



            noise_sh = LCT.noise_sh
            noise_sh = q * R * P_r * BW
            noise_th = (2 * k * T_s * BW / R_L)
            noise_beat = 2 * R**2 * (h*v/2)**2 * 1.5 * BW**2
            noise = M**2 * noise_factor * (noise_sh + noise_bg) + noise_th + noise_beat

            Q_thres = np.sqrt(2) * erfcinv(2*BER_thres)
            P_r_thres = Q_thres * np.sqrt( M**2 * noise_factor * noise_sh + noise_th + noise_beat) / (M * R)
            Np_thres = Np_func(P_r_thres, BW)
            SNR = np.sqrt((M * P_r * R)**2 / (M**2*(noise_sh+noise_bg) + noise_th))

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(data_rate/BW, LCT.PPB_thres_check, label='BER='+str(BER_thres))
            ax.grid()
            ax.set_ylabel('Average number of photons/bit (PPB)')
            ax.set_xlabel('Spectral efficiency ($R_B/B_o$)')
            ax.legend()
            plt.show()


            ax.grid()
            ax.set_ylabel('-$LOG_{10}$(BER)')
            ax.set_xlabel('Photons/bit (dB)')
            # ax.set_xscale('log')
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.legend()
            plt.show()

# -------------------------------------------------------------------------------
# Turbulence (Cn and wind profile)
# -------------------------------------------------------------------------------

def test_windspeed():
    slew_rate = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)
    speed_AC = [10, 100, 150, 200]
    axs = plt.subplot(111)

    turb = turbulence()
    axs.set_title(f'Windspeed (Bufton)  vs  heights')
    for Vg in speed_AC:
        turb.windspeed_func(slew=slew_rate, Vg=Vg, wind_model_type=wind_model_type)
        turb.Cn_func(turbulence_model=turbulence_model)
        heights = turb.heights[turb.heights<20000]
        windspeed = turb.windspeed[turb.heights<20000]
        axs.plot(heights, windspeed, label='V wind (m/s rms) = ' + str(np.round(turb.windspeed_rms,2))+', V ac (m/s) = '+str(Vg)+', slew: '+str(np.round(slew_rate,4)))


    turb.windspeed_func(slew=slew_rate*2, Vg=Vg, wind_model_type=wind_model_type)
    turb.Cn_func(turbulence_model=turbulence_model)
    heights = turb.heights[turb.heights<20000]
    windspeed = turb.windspeed[turb.heights<20000]
    axs.plot(heights, windspeed, label='V wind (m/s rms) = ' + str(np.round(turb.windspeed_rms,2))+', V ac (m/s) = '+str(Vg)+', slew: '+str(np.round(slew_rate*2,4)))

    turb.windspeed_func(slew=slew_rate, Vg=speed_AC[0], wind_model_type=wind_model_type)
    turb.Cn_func(turbulence_model="HVB_57")
    axs.plot(turb.heights, turb.windspeed, label='V wind (m/s rms) = ' + str(np.round(turb.windspeed_rms,2))+', V ac (m/s) = '+str(speed_AC[0])+', A: yes')

    axs.set_yscale('log')
    axs.set_ylim(1.0E-20, 1.0E-14)
    axs.set_xscale('log')
    axs.set_ylabel('Wind speed [m/s]')
    axs.set_xlabel('Altitude $h$ [m]')
    axs.legend()
    plt.show()

# -------------------------------------------------------------------------------
# Turbulence (Strehl ratio)
# -------------------------------------------------------------------------------

def test_strehl_ratio():
    r_0 = 0.10
    w0 = np.linspace(0, 5*r_0, 100)
    D = np.e * w0

    var_tc = 1.03  * (D / r_0) ** (5 / 3)
    var = 0.134 * (D / r_0) ** (5 / 3)
    S_tc = np.exp(-var_tc)
    S = np.exp(-var)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(D/r_0, S_tc, label='Parenti/Andrews TC')
    ax.plot(D/r_0, S, label='Parenti/Andrews NC')
    ax.grid()
    ax.set_ylabel('Strehl ratio')
    ax.set_xlabel('D/r0')

    ax.legend()
    plt.show()

# -------------------------------------------------------------------------------
# Turbulence (Beam wander)
# -------------------------------------------------------------------------------

def test_beamwander():
    zenith_angle = np.deg2rad(0.0)
    r0 = 0.1
    w0 = np.linspace(0.01, 1.0, 1000)
    L = h_SC - h_AC
    heights = np.linspace(h_AC, h_SC, 1000)
    w_r, z_r = beam_spread(w0, L)

    windspeed_rms = 150.0
    Cn = 5.94E-53 * (windspeed_rms / 27) ** 2 * (heights) ** 10 * np.exp(-heights / 1000.0) + \
                      2.75E-16 * np.exp(-heights / 1500.0)
    # r0 = (0.423 * k_number ** 2 / abs(np.cos(zenith_angle)) *
    #                    np.trapz(Cn * ((L - heights) / L) ** (5/3), x=heights)) ** (-3 / 5)

    # REF: ANDREWS
    var_bw = 0.54 * L**2 * 1/np.cos(zenith_angle)**2 * (wavelength / (2*w0))**2 * (2*w0 / r0)**(5/3)
                          # * (1 - ((kappa0**2 * w0**2) / (1 + kappa0**2 * w0**2))**(1/6))

    # REF: F. Dios, SCINTILLATION AND BEAM-WANDER ANALSIS IN AN OPTICAL GROUND STATION-SATELLITE UPLINK, 2004
    w0 = 0.05
    w_ST = w0**2 * (1 + (L / z_r)**2) + 2 * (4.2 * L / (k_number * r0) * (1 - 0.26 * (r0/w0)**(1/3)))**2
    var_bw1 = 1/2 * 2.07 * 1/np.cos(zenith_angle) * np.trapz(Cn * (L - heights)**2 * (1 / w_r)**(1/3), x=heights)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(w0*100, np.sqrt(var_bw)/L*1.0E6)
    ax.plot(w0*100, np.sqrt(var_bw1)/L*1.0E6)
    # ax.plot(w0*100, w_ST)
    # ax.plot(D/r_0, S, label='Parenti/Andrews NC')
    ax.grid()
    ax.set_xscale('log')
    # ax.set_ylabel('Strehl ratio')
    ax.set_xlabel('w0 (cm)')

    ax.legend()
    plt.show()

# -------------------------------------------------------------------------------
# Jitter
# -------------------------------------------------------------------------------

def test_jitter():
    from LCT import terminal_properties
    LCT = terminal_properties
    steps = 10000
    angle_div = np.linspace(2.0E-6, 100.0E-6, 1000)
        # [10.0E-6, 20E-6, 30E-6, 40E-6]
    w0 = wavelength / (np.pi * angle_div)
    L = 900.0E3
    w_r, z_r = beam_spread(w0, L)
    std_urad = std_pj_t*1.0E6
    Lambda0 = 2 * L / (k_number * w0 ** 2)
    Lambda = 2 * L / (k_number * w_r ** 2)
    var_rytov = 0.4
    w_ST = beam_spread_turbulence_ST(Lambda0, Lambda, var_rytov, w_r)

    angle_pe_t_x = dist.norm_rvs(sigma=std_pj_t, steps=steps) + angle_pe_t
    angle_pe_t_y = dist.norm_rvs(sigma=std_pj_t, steps=steps) + angle_pe_t

    angle_pe_t_radial = np.sqrt(angle_pe_t_x ** 2 + angle_pe_t_y ** 2)
    angle_pe_t_radial = np.mean(angle_pe_t_radial)
    r_pj_t = angle_pe_t_radial * L
    h_pj_t = np.exp(-r_pj_t ** 2 / w_ST ** 2)

    fig, ax = plt.subplots(2,1)
    ax[0].set_title('Top: Scatter plot of jitter+error TX, \n Bottom: Normalized intensity profile for different values of PE/div ratio')
    ax[0].scatter(angle_pe_t_x*1.0E6, angle_pe_t_y*1.0E6, label='std='+str(np.round(std_urad,2))+' urad, static error='+str(angle_pe_t)+' urad')
    ax[0].set_xlabel('Pointing + jitter error X (urad)')
    ax[0].set_ylabel('Pointing + jitter error Y (urad)')

    ax[1].plot(angle_pe_t_radial/angle_div, h_pj_t)
    ax[1].set_xlabel('$angle_P$ / $angle_{div}$')
    ax[1].set_ylabel('Pointing + jitter (normalized intensity)')

    # ax.plot(w0*100, w_ST)
    # ax.plot(D/r_0, S, label='Parenti/Andrews NC')
    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    plt.show()


# -------------------------------------------------------------------------------
# PDF gaussian-rayleigh
# -------------------------------------------------------------------------------

def test_PDF():
    std_normal = 3.3E-6
    mean_normal = 0.0
    samples = 100000
    f_sampling = 10E3
    f_cutoff = 1E3
    f_cutoff_band = [100.0, 200.0]
    L = 500E3
    w_r = beam_spread(w0, L)

    X1 = np.empty((2, samples))
    X2 = np.empty((2, samples))
    for i in range(2):
        X1[i] = np.random.standard_normal(size=samples)
        X2[i] = np.random.standard_normal(size=samples)

    X1_norm = filtering(effect='beam wander', order=2, data=X1, f_cutoff_low=f_cutoff,
                        filter_type='lowpass', f_sampling=f_sampling, plot='no')

    X2_norm = filtering(effect='beam wander', order=2, data=X2, f_cutoff_low=f_cutoff,
                        filter_type='lowpass', f_sampling=f_sampling, plot='no')

    X1_norm = X1_norm[0]
    X2_norm = X2_norm[0]

    # ----------------------------------------------------------------------------
    # Standard normal > lognormal distribution
    mean_lognorm = -0.5 *  np.log(std_normal**2+ 1)
    std_lognorm  = np.sqrt(np.log(std_normal**2 + 1))
    # mean_lognorm = np.exp(mean_normal + std_normal**2 / 2)
    # std_lognorm = np.sqrt((np.exp(std_normal**2) - 1) * np.exp(std_normal**2))
    X1_lognorm = np.exp(mean_lognorm + std_lognorm * X1_norm)

    # ----------------------------------------------------------------------------
    # Standard normal > normal distribution > rice distribution: for TX jitter and RX jitter
    X1_normal = std_normal * X1_norm + mean_normal
    X2_normal = std_normal * X2_norm + mean_normal
    mean_rice = np.sqrt(mean_normal**2 + mean_normal**2)
    std_rice = std_normal
    X_rice = np.sqrt(X1_normal**2 + X2_normal**2)

    # ----------------------------------------------------------------------------
    # Standard normal > rayleigh distribution: for Beam wander or Angle-of-Arrival
    std_rayleigh = np.sqrt(2 / (4 - np.pi) * std_normal**2)
    mean_rayleigh = np.sqrt(np.pi / 2) * std_rayleigh
    X_rayleigh = std_rayleigh * np.sqrt(X1_norm**2 + X2_norm**2)


    # X domains
    x_0 = np.linspace(-3, 3, 100)
    x_normal = np.linspace(-angle_div, angle_div, 100)
    x_lognorm = np.linspace(0.0, 3.0, 100)
    x_rayleigh = np.linspace(0.0, angle_div, 100)
    x_rice = np.linspace(0.0, angle_div, 100)

    # Theoretical distributions
    pdf_0 = 1 / np.sqrt(2 * np.pi * 1 ** 2) * np.exp(-((x_0 - 0) / 1) ** 2 / 2)
    pdf_normal = 1/np.sqrt(2 * np.pi * std_normal**2) * np.exp(-((x_normal - mean_normal) / std_normal)**2/2)
    pdf_lognorm = 1 / (x_lognorm * std_lognorm * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x_lognorm) - mean_lognorm)**2 / (2 * std_lognorm**2))
    pdf_rayleigh = x_rayleigh / std_rayleigh**2 * np.exp(-x_rayleigh**2 / (2 * std_rayleigh**2))
    pdf_rice = x_rice / std_rice**2 * np.exp(-(x_rice**2 + mean_rice**2) / (2*std_rice**2)) * i0(x_rice * mean_rice / std_rice**2)
    b = mean_rice / std_rice**2

    fig, ax = plt.subplots(4, 1)

    # plot normalized samples
    ax[0].hist(X2_norm, density=True)
    loc, scale = norm.fit(X2_norm)
    pdf_data = norm.pdf(x_0, loc, scale)
    std = norm.std(loc=loc, scale=scale)
    mean = norm.mean(loc=loc, scale=scale)
    ax[0].plot(x_0, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ', $\sigma$=' + str(std), color='red')
    ax[0].plot(x_0, pdf_0, label='$\mu$=' + str(0) + ' $\sigma$=' + str(1))


    # plot normal distribution
    ax[1].hist(X1_normal, density=True)
    loc, scale = norm.fit(X1_normal)
    std = norm.std(loc=loc, scale=scale)
    mean = norm.mean(loc=loc, scale=scale)
    pdf_data = norm.pdf(x=x_normal, loc=loc, scale=scale)
    ax[1].plot(x_normal, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ', $\sigma$=' + str(std), color='red')

    ax[1].hist(X2_normal, density=True)
    loc, scale = norm.fit(X2_normal)
    std = norm.std(loc=loc, scale=scale)
    mean = norm.mean(loc=loc, scale=scale)
    pdf_data = norm.pdf(x=x_normal, loc=loc, scale=scale)
    ax[1].plot(x_normal, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ', $\sigma$=' + str(std), color='red')
    ax[1].plot(x_normal, pdf_normal, label='$\mu$=' + str(mean_normal) + ' $\sigma$=' + str(std_normal))

    # plot lognormal distribution
    # ax[2].hist(X1_lognorm, density=True, range=(x_lognorm.min(), x_lognorm.max()))
    # shape, loc, scale = lognorm.fit(X1_lognorm)
    # std = lognorm.std(s=shape, loc=loc, scale=scale)
    # mean = lognorm.mean(s=shape, loc=loc, scale=scale)
    # pdf_data = lognorm.pdf(x=x_lognorm, s=shape, loc=loc, scale=scale)
    # ax[2].plot(x_lognorm, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ' $\sigma$=' + str(std), color='red')
    # ax[2].plot(x_lognorm, pdf_lognorm, label='$\mu$='+ str(mean_lognorm)+', $\sigma$=' + str(std_lognorm))

    # plot rayleigh distribution
    ax[2].hist(X_rayleigh, density=True, bins=1000)
    loc, scale = rayleigh.fit(X_rayleigh)
    std = rayleigh.std(loc=loc, scale=scale)
    # mean = rayleigh.mean(loc=loc, scale=scale)
    # std = np.std(X_rayleigh)
    mean = np.mean(X_rayleigh)

    pdf_data = rayleigh.pdf(x=x_rayleigh, loc=loc, scale=scale)
    ax[2].plot(x_rayleigh, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ' $\sigma$=' + str(scale), color='red')
    ax[2].plot(x_rayleigh, pdf_rayleigh, label='$\mu$='+str(mean_rayleigh)+', $\sigma$=' + str(std_rayleigh))

    # plot rician distribution
    # ax[3].hist(X_rice, density=True)
    # b, loc, scale = rice.fit(X_rice)
    # std = rice.std(b=b, loc=loc, scale=scale)
    # mean = rice.mean(b=b, loc=loc, scale=scale)
    # pdf_data = rice.pdf(x=x_rice, b=b, loc=loc, scale=scale)
    # ax[3].plot(x_rice, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(mean) + ' $\sigma$=' + str(std), color='red')
    # ax[3].plot(x_rice, pdf_rice, label='$\mu$=' +str(mean_rice)+' $\sigma$=' + str(std_rice))

    ax[0].set_ylabel('Prob. denisty \n unfiltered stand. Gauss')
    ax[1].set_ylabel('Prob. denisty \n filtered & norm. stand. Gauss')
    ax[2].set_ylabel('Prob. denisty \n Rayleigh')
    # ax[2].set_ylabel('Prob. denisty \n Lognormal')
    ax[3].set_ylabel('Prob. denisty \n Rice')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()

def test_noise_types():
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig_noise, ax_noise = plt.subplots(1, 1)
    from input import data_rate

    SNR_dict = dict()
    BER_dict = dict()
    noise_dict = dict()
    BER_coded_interleaved_dict = dict()
    errors_dict = dict()
    errors_coded_interleaved_dict = dict()
    axis = 0
    P_r = np.logspace(-(60+30)/10, -(0+30)/10, num=100, base=10)
    P_r_length = len(P_r)
    PPB = PPB_func(P_r, data_rate)
    detection_list = ['PIN', 'APD', 'quantum limit']
    modulation_list = ['OOK-NRZ', 'BPSK', '2-PPM']
    # ------------------------------------------------------------------------
    LCT = terminal_properties()
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
    ax_noise.set_title('Noise types vs. $P_{RX}$', fontsize=15)
    ax[0].set_title('SNR vs. $P_{RX}$', fontsize=15)
    ax[1].set_title('BER vs. $P_{RX}$', fontsize=15)
    ax_noise.plot(W2dBm(P_r), W2dB(noise_sh), label='shot noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_th), label='thermal noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_bg), label='background noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_beat), label='beat noise')

    for detection in detection_list:
        SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        ax[0].plot(W2dBm(P_r), W2dB(SNR), label=str(detection))

        for modulation in modulation_list:
            LCT.threshold(BER_thres=BER_thres,
                          modulation=modulation,
                          detection=detection)

            BER = LCT.BER_func(Q=Q, modulation=modulation)
            BER[BER < 1e-100] = 1e-100
            x_BER = np.linspace(-30.0, 0.0, 1000)
            BER_pdf = pdf_function(np.log10(BER), len(BER), x_BER)

            total_bits = LCT.data_rate * interval_channel_level         # Should be a scalar
            total_bits_per_sample = total_bits / samples_channel_level  # Should be a scalar
            errors = total_bits_per_sample * BER

            ax[1].plot(W2dBm(P_r), BER, label=str(detection) + ', ' + str(modulation))

    ax[1].set_yscale('log')
    ax[0].set_ylabel('Signal-to-Noise ratio \n [$i_r$ / $\sigma_n$] (dB)', fontsize=15)
    ax[1].set_ylabel('Error probability \n [# Error bits / total bits]', fontsize=15)
    ax[0].set_xlabel('Received power (dBm)', fontsize=15)
    ax[1].set_xlabel('Received power (dBm)', fontsize=15)
    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    ax[0].grid()
    ax[1].grid()

    ax_noise.set_ylabel('noise (dBm)', fontsize=15)
    ax_noise.set_xlabel('Received power (dBm)', fontsize=15)

    ax_noise.legend(fontsize=10)
    ax_noise.grid()

    plt.show()


def test_airy_disk():
    angle = np.linspace(1.0E-6, 20.0E-5, 100)
    I_norm_airy, I_norm_gaussian_approx = h_p_airy(angle, D_r, focal_length)

    r = focal_length * np.sin(angle)
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Spot profile, focal length='+str(focal_length)+'m, Dr='+str(np.round(D_r,3))+'m, Div. angle='+str(angle_div*1.0E6)+'$\mu$rad')
    ax.plot(r*1.0E6, I_norm_airy, label= 'Airy disk profile')
    ax.plot(r*1.0E6, I_norm_gaussian_approx, label='Gaussian approx. of Airy disk profile')
    ax.set_xlabel('r-position ($\mu$m) [r= focal length * sin(angle)]')
    ax.set_ylabel('Normalized intensity at r=0.0 $\mu$m')
    ax.legend()
    ax.grid()
    plt.show()

def test_mapping_dim1_dim2():
    elevation_angles = np.deg2rad(np.arange(0.0, 90.0, 0.1))
    time = np.linspace(0, 60 * 5, len(elevation_angles))

    data_dim1 = get_data('elevation')

    # data_dim1 = data_dim1[1:]
    a_s = (data_dim1[-1] - data_dim1[1]) / (len(data_dim1) - 1)
    a_0 = data_dim1[1]
    mapping_lb = (np.rad2deg(elevation_angles) - a_0).astype(int) + 1
    mapping_ub = mapping_lb + 1

    for i in range(len(elevation_angles)):
        if elevation_angles[i] < elevation_min:
            mapping_lb[i] = 0.0
            mapping_ub[i] = 0.0

    mapping_lb = list(mapping_lb)
    mapping_ub = list(mapping_ub)

    performance_data = get_data('all', mapping_lb, mapping_ub)
    # performance_data = get_data('all', mapping_lb)

    fig, ax = plt.subplots(1, 1)
    ax.plot(time, np.ones(len(time)) * np.rad2deg(elevation_min), label='Minimum elevation angle')
    ax.plot(time, np.rad2deg(elevation_angles), label='geometrical data')
    ax.plot(time, performance_data['elevation'], label='mapped data')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Elevation angle ($\degree$)')
    ax.legend()

    plt.show()

def test_interleaving():
    latency_interleaving = 5.0E-3
    interval = 1
    samples = 1000
    time = np.linspace(0, interval, samples)
    errors = np.vstack((np.random.rand(samples), np.random.rand(samples)))
    dt = interval / samples
    spread = int(np.round(latency_interleaving / dt,0) + 1)
    errors_per_sample = errors / spread

    errors_interleaved = np.empty(np.shape(errors))

    print(errors.sum())
    for i in range(0,spread):
        errors_interleaved += np.roll(errors_per_sample,i,axis=1)
    print(errors_interleaved.sum())

    fig, ax = plt.subplots(1, 1)
    ax.scatter(time, errors[0])
    ax.scatter(time, errors_interleaved[0])
    plt.show()

def simple_losses():
    Sn = noise_factor * M * h * v / 2
    Be = BW / 2
    m = 1
    R = eff_quantum * q / (h * v)
    BER_thres = 1.0E-9
    SNR_thres = (np.sqrt(2) * erfcinv(2 * BER_thres)) ** 2
    Q_thres = np.sqrt(SNR_thres)
    P_thres = Q_thres * Sn * 2 * Be / M * (Q_thres + np.sqrt(m/2 * (2*BW/(2*Be) - 1/2) + 2 * k * T_s / (R_L * 4 * Be * R**2 * Sn**2)))


    sig = 3.3E-6
    print(w0, sig)
    # Convert to rayleigh sigma
    sig_rayleigh = np.sqrt(2 / (4 - np.pi)) * sig
    beta = angle_div**2 / (4*sig**2)
    T_jit = beta / (beta + 1)
    # T_jit = angle_div**2 / (4*sig**2 + angle_div**2)
    PSI = 0.5

    a_scint = 4.343 * ( erfinv(2 * P_thres - 1) * (2 * np.log(PSI + 1))**(1/2) - 1/2 * np.log(PSI + 1))

def multiscaling():
    from channel_level import channel_level
    from bit_level import bit_level
    from Link_geometry import link_geometry
    from Routing_network import routing_network
    from Atmosphere import attenuation

    loop_through_range_of_stepsizes = 'no'
    probability_domain = 'yes'
    plot_index = 20

    if loop_through_range_of_stepsizes == 'yes':
        range_delta_t = np.arange(1.0, 10.0, 0.05)
        fig_T, ax_T = plt.subplots(4, 1)
        ax_T[0].set_title('Macroscale $\Delta$T determination')
        fractional_fade_time_list = []
        var_BER_list = []
        var_P_r_list = []
        mean_P_r_list = []

    else:
        range_delta_t = [5.0]
        fig_T, ax_T = plt.subplots(2, 2)
        ax_T[0,0].set_title('Probability domain analysis')

    for delta_t in range_delta_t:

        # Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
        link = link_geometry()
        link.propagate(stepsize_AC=delta_t, stepsize_SC=delta_t)
        link.geometrical_outputs()
        time = link.time
        network = routing_network(time)
        slew_rate = 1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth)
        geometrical_output, mask = network.routing(link.geometrical_output, time)
        time_hr = time[mask] / 3600.0
        # link.plot(type='satellite sequence', sequence=geometrical_output['pos SC'],
        #                    links=network.number_of_links)
        # plt.show()

        # Create slow variable: Attenuation (h_ext)
        att = attenuation()
        # samples = len(time)
        # duration = end_time - start_time
        # sampling_frequency = 1 / step_size_link
        # ext_frequency = 1/600 # 10 minute frequency of the transmission due to clouds
        # h_clouds = norm.rvs(scale=1, loc=0, size=samples)
        # order = 2
        # h_clouds = filtering(effect='extinction', order=order, data=h_clouds, f_cutoff_low=ext_frequency,
        #                     filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
        #
        att.h_ext_func(range_link=geometrical_output['ranges'], zenith_angles=geometrical_output['zenith'], method="standard_atmosphere")
        # att.h_clouds_func(data=h_clouds)
        # T_fs = (wavelength / (4 * np.pi * geometrical_output['ranges'])) ** 2

        # att.h_clouds = att.h_clouds[mask]
        # h_ext = att.h_ext #* att.h_clouds
        # fig, ax = plt.subplots(3,1)
        # ax[0].scatter(time_hr, W2dB(att.h_clouds), s=1,label='$h_{clouds}$')
        # ax[0].scatter(time_hr, W2dB(att.h_ext), s=1,label='$h_{ext}$')
        # ax[1].scatter(time_hr, W2dB(h_ext), s=1,label='$h_{ext}$')
        # ax[2].scatter(time_hr, W2dB(T_fs), s=1)
        #
        # ax[0].legend()
        # ax[1].legend()
        # ax[2].legend()
        # ax[0].grid()
        # ax[1].grid()
        # ax[2].grid()
        # plt.show()

        # x_ext = np.linspace(0.0, h_clouds.max(), 1000)
        # hist_ext = np.histogram(h_clouds, bins=1000)
        # rv = rv_histogram(hist_ext, density=True)
        # pdf_ext = rv.pdf(x_ext)
        # cdf_ext = rv.cdf(x_ext)

        # Create slow variable: Elevation (e) and Range (R)
        elevation = geometrical_output['elevation']
        ranges = geometrical_output['ranges']
        samples = len(elevation)
        bins = 100
        interval = len(time) * step_size_AC

        # CREATE DISTRIBUTION (HISTOGRAM) OF GEOMETRICAL DATA
        x_elev = np.linspace(elevation.min(), elevation.max(), bins)
        hist_e = np.histogram(elevation, bins=bins)
        rv   = rv_histogram(hist_e, density=False)
        pdf_elev  = rv.pdf(x_elev)
        cdf_elev  = rv.cdf(x_elev)
        hist_e_midpoints = hist_e[1][:-1] + np.diff(hist_e[1])/2
        elev_counts = hist_e[0]

        x_R = np.linspace(ranges.min(), ranges.max(), bins)
        hist_R = np.histogram(ranges, bins=bins)
        rv = rv_histogram(hist_R, density=False)
        pdf_R = rv.pdf(x_R)
        cdf_R = rv.cdf(x_R)
        hist_R_midpoints = hist_R[1][:-1] + np.diff(hist_R[1]) / 2
        R_counts = hist_R[0]
        T_fs = W2dB((wavelength / (4 * np.pi * hist_R_midpoints)) ** 2)

        # fig, ax = plt.subplots(2,2)
        # ax[0,0].set_title('bins: ' + str(bins) + '\n'
        #                   '$\Delta L$: '+ str(np.round(np.diff(hist_R_midpoints)[0]/1000,2)) +
        #                   'km, $T_{fs}$='+str(np.round(np.diff(T_fs)[0],2)))
        #
        # ax[0,1].set_title('Samples: '+str(samples))
        # ax[0,0].plot(x_R, pdf_R)
        # ax[0,0].plot(x_R, cdf_R)
        # ax[0,0].hist(ranges, density=True, bins=bins)
        # ax[0,0].set_ylabel('Prob. density')
        # ax[0,0].set_xlabel('Range (m)')
        # ax[0,1].scatter(time_hr, ranges/1000, s=0.5)
        # ax[0,1].set_ylabel('Range (km)')
        # ax[0,1].set_xlabel('Time (hours)')
        #
        # ax[1, 0].set_title('$\Delta \epsilon$: ' + str(np.round(np.diff(hist_e_midpoints)[0], 2)) + '$\degree$')
        # ax[1, 0].plot(x_elev, pdf_elev)
        # ax[1, 0].plot(x_elev, cdf_elev)
        # ax[1, 0].hist(elevation, density=True, bins=bins)
        # ax[1, 0].set_ylabel('Prob. density')
        # ax[1, 0].set_xlabel('Elevation (rad)')
        # ax[1, 1].scatter(time_hr, elevation, s=0.5)
        # ax[1, 1].set_ylabel('Elevation (degrees)')
        # ax[1, 1].set_xlabel('Time (hours)')

        # ax[0, 0].grid()
        # ax[0, 1].grid()
        # ax[1, 0].grid()
        # ax[1, 1].grid()
        # plt.show()

        # # Probability domain
        # elevation = hist_e_midpoints
        # ranges = hist_R_midpoints
        # zenith = np.pi/2 - elevation

        # ------------------------------------------------------------------------
        # -----------------------------------LCT----------------------------------
        # ------------------------------------------------------------------------
        LCT = terminal_properties()
        LCT.threshold(BER_thres=BER_thres,
                      modulation="OOK-NRZ",
                      detection="APD")
        # ------------------------------------------------------------------------
        # -------------------------------TURBULENCE-------------------------------
        # ------------------------------------------------------------------------

        # The turbulence class is initiated here. Inside the turbulence class, there are multiple functions that are run.
        turb = turbulence(ranges=geometrical_output['ranges'], link=link)
        turb.windspeed_func(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
        turb.Cn_func(turbulence_model=turbulence_model)
        r0 = turb.r0_func(zenith_angles=geometrical_output['zenith'])
        turb.var_rytov_func(zenith_angles=geometrical_output['zenith'])
        turb.var_scint_func(zenith_angles=geometrical_output['zenith'])
        turb.Strehl_ratio_func(tip_tilt="YES")
        turb.beam_spread()
        turb.var_bw_func(zenith_angles=geometrical_output['zenith'])
        turb.var_aoa_func(zenith_angles=geometrical_output['zenith'])
        turb.print(index=plot_index, elevation=np.rad2deg(geometrical_output['elevation']), ranges=geometrical_output['ranges'])
        # ------------------------------------------------------------------------
        # -------------------------------LINK-BUDGET------------------------------
        # ------------------------------------------------------------------------
        # The link budget class is initiated here.
        link = link_budget(ranges=ranges, h_strehl=turb.h_strehl, w_ST=turb.w_ST,
                           h_beamspread=turb.h_beamspread, h_ext=att.h_ext)
        # The power at the receiver is computed from the link budget.
        P_r_0 = link.P_r_0_func()
        # ------------------------------------------------------------------------
        # ------------------------SNR--BER--COARSE-SOLVER-------------------------
        # ------------------------------------------------------------------------
        noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun)
        SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)
        margin_0 = LCT.P_r_thres[0] / P_r_0
        errors_0 = BER_0 * data_rate

        # ------------------------------------------------------------------------
        # --------------------------SNR--BER--FINE-SOLVER-------------------------
        # ------------------------------------------------------------------------

        P_r, PPB, elevation_angles, pdf_h_tot, pdf_P_r, h_tot, h_scint, h_RX, h_TX = \
            channel_level(plot_index=int(plot_index),
                          LCT=LCT,
                          turb=turb,
                          P_r_0=P_r_0,
                          ranges=geometrical_output['ranges'],
                          elevation_angles=geometrical_output['elevation'],
                          zenith_angles=geometrical_output['zenith'],
                          samples=samples_channel_level)

        # P_r, BER, pdf_BER, errors, margin = \
        P_r, BER, pdf_BER, BER_coded, pdf_BER_coded, errors, errors_coded, margin = \
            bit_level(LCT=LCT,
                      link_budget=link,
                      plot_index=int(plot_index),
                      divide_index=N,
                      samples=samples_channel_level,
                      P_r_0=P_r_0,
                      pdf_P_r=pdf_P_r,
                      P_r=P_r,
                      PPB=PPB,
                      elevation_angles=geometrical_output['elevation'],
                      pdf_h_tot=pdf_h_tot,
                      h_tot=h_tot,
                      h_scint=h_scint,
                      h_RX=h_RX,
                      h_TX=h_TX,
                      coding='yes')

        P_r_total = P_r.flatten()
        x_P_r_total = np.linspace(W2dBm(P_r_total.min()), W2dBm(P_r_total.max()), 10000)
        P_r_pdf_total = pdf_function(data=W2dBm(P_r_total), length=1, x=x_P_r_total)
        BER_total = BER.flatten()
        x_BER_total = np.linspace(-30.0, 0.0, 10000)
        BER_pdf_total = pdf_function(data=np.log10(BER_total), length=1, x=x_BER_total)
        BER_coded_total = BER_coded.flatten()
        BER_coded_total_pdf = pdf_function(data=np.log10(BER_coded_total), length=1, x=x_BER_total)
        errors_total = errors.flatten()
        errors_coded_total = errors_coded.flatten()

        if loop_through_range_of_stepsizes == 'yes':
            mean_P_r = P_r.mean()
            mean_P_r_dB = W2dB(mean_P_r)
            var_P_r = P_r.var()
            var_P_r_dB = W2dB(var_P_r)
            # var_BER = BER.var()
            fractional_fade_time = np.count_nonzero((P_r_total < LCT.P_r_thres[1]), axis=0) / len(P_r_total)
            mean_P_r_list.append(mean_P_r_dB)
            var_P_r_list.append(var_P_r_dB)
            # var_BER_list.append(var_BER)
            fractional_fade_time_list.append(fractional_fade_time)
            if list(range_delta_t).index(delta_t) == 0:
                ax_T[0].plot(W2dBm(x_P_r_total), P_r_pdf_total, label='True distribution= $\mu$='+ str(
                    np.round(mean_P_r_dB, 3)) + 'dB' + ', $\sigma^2$=' + str(np.round(var_P_r_dB, 3)) + 'dB')
            if list(range_delta_t).index(delta_t) == 19 or list(range_delta_t).index(delta_t) == 79 or list(range_delta_t).index(delta_t) == -19:
                ax_T[0].plot(W2dBm(x_P_r_total), P_r_pdf_total, label='$\Delta$T=' + str(np.round(delta_t,2)) + ', $\mu$=' + str(np.round(mean_P_r_dB, 3)) + 'dB'+ ', $\sigma^2$=' + str(np.round(var_P_r_dB, 3)) + 'dB')


        elif probability_domain == 'yes':
            time_cross_section = [0.52, 0.545]
            indices = []
            for t in time_cross_section:
                t = t * 3600
                index = np.argmin(abs(time[mask] - t))
                indices.append(index)

            fractional_fade_time_total = np.count_nonzero((P_r_total < LCT.P_r_thres[1]), axis=0) / len(P_r_total)
            fractional_BER_fade_total = np.count_nonzero((BER_total > BER_thres[1]), axis=0) / len(BER_total)
            fractional_BER_coded_fade_total = np.count_nonzero((BER_coded_total > BER_thres[1]), axis=0) / len(BER_coded_total)
            for p in range(len(P_r_0)):
                if p == 3 or p == 20:
                    fractional_fade_time = np.count_nonzero((P_r[p] < LCT.P_r_thres[1]), axis=0) / samples_channel_level
                    fractional_BER_fade = np.count_nonzero((BER[p] > BER_thres[1]), axis=0) / samples_channel_level
                    fractional_BER_coded_fade = np.count_nonzero((BER_coded[p] > BER_thres[1]), axis=0) / samples_channel_level
                    ax_T[1,0].plot(pdf_P_r[1], pdf_P_r[0][p], label='$\epsilon$='+str(np.round(np.rad2deg(geometrical_output['elevation'][p]),2))+'\n'
                                    'Frac. fade time='+str(np.round(fractional_fade_time,2)))
                    ax_T[1,1].plot(pdf_BER[1], pdf_BER[0][p], label='Uncoded, $\epsilon$='+str(np.round(np.rad2deg(geometrical_output['elevation'][p]),2))+'\n'
                                    '% BER over threshold: '+str(np.round(fractional_BER_fade*100,2)))
                    # ax_T[1,1].plot(pdf_BER_coded[1], pdf_BER_coded[0][p], label='Coded, $\epsilon$='+str(np.round(np.rad2deg(geometrical_output['elevation'][p]),2))+'\n'
                    #                 '% BER over threshold: '+str(np.round(fractional_BER_coded_fade*100,2)))

            ax_T[0,0].plot(x_P_r_total, P_r_pdf_total, label='One distribution, $\mu$=' + str(np.round(W2dBm(P_r_total.mean()),2)) +
                                                                    'dB' + ', $\sigma^2$=' + str(np.round(W2dBm(P_r_total.var()),2)) + 'dB')
            ax_T[0,0].plot(np.ones(2) * W2dBm(P_r.mean()), [P_r_pdf_total.min(), P_r_pdf_total.max()], color='grey')
            ax_T[0,0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [P_r_pdf_total.min(), P_r_pdf_total.max()], c='black',
                         linewidth=3, label='treshold BER=1.0E-6')
            ax_T[0,0].set_ylabel('PDF of all dist combined')

            ax_T[0,1].plot(x_BER_total, BER_pdf_total, label='Uncoded, $\mu$=1E' + str(np.round(np.log10(BER_total.mean()),2)) +
                                                             ', $\sigma^2$=1E' + str(np.round(np.log10(BER_total.var()),2)) + '\n'
                                  '% BER over threshold: '+str(np.round(fractional_BER_fade_total*100,2)))
            # ax_T[0,1].plot(x_BER_total, BER_coded_total_pdf,
            #                 label='Coded, $\mu$=1E' + str(np.round(np.log10(BER_coded_total.mean()), 2)) +
            #                       ', $\sigma^2$=1E' + str(np.round(np.log10(BER_coded_total.var()), 2)) +'\n'
            #                       '% BER over threshold: '+str(np.round(fractional_BER_coded_fade_total*100,2)))
            ax_T[0,1].plot(np.ones(2) * np.log10(BER_thres[1]), [BER_pdf_total.min(), BER_pdf_total.max()], c='black',
                         linewidth=3, label='treshold BER=1.0E-6')
            ax_T[0,1].set_ylabel('PDF of all dist combined')


            ax_T[1,0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [(pdf_P_r[0][-1]).min(), (pdf_P_r[0][-1]).max()], c='black',
                         linewidth=3, label='treshold BER=1.0E-6')
            ax_T[1,0].set_ylabel('PDF of each Pr dist')
            ax_T[1,0].set_xlabel('Pr (dBm)')

            ax_T[1,1].plot(np.ones(2) * np.log10(BER_thres[1]), [(pdf_BER[0][20]).min(), (pdf_BER[0][20]).max()], c='black',
                         linewidth=3, label='treshold BER=1.0E-6')
            ax_T[1,1].set_ylabel('PDF of each BER dist')
            ax_T[1,1].set_xlabel('BER')
            # ax_T[0,1].set_xscale('log')
            # ax_T[1,1].set_xscale('log')

            ax_T[0, 0].legend()
            ax_T[0, 1].legend()
            ax_T[1, 0].legend()
            ax_T[1, 1].legend()
            plt.show()



    if loop_through_range_of_stepsizes == 'yes':
        # ax_T[0].plot(np.ones(2) * BER_thres[1], [BER_total_pdf.min(), BER_total_pdf.max()], c='black', linewidth=3)
        ax_T[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [P_r_pdf_total.min(), P_r_pdf_total.max()], c='black', linewidth=3)
        # ax_T[0].set_ylabel('PDF')
        # ax_T[0].set_xlabel('BER')
        # ax_T[0].set_xscale('log')
        ax_T[0].set_ylabel('PDF')
        ax_T[0].set_xlabel('Pr (dBm)')

        ax_T[1].set_ylabel('Fractional fade time \n for $BER_{thres}$=1E-6')
        ax_T[1].plot([range_delta_t[0], range_delta_t[-1]], fractional_fade_time_list[0] * 0.9 * np.ones(2), label='-10% diff from true', color='black')
        ax_T[1].plot(range_delta_t, fractional_fade_time_list)
        ax_T[1].plot([range_delta_t[0], range_delta_t[-1]], fractional_fade_time_list[0] * 1.1 * np.ones(2), label='+10% diff from true', color='black')

        ax_T[2].set_ylabel('Sample variance of $P_r$ (dB)')
        ax_T[2].plot(range_delta_t, var_P_r_list)
        ax_T[2].plot([range_delta_t[0], range_delta_t[-1]], (var_P_r_list[0] - 0.1) * np.ones(2), label='-0.1 dB diff from true', color='black')
        ax_T[2].plot([range_delta_t[0], range_delta_t[-1]], (var_P_r_list[0] + 0.1) * np.ones(2), label='+0.1 dB diff from true', color='grey')
        ax_T[3].set_ylabel('Sample mean  of $P_r$ (dB)')
        ax_T[3].set_xlabel('Macroscale $\Delta$T (sec)')
        ax_T[3].plot(range_delta_t, mean_P_r_list)
        ax_T[3].plot([range_delta_t[0], range_delta_t[-1]], (mean_P_r_list[0] - 0.1) * np.ones(2), label='-0.1 dB diff from true', color='black')
        ax_T[3].plot([range_delta_t[0], range_delta_t[-1]], (mean_P_r_list[0] + 0.1) * np.ones(2), label='+0.1 dB diff from true', color='grey')

        ax_T[0].legend(loc='upper right')
        ax_T[0].grid()
        ax_T[1].legend(loc='upper right')
        ax_T[1].grid()
        ax_T[2].legend(loc='upper right')
        ax_T[2].grid()
        ax_T[3].legend(loc='upper right')
        ax_T[3].grid()
        plt.show()

    elif probability_domain == 'yes':
        # # USING PROBABILITY DOMAIN
        # total_errors = errors * (step_size_link / interval_channel_level) * elev_counts
        # total_bits = data_rate * step_size_AC * elev_counts
        # throughput = total_bits - total_errors
        #
        # fig, ax = plt.subplots(3, 1)
        # ax[0].plot(np.rad2deg(geometrical_output['elevation']), W2dBm(P_r.mean(axis=1)), label='Pr (average from bit level)')
        # ax[0].plot(np.rad2deg(geometrical_output['elevation']), W2dBm(np.ones(len(P_r_0)) * LCT.P_r_thres[0]), label='Sensitivty comm. (BER=1.0E-9)')
        # ax[0].plot(np.rad2deg(geometrical_output['elevation']), W2dBm(np.ones(len(P_r_0)) * LCT.P_r_thres[1]), label='Sensitivty comm. (BER=1.0E-6)')
        # ax[0].plot(np.rad2deg(geometrical_output['elevation']), W2dBm(np.ones(len(P_r_0)) * LCT.P_r_thres[2]), label='Sensitivty comm. (BER=1.0E-3)')
        # ax[0].set_ylabel('Comm \n Power (dBm)')
        #
        # ax[1].plot(geometrical_output['elevation'], total_bits / 1.0E6, label='Total bits=' + str(total_bits.sum() / 1.0E12) + 'Tb')
        # ax[1].plot(geometrical_output['elevation'], total_errors / 1.0E6, label='Total erroneous bits=' + str(total_errors.sum() / 1.0E9) + 'Gb')
        # ax[1].plot(geometrical_output['elevation'], throughput / 1.0E6, label='Throughput=' + str(throughput.sum() / 1.0E12) + 'Tb')
        # # ax[2].plot(time_hr, performance_output['total_errors_coded'] / 1.0E6,  label='RS coded (' + str(N) + ',' + str(K) + '), interleaving=' + str(
        # #                     latency_interleaving) + 's, \n total throughput=' + str(performance_output['throughput_coded'].sum() / 1.0E12) + 'Tb')
        # ax[1].set_ylabel('Transferred bits \n per $\Delta t_{mission}$ (Mb/s)')
        # ax[1].set_yscale('log')
        # ax[1].set_ylim(total_errors.min() / 1.0E6, total_errors.max() / 1.0E6)
        #
        # ax[0].legend()
        # ax[1].legend()
        # plt.show()


        # USING TIME DOMAIN
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(time_hr, W2dBm(P_r.mean(axis=1)), label='Pr (average from bit level)')
        ax[0].plot(time_hr, W2dBm(P_r_0), label='Pr0')
        ax[0].set_ylabel('Comm \n Power (dBm)')

        ax[1].plot(time_hr, BER_0, label='BER from Pr0')
        ax[1].plot(time_hr, BER.mean(axis=1), label='BER mean from bit level')
        ax[1].set_ylabel('BER')
        ax[1].set_yscale('log')

        ax[0].legend()
        ax[1].legend()
        ax[0].grid()
        ax[1].grid()
        plt.show()


def channel_level_verification():
    from channel_level import channel_level

    LCT = terminal_properties()
    LCT.threshold(BER_thres=BER_thres, modulation="OOK-NRZ", detection="APD")

    h_AC = 10.0E3
    h_SC = 600.0E3
    P_r_0 = np.array([dBm2W(-25)])
    elevation = np.array([np.deg2rad(60.0)])
    zenith = np.pi/2 - elevation
    ranges = (h_SC - h_AC) / np.tan(elevation)
    V_ac = 150.0
    slew_rate = 1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth)
    plot_index = 0

    turb = turbulence(ranges=ranges, link=link)
    turb.windspeed_func(slew=slew_rate, Vg=V_ac, wind_model_type=wind_model_type)
    turb.Cn_func(turbulence_model=turbulence_model)
    r0 = turb.r0_func(zenith_angles=zenith)
    turb.var_rytov_func(zenith_angles=zenith)
    turb.var_scint_func(zenith_angles=zenith)
    turb.Strehl_ratio_func(tip_tilt="YES")
    turb.beam_spread()
    turb.var_bw_func(zenith_angles=zenith)
    turb.var_aoa_func(zenith_angles=zenith)
    turb.print(index=plot_index, elevation=np.rad2deg(elevation), ranges=ranges)

    interval_channel_level = 500.0
    t = np.arange(0.0, interval_channel_level, step_size_channel_level)
    samples_channel_level = int(len(t))
    P_r, PPB, elevation_angles, pdf_h_tot, pdf_P_r, h_tot, turb.h_scint, h_RX, h_TX = channel_level(
        plot_index=plot_index, LCT=LCT, turb=turb, P_r_0=P_r_0, ranges=ranges, elevation_angles=elevation,
        zenith_angles=zenith, samples=samples_channel_level)
    mean_0 = P_r.mean()
    mean_0_dB = W2dB(mean_0)
    var_0 = P_r.var()
    var_0_dB = W2dB(var_0)
    threshold_condition = pdf_P_r[1] < LCT.P_r_thres[1]
    fractional_fade_time_0 = np.trapz((pdf_P_r[0])[threshold_condition], x=(pdf_P_r[1])[threshold_condition])

    # TURBULENCE DISTRIBUTION FOR 1 MILLION SAMPLES
    fig_T, ax_T = plt.subplots(4,1)
    ax_T[0].set_title('Channel sample size determination \n'
                      '($\epsilon$='+str(np.round(np.rad2deg(elevation[0]),2))+', $\sigma_{Rytov}^2$='+str(np.round(turb.var_rytov[0],3))+')')
    ax_T[0].set_ylabel('PDF')
    ax_T[0].set_xlabel('Pr (dBm)')

    interval_channel_level = np.arange(0.1,50.0,0.1)
    fractional_fade_time_list = []
    number_of_fades_list = []
    mean_fade_time_list = []
    var_list = []
    mean_list = []
    for interval in interval_channel_level:
        print(interval, list(interval_channel_level).index(interval))
        t = np.arange(0.0, interval, step_size_channel_level)
        v = int(len(t)) - 1
        P_r, PPB, elevation_angles, pdf_h_tot, pdf_P_r, h_tot, turb.h_scint, h_RX, h_TX = channel_level(
            plot_index=plot_index, LCT=LCT, turb=turb, P_r_0=P_r_0, ranges=ranges, elevation_angles=elevation,
            zenith_angles=zenith, samples=v)
        x = pdf_P_r[1]
        pdf_T = gamma((v+1)/2) / (np.sqrt(v*np.pi)*(v/2)) * (1+x**2/v)**(-(v+1)/2)
        # ax_T.plot(W2dBm(x), pdf_T, label='v='+str(int(v/100))+'E3 samples')
        mean = P_r.mean()
        mean_dB = W2dB(mean)
        var = P_r.var()
        var_dB = W2dB(var)

        number_of_fades = np.count_nonzero((P_r < LCT.P_r_thres[1]))
        threshold_condition = pdf_P_r[1] < LCT.P_r_thres[1]
        fractional_fade_time = np.trapz((pdf_P_r[0])[threshold_condition], x=(pdf_P_r[1])[threshold_condition])
        mean_fade_time = fractional_fade_time / number_of_fades

        var_list.append(var_dB)
        mean_list.append(mean_dB)
        fractional_fade_time_list.append(fractional_fade_time)
        number_of_fades_list.append(number_of_fades)
        mean_fade_time_list.append(mean_fade_time)

        if list(interval_channel_level).index(interval) == 99 or list(interval_channel_level).index(interval) == 299 or list(interval_channel_level) == 499:
            print('plot pdf')
        # if abs(interval - 5)<0.05 or abs(interval - 15)<0.05 or abs(interval - 25)<0.05 or abs(interval - 50)<0.05:
            ax_T[0].plot(W2dBm(x), pdf_P_r[0],
                         label='v='+str(int(v/1E3))+'E3, $\mu$=' + str(
                     np.round(W2dB(mean), 3)) + 'dB, $\sigma^2$=' + str(np.round(W2dB(var), 3)) + 'dB')

    ax_T[0].plot(W2dBm(pdf_P_r[1]), pdf_P_r[0],
                 label='True, $\mu$=' + str( np.round(W2dB(mean_0), 3)) + 'dB, $\sigma^2$=' + str(np.round(W2dB(var_0), 3)) + 'dB')
    ax_T[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [(pdf_P_r[0]).min(), (pdf_P_r[0]).max()],
                 c='black', linewidth=3, label='Treshold BER=1E-6')
    sample_list = interval_channel_level / step_size_channel_level
    ax_T[1].set_ylabel('Fractional fade time \n for $BER_{thres}$=1E-6')
    ax_T[1].plot(sample_list/1E3, fractional_fade_time_list, linewidth=0.5)
    ax_T[1].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], fractional_fade_time_0 * np.ones(2), label='True fractional fade time')
    ax_T[1].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], fractional_fade_time_0 * 0.90 * np.ones(2), label='-10% diff from true', color='black')
    ax_T[1].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], fractional_fade_time_0 * 1.10 * np.ones(2), label='+10% diff from true', color='grey')

    ax_T[2].set_ylabel('Sample variance (dB)')
    ax_T[2].plot(sample_list/1E3, var_list)
    ax_T[2].plot([(sample_list[0])/1E3, (sample_list[-1])/1E3], var_0_dB * np.ones(2), label='True variance')
    ax_T[2].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], (var_0_dB - 0.1) * np.ones(2), label='-0.1 dB diff from true', color='black')
    ax_T[2].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], (var_0_dB + 0.1) * np.ones(2), label='+0.1 dB diff from true', color='grey')

    ax_T[3].set_ylabel('Sample mean (dB)')
    ax_T[3].set_xlabel('sample size of simulation (x1E3)')
    ax_T[3].plot(sample_list/1E3, mean_list)
    ax_T[3].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], mean_0_dB * np.ones(2), label='True mean')
    ax_T[3].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], (mean_0_dB - 0.1) * np.ones(2), label='-0.1 dB diff from true', color='black')
    ax_T[3].plot([(sample_list[0]) / 1E3, (sample_list[-1]) / 1E3], (mean_0_dB + 0.1) * np.ones(2), label='+0.1 dB diff from true', color='grey')

    ax_T[0].legend(loc='upper right')
    ax_T[0].grid()
    ax_T[1].legend(loc='upper right')
    ax_T[1].grid()
    ax_T[2].legend(loc='upper right')
    ax_T[2].grid()
    ax_T[3].legend(loc='upper right')
    ax_T[3].grid()
    plt.show()



    number_of_fades = np.count_nonzero((P_r < LCT.P_r_thres[1]))
    threshold_condition = pdf_P_r[1] < LCT.P_r_thres[1]
    fractional_fade_time = np.trapz((pdf_P_r[0][0])[threshold_condition], x=pdf_P_r[1][threshold_condition])
    mean_fade_time = fractional_fade_time / number_of_fades



    fig, ax = plt.subplots(2,1)
    ax[0].plot(t[10:], W2dBm(P_r[plot_index, 10:]),
               label='Dynamic power $P_{r}$, mean: ' + str(np.round(W2dBm(np.mean(P_r[plot_index])), 2)) + ' dBm')
    ax[0].plot(t[10:], (np.ones(len(t)) * W2dBm(P_r_0[plot_index]))[10:],
               label='Static power $P_{r0}$, mean: ' + str(np.round(W2dBm(np.mean(P_r_0[plot_index])), 2)) + ' dBm',
               linewidth=2.0)
    ax[0].plot(t[10:], (np.ones(len(t)) * W2dBm(LCT.P_r_thres[1]))[10:], label='BER: 1E-6')

    ax[1].plot(W2dBm(pdf_P_r[1]), pdf_P_r[0])
    ax[1].plot(W2dBm(np.ones(2) * LCT.P_r_thres[1]), [(pdf_P_r[0]).min(), (pdf_P_r[0]).max()])


    ax[0].set_ylabel('P at RX [dBm]')
    ax[0].set_xlabel('Time [s]')
    ax[0].legend()
    plt.show()


# test_constellation(constellation)
# test_filter()
# test_RS_coding('simple')
# test_sensitivity(method='Gallion')
# test_windspeed()
# test_beamwander()
# test_jitter()
# test_PDF()
# test_airy_disk()
# test_noise_types()
# test_mapping_dim1_dim2()
# simple_losses()
# test_interleaving()

multiscaling()
# channel_level_verification()