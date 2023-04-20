import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim, lsim2, bessel, butter, filtfilt, lfilter, lfilter_zi
from scipy.fft import fft, rfft, ifft, fftfreq, rfftfreq
from scipy.special import binom, i0, i1, erfc, erfinv, erfcinv
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
    from Routing_network import network
    from Atmosphere import attenuation
    # Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
    link = link_geometry()
    link.propagate()
    link.geometrical_outputs()
    time = link.time
    network = network(time)
    slew_rate = 1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth)
    geometrical_output, mask = network.hand_over_strategy(number_of_planes,
                                                        number_sats_per_plane,
                                                        link.geometrical_output,
                                                        time)

    time_hr = time[mask] / 3600.0
    # link.plot(type='angles')
    # link.plot(type='satellite sequence', sequence=geometrical_output['pos SC'],
    #                    handovers=network.number_of_handovers)

    # Create slow variable: Attenuation (h_ext)
    att = attenuation()
    samples = len(time)
    duration = end_time - start_time
    sampling_frequency = 1 / step_size_link
    ext_frequency = 1/600 # 10 minute frequency of the transmission due to clouds
    h_clouds = norm.rvs(scale=1, loc=0, size=samples)
    order = 2
    h_clouds = filtering(effect='extinction', order=order, data=h_clouds, f_cutoff_low=ext_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')

    att.h_ext_func(range_link=geometrical_output['ranges'], zenith_angles=geometrical_output['zenith'], method="standard_atmosphere")
    att.h_clouds_func(data=h_clouds)
    T_fs = (wavelength / (4 * np.pi * geometrical_output['ranges'])) ** 2

    # att.h_ext = att.h_ext[mask]
    att.h_clouds = att.h_clouds[mask]
    h_ext = att.h_ext #* att.h_clouds
    fig, ax = plt.subplots(3,1)
    ax[0].scatter(time_hr, W2dB(att.h_clouds), s=1,label='$h_{clouds}$')
    ax[0].scatter(time_hr, W2dB(att.h_ext), s=1,label='$h_{ext}$')
    ax[1].scatter(time_hr, W2dB(h_ext), s=1,label='$h_{ext}$')
    ax[2].scatter(time_hr, W2dB(T_fs), s=1)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    plt.show()

    x_ext = np.linspace(0.0, h_clouds.max(), 1000)
    hist_ext = np.histogram(h_clouds, bins=1000)
    rv = rv_histogram(hist_ext, density=True)
    pdf_ext = rv.pdf(x_ext)
    cdf_ext = rv.cdf(x_ext)

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
    #
    # ax[0, 0].grid()
    # ax[0, 1].grid()
    # ax[1, 0].grid()
    # ax[1, 1].grid()
    # plt.show()

    # # Probability domain
    # elevation = hist_e_midpoints
    # ranges = hist_R_midpoints
    # zenith = np.pi/2 - elevation

    # Time domain
    N = 10
    plot_index = 10
    elevation = geometrical_output['elevation']
    ranges = geometrical_output['ranges']
    zenith = geometrical_output['zenith']

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
    turb = turbulence(ranges=ranges, link=link)
    turb.windspeed_func(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
    turb.Cn_func(turbulence_model=turbulence_model)
    r0 = turb.r0_func(zenith_angles=zenith)
    turb.var_rytov_func(zenith_angles=zenith)
    turb.var_scint_func(zenith_angles=zenith)
    turb.Strehl_ratio_func(tip_tilt="YES")
    turb.beam_spread()
    turb.var_bw_func(zenith_angles=zenith)
    turb.var_AoA(zenith_angles=zenith)
    turb.print(index=plot_index, elevation=np.rad2deg(elevation[::N]), ranges=ranges[::N])
    # ------------------------------------------------------------------------
    # -------------------------------LINK-BUDGET------------------------------
    # ------------------------------------------------------------------------
    # The link budget class is initiated here.
    link = link_budget(ranges=ranges, h_strehl=turb.h_strehl, w_ST=turb.w_ST,
                       h_beamspread=turb.h_beamspread, h_ext=h_ext)
    # The power at the receiver is computed from the link budget.
    P_r_0 = link.P_r_0_func()
    # ------------------------------------------------------------------------
    # ------------------------SNR--BER--COARSE-SOLVER-------------------------
    # ------------------------------------------------------------------------
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun)
    SNR, Q = LCT.SNR_func(P_r=P_r_0, detection=detection,
                               noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation)
    margin = LCT.P_r_thres[0] / P_r_0
    errors = BER * data_rate

    # ------------------------------------------------------------------------
    # --------------------------SNR--BER--FINE-SOLVER-------------------------
    # ------------------------------------------------------------------------
    P_r_0_fine = P_r_0[::N]
    ranges_fine = geometrical_output['ranges'][::N]
    elevation_fine = geometrical_output['elevation'][::N]

    P_r, PPB,elevation_angles, pdf_h_tot, h_tot, h_scint, h_RX, h_TX = \
        channel_level(plot_index=plot_index, LCT=LCT, turb=turb, P_r_0=P_r_0_fine, ranges=ranges_fine,
                      slew_rate=slew_rate, elevation_angles=elevation_fine)
    P_r_mean, BER_mean, total_errors, margin = \
        bit_level(LCT=LCT, N=N, link_budget=link, plot_index=plot_index, P_r_0=P_r_0_fine, P_r=P_r, PPB=PPB,
                  elevation_angles=elevation_fine, pdf_h_tot=pdf_h_tot, h_tot=h_tot, h_scint=h_scint,
                  h_RX=h_RX, h_TX=h_TX)

    # # USING PROBABILITY DOMAIN
    # total_errors = total_errors * (step_size_AC / interval_dim1) * elev_counts
    # total_bits = data_rate * step_size_AC * elev_counts
    # throughput = total_bits - total_errors

    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(np.rad2deg(elevation), W2dBm(P_r_mean), label='Pr (average from bit level)')
    # ax[0].plot(np.rad2deg(elevation), W2dBm(np.ones(len(elevation)) * LCT.P_r_thres[0]), label='Sensitivty comm. (BER=1.0E-9)')
    # ax[0].plot(np.rad2deg(elevation), W2dBm(np.ones(len(elevation)) * LCT.P_r_thres[1]), label='Sensitivty comm. (BER=1.0E-6)')
    # ax[0].plot(np.rad2deg(elevation), W2dBm(np.ones(len(elevation)) * LCT.P_r_thres[2]), label='Sensitivty comm. (BER=1.0E-3)')
    # ax[0].set_ylabel('Comm \n Power (dBm)')
    #
    # ax[1].plot(elevation, total_bits / 1.0E6, label='Total bits=' + str(total_bits.sum() / 1.0E12) + 'Tb')
    # ax[1].plot(elevation, total_errors / 1.0E6, label='Total erroneous bits=' + str(total_errors.sum() / 1.0E9) + 'Gb')
    # ax[1].plot(elevation, throughput / 1.0E6, label='Throughput=' + str(throughput.sum() / 1.0E12) + 'Tb')
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
    ax[0].plot(time[mask], W2dBm(P_r_mean), label='Pr (average from bit level)')
    ax[0].plot(time[mask], W2dBm(P_r_0), label='Pr0')
    ax[0].set_ylabel('Comm \n Power (dBm)')

    ax[1].plot(time[mask], BER, label='BER from Pr0')
    ax[1].plot(time[mask], BER_mean, label='BER mean from bit level')
    ax[1].set_ylabel('BER')
    ax[1].set_yscale('log')

    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    plt.show()

def compare_backward_threshold_function_with_forward_BER_function():
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

    fig, ax = plt.subplots(1,1)
    ax.plot(W2dBm(P_r), BER)
    ax.plot(W2dBm(LCT.P_r_thres * np.ones(2)), [BER.min(), BER.max()], label='Pr threshold')
    ax.plot([W2dBm(P_r.min()), W2dBm(P_r.max())], BER_thres * np.ones(2), label='BER threshold')

    ax.set_ylabel('BER')
    ax.set_yscale('log')
    ax.set_xlabel('Pr')
    ax.grid()
    ax.legend()
    plt.show()

def turbulence_interval():
    from channel_level import channel_level

    LCT = terminal_properties()

    P_r_0 = dBm2W(-25)
    ranges = 600.0
    elevation = np.deg2rad(30.0)
    zenith = np.pi/2 - elevation
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
    turb.print(index=plot_index, elevation=zenith, ranges=ranges)

    P_r, PPB, elevation_angles, pdf_h_tot, h_tot, turb.h_scint, h_RX, h_TX = channel_level(
        plot_index=plot_index, LCT=LCT, turb=turb, P_r_0=P_r_0, ranges=ranges, elevation_angles=elevation, zenith_angles=zenith)


# test_constellation(constellation)
# test_filter()
# test_RS_coding('simple')
# test_sensitivity(method='Gallion')
compare_backward_threshold_function_with_forward_BER_function()
# test_windspeed()
# test_beamwander()
# test_jitter()
# test_PDF()
# test_airy_disk()
# test_mapping_dim1_dim2()
# simple_losses()
# test_interleaving()

# multiscaling()