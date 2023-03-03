import numpy as np
from scipy.signal import lsim, lsim2, bessel, butter, filtfilt, lfilter, lfilter_zi
from scipy.fft import fft, rfft, ifft, fftfreq, rfftfreq
from scipy.special import binom, i0, i1
from scipy.stats import norm, lognorm, rayleigh, rice

from input import *
from helper_functions import *
from Constellation import constellation
from LCT import terminal_properties
from PDF import dist
from Atmosphere import turbulence

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
    modulation = "2-PPM"
    detection = "APD"
    data_rate = 10.0E9

    LCT = terminal_properties()

    P_r = np.linspace(dBm2W(-50), dBm2W(-20), 1000)
    N_p = Np_func(P_r, data_rate)
    N_p = W2dB(N_p)

    noise_sh = LCT.noise(noise_type="shot", P_r=P_r)
    noise_th = LCT.noise(noise_type="thermal")
    noise_bg = LCT.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
    noise_beat = LCT.noise(noise_type="beat")
    LCT.threshold(modulation=modulation, detection=detection)

    SNR = LCT.SNR_func(P_r=P_r, detection=detection)
    BER = LCT.BER_func()

    if method == "BER/PPB":
        plt.plot(N_p, BER)
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

        LCT.PPB_func(P_r=P_r, data_rate=data_rate)
        LCT.SNR_func(P_r=P_r, detection='APD')
        LCT.BER_func()

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
    L = 500E3
    w_r = beam_spread(w0, L)
    X1 = np.random.standard_normal(size=samples)
    X2 = np.random.standard_normal(size=samples)

    X1_norm = filtering(effect='TX jitter', order=5, data=X1, f_cutoff=f_cutoff,
                        filter_type='lowpass', f_sampling=f_sampling, plot='no')

    X2_norm = filtering(effect='TX jitter', order=5, data=X2, f_cutoff=f_cutoff,
                        filter_type='lowpass', f_sampling=f_sampling, plot='no')

    # Standard normal > normal distribution > rice distribution
    X1_normal = std_normal * X1_norm + mean_normal
    X2_normal = std_normal * X2_norm + mean_normal
    mean_rice = np.sqrt(mean_normal**2 + mean_normal**2)
    std_rice = std_normal
    X_rice = np.sqrt(X1_normal**2 + X2_normal**2)

    # Standard normal > rayleigh distribution
    std_rayleigh = np.sqrt(2 / (4 - np.pi) * std_normal**2)
    X_rayleigh = std_rayleigh * np.sqrt(X1_norm**2 + X2_norm**2)


    # Power vectors
    # h1 = np.exp(-2 * X_rayleigh / angle_div ** 2)
    # h2 = np.exp(-2 * X_rayleigh2 / angle_div ** 2)

    # X domains
    x_0 = np.linspace(-3, 3, 100)
    x_normal = np.linspace(-angle_div, angle_div, 100)
    x_rayleigh = np.linspace(0.0, angle_div, 100)
    x_rice = np.linspace(0.0, angle_div, 100)

    # Theoretical distributions
    pdf_0 = 1 / np.sqrt(2 * np.pi * 1 ** 2) * np.exp(-((x_0 - 0) / 1) ** 2 / 2)
    pdf_normal = 1/np.sqrt(2 * np.pi * std_normal**2) * np.exp(-((x_normal - mean_normal) / std_normal)**2/2)
    pdf_rayleigh = x_rayleigh / std_rayleigh**2 * np.exp(-x_rayleigh**2 / (2 * std_rayleigh**2))
    pdf_rice = x_rice / std_rice**2 * np.exp(-(x_rice**2 + mean_rice**2) / (2*std_rice**2)) * i0(x_rice * mean_rice / std_rice**2)
    # b = mean_rice**2 / (2*std_rice**2)
    # pdf_rice = rice.pdf(x_rice, b)

    fig, ax = plt.subplots(4, 1)

    # plot normalized samples
    ax[0].hist(X2_norm, density=True)
    loc, scale = norm.fit(X2_norm)
    pdf_data = lognorm.pdf(x_0, loc, scale)
    # ax[0].plot(x_0, pdf_data, label='pdf fitted to histogram, loc=' + str(loc) + ', scale=' + str(scale), color='red')
    ax[0].plot(x_0, pdf_0, label='loc=' + str(0) + ' scale=' + str(1))


    # plot normal distribution
    ax[1].hist(X1_normal, density=True)
    loc, scale = norm.fit(X1_normal)
    pdf_data = norm.pdf(x_normal, loc, scale)
    ax[1].plot(x_normal, pdf_data, label='pdf fitted to histogram, loc=' + str(loc) + ', scale=' + str(scale), color='red')

    ax[1].hist(X2_normal, density=True)
    loc, scale = norm.fit(X2_normal)
    pdf_data = norm.pdf(x_normal, loc, scale)
    ax[1].plot(x_normal, pdf_data, label='pdf fitted to histogram, loc=' + str(loc) + ', scale=' + str(scale), color='red')
    ax[1].plot(x_normal, pdf_normal, label='loc=' + str(mean_normal) + ' scale=' + str(std_normal))

    # plot rayleigh distribution
    ax[2].hist(X_rayleigh, density=True)
    loc, scale = rayleigh.fit(X_rayleigh)
    pdf_data = rayleigh.pdf(x_rayleigh, loc, scale)
    ax[2].plot(x_rayleigh, pdf_data, label='pdf fitted to histogram, loc=' + str(loc) + ' scale=' + str(scale), color='red')
    ax[2].plot(x_rayleigh, pdf_rayleigh, label='loc= - scale=' + str(std_rayleigh))

    # plot rician distribution
    ax[3].hist(X_rice, density=True)
    shape, loc, scale = rice.fit(X_rice)
    pdf_data = rice.pdf(x_rice, shape, loc, scale)
    ax[3].plot(x_rice, pdf_data, label='pdf fitted to histogram, loc=' + str(loc) + ' scale=' + str(scale), color='red')
    ax[3].plot(x_rice, pdf_rice, label='loc=' +str(mean_rice)+' scale=' + str(std_rice))


    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()


def test_airy_disk():
    angle = np.linspace(1.0E-6, 20.0E-5, 100)
    I_norm_airy, I_norm_gaussian_approx = airy_profile(angle, D_r, focal_length)

    I_norm_gaussian = np.exp(-2*angle ** 2 / angle_div ** 2)

    fig, ax = plt.subplots(1, 1)
    ax.plot(angle, I_norm_airy, label= 'Airy disk profile with Dr='+str(D_r)+'m, focal length='+str(focal_length)+'m')
    ax.plot(angle, I_norm_gaussian_approx, label='Gaussian approx. of Airy disk profile')
    ax.plot(angle, I_norm_gaussian, label='Gaussian profile with div=25.0 urad')
    ax.legend()
    plt.show()

# test_constellation(constellation)
# test_filter()
test_RS_coding('simple')
# test_sensitivity()
# test_windspeed()
# test_beamwander()
# test_jitter()
# test_PDF()
# test_airy_disk()