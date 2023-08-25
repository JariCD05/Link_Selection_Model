import matplotlib.pyplot as plt
from matplotlib import ticker
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
    LCT = terminal_properties()
    P_r = np.logspace(-(60 + 30) / 10, -(0 + 30) / 10, num=100, base=10)
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
    SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection, noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg,
                          noise_beat=noise_beat)

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
        # Pr = np.linspace(0, 50, samples)
        # SNR = np.linspace(0, 50, samples)

        BER = LCT.BER_func(Q=Q, modulation=modulation)
        # BER = 1 / 2 * erfc(np.sqrt(SNR))
        # Vs = 1 - (1 - Vb) ** symbol_length

        # k_values = np.arange(E, N - 1)
        # binom_values = binom(N - 1, k_values)
        # SER_coded = Vs * (binom_values * np.power.outer(Vs, k_values) * np.power.outer(1 - Vs, N - k_values - 1)).sum(axis=1)
        # BER_coded = 2**(symbol_length-1)/N * SER_coded
        #
        BER_c = LCT.coding(K, N, BER)


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(W2dBm(P_r), BER, label='Uncoded (Channel BER)')
        ax.plot(W2dBm(P_r), BER_c, label='(255,223) RS coded')
        ax.invert_yaxis()
        ax.set_ylim(1.0E-9, 1.0E-1)
        ax.set_yscale('log')
        ax.grid()
        ax.set_title('Uncoded BER vs RS coded BER (255,223)')
        ax.set_xlabel('Pr (dB)')
        ax.set_ylabel('Probability of Error (# Error bits/total bits)')
        ax.legend()
        plt.show()

    elif method == 'integrated':
        return

# -------------------------------------------------------------------------------
# Sensitivity verification
# -------------------------------------------------------------------------------
def test_sensitivity(method):
    modulation = "OOK-NRZ"
    detection = "APD"
    data_rate = 10.0E9

    LCT = terminal_properties()

    P_r = np.logspace(-(60+30)/10, -(0+30)/10, num=100, base=10)
    N_p = Np_func(P_r, data_rate)
    N_p = W2dB(N_p)


    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
    LCT.BER_to_P_r(modulation=modulation, detection=detection, BER=BER_thres, threshold=True)
    SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection, noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation)

    if method == "BER/PPB":
        fig, ax = plt.subplots(1, 1)
        ax.plot(W2dB(SNR), BER)
        ax.yscale('log')
        ax.grid()
        ax.set_title('BER vs PPB')
        ax.set_xlabel('PPB (dB)')
        ax.set_ylabel('-$log_{10}$(BER)')
        plt.show()

    elif method == "compare_backward_threshold_function_with_forward_BER_function":
        fig, ax = plt.subplots(1, 1)
        detection = ['PIN', 'APD', 'quantum limit']
        ax.plot([W2dBm(P_r.min()), W2dBm(P_r.max())], BER_thres[1] * np.ones(2), label='BER thres')

        for d in detection:
            label = str(d)
            noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
            LCT.BER_to_P_r(modulation=modulation, detection=d, BER=BER_thres, threshold=True)
            SNR, Q = LCT.SNR_func(P_r=P_r, detection=d, noise_sh=noise_sh, noise_th=noise_th,
                                  noise_bg=noise_bg, noise_beat=noise_beat)
            BER = LCT.BER_func(Q=Q, modulation=modulation)
            ax.plot(W2dBm(P_r), BER, label=label)
            ax.plot(W2dBm(LCT.P_r_thres[1] * np.ones(2)), [1.0E-10, 0.5], label='Pr thres,' + label)



        # ax.set_ylim(1.0E-15, 0.0)
        ax.set_ylabel('BER')
        ax.set_yscale('log')
        ax.set_ylim(1.0E-10, 0.5)
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
    # slew_rate = [0, np.ones(2) * (1 / np.sqrt((R_earth+h_SC)**3 / mu_earth))]
    # speed_AC = [0, 250]
    # axs = plt.subplot(111)
    # h_AC = 10.0E3
    # ranges = np.ones(2) * (h_SC - h_AC)
    # heights_SC = np.ones(2) * h_SC
    # heights_AC = np.ones(2) * h_AC
    # zenith = np.ones(2) * np.pi/4
    #
    # turb = turbulence(ranges=ranges, h_AC=heights_AC, h_SC=heights_SC, zenith_angles=zenith, angle_div=angle_div)
    # # axs.set_title(f'Windspeed (Bufton)  vs  heights')
    # for Vg in speed_AC:
    #     turb.windspeed_func(slew=slew_rate, Vg=Vg, wind_model_type=wind_model_type)
    #     turb.Cn_func(turbulence_model=turbulence_model)
    #     heights = turb.height_profiles[0]
    #     windspeed = turb.windspeed[0]
    #     axs.plot(heights, windspeed, label='V ac (m/s) = '+str(Vg)+', slew ($\degree$/s): '+str(np.round(np.rad2deg(slew_rate[0]),3))+', $V_{rms}$='+str(np.round(turb.windspeed_rms[0],1)))
    #
    #
    # turb.windspeed_func(slew=slew_rate*2, Vg=Vg, wind_model_type=wind_model_type)
    # turb.Cn_func(turbulence_model=turbulence_model)
    # heights = turb.height_profiles[0]
    # windspeed = turb.windspeed[0]
    # axs.plot(heights, windspeed, label='V ac (m/s) = '+str(Vg)+', slew ($\degree$/s): '+str(np.round(np.rad2deg(slew_rate[0])*2,3))+', $V_{rms}$='+str(np.round(turb.windspeed_rms[0],1)))
    #
    # # turb.windspeed_func(slew=slew_rate, Vg=speed_AC[0], wind_model_type=wind_model_type)
    # # turb.Cn_func(turbulence_model="HVB_57")
    # # axs.plot(turb.heights, turb.windspeed, label='V wind (m/s rms) = ' + str(np.round(turb.windspeed_rms,2))+', V ac (m/s) = '+str(speed_AC[0])+', A: yes')
    # axs.set_ylabel('Wind speed [m/s]')
    # axs.set_xlabel('Altitude $h$ [m]')
    # axs.legend(fontsize=8)
    # axs.grid()
    # plt.show()

    fig, axs = plt.subplots(1,2)
    h_AC = 0.0E3
    ranges = np.ones(2) * (h_SC - h_AC)
    heights_SC = np.ones(2) * h_SC
    heights_AC = np.ones(2) * h_AC
    zenith = np.ones(2) * np.pi/4
    slew_rate = np.array([np.deg2rad(0.05), np.deg2rad(0.3)])
    speed_AC = np.array([200, 200])


    turb = turbulence(ranges=ranges, h_AC=heights_AC, h_SC=heights_SC, zenith_angles=zenith, angle_div=angle_div)
    turb.windspeed_func(slew=slew_rate, Vg=speed_AC)
    heights = turb.height_profiles_masked[0]


    turb.Cn_func()
    axs[0].plot(heights*1e-3, turb.windspeed[0],
                label='$V_{AC}$=0 m/s, slew=0$\degree$/s')
    axs[1].plot(heights*1e-3, turb.Cn2[0], label='$V_{rms}$=' + str(np.round(turb.windspeed_rms[0], 1))+'m/s', linewidth=2)


    for i in range(len(speed_AC)):
        print(slew_rate[i], speed_AC[i])
        heights = turb.height_profiles_masked[i]
        windspeed = turb.windspeed_total[i]
        axs[0].plot(heights*1e-3, windspeed, label='$V_{AC}$=' + str(speed_AC[i]) + 'm/s, slew=' + str(
                    np.round(np.rad2deg(slew_rate[i]), 3))+'$\degree$/s')
        axs[1].plot(heights*1e-3, turb.Cn2_total[i], label='$V_{rms}$=' + str(np.round(turb.windspeed_rms_total[i], 1))+'m/s')


    turb.Cn_func(A='yes')
    axs[1].plot(heights*1e-3, turb.Cn2[0], ':', label='$V_{rms}$=' + str(np.round(turb.windspeed_rms[0], 1))+'m/s, A=1.7e-14', linewidth=2)

    axs[0].set_ylabel('Wind speed [m/s]')
    axs[1].set_ylabel('$C_n^2$ [$m^{-2/3}$]')
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[0].set_xlabel('Altitude $h$ [km]')
    axs[1].set_xlabel('Altitude $h$ [km]')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[0].legend(fontsize=10)
    axs[1].legend(fontsize=10, loc='lower left')
    axs[0].grid()
    axs[1].grid()
    plt.show()

def Cn_profile():
    # fig,axs = plt.subplots(1,1)
    # slew_rate = np.array(np.deg2rad([0.0, 0.2]))
    # speed_AC = np.array([20.0, 220.0])
    # h_ac = np.array([0.0])
    # h_sc = np.array([h_SC])
    # zenith = np.array([0.0])
    # elevation = np.array(np.pi/2 - zenith)
    # ranges = np.array((h_SC - h_ac) / np.sin(elevation))
    #
    # for Vg in speed_AC:
    #     for slew in slew_rate:
    #         turb = turbulence(ranges=ranges, h_AC=h_ac, h_SC=h_sc, zenith_angles=zenith, angle_div=angle_div)
    #         axs.set_title(f'Cn2  vs  heights')
    #         turb.windspeed_func(slew=np.array([slew]), Vg=Vg, wind_model_type=wind_model_type)
    #         turb.Cn_func(turbulence_model=turbulence_model)
    #         heights = turb.height_profiles_masked/1000.0
    #         axs.plot(heights[0], turb.Cn2[0],
    #                  label='$V_{ac}$(m/s)=' + str(Vg) + ', slew($\degree$/s)=' + str(np.round(np.rad2deg(slew), 2)))
    #
    # turb.windspeed_func(slew=np.array([slew]), Vg=Vg, wind_model_type=wind_model_type)
    # turb.Cn_func(turbulence_model=turbulence_model, A='yes')
    # heights = turb.height_profiles_masked / 1000.0
    # axs.plot(heights[0], turb.Cn2[0], label='$V_{ac}$(m/s)='+str(speed_AC[0])+', slew($\degree$/s)=' + str(np.round(np.rad2deg(slew), 2))+ ', A=1.7e-14')
    #
    # axs.set_xlabel('Altitude $h$ (km)', fontsize=10)
    # axs.set_ylabel('$C_n^2$ ($m^{-2/3}$)', fontsize=10)
    # axs.set_yscale('log')
    # axs.set_xscale('log')
    # axs.grid()
    # axs.legend(fontsize=10, loc='lower left')
    # plt.show()

    fig, axs = plt.subplots(1, 1)
    slew_rate = np.array(np.deg2rad([0.0, 0.2]))
    speed_AC = np.array([20.0, 220.0])
    h_ac = np.array([0.0])
    h_sc = np.array([h_SC])
    zenith = np.array([0.0])
    elevation = np.array(np.pi / 2 - zenith)
    ranges = np.array((h_SC - h_ac) / np.sin(elevation))

    for Vg in speed_AC:
        for slew in slew_rate:
            turb = turbulence(ranges=ranges, h_AC=h_ac, h_SC=h_sc, zenith_angles=zenith, angle_div=angle_div)
            axs.set_title(f'Cn2  vs  heights')
            turb.windspeed_func(slew=np.array([slew]), Vg=Vg, wind_model_type=wind_model_type)
            turb.Cn_func(turbulence_model=turbulence_model)
            heights = turb.height_profiles_masked / 1000.0
            axs.plot(heights[0], turb.Cn2[0],
                     label='$V_{ac}$(m/s)=' + str(Vg) + ', slew($\degree$/s)=' + str(np.round(np.rad2deg(slew), 2)))

    turb.windspeed_func(slew=np.array([slew]), Vg=Vg, wind_model_type=wind_model_type)
    turb.Cn_func(turbulence_model=turbulence_model, A='yes')
    heights = turb.height_profiles_masked / 1000.0
    axs.plot(heights[0], turb.Cn2[0], label='$V_{ac}$(m/s)=' + str(speed_AC[0]) + ', slew($\degree$/s)=' + str(
        np.round(np.rad2deg(slew), 2)) + ', A=1.7e-14')

    axs.set_xlabel('Altitude $h$ (km)', fontsize=10)
    axs.set_ylabel('$C_n^2$ ($m^{-2/3}$)', fontsize=10)
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.grid()
    axs.legend(fontsize=10, loc='lower left')
    plt.show()




# -------------------------------------------------------------------------------
# Turbulence (Scintillation variance)
# -------------------------------------------------------------------------------

def test_scintillation():
    from Routing_network import routing_network
    from Link_geometry import link_geometry

    t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
    link_geometry = link_geometry()
    link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                            aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
    link_geometry.geometrical_outputs()
    link_geometry.geometrical_outputs()
    time = link_geometry.time


    routing_network = routing_network(time)
    routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_output, time)
    time_hrs = time[mask] / 3600

    time_links = routing_output['time'][link_number]
    time_links_hrs = time_links / 3600.0
    ranges = routing_output['ranges'][link_number]
    elevation = routing_output['elevation'][link_number]
    zenith = routing_output['zenith'][link_number]
    slew_rates = routing_output['slew rates'][link_number]
    heights_SC = routing_output['heights SC'][link_number]
    heights_AC = routing_output['heights AC'][link_number]
    speeds_AC = routing_output['speeds AC'][link_number]

    slew_rates_simple = np.ones(np.shape(slew_rates)) * (1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth))
    slew_rates_0      = np.zeros(np.shape(slew_rates))

    turb = turbulence(ranges=ranges,
                      h_AC=heights_AC,
                      h_SC=heights_SC,
                      zenith_angles=zenith,
                      angle_div=angle_div)
    turb.windspeed_func(slew=slew_rates,
                        Vg=speeds_AC,
                        wind_model_type=wind_model_type)

    turb.Cn_func()
    r0 = turb.r0_func()
    turb.var_rytov_func()


    D_r = [1.0]
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Scintillation variance for uplink and downlink', fontsize=13)
    # Plotting irridiance scintillation uplink and downlink
    turb.var_scint_func(link='up')
    ax[0].plot(np.rad2deg(zenith), turb.var_rytov, label='$\sigma_I^2$, weak theory')
    ax[0].plot(np.rad2deg(zenith), turb.var_scint_I, label='$\sigma_I^2$, general theory')
    ax[0].plot(np.rad2deg(zenith), turb.var_scint_P, label='$\sigma_P^2$')

    turb.var_scint_func(link='down')
    ax[1].plot(np.rad2deg(zenith), turb.var_Bu, label='$\sigma_I^2$, weak theory')
    ax[1].plot(np.rad2deg(zenith), turb.var_scint_I, label='$\sigma_I^2$, general theory')
    ax[1].plot(np.rad2deg(zenith), turb.var_scint_P, label='$\sigma_P^2$')

    ax[0].set_ylabel('Uplink   $\sigma_I^2$ (-)', fontsize=12)
    ax[1].set_ylabel('Downlink $\sigma_I^2$ (-)', fontsize=12)
    ax[1].set_xlabel('Zenith angle ($\degree$)', fontsize=12)

    # ax[0].set_ylim(0, 2)
    ax[0].set_yscale('log')
    # ax[1].set_ylim(0, 2)
    ax[1].set_yscale('log')

    ax[0].legend(fontsize=10, loc='upper left')
    ax[1].legend(fontsize=10, loc='upper left')
    ax[0].grid()
    ax[1].grid()

    plt.show()

    D_r = [D_sc, 0.15, 0.30, 1.0]
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Scintillation: Intensity variance vs Power variance', fontsize=13)
    for D in D_r:
        # Plotting irridiance scintillation uplink and downlink
        turb.var_scint_func(D_r=D, link='up')
        ax[0].plot(np.rad2deg(zenith), turb.var_scint_P/turb.var_scint_I, label='$D_R$='+str(D*100)+'cm')
        turb.var_scint_func(D_r=D, link='down')
        ax[1].plot(np.rad2deg(zenith), turb.var_scint_P/turb.var_scint_I, label='$D_R$='+str(D*100)+'cm')

    ax[0].set_ylabel('Uplink   $\sigma_{P}^2$ / $\sigma_{I}^2$ (-)', fontsize=12)
    ax[1].set_ylabel('Downlink $\sigma_{P}^2$ / $\sigma_{I}^2$ (-)', fontsize=12)
    ax[1].set_xlabel('Zenith angle ($\degree$)', fontsize=12)

    ax[0].set_ylim(0,2)
    ax[1].set_ylim(0,2)

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=10)
    ax[0].grid()
    ax[1].grid()
    plt.show()


    turb.Strehl_ratio_func(tip_tilt="YES")
    turb.beam_spread(zenith_angles=zenith)
    turb.var_bw_func(zenith_angles=zenith)
    turb.var_aoa_func(zenith_angles=zenith)

    elevation_cross_section = [25.0, 50.0]
    time_cross_section = []
    indices = []
    for e in elevation_cross_section:
        index = np.argmin(abs(elevation - np.deg2rad(e)))
        indices.append(index)
        t = time_links[index] / 3600
        time_cross_section.append(t)
        turb.print(index=index, elevation=np.rad2deg(elevation), ranges=ranges)





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


def test_atmosphere():
    from Atmosphere import attenuation
    fig, ax = plt.subplots(1, 1)
    # ax.set_title('Attenuation coefficient based on ISA model \n'
    #              'For different elevation angles',fontsize=20)

    h_0 = 0.0
    h_1 = 100.0E3
    h_profile = np.linspace(h_0, h_1, 1000)
    zenith_angles = np.linspace(0.0, 80.0, 5)
    att = attenuation(att_coeff=att_coeff, H_scale=scale_height)

    for zenith_angle in zenith_angles:
        range_link = h_profile / np.cos(np.deg2rad(zenith_angle))
        h_ext = att.h_ext_func(range_link=range_link, zenith_angles=np.deg2rad(zenith_angle), method="ISA profile")

        number_density = np.exp(-range_link / scale_height)
        att_coeff_range = att_coeff * number_density

        h_ext_test = np.exp(- np.trapz(att_coeff_range, x=range_link))

        ax.plot(h_ext, h_profile / 1E3, label=str(np.round(zenith_angle,1)))
        # ax.plot(h_ext_test*np.ones(2), np.array([h_0, h_1]) / 1E3, label=zenith_angle)

        # ax.plot(number_density, h_profile/1E3)



    ax.set_xlabel('Att. coefficient (I/I0)',fontsize=10)
    ax.set_ylabel('Altitude (km)',fontsize=10)
    ax.grid()
    ax.legend(fontsize=10)
    plt.show()


def test_PDF():
    global interval_channel_level
    sample_size_determination = 'no'
    Monte_Carlo_generation = 'no'
    pdf_verification = 'yes'
    if Monte_Carlo_generation == 'yes':
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
        var_lognorm_0 = 0.9
        std_lognorm_0 = np.sqrt(var_lognorm_0)
        # Standard normal > lognormal distribution
        mean_lognorm = -0.5 *  np.log(std_lognorm_0**2+ 1)
        std_lognorm  = np.sqrt(1/4 * np.log(std_lognorm_0**2 + 1))
        # mean_lognorm = np.exp(mean_normal + std_normal**2 / 2)
        # std_lognorm = np.sqrt((np.exp(std_normal**2) - 1) * np.exp(std_normal**2))
        X_lognorm = np.exp(mean_lognorm + std_lognorm * X1_norm)

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
        ax[0].set_title('Generation of distributions with $\sigma$='+str(std_normal)+ ' \n'
                        'sampling='+str(f_sampling/1000)+'kHz, cut-off='+str(f_cutoff/1000)+'kHz')

        # plot normalized samples
        ax[0].hist(X2_norm, density=True, bins=1000)
        loc, scale = norm.fit(X2_norm)
        pdf_data = norm.pdf(x_0, loc, scale)
        std = norm.std(loc=loc, scale=scale)
        mean = norm.mean(loc=loc, scale=scale)
        ax[0].plot(x_0, pdf_data, label='numerical, $\mu$=' + str(np.round(mean,1)) + ', $\sigma$=' + str(np.round(std,1)), color='red')
        ax[0].plot(x_0, pdf_0, label='theoretical, $\mu$=' + str(0) + ' $\sigma$=' + str(1))


        # plot normal distribution
        ax[1].hist(X1_normal, density=True, bins=1000)
        loc, scale = norm.fit(X1_normal)
        std = norm.std(loc=loc, scale=scale)
        mean = norm.mean(loc=loc, scale=scale)
        pdf_data = norm.pdf(x=x_normal, loc=loc, scale=scale)
        ax[1].plot(x_normal, pdf_data, label='numerical, $\mu$=' + str(np.round(mean,1)) + ', $\sigma$=' + str(np.round(std*1.0E6,1))+'e-06', color='red')
        ax[1].plot(x_normal, pdf_normal, label='theoretical, $\mu$=' + str(mean_normal) + ' $\sigma$=' + str(np.round(std_normal*1.0E6,1))+'e-06')
        # ax[1].hist(X2_normal, density=True, bins=1000)
        # loc, scale = norm.fit(X2_normal)
        # std = norm.std(loc=loc, scale=scale)
        # mean = norm.mean(loc=loc, scale=scale)
        # pdf_data = norm.pdf(x=x_normal, loc=loc, scale=scale)
        # ax[1].plot(x_normal, pdf_data, label='pdf fitted to histogram, $\mu$=' + str(np.round(mean,2)) + ', $\sigma$=' + str(std), color='red')

        # plot rayleigh distribution
        ax[2].hist(X_rayleigh, density=True, bins=1000)
        loc, scale = rayleigh.fit(X_rayleigh)
        std = rayleigh.std(loc=loc, scale=scale)
        std = np.sqrt(2 / (4 - np.pi) * std ** 2)
        mean = rayleigh.mean(loc=loc, scale=scale)
        pdf_data = rayleigh.pdf(x=x_rayleigh, loc=loc, scale=scale)
        ax[2].plot(x_rayleigh, pdf_data, label='numerical, $\mu$=' + str(np.round(mean*1e6,1)) + 'e-6 $\sigma$=' + str(np.round(std*1E6,1))+'e-06', color='red')
        ax[2].plot(x_rayleigh, pdf_rayleigh, label='theoretical, $\mu$=' + str(np.round(mean_rayleigh*1e6,1)) + 'e-6 $\sigma$=' + str(np.round(std_rayleigh*1E6,1))+'e-06')

        # plot lognormal distribution
        ax[3].hist(X_lognorm, density=True, bins=1000)
        shape, loc, scale = lognorm.fit(X_lognorm)
        std = lognorm.std(s=shape, loc=loc, scale=scale)
        mean = lognorm.mean(s=shape, loc=loc, scale=scale)
        pdf_data = lognorm.pdf(x=x_lognorm, s=shape, loc=loc, scale=scale)
        ax[3].plot(x_lognorm, pdf_data, label='numerical, $\sigma$=' + str(np.round(std,1)), color='red')
        ax[3].plot(x_lognorm, pdf_lognorm, label='theoretical, $\sigma$=' + str(np.round(std_lognorm,1)))

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax[1].yaxis.set_major_formatter(formatter)
        ax[2].yaxis.set_major_formatter(formatter)

        ax[0].set_ylabel('Stand. \n Gaussian', fontsize=10)
        ax[1].set_ylabel('Normalized \n Filtered', fontsize=10)
        ax[2].set_ylabel('Rayleigh', fontsize=10)
        ax[3].set_ylabel('Lognormal \n'
                         '$\sigma_I^2$='+str(np.round(std_lognorm_0**2,1)), fontsize=10)
        # ax[2].set_xlabel('$\mu$rad')
        ax[3].set_xlabel('$I/I_0$')


        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[2].legend(loc='upper right')
        ax[3].legend(loc='upper right')
        plt.show()

    elif pdf_verification == 'yes':

        # effect = 'beam wander'
        effect = 'scintillation'
        # effect = 'angle of arrival'
        # effect = 'TX jitter'
        LCT = terminal_properties()
        LCT.BER_to_P_r(BER=BER_thres,
                       modulation="OOK-NRZ",
                       detection="APD",
                       threshold=True)
        # h_AC = np.array([10.0E3, 10.0E3, 10.0E3])
        # h_SC = np.array([600.0E3, 600E3, 600E3])
        # P_r_0 = np.array([dBm2W(-25), dBm2W(-25), dBm2W(-25)])
        # elevation = np.array([np.deg2rad(25.0), np.deg2rad(50.0), np.deg2rad(70.0)])

        h_AC = np.array([10.0E3, 10.0E3, 10.0E3, 10.0E3, 10.0E3])
        h_SC = np.array([1200.0E3, 1200.0E3, 1200.0E3, 1200.0E3, 1200.0E3])
        P_r_0 = np.array([dBm2W(-25)])
        elevation = np.array([np.deg2rad(5.0), np.deg2rad(10.0), np.deg2rad(20.0), np.deg2rad(30.0), np.deg2rad(60.0)])

        zenith = np.pi / 2 - elevation
        ranges = (h_SC - h_AC) / np.sin(elevation)
        V_ac = np.array([220.0, 220.0, 220.0, 220.0, 220.0])
        slew_rate = np.ones(np.shape(h_AC)) * (1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth))
        index = 0

        turb = turbulence(ranges=ranges, zenith_angles=zenith, h_AC=h_AC, h_SC=h_SC, angle_div=angle_div)
        turb.windspeed_func(slew=slew_rate, Vg=V_ac, wind_model_type=wind_model_type)
        turb.Cn_func()
        turb.frequencies()
        r0 = turb.r0_func()
        turb.var_rytov_func()
        turb.var_scint_func()
        turb.WFE(tip_tilt="YES")
        turb.beam_spread()
        turb.var_bw_func()
        turb.var_aoa_func()
        # turb.print(index=index, elevation=np.rad2deg(elevation), ranges=ranges)

        if pdf_verification == 'yes':
            if effect == 'TX jitter' or effect == 'RX jitter':
                fig, ax = plt.subplots(1, 1)
            else:
                fig, ax = plt.subplots(3, 1)
            interval_channel_level = [10.0]

        elif sample_size_determination == 'yes':
            fig, ax = plt.subplots(3, 1)
            # interval_channel_level = np.arange(0.1, 20.0, 0.1)
            interval_channel_level = np.arange(0.1,10.1,0.1)
            std_list = []
            mean_list = []

        for interval in interval_channel_level:
            t = np.arange(0.0, interval, step_size_channel_level)
            samples = len(t)
            np.random.seed(seed=random.randint(0, 1000))
            angle_pj_t_X = norm.rvs(scale=1, loc=0, size=samples)
            angle_pj_t_Y = norm.rvs(scale=1, loc=0, size=samples)
            angle_pj_r_X = norm.rvs(scale=1, loc=0, size=samples)
            angle_pj_r_Y = norm.rvs(scale=1, loc=0, size=samples)

            h_scint = np.empty((len(P_r_0), samples))      # Should have 2D with size ( len of P_r_0 list, # of samples )
            angle_bw_X = np.empty((len(P_r_0), samples))   # Should have 2D with size ( len of P_r_0 list, # of samples )
            angle_bw_Y = np.empty((len(P_r_0), samples))   # Should have 2D with size ( len of P_r_0 list, # of samples )
            angle_aoa_X = np.empty((len(P_r_0), samples))  # Should have 2D with size ( len of P_r_0 list, # of samples )
            angle_aoa_Y = np.empty((len(P_r_0), samples))  # Should have 2D with size ( len of P_r_0 list, # of samples )

            for i in range(len(P_r_0)):
                h_scint[i] = norm.rvs(scale=1, loc=0, size=samples)
                angle_bw_X[i] = norm.rvs(scale=1, loc=0, size=samples)
                angle_bw_Y[i] = norm.rvs(scale=1, loc=0, size=samples)
                angle_aoa_X[i] = norm.rvs(scale=1, loc=0, size=samples)
                angle_aoa_Y[i] = norm.rvs(scale=1, loc=0, size=samples)

            sampling_frequency = 1 / step_size_channel_level  # 0.1 ms intervals
            nyquist = sampling_frequency / 2

            h_scint = filtering(effect='scintillation', order=frequency_filter_order, data=h_scint,
                                f_cutoff_low=turb.freq,
                                filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            angle_bw_X = filtering(effect='beam wander', order=frequency_filter_order, data=angle_bw_X,
                                   f_cutoff_low=turb.freq,
                                   filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            angle_bw_Y = filtering(effect='beam wander', order=frequency_filter_order, data=angle_bw_Y,
                                   f_cutoff_low=turb.freq,
                                   filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            angle_aoa_X = filtering(effect='angle of arrival', order=frequency_filter_order, data=angle_aoa_X,
                                    f_cutoff_low=turb.freq,
                                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            angle_aoa_Y = filtering(effect='angle of arrival', order=frequency_filter_order, data=angle_aoa_Y,
                                    f_cutoff_low=turb.freq,
                                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            angle_pj_t_X = filtering(effect='TX jitter', order=frequency_filter_order, data=angle_pj_t_X,
                                     f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1,f_cutoff_band1=jitter_freq2,
                                     filter_type='multi', f_sampling=sampling_frequency, plot='no')
            angle_pj_t_Y = filtering(effect='TX jitter', order=frequency_filter_order, data=angle_pj_t_Y,
                                     f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1,f_cutoff_band1=jitter_freq2,
                                     filter_type='multi', f_sampling=sampling_frequency, plot='no')
            angle_pj_r_X = filtering(effect='RX jitter', order=frequency_filter_order, data=angle_pj_r_X,
                                     f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                                     filter_type='multi', f_sampling=sampling_frequency, plot='no')
            angle_pj_r_Y = filtering(effect='RX jitter', order=frequency_filter_order, data=angle_pj_r_Y,
                                     f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                                     filter_type='multi', f_sampling=sampling_frequency, plot='no')

            h_scint, std_scint_dist, mean_scint_dist = turb.create_turb_distributions(data=h_scint,
                                                         steps=samples,
                                                         effect="scintillation")
            angle_bw_R, std_bw_dist, mean_bw_dist = turb.create_turb_distributions(data=[angle_bw_X, angle_bw_Y],
                                                         steps=samples,
                                                         effect="beam wander")
            angle_aoa_R, std_aoa_dist, mean_aoa_dist = turb.create_turb_distributions(data=[angle_aoa_X, angle_aoa_Y],
                                                         steps=samples,
                                                         effect="angle of arrival")
            angle_pj_t_R, std_pj_t_dist, mean_pj_t_dist = LCT.create_pointing_distributions(data=[angle_pj_t_X, angle_pj_t_Y],
                                                             steps=samples,
                                                             effect='TX jitter')
            angle_pj_r_R, std_pj_r_dist, mean_pj_r_dist = LCT.create_pointing_distributions(data=[angle_pj_r_X, angle_pj_r_Y],
                                                             steps=samples,
                                                             effect='RX jitter')

            h_bw   = h_p_gaussian(angle_bw_R, angle_div)
            h_pj_t = h_p_gaussian(angle_pj_t_R, angle_div)
            h_pj_r = h_p_airy(angle_pj_r_R, D_r, focal_length)
            h_aoa  = h_p_airy(angle_aoa_R, D_r, focal_length)

            if effect == "scintillation":
                # Analytical solution
                std_norm = turb.std_scint_I[:, None]
                std_dist = std_scint_dist[:, None]
                mean_dist = turb.mean_scint_X[:, None]
                # x = turb.x_scint
                # pdf = turb.pdf_scint
                # Numerical solution
                data = h_scint
                # dist_data, x_data = distribution_function(data, length=1, min=data.min(), max=data.max(), steps=100)
                # pdf_numerical = dist_data.pdf(x_data)
                # std_numerical = dist_data.std()
                # mean_numerical = dist_data.mean()
                data = h_scint
                pdf_numerical, cdf_numerical, x_data, std_numerical, mean_numerical = \
                    distribution_function(data=data, length=len(elevation), min=0, max=2, steps=200)
                # std_numerical = np.sqrt(1 / 4 * np.log(std_numerical + 1))
                # std_numerical = np.sqrt(1 / 4 * np.log(std_numerical ** 2 + 1))
                # std_numerical = np.sqrt(np.exp(4*std_numerical ** 2) - 1)
                # mean_numerical = -0.5 * np.log(std_numerical + 1)

                std_numerical =  np.sqrt(np.exp(4*std_numerical ** 2) - 1)

                if dist_scintillation == 'lognormal':
                    x, pdf = dist.lognorm_pdf(sigma=std_dist, mean=mean_dist, steps=samples)

                std_dist = np.sqrt(np.exp(4*std_dist ** 2) - 1)

            elif effect == "beam wander":
                std_norm = turb.std_bw[:, None]
                std_dist = std_bw_dist[:, None]
                mean_dist = mean_bw_dist[:, None]
                data = angle_bw_R
                pdf_numerical, cdf_numerical, x_data, std_numerical, mean_numerical = \
                    distribution_function(data=data, length=len(elevation), min=0, max=angle_div, steps=500)

                std_numerical = np.sqrt(2 / (4 - np.pi) * std_numerical ** 2)


                if dist_beam_wander == 'rayleigh':
                    x, pdf = dist.rayleigh_pdf(sigma=std_dist, steps=samples)
                elif dist_beam_wander == 'rice':
                    x, pdf = dist.rice_pdf(sigma=std_dist, mean=mean_dist, steps=samples)


            elif effect == 'angle of arrival':
                std_norm = turb.std_aoa[:, None]
                std_dist = std_aoa_dist[:, None]
                mean_dist = mean_aoa_dist[:, None]
                data = angle_aoa_R
                pdf_numerical, cdf_numerical, x_data, std_numerical, mean_numerical = \
                    distribution_function(data=data, length=len(elevation), min=0, max=angle_div, steps=500)

                std_numerical = np.sqrt(2 / (4 - np.pi) * std_numerical ** 2)

                if dist_AoA == 'rayleigh':
                    x, pdf = dist.rayleigh_pdf(sigma=std_dist, steps=samples)
                elif dist_AoA == 'rice':
                    x, pdf = dist.rice_pdf(sigma=std_dist, mean=mean_dist, steps=samples)


            elif effect == 'TX jitter':
                std_norm = std_pj_t
                std_dist = std_pj_t_dist
                mean_dist = mean_pj_t_dist
                data = LCT.angle_pe_t_R
                pdf_numerical, cdf_numerical, x_data, std_numerical, mean_numerical = \
                    distribution_function(data=data, length=1, min=0, max=angle_div*3, steps=500)

                # mean_numerical = np.sqrt(np.pi / 2 * std_numerical ** 2)
                std_numerical = np.sqrt(2 / (4 - np.pi) * std_numerical ** 2)

                if dist_pointing == 'rayleigh':
                    x, pdf = dist.rayleigh_pdf(sigma=std_dist, steps=samples)
                elif dist_pointing == 'rice':
                    x, pdf = dist.rice_pdf(sigma=std_dist, mean=mean_dist, steps=samples)

            elif effect == 'RX jitter':
                std_norm = std_pj_r
                std_dist = std_pj_r_dist
                mean_dist = mean_pj_r_dist
                data = LCT.angle_pe_r_R
                pdf_numerical, cdf_numerical, x_data, std_numerical, mean_numerical = \
                    distribution_function(data=data, length=1, min=data.min(), max=data.max(), steps=500)
                std_numerical = np.sqrt(2 / (4 - np.pi) * std_numerical ** 2)

                if dist_pointing == 'rayleigh':
                    x, pdf = dist.rayleigh_pdf(sigma=std_dist, steps=samples)
                elif dist_pointing == 'rice':
                    x, pdf = dist.rice_pdf(sigma=std_dist, mean=mean_dist, steps=samples)

            print('theory: ', std_dist**2, ', simulated: ', std_numerical**2 )
            if pdf_verification == 'yes':
                dist.plot_pdf_verification(ax=ax,
                                           sigma=std_dist,
                                           mean=mean_dist,
                                           x=x,
                                           pdf=pdf,
                                           sigma_num=std_numerical,
                                           mean_num=mean_numerical,
                                           pdf_num=pdf_numerical,
                                           x_num=x_data,
                                           data=data,
                                           elevation=elevation,
                                           effect=effect)
                plt.show()

            else:
                ax[0].set_title("Stability testing of Monte Carlo sampling for "+effect+" effect \n"
                                                                                        "Sample step size = "+str(np.round(step_size_channel_level*1000,1))+" ms", fontsize=15)
                # Analytical solution
                std_analytical  = std_dist
                mean_analytical = mean_dist
                pdf_analytical  = pdf

                if effect == "scintillation" or effect == "beam wander" or effect == "angle of arrival":
                    # Analytical solution
                    std_analytical = std_analytical[0]
                    mean_analytical = mean_analytical[0]
                    pdf_analytical = pdf[0]
                    # Numerical solution
                    pdf_numerical = pdf_numerical
                    std_numerical = std_numerical
                    mean_numerical = mean_numerical

                # print('--------------------------------------------')
                # print('Analytical', std_analytical, mean_analytical)
                # print('Numerical', std_numerical, mean_numerical)

                std_list.append(std_numerical)
                mean_list.append(mean_numerical)
                print(interval)
                if interval == 1 or interval == 5 or interval == 10 or interval == 20 or interval == 50:
                    ax[0].plot(x_data, pdf_numerical,
                                 label='Interval size=' + str(np.round(interval,1)) + 's')
        ax[0].plot(x, pdf_analytical,
                     label='Analytical')


        sample_list = interval_channel_level / step_size_channel_level

        ax[1].set_ylabel('Sample standard deviation ($\sigma$)', fontsize=15)
        ax[1].plot(interval_channel_level, std_list)
        ax[1].plot([(interval_channel_level[0]), (interval_channel_level[-1])], std_analytical * np.ones(2),
                     label='Analytical std')
        ax[1].plot([(interval_channel_level[0]), (interval_channel_level[-1])], (std_analytical * 1.05) * np.ones(2),
                     label='-5% diff', color='black')
        ax[1].plot([(interval_channel_level[0]), (interval_channel_level[-1])], (std_analytical * 0.95) * np.ones(2),
                     label='+5% diff', color='grey')

        ax[2].set_ylabel('Sample mean ($\mu$)', fontsize=15)
        ax[2].plot(interval_channel_level, mean_list)
        ax[2].plot([(interval_channel_level[0]), (interval_channel_level[-1])], mean_analytical * np.ones(2), label='True mean')
        ax[2].plot([(interval_channel_level[0]), (interval_channel_level[-1])], (mean_analytical * 1.05) * np.ones(2),
                     label='-5% diff', color='black')
        ax[2].plot([(interval_channel_level[0]), (interval_channel_level[-1])], (mean_analytical * 0.95) * np.ones(2),
                     label='+5% diff', color='grey')

        ax[2].set_xlabel('Interval of Monte Carlo simulation (sec)', fontsize=15)
        ax[0].legend(loc='upper right', fontsize=15)
        ax[0].grid()
        ax[1].legend(loc='upper right', fontsize=15)
        ax[1].grid()
        ax[2].legend(loc='upper right', fontsize=15)
        ax[2].grid()

        plt.show()

def noise_and_SNR_BER_verification():
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig_noise, (ax_noise, ax_noise1) = plt.subplots(1, 2)
    from input import data_rate

    t = np.arange(0.0, interval_channel_level, step_size_channel_level)
    samples_channel_level = int(len(t))

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
    ax_noise.set_title('NF='+str(noise_factor)+', M='+str(M), fontsize=13)
    ax_noise.plot(W2dBm(P_r), W2dB(noise_sh), label='shot noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_th), label='thermal noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_bg), label='background noise')
    ax_noise.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_beat), label='beat noise')

    LCT = terminal_properties(noise_factor=8, M=300)
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
    ax_noise1.set_title('NF=' + str(8) + ', M=' + str(300), fontsize=13)
    ax_noise1.plot(W2dBm(P_r), W2dB(noise_sh), label='shot noise')
    ax_noise1.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_th), label='thermal noise')
    ax_noise1.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_bg), label='background noise')
    ax_noise1.plot(W2dBm(P_r), np.ones(len(P_r)) * W2dB(noise_beat), label='beat noise')


    for detection in detection_list:
        noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)
        SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        if detection == 'APD':
            ax[0].plot(W2dBm(P_r), W2dB(SNR), label=str(detection)+' (NF='+str(noise_factor)+',G='+str(M)+')')
        else:
            ax[0].plot(W2dBm(P_r), W2dB(SNR), label=str(detection))

        for modulation in modulation_list:
            LCT.BER_to_P_r(BER=BER_thres,
                           modulation=modulation,
                           detection=detection,
                           threshold=True)

            BER = LCT.BER_func(Q=Q, modulation=modulation)
            BER[BER < 1e-100] = 1e-100
            x_BER = np.linspace(-30.0, 0.0, 1000)
            # BER_pdf = pdf_function(np.log10(BER), len(BER), x_BER, steps=100)

            total_bits = LCT.data_rate * interval_channel_level         # Should be a scalar
            total_bits_per_sample = total_bits / samples_channel_level  # Should be a scalar
            errors = total_bits_per_sample * BER

            ax[1].plot(W2dBm(P_r), BER, label=str(detection) + ', ' + str(modulation))

    ax[1].set_yscale('log')
    ax[0].set_ylabel('Signal-to-Noise ratio \n [$i_r$ / $\sigma_n$] (dB)', fontsize=10)
    ax[1].set_ylabel('Error probability \n [# Error bits / total bits]', fontsize=10)
    ax[0].set_xlabel('Received power (dBm)', fontsize=10)
    ax[1].set_xlabel('Received power (dBm)', fontsize=10)
    ax[0].legend(fontsize=10)
    # ax[1].legend(fontsize=10)
    ax[0].grid()
    ax[1].grid()

    ax_noise.set_ylabel('noise (dBm)', fontsize=10)
    ax_noise.set_xlabel('Received power (dBm)', fontsize=10)
    ax_noise1.set_xlabel('Received power (dBm)', fontsize=10)

    ax_noise.legend(fontsize=10)
    ax_noise.grid()
    ax_noise1.legend(fontsize=10)
    ax_noise1.grid()

    plt.show()

def test_airy_disk():
    angle = np.linspace(1.0E-6, 100.0E-6, 1000)
    I_norm_airy, P_norm_airy = h_p_airy(angle, D_r, focal_length)
    r = focal_length * np.sin(angle)
    fig, ax = plt.subplots(1, 1)
    # ax.set_title('Spot profile, focal length='+str(focal_length)+'m, Dr='+str(np.round(D_r,3))+'m')
    ax.plot(angle*1.0E6, I_norm_airy, label='I(r)/$I_0$')
    # ax.plot(angle*1.0E6, I_norm_gaussian_approx, label='Gaussian approx. I(r)/$I_0$')
    ax.plot(angle*1.0E6, P_norm_airy, label='P(r)/$P_0$')
    ax.set_xlabel('Angular displacement ($\mu$rad)', fontsize=12)
    ax.set_ylabel('Normalized \n intensity and power', fontsize=12)
    ax.legend(fontsize=15)
    ax.grid()
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

def coverage():
    LOS_verification = False

    from Link_geometry import link_geometry
    from Routing_network import routing_network

    t_macro = np.arange(0.0, (end_time - start_time), step_size_link)

    routes = [r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\traffic_trajectories\OSL_ENEV.csv",
              r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\traffic_trajectories\SYD_MEL.csv"]

    fig = plt.figure(figsize=(6, 6), dpi=125)
    fig1, ax1 = plt.subplots(2, 1)
    fig2, ax2 = plt.subplots(2, 1)

    ax = fig.add_subplot(111, projection='3d')
    for route in routes:
        print(route)
        link = link_geometry()
        link.propagate(step_size_AC=step_size_AC,
                       step_size_SC=step_size_SC,
                       step_size_analysis=False,
                       verification_cons=False,
                       aircraft_filename=route,
                       time=t_macro)
        link.geometrical_outputs()

        time = link.time
        routing = routing_network(time=time)
        routing_output, routing_total_output, mask = routing.routing(link.geometrical_output, time, step_size_link)
        link.plot(type='satellite sequence', routing_output=routing_output, fig=fig,ax=ax, aircraft_filename=route)

        def plot_mission_geometrical_output_slew_rates():
            if link_number == 'all':
                pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                    data=routing_total_output['slew rates'],
                    length=1,
                    min=routing_total_output['slew rates'].min(),
                    max=routing_total_output['slew rates'].max(),
                    steps=1000)
                # ax[1, 0].plot(x_elev, pdf_elev)
                ax1[0].set_title('Slew rate analysis', fontsize=15)
                ax1[0].plot(np.rad2deg(x_slew), cdf_slew, label=str(route[84:-4]))

                for i in range(len(routing_output['link number'])):
                    if np.any(np.isnan(routing_output['slew rates'][i])) == False:
                        pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                            data=routing_output['slew rates'][i],
                            length=1,
                            min=routing_output['slew rates'][i].min(),
                            max=routing_output['slew rates'][i].max(),
                            steps=1000)
                        ax1[1].plot(np.rad2deg(x_slew), cdf_slew)


                ax1[0].set_ylabel('Prob. density \n all links combined', fontsize=15)
                ax1[1].set_ylabel('Prob. density \n for each link', fontsize=15)
                ax1[1].set_xlabel('Slew rate (deg/sec)', fontsize=15)
                ax1[0].grid()
                ax1[1].grid()
                ax1[0].legend(fontsize=10)
                ax1[1].legend(fontsize=10)

        def plot_mission_geometrical_output_coverage():
            if link_number == 'all':
                elevation = flatten(routing_output['elevation'])

                pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=elevation, length=1,
                                                                                        min=elevation.min(),
                                                                                        max=elevation.max(), steps=1000)

                fig2.suptitle('Coverage analysis \n Total mission time (hrs): ' + str(
                    np.round((time[-1] - time[0]) / 3600, 2)) + ', '
                                                                'total acquisition time (min)=' + str(
                    np.round(routing.total_acquisition_time / 60, 2)) + '\n '
                                                                                'Fractional link time (%):' + str(
                    np.round(routing.frac_comm_time * 100, 2)), fontsize=15)
                ax2[0].plot(np.rad2deg(x_elev), cdf_elev, label=str(route[84:-4]))

                for e in range(len(routing_output['elevation'])):
                    if np.any(np.isnan(routing_output['elevation'][e])) == False:
                        elevation = routing_output['elevation'][e]
                        pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=elevation,
                                                                                                length=1,
                                                                                                min=elevation.min(),
                                                                                                max=elevation.max(),
                                                                                                steps=1000)
                        ax2[1].plot(np.rad2deg(x_elev), cdf_elev)

                ax2[0].set_ylabel('Prob. density \n all links combined', fontsize=15)
                ax2[1].set_ylabel('Prob. density \n for each link', fontsize=15)
                ax2[1].set_xlabel('Elevation (rad)', fontsize=15)

                ax2[0].grid()
                ax2[1].grid()
                ax2[0].legend(fontsize=10)
                ax2[1].legend(fontsize=10)

        plot_mission_geometrical_output_coverage()
        plot_mission_geometrical_output_slew_rates()

    plt.show()

    if LOS_verification == True:
        link = link_geometry()
        link.propagate(step_size_AC=step_size_AC,
                       step_size_SC=step_size_SC,
                       step_size_analysis=False,
                       verification_cons=False,
                       aircraft_filename=aircraft_filename_load,
                       time=t_macro)

        link.geometrical_outputs()
        elevation = link.geometrical_output['elevation']
        # fig, ax = plt.subplots(1,1)
        # for i in range(len(elevation)):

def stability_verification():
    global range_delta_t, angle_div, w0, turbulence_freq_lowpass, interval_channel_level
    from channel_level import channel_level
    from bit_level import bit_level
    from Link_geometry import link_geometry
    from Routing_network import routing_network
    from Atmosphere import attenuation

    input_variable = 'aperture' #'frequency' or 'aperture' or 'none'
    model_variable = 'macro: step size'  #'micro: sample size' or 'macro: step size'
    test_scope = 'global' #'global' or 'local'
    elevation_cross_section = [40.0]

    # Define input variable which is used for the verification test
    if input_variable == 'aperture':
        unit = 'm'
        variable_range = np.array([0.02, 0.05, 0.08, 0.1])
        # variable_range = np.array([0.05])

    elif input_variable == 'frequency':
        unit = 'Hz'
        variable_range = np.array([500, 750.0, 1000.0, 1500.0])
        # variable_range = np.array([1000.0])

    else:
        variable_range = [0.0]


    # Define model variable which is used for the verification test
    if model_variable == 'macro: step size':
        range_delta_t = np.arange(1.0, 10.5, 0.5)
        # range_delta_t = np.arange(3.0, 10.0, 1.0)
        interval_channel_level = [5.0]

        fig_T, ax_T = plt.subplots(4, 1)
        ax_T[0].set_title('Model stability: Macroscale step size \n Input variable: '+str(input_variable), fontsize=15)

    elif model_variable == 'micro: sample size':
        range_delta_t = [5.0]
        interval_channel_level = np.arange(0.1, 10.6, 0.5)
        interval_channel_level = np.array([5.0, 10.0, 25.0, 50.0, 80.0])
        # interval_channel_level = [1.0]

        fig_T, ax_T = plt.subplots(4, 1)
        ax_T[0].set_title('Model stability: Microscale population size \n Input variable: '+str(input_variable), fontsize=15)


    fig_fades, ax_fades = plt.subplots(3, 1)
    for variable in variable_range:
        fractional_fade_time_list = []
        mean_fade_time_list = []
        var_P_r_list = []
        mean_P_r_list = []

        if input_variable == 'frequency':
            turbulence_freq_lowpass = variable
        elif input_variable == 'aperture':
            w0 = variable / clipping_ratio / 2
            angle_div = wavelength / (np.pi * w0)

        for delta_t in range_delta_t:
            interval_link_level = int(end_time - start_time)
            t_macro = np.arange(0.0, interval_link_level, delta_t)
            samples_macro = int(len(t_macro))

            print('--------------------------')
            print(delta_t, 'of', range_delta_t[-1])

            # Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
            link = link_geometry()
            link.propagate(step_size_AC=delta_t,
                           step_size_SC=step_size_SC,
                           step_size_analysis=False,
                           verification_cons=False,
                           aircraft_filename=aircraft_filename_load,
                           time=t_macro)
            link.geometrical_outputs()
            time = link.time
            network = routing_network(time)
            routing_output, routing_total_output, mask = network.routing(link.geometrical_output, time, delta_t)


            time_links = flatten(routing_output['time'      ][:1])
            elevation  = flatten(routing_output['elevation' ][:1])
            ranges     = flatten(routing_output['ranges'    ][:1])
            zenith     = flatten(routing_output['zenith'    ][:1])
            heights_SC = flatten(routing_output['heights SC'][:1])
            heights_AC = flatten(routing_output['heights AC'][:1])
            slew_rates = flatten(routing_output['slew rates'][:1])

            time_cross_section = []
            indices = []
            for e in elevation_cross_section:
                index = np.argmin(abs(elevation - np.deg2rad(e)))
                indices.append(index)
                t = time_links[index] / 3600
                time_cross_section.append(t)

            if test_scope == 'local':
                time_links = time_links[indices]
                elevation = elevation[indices]
                ranges = ranges[indices]
                zenith = zenith[indices]
                heights_SC = heights_SC[indices]
                heights_AC = heights_AC[indices]
                slew_rates = slew_rates[indices]

            def plot_mission_geometrical_output_coverage():
                if link_number == 'all':
                    pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=elevation, length=1,
                                                                                            min=elevation.min(),
                                                                                            max=elevation.max(),
                                                                                            steps=1000)

                    fig, ax = plt.subplots(2, 1)
                    fig.suptitle('Coverage analysis \n Total mission time (hrs): ' + str(
                        np.round((time[-1] - time[0]) / 3600, 2)) + ', '
                                                                    'total acquisition time (min)=' + str(
                        np.round(network.total_acquisition_time / 60, 2)) + '\n '
                                                                                    'Fractional link time (%):' + str(
                        np.round(network.frac_comm_time * 100, 2)), fontsize=15)
                    ax[0].plot(np.rad2deg(x_elev), cdf_elev)
                    ax[0].plot(np.ones(2) * elevation_cross_section[0], [0, 1], color='black',
                               label='cross section for $\epsilon$=' + str(elevation_cross_section[0]) + 'deg')
                    ax[0].plot(np.ones(2) * elevation_cross_section[1], [0, 1], color='black',
                               label='cross section for $\epsilon$=' + str(elevation_cross_section[1]) + 'deg')

                    for e in range(len(routing_output['elevation'])):
                        if np.any(np.isnan(routing_output['elevation'][e])) == False:
                            pdf_elev, x_elev = pdf_function(data=routing_output['elevation'][e], length=1,
                                                            min=elevation.min(), max=elevation.max(), steps=1000)
                            cdf_elev, x_elev = cdf_function(data=routing_output['elevation'][e], length=1,
                                                            min=elevation.min(), max=elevation.max(), steps=1000)
                            ax[1].plot(np.rad2deg(x_elev), cdf_elev,
                                       label='link ' + str(routing_output['link number'][e]))

                    ax[0].set_ylabel('Prob. density \n all links combined', fontsize=15)
                    ax[1].set_ylabel('Prob. density \n for each link', fontsize=15)
                    ax[1].set_xlabel('Elevation (rad)', fontsize=15)

                    ax[0].grid()
                    ax[1].grid()
                    ax[0].legend(fontsize=10)
                    ax[1].legend(fontsize=10)

                else:
                    fig, ax = plt.subplots(1, 1)
                    fig.suptitle(
                        'Coverage analysis \n Total mission time (hrs): ' + str(
                            np.round((time[-1] - time[0]) / 3600, 2)) + ', '
                                                                        'total acquisition time (min)=' + str(
                            np.round(network.total_acquisition_time / 60, 2)) + '\n '
                                                                                        'Fractional link time (%):' + str(
                            np.round(network.frac_comm_time * 100, 2)), fontsize=20)
                    for e in range(len(routing_output['elevation'])):
                        if np.any(np.isnan(routing_output['elevation'][e])) == False:
                            pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(
                                data=routing_output['elevation'][e],
                                length=1,
                                min=routing_output['elevation'][e].min(),
                                max=routing_output['elevation'][e].max(),
                                steps=1000)
                            ax.plot(np.rad2deg(x_elev), cdf_elev, label='link ' + str(routing_output['link number'][e]))

                    ax.set_ylabel('Prob. density \n for each link', fontsize=20)
                    ax.set_xlabel('Elevation (rad)', fontsize=20)
                    ax.grid()
                    ax.legend(fontsize=20)

                plt.show()
            def plot_mission_geometrical_output_slew_rates():
                if link_number == 'all':
                    pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                        data=routing_total_output['slew rates'],
                        length=1,
                        min=routing_total_output['slew rates'].min(),
                        max=routing_total_output['slew rates'].max(),
                        steps=1000)
                    fig, ax = plt.subplots(2, 1)
                    # ax[1, 0].plot(x_elev, pdf_elev)
                    ax[0].set_title('Slew rate analysis', fontsize=15)
                    ax[0].plot(np.rad2deg(x_slew), cdf_slew)

                    for i in range(len(routing_output['link number'])):
                        if np.any(np.isnan(routing_output['slew rates'][i])) == False:
                            pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                                data=routing_output['slew rates'][i],
                                length=1,
                                min=routing_output['slew rates'][i].min(),
                                max=routing_output['slew rates'][i].max(),
                                steps=1000)
                            ax[1].plot(np.rad2deg(x_slew), cdf_slew,
                                       label='link ' + str(routing_output['link number'][i]))

                    ax[0].set_ylabel('Prob. density \n all links combined', fontsize=15)
                    ax[1].set_ylabel('Prob. density \n for each link', fontsize=15)
                    ax[1].set_xlabel('Slew rate (deg/sec)', fontsize=15)
                    ax[0].grid()
                    ax[1].grid()
                    ax[0].legend(fontsize=10)
                    ax[1].legend(fontsize=10)

                else:
                    fig, ax = plt.subplots(1, 1)
                    fig.suptitle('Slew rate analysis', fontsize=20)

                    for i in range(len(routing_output['link number'])):
                        if np.any(np.isnan(routing_output['slew rates'][i])) == False:
                            pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                                data=routing_output['slew rates'][i],
                                length=1,
                                min=routing_output['slew rates'][i].min(),
                                max=routing_output['slew rates'][i].max(),
                                steps=1000)
                            ax.plot(np.rad2deg(x_slew), cdf_slew, label='link ' + str(routing_output['link number'][i]))

                    ax.set_ylabel('Prob. density \n for each link', fontsize=20)
                    ax.set_xlabel('Slew rate (deg/sec)', fontsize=20)
                    ax.grid()
                    ax.legend(fontsize=20)

                plt.show()
            # plot_mission_geometrical_output_coverage()
            # plot_mission_geometrical_output_slew_rates()
            # link.plot(type='satellite sequence', routing_output=routing_output,
                               # links=network.number_of_links)

            att = attenuation()
            att.h_ext_func(range_link=ranges, zenith_angles=zenith, method=method_att)
            att.h_clouds_func(method=method_clouds)
            h_ext = att.h_ext * att.h_clouds

            # ------------------------------------------------------------------------
            # -----------------------------------LCT----------------------------------
            # ------------------------------------------------------------------------
            LCT = terminal_properties()
            LCT.BER_to_P_r(BER=BER_thres,
                          modulation="OOK-NRZ",
                          detection="APD",
                          threshold=True)
            # ------------------------------------------------------------------------
            # -------------------------------TURBULENCE-------------------------------
            # ------------------------------------------------------------------------

            # The turbulence class is initiated here. Inside the turbulence class, there are multiple functions that are run.
            turb = turbulence(ranges=ranges,
                              zenith_angles=zenith,
                              angle_div=angle_div,
                              h_AC=heights_AC,
                              h_SC=heights_SC)
            # link.speed_AC.mean()
            turb.windspeed_func(slew=slew_rates, Vg=link.speed_AC.mean(), wind_model_type=wind_model_type)
            turb.Cn_func(turbulence_model=turbulence_model)
            r0 = turb.r0_func(zenith_angles=zenith)
            turb.var_rytov_func(zenith_angles=zenith)
            turb.var_scint_func(zenith_angles=zenith, D_r=D_r)
            turb.WFE(tip_tilt="YES")
            turb.beam_spread(zenith_angles=zenith)
            turb.var_bw_func(zenith_angles=zenith)
            turb.var_aoa_func(zenith_angles=zenith)
            # turb.print(index=plot_index, elevation=np.rad2deg(elevation), ranges=ranges)
            # ------------------------------------------------------------------------
            # -------------------------------LINK-BUDGET------------------------------
            # ------------------------------------------------------------------------
            # The link budget class is initiated here.
            link = link_budget(angle_div=angle_div, w0=w0, ranges=ranges, h_WFE=turb.h_WFE, w_ST=turb.w_ST,
                               h_beamspread=turb.h_beamspread, h_ext=att.h_ext)
            # The power at the receiver is computed from the link budget.
            P_r_0 = link.P_r_0_func()

            # link.print(elevation=elevation, index=indices[0], static=True)


            for interval in interval_channel_level:
                t_micro = np.arange(0.0, interval, step_size_channel_level)
                samples_micro = int(len(t_micro))

                P_r, P_r_no_pointing_errors, PPB, elevation_angles, losses, angles = \
                    channel_level(plot_index=int(0),
                                  LCT=LCT,
                                  t=t_micro,
                                  turb=turb,
                                  P_r_0=P_r_0,
                                  ranges=ranges,
                                  angle_div=link.angle_div,
                                  elevation_angles=elevation,
                                  zenith_angles=zenith,
                                  samples=samples_micro,
                                  turb_cutoff_frequency=turbulence_freq_lowpass)

                h_tot = losses[0]

                # Fade correction
                # desired_frac_BER_fade = 0.05
                P_min = penalty(P_r=P_r, desired_frac_fade_time=desired_frac_fade_time)
                h_penalty = (P_min / P_r.mean(axis=1)).clip(min=0.0, max=1.0)
                # P_r = P_r * h_penalty[:, None]
                correction = (LCT.P_r_thres[1] / P_min).clip(min=1.0)
                # P_r = P_r * correction[:, None]

                P_r_total = P_r.flatten()
                P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r, mean_P_r = distribution_function(data=W2dBm(P_r_total),
                                                                                                     length=1, min=-80.0,
                                                                                                     max=0.0, steps=10000)

                # # In case of TOTAL variance
                if test_scope == 'global':
                    mean_P_r = P_r_total.mean()
                    mean_P_r_dB = W2dB(mean_P_r)
                    var_P_r = P_r_total.var()
                    var_P_r_dB = W2dB(var_P_r)

                # In case of time step-specific variance
                elif test_scope == 'local':
                    mean_P_r = P_r.mean()
                    mean_P_r_dB = W2dB(mean_P_r)
                    var_P_r = P_r.var()
                    var_P_r_dB = W2dB(var_P_r)

                number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
                fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_micro
                mean_fade_time = fractional_fade_time / number_of_fades * interval


                ax_fades[0].plot(np.rad2deg(elevation), fractional_fade_time, label=str(interval)+'s')
                ax_fades[1].plot(np.rad2deg(elevation), mean_fade_time * 1000)
                ax_fades[2].plot(np.rad2deg(elevation), number_of_fades/interval)

                # if interval == 80.0:
                #     ax_fades[0].plot(np.rad2deg(elevation), np.ones(elevation.shape) * fractional_fade_time[0] * 1.05,
                #                      label='5% diff', color='black')
                #     ax_fades[0].plot(np.rad2deg(elevation), np.ones(elevation.shape) * fractional_fade_time[0] * 0.95,
                #                      color='black')
                #     ax_fades[1].plot(np.rad2deg(elevation), np.ones(elevation.shape) * mean_fade_time[0] * 1.05,
                #                      label='5% diff', color='black')
                #     ax_fades[1].plot(np.rad2deg(elevation), np.ones(elevation.shape) * mean_fade_time[0] * 0.95,
                #                      color='black')
                #     ax_fades[2].plot(np.rad2deg(elevation),
                #                      np.ones(elevation.shape) * number_of_fades[0] / interval * 1.05, label='5% diff',
                #                      color='black')
                #     ax_fades[2].plot(np.rad2deg(elevation),
                #                      np.ones(elevation.shape) * number_of_fades[0] / interval * 0.95, color='black')

                # # In case of TOTAL VARIANCE
                if test_scope == 'global':
                    fractional_fade_time = fractional_fade_time.mean()
                    mean_fade_time = mean_fade_time.mean()

                mean_P_r_list.append(mean_P_r_dB)
                var_P_r_list.append(var_P_r_dB)
                fractional_fade_time_list.append(fractional_fade_time)
                mean_fade_time_list.append(mean_fade_time)

                # Plotting
                if input_variable == 'frequency' or input_variable == 'aperture':
                    label = str(variable)+' '+str(unit)
                else:
                    label = None

                if model_variable == 'macro: step size':
                    if np.round(delta_t, 2) == 10.0:
                        ax_T[0].plot(x_P_r_total, P_r_pdf_total, label=label)

                elif model_variable == 'micro: sample size':
                    if np.round(interval, 2) == 10.1:
                        ax_T[0].plot(x_P_r_total, P_r_pdf_total, label=label)


        if model_variable == 'macro: step size':

            ax_T[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [P_r_pdf_total.min(), P_r_pdf_total.max()], c='black',
                         linewidth=3)

            ax_T[1].set_ylabel('Frac. fade time', fontsize=10)
            ax_T[1].plot(range_delta_t, fractional_fade_time_list)

            ax_T[2].set_ylabel('Mean fade time (ms)', fontsize=10)
            ax_T[2].plot(range_delta_t, np.array(mean_fade_time_list)*1E3)

            ax_T[3].set_ylabel('Sample variance \n of $P_r$ (dB)', fontsize=10)
            ax_T[3].plot(range_delta_t, var_P_r_list)
            ax_T[3].set_xlabel('Macroscale $\Delta$T (sec)', fontsize=15)

            if input_variable == 'frequency' and variable == 1000.0 or \
                    input_variable == 'aperture' and variable == 0.08 or input_variable == 'none':
                ax_T[1].plot([range_delta_t[0], range_delta_t[-1]], fractional_fade_time_list[0] * 0.95 * np.ones(2),
                             label='5% diff', color='black')
                ax_T[1].plot([range_delta_t[0], range_delta_t[-1]], fractional_fade_time_list[0] * 1.05 * np.ones(2),
                             color='black')
                ax_T[2].plot([range_delta_t[0], range_delta_t[-1]], mean_fade_time_list[0] * 1E3 * 0.95 * np.ones(2),
                             label='-5% diff ', color='black')
                ax_T[2].plot([range_delta_t[0], range_delta_t[-1]], mean_fade_time_list[0] * 1E3 * 1.05 * np.ones(2),
                             color='black')
                ax_T[3].plot([range_delta_t[0], range_delta_t[-1]], (var_P_r_list[0] - 0.5) * np.ones(2),
                             label='-0.5 dB diff', color='black')
                ax_T[3].plot([range_delta_t[0], range_delta_t[-1]], (var_P_r_list[0] + 0.5) * np.ones(2),
                             color='black')

        elif model_variable == 'micro: sample size':

            ax_T[1].set_ylabel('Frac fade \n time (-)', fontsize=10)
            ax_T[1].plot(interval_channel_level, fractional_fade_time_list)

            ax_T[2].set_ylabel('Mean fade \n time (ms)', fontsize=10)
            ax_T[2].plot(interval_channel_level, np.array(mean_fade_time_list)*1E3)

            ax_T[3].set_ylabel('Variance (dB)', fontsize=10)
            ax_T[3].plot(interval_channel_level, var_P_r_list)
            ax_T[3].set_xlabel('Population size (sec)', fontsize=15)

            print('frequency =', variable)
            if input_variable == 'frequency' and variable == 1000.0 or \
                    input_variable == 'aperture' and variable == 0.08 or input_variable == 'none':
                print('PLOTTING--------------------------------------')
                ax_T[1].plot([interval_channel_level[0], interval_channel_level[-1]], fractional_fade_time_list[-1] * 0.95 * np.ones(2),
                             label='5% diff', color='black')
                ax_T[1].plot([interval_channel_level[0], interval_channel_level[-1]], fractional_fade_time_list[-1] * 1.05 * np.ones(2),
                             color='black')
                ax_T[2].plot([interval_channel_level[0], interval_channel_level[-1]], mean_fade_time_list[-1] * 1E3 * 0.95 * np.ones(2),
                             label='-5% diff ', color='black')
                ax_T[2].plot([interval_channel_level[0], interval_channel_level[-1]], mean_fade_time_list[-1] * 1E3 * 1.05 * np.ones(2),
                             color='black')
                ax_T[3].plot([interval_channel_level[0], interval_channel_level[-1]], (var_P_r_list[-1] - 0.5) * np.ones(2),
                             label='-0.5 dB diff', color='black')
                ax_T[3].plot([interval_channel_level[0], interval_channel_level[-1]], (var_P_r_list[-1] + 0.5) * np.ones(2),
                             color='black')



    ax_T[0].legend(loc='upper right')
    ax_T[0].grid()
    ax_T[1].legend(loc='upper right')
    ax_T[1].grid()
    ax_T[2].legend(loc='upper right')
    ax_T[2].grid()
    ax_T[3].legend(loc='upper right')
    ax_T[3].grid()

    ax_T[0].set_ylabel('PDF')
    ax_T[0].set_xlabel('Pr (dBm)', fontsize=15)
    ax_T[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [0.0, 0.5],
                 c='black', linewidth=3, label='Treshold BER=1E-6')

    ax_fades[0].set_ylabel('frac fade \n time (-)', fontsize=10)
    ax_fades[1].set_ylabel('mean fade \n time (ms)', fontsize=10)
    ax_fades[2].set_ylabel('number of \n fades (#/s)', fontsize=10)
    ax_fades[2].set_xlabel('elevation (deg)', fontsize=10)
    ax_fades[0].legend(fontsize=10)
    ax_fades[0].grid()
    ax_fades[1].grid()
    ax_fades[2].grid()
    plt.show()


# test_constellation(constellation)
# test_filter()
# test_RS_coding('simple')
# test_sensitivity(method='compare_backward_threshold_function_with_forward_BER_function')
# test_windspeed()
# Cn_profile()
# test_scintillation()
# test_beamwander()
# test_jitter()
# test_atmosphere()
# test_PDF()
# test_airy_disk()
noise_and_SNR_BER_verification()
# simple_losses()
# test_interleaving()
# channel_level_verification()
# coverage()
# stability_verification()