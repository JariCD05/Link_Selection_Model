
from helper_functions import *

def channel_level(LCT,
                  turb,
                  link_budget,
                  t:np.array,
                  plot_indices: list,
                  ranges: np.array,
                  angle_div: float,
                  P_r_0: np.array,
                  elevation_angles: np.array,
                  samples,
                  turb_cutoff_frequency=1.0E4):
    # print('')
    # print('-----------------------------------CHANNEL-LEVEL-----------------------------------------')
    # print('')
    plot_index = plot_indices[0]

    # ------------------------------------------------------------------------
    # ------------------------INITIALIZING-ALL-VECTORS------------------------
    # ----------------------START-MONTE-CARLO-SIMULATIONS---------------------
    # ------------------------------------------------------------------------
    # Seed is randomized
    np.random.seed(seed=random.randint(0,1000))

    # For each fluctuating variable, a vector is initialized with a standard normal distribution (std=1, mean=0)
    # For all jitter related vectors (beam wander, angle-of-arrival, mechanical TX jitter, mechanical RX jitter), two variables are initialized for both X- and Y-components
    # And stored in a 1D array

    angle_pj_t_X = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_t_Y = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_r_X = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_r_Y = norm.rvs(scale=1, loc=0, size=samples)

    # The turbulence vectors (beam wander and angle-of-arrival) are range-dependent and must be evaluated for each macro time step
    # And stored in a 2D array.

    h_scint     = np.empty((len(P_r_0), samples))                            # Should have 2D with size ( len of P_r_0 list, # of samples )
    angle_bw_X  = np.empty((len(P_r_0), samples))                            # Should have 2D with size ( len of P_r_0 list, # of samples )
    angle_bw_Y  = np.empty((len(P_r_0), samples))                            # Should have 2D with size ( len of P_r_0 list, # of samples )
    angle_aoa_X = np.empty((len(P_r_0), samples))                            # Should have 2D with size ( len of P_r_0 list, # of samples )
    angle_aoa_Y = np.empty((len(P_r_0), samples))                            # Should have 2D with size ( len of P_r_0 list, # of samples )
    for i in range(len(P_r_0)):
        h_scint[i]     = norm.rvs(scale=1, loc=0, size=samples)
        angle_bw_X[i]  = norm.rvs(scale=1, loc=0, size=samples)
        angle_bw_Y[i]  = norm.rvs(scale=1, loc=0, size=samples)
        angle_aoa_X[i] = norm.rvs(scale=1, loc=0, size=samples)
        angle_aoa_Y[i] = norm.rvs(scale=1, loc=0, size=samples)


    # -----------------------------------------------------------------------------------------------
    # ------------------FREQUENCY-FILTERING-&-NORMALIZATION-OF-FLUCTUATION-VECTORS-------------------
    # -----------------------------------------------------------------------------------------------
    sampling_frequency = 1 / step_size_channel_level  # 0.1 ms
    nyquist = sampling_frequency / 2

    # The frequency of all vectors is filtered and normalized, such that we end up with a standard normal distribution again, but now with a defined spectrum.
    # The turbulence vectors are filtered with a low-pass filter with a default cut-off frequency of 1 kHz.
    # The turbulence vectors are filtered with a band-pass filter with a default cut-off frequency ranges of [0.1- 0.2] Hz, [1.0- 1.1] Hz.
    h_scint     = filtering(effect='scintillation', order=frequency_filter_order, data=h_scint, f_cutoff_low=turb.freq,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_bw_X  = filtering(effect='beam wander', order=frequency_filter_order, data=angle_bw_X, f_cutoff_low=turb.freq,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_bw_Y  = filtering(effect='beam wander', order=frequency_filter_order, data=angle_bw_Y, f_cutoff_low=turb.freq,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_aoa_X = filtering(effect='angle of arrival', order=frequency_filter_order, data=angle_aoa_X, f_cutoff_low=turb.freq,
                            filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_aoa_Y = filtering(effect='angle of arrival', order=frequency_filter_order, data=angle_aoa_Y, f_cutoff_low=turb.freq,
                            filter_type='lowpass', f_sampling=sampling_frequency, plot='no')

    angle_pj_t_X = filtering(effect='TX jitter', order=frequency_filter_order, data=angle_pj_t_X, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_t_Y = filtering(effect='TX jitter', order=frequency_filter_order, data=angle_pj_t_Y, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_r_X = filtering(effect='RX jitter', order=frequency_filter_order, data=angle_pj_r_X, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_r_Y = filtering(effect='RX jitter', order=frequency_filter_order, data=angle_pj_r_Y, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')


    # -----------------------------------------------------------------------------------------------
    # -------------------------------REDISTRIBUTE-FLUCTUATION-VECTORS--------------------------------
    # -----------------------------------------------------------------------------------------------

    # Each vector is redistribution to a defined distribution. All distributions are defined in input.py
    # For scintillation vector, the default distribution is LOGNORMAL.
    # For the beam wander vectors (X- and Y-comp.) and the angle-of-arrival vectors (X- and Y-comp.), the default distribution is RAYLEIGH.
    # For the TX jitter vectors (X- and Y-comp.) and the TX jitter vectors (X- and Y-comp.), the default distribution is RAYLEIGH.

    h_scint, std_scint_dist, mean_scint_dist = turb.create_turb_distributions(data=h_scint,
                                                                              steps=samples,
                                                                              effect="scintillation")

    angle_bw_R, std_bw_dist, mean_bw_dist    = turb.create_turb_distributions(data=[angle_bw_X, angle_bw_Y],
                                                                              steps=samples,
                                                                              effect="beam wander")

    angle_aoa_R, std_aoa_dist, mean_aoa_dist = turb.create_turb_distributions(data=[angle_aoa_X, angle_aoa_Y],
                                                                              steps=samples,
                                                                              effect="angle of arrival")
    # First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
    angle_pj_t_R, std_pj_t_dist, mean_pj_t_dist = LCT.create_pointing_distributions(data=[angle_pj_t_X, angle_pj_t_Y],
                                                                                    steps=samples,
                                                                                    effect='TX jitter')
    # First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
    angle_pj_r_R, std_pj_r_dist, mean_pj_r_dist = LCT.create_pointing_distributions(data=[angle_pj_r_X, angle_pj_r_Y],
                                                                                    steps=samples,
                                                                                    effect='RX jitter')
    # filter_PSD(angle_pj_t, f_sampling=sampling_frequency, order=2)
    # -----------------------------------------------------------------------------------------------
    # ---------------------------------COMBINE-FLUCTUATION-VECTORS-----------------------------------
    # -----------------------------------------------------------------------------------------------
    # The radial angle fluctuations for TX (mech. TX jitter and beam wander) are super-positioned
    # The radial angle fluctuations for RX (mech. RX jitter and angle-of-arrival) are super-positioned
    angle_TX = angle_pj_t_R + angle_bw_R
    angle_RX = angle_pj_r_R + angle_aoa_R

    # The super-positioned vectors for TX are projected over a Gaussian beam profile to obtain the loss fraction h_TX(t)
    # The super-positioned vectors for RX are projected over a Airy disk profile to obtain the loss fraction h_RX(t)
    h_TX = h_p_gaussian(angle_TX, angle_div)
    h_RX = h_p_airy(angle_RX, D_r, focal_length)

    # The combined power vector is obtained by multiplying all three power vectors with each other (under the assumption of statistical independence between the three vectors)
    # REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
    h_tot = h_scint * h_TX * h_RX
    pdf_h_tot, cdf_h_tot, x_h_tot, std_h_tot, mean_h_tot = distribution_function(h_tot, len(ranges), min=0.0, max=2.0, steps=100)

    # For each jitter related vector, the power vector is also computed separately for analysis of the separate contributions
    h_bw = h_p_gaussian(angle_bw_R, angle_div)
    h_pj_t = h_p_gaussian(angle_pj_t_R, angle_div)
    h_pj_r = h_p_airy(angle_pj_r_R, D_r, focal_length)
    h_aoa = h_p_airy(angle_aoa_R, D_r, focal_length)

    # This is the case of perfect pointing (no platform jitter effects)
    h_tot_no_pointing_errors = h_scint * h_bw * h_aoa

    angles = [angle_TX, angle_RX]
    losses = [h_tot, h_scint, h_RX, h_TX, h_bw, h_aoa, h_pj_t, h_pj_r, h_tot_no_pointing_errors]


    print('BEAM PROPAGATION MODEL')
    print('------------------------------------------------')
    print('Signal through channel: Gaussian beam profile')
    print('Signal at RX fiber coupling: Airy disk')
    print('------------------------------------------------')
    #------------------------------------------------------------------------
    #------------------------------COMPUTING-P_r-----------------------------
    #------------------------------------------------------------------------
    # The fluctuating signal power at the receiver is computed by multiplying the static power (P_r_0) with the combined power vector (h_tot)
    # The PPB is computed with P_r and data_rate, according to (A.MAJUMDAR, 2008, EQ.3.29, H.HEMMATI, 2004, EQ.4.1-1)
    P_r = (h_tot.transpose() * P_r_0).transpose()
    P_r_no_pointing_errors = (h_tot_no_pointing_errors.transpose() * P_r_0).transpose()
    PPB = PPB_func(P_r, data_rate)
    pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r), len(P_r_0), min=-60.0, max=-10.0, steps=100)


    print('MONTE CARLO  POWER VECTOR TOOL')
    print('------------------------------------------------')
    print('3 Dynamic turbulence effects used  : Scintillation, Beam wander, Angle of arrival (AoA)')
    print('2 Platform jitter effects used     : TX platform & RX platform microvibrations')
    print('Population size sampling           : ' + str(samples))
    print('Low-pass frequency turbulence (at '+str(np.round(np.rad2deg(elevation_angles[plot_index]),0))+') : ' + str(turb.freq[plot_index])+' Hz')
    print('Low-pass frequency jitter          : ' + str(jitter_freq_lowpass)+' Hz')
    print('Band-pass frequencies jitter       : ' + str(jitter_freq1)+' Hz, '+str(jitter_freq2)+' Hz')
    print('Distribution for scintillation     : ' + dist_scintillation)
    print('Distribution for beam wander & AoA : ' + dist_beam_wander)
    print('Distribution for platform jitter   : ' + dist_pointing)

    #------------------------------------------------------------------------
    #-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
    #------------------------------------------------------------------------

    def plot_turbulence_data():
        fig_cn, axs = plt.subplots(1, 2, dpi=125)
        axs[0].set_title(f'$C_n^2$ vs  heights')
        axs[0].plot(turb.height_profiles_masked[plot_index]*1.0E-3, turb.Cn2[plot_index], label='$V_{wind}$ (rms) = ' + str(np.round(turb.windspeed_rms[plot_index],1))+' m/s')
        axs[0].set_yscale('log')
        axs[0].set_ylabel('$C_n^2$ ($m^{2/3}$)')
        axs[0].set_xlabel('heights (km)')
        # axs[0].set_xlim(turb.height_profiles_masked[plot_index,0]*1.0E-3, 20.0)
        # axs[0].set_ylim(turb.Cn2[plot_index,-1], turb.Cn2[plot_index,0])
        axs[0].legend()
        axs[0].grid()

        axs[1].set_title(f'Wind speed  vs  heights')
        # axs[1].plot(turb.windspeed, turb.heights)
        axs[1].plot(turb.height_profiles_masked[plot_index]*1.0E-3, turb.windspeed[plot_index])
        axs[1].set_xlabel('heights (km)')
        axs[1].set_ylabel('wind speed (m/s)')
        # axs[1].set_ylim(turb.windspeed[plot_index,0], turb.windspeed[plot_index,-1])
        axs[1].grid()

        # fig_scint, axs_scint = plt.subplots(3, 1, dpi=125)
        # axs_scint[0].plot(np.rad2deg(elevation_angles), turb.var_rytov, label='$\sigma_{rytov}^2$')
        # axs_scint[0].set_title(f'Scintillation variance (Intensity and Power)')
        #
        # axs_scint[0].plot(np.rad2deg(elevation_angles), turb.var_scint_gg, linestyle="--", label='$\sigma_{I}^2$')
        # axs_scint[0].set_ylabel('$\sigma_{I}^2$')
        # axs_scint[0].legend()
        #
        # axs_scint[1].plot(np.rad2deg(elevation_angles), turb.Af, linestyle="--", label='Dr = ' + str(D_r))
        # axs_scint[1].set_ylabel('Aperture averaging factor \n [$A_f = \sigma_P / \sigma_I$]')
        # axs_scint[1].set_xlabel('Elevation [deg]')
        #
        # axs_scint[2].plot(turb.var_rytov, turb.var_scint, linestyle="--", label='Dr = ' + str(D_r))
        # axs_scint[2].set_ylabel('$\sigma_{P}^2$')
        # axs_scint[2].set_xlabel('$\sigma_{I}^2$')
        # axs_scint[2].legend()
        plt.show()

    def plot_TX_losses():
        fig_TX, ax = plt.subplots(2, 1)
        ax[0].set_title('Micro-scale fluctuations at $\epsilon$='+str(np.round(np.rad2deg(elevation_angles[plot_index]),1))+'$\degree$', fontsize=13)

        pdf_bw_angle, cdf_bw, x_bw, std_bw, mean_bw = distribution_function(data=angle_bw_R, length=len(ranges), min=0, max=angle_div*2, steps=200)
        pdf_pj_t_angle, cdf_pj, x_pj, std_pj, mean_pj = distribution_function(data=angle_pj_t_R, length=1, min=0, max=angle_div*2, steps=200)
        pdf_TX_angle, cdf_TX, x_TX, std_TX, mean_TX = distribution_function(data=angle_TX, length=len(ranges), min=0, max=angle_div*2, steps=200)
        pdf_h_bw, cdf_h_bw, x, std_h_bw, mean_h_bw = distribution_function(data=h_bw, length=len(ranges), min=0, max=1, steps=200)
        pdf_h_pj_t, cdf_h_pj, x, std_h_pj, mean_h_pj = distribution_function(data=h_pj_t, length=1, min=0, max=1, steps=200)
        pdf_h_TX, cdf_h_TX, x, std_h_TX, mean_h_TX = distribution_function(data=h_TX, length=len(ranges), min=0, max=1, steps=200)

        ax[0].plot(x_bw*1e6, pdf_bw_angle[plot_index],
                   label='$\sigma^2$: '+str(np.round(std_bw[plot_index]**2*1.0E9,3))+'nrad, '+
                         '$\mu$: '+str(np.round(np.mean(angle_bw_R[plot_index])*1.0E6, 2))+'urad')

        ax[0].plot(x_pj*1e6, pdf_pj_t_angle,
                   label='$\sigma^2$: '+str(np.round(std_pj**2*1.0E9,3))+ 'nrad, '+
                         '$\mu$: '+str(np.round(np.mean(angle_pj_t_R) * 1.0E6, 2))+'urad')
        ax[0].plot(x_TX*1e6, pdf_TX_angle[plot_index], label=
                         '$\sigma^2$: '+str(np.round(std_TX[plot_index]**2*1.0E9, 3)) + 'nrad, ' +
                         '$\mu$: ' + str(np.round(np.mean(angle_TX[plot_index]) * 1.0E6, 2)) + 'urad')

        ax[1].plot(x, pdf_h_bw[plot_index], label='BW, $\mu$: '+str(np.round(np.mean(W2dB(h_bw[plot_index])), 2))+'dB')
        ax[1].plot(x, pdf_h_pj_t, label='Platform jitter, $\mu$: ' +str(np.round(np.mean(W2dB(h_pj_t)), 2))+'dB')
        ax[1].plot(x, pdf_h_TX[plot_index], label='Combined, $\mu$: '+str(np.round(np.mean(W2dB(h_TX[plot_index])), 2))+'dB')

        ax[0].set_ylabel('PDF \n'
                         '(Rayleigh)', fontsize=12)
        ax[1].set_ylabel('PDF \n'
                         '(Beta)', fontsize=12)
        ax[0].set_xlabel('Angular displacement (urad)', fontsize=12)
        ax[1].set_xlabel('Power loss (P/P0)', fontsize=12)

        ax[0].grid()
        ax[1].grid()
        ax[0].legend(fontsize=11)
        ax[1].legend(fontsize=11)
        plt.show()


    def plot_RX_losses():
        fig_RX, ax = plt.subplots(2, 1)
        ax[0].set_title('RX fluctuations at $\epsilon$='+str(np.round(np.rad2deg(elevation_angles[plot_index]),1))+'$\degree$', fontsize=13)

        pdf_aoa_angle, cdf_aoa, x_aoa, std_aoa, mean_aoa = distribution_function(data=angle_aoa_R, length=len(ranges), min=0, max=angle_div*2, steps=200)
        pdf_pj_r_angle, cdf_pj, x_pj, std, mean  = distribution_function(data=angle_pj_r_R, length=1, min=0, max=angle_div*2, steps=200)
        pdf_RX_angle, cdf_RX, x_RX, std, mean    = distribution_function(data=angle_RX, length=len(ranges), min=0, max=angle_div*2, steps=200)

        pdf_h_aoa, cdf_h_aoa, x, std_h_aoa, mean_h_aoa = distribution_function(data=h_aoa, length=len(ranges), min=0, max=1, steps=200)
        pdf_h_pj_r, cdf_pj, x, std_h_pj, mean_h_pj   = distribution_function(data=h_pj_r, length=1, min=0, max=1, steps=200)
        pdf_h_RX, cdf_RX, x, std_h_RX, mean_h_RX     = distribution_function(data=h_RX, length=len(ranges), min=0, max=1, steps=200)

        ax[0].plot(x_aoa*1e6, pdf_aoa_angle[plot_index],
                   label='$\sigma^2$: ' + str(np.round(np.var(angle_aoa_R[plot_index]) * 1.0E9, 3)) + 'nrad, ' +
                         '$\mu$: ' + str(np.round(np.mean(angle_aoa_R[plot_index]) * 1.0E6, 2)) + 'urad', linewidth=2)

        ax[0].plot(x_pj*1e6, pdf_pj_r_angle,
                   label='$\sigma^2$: ' + str(np.round(np.var(angle_pj_t_R) * 1.0E9, 3)) + 'nrad, ' +
                         '$\mu$: ' + str(np.round(np.mean(angle_pj_t_R) * 1.0E6, 2)) + 'urad', linewidth=2)
        ax[0].plot(x_RX*1e6, pdf_RX_angle[plot_index], label='Combined, '
                         '$\sigma^2$: '+str(np.round(np.var(angle_RX) * 1.0E9, 3)) +
                         '$\mu$: ' + str(np.round(np.mean(angle_RX) * 1.0E6, 2)) + 'urad', linewidth=2)

        ax[1].plot(x, pdf_h_aoa[plot_index], label='AoA, $\mu$: ' + str(np.round(np.mean(W2dB(h_aoa[plot_index])), 2))+'dB', linewidth=2)
        ax[1].plot(x, pdf_h_pj_r, label='Platform jitter, $\mu$: ' + str(np.round(np.mean(W2dB(h_pj_r)), 2))+'dB', linewidth=2)
        ax[1].plot(x, pdf_h_RX[plot_index], label='Combined, $\mu$: ' + str(np.round(np.mean(W2dB(h_RX[plot_index])), 2))+'dB', linewidth=2)

        ax[0].set_ylabel('probability density \n'
                         '(Rayleigh)', fontsize=12)
        ax[1].set_ylabel('probability density \n'
                         '(Beta)', fontsize=12)
        ax[0].set_xlabel('Angular displacement (urad)', fontsize=12)
        ax[1].set_xlabel('Power loss (P/P0)', fontsize=12)

        ax[0].grid()
        ax[1].grid()
        ax[0].legend(fontsize=11)
        ax[1].legend(fontsize=11)
        plt.show()
    def plot_all_losses_time_series():

        # Plot time series losses
        fig_losses_tot, ax_losses_tot = plt.subplots(4, 1)
        ax_losses_tot[0].set_title('Channel level Power Vectors (Scint, TX & RX) for 1 timestep ($\epsilon$ = ' + str(
            np.rad2deg(np.round(elevation_angles[plot_index], 2)))+ ')')
        ax_losses_tot[0].plot(t[10:], W2dB(h_tot[plot_index])[10:], linewidth='0.5')
        ax_losses_tot[0].set_ylabel('h tot [dB]')
        ax_losses_tot[1].plot(t[10:], W2dB(h_scint[plot_index])[10:], linewidth='0.5')
        ax_losses_tot[1].set_ylabel('h scint [dB]')
        ax_losses_tot[2].plot(t[10:], W2dB(h_TX[plot_index])[10:], linewidth='0.5')
        ax_losses_tot[2].set_ylabel('h TX [dB] \n (BW + TX jitter)')
        ax_losses_tot[3].plot(t[10:], W2dB(h_RX[plot_index])[10:], linewidth='0.5')
        ax_losses_tot[3].set_ylabel('h RX [dB]  \n (AoA + RX jitter)')
        ax_losses_tot[3].set_xlabel('Time [s]')

        ax_losses_tot[0].legend()
        ax_losses_tot[1].legend()
        ax_losses_tot[2].legend()
        ax_losses_tot[3].legend()

        fig, ax = plt.subplots(1,1)
        ax.set_title('Combined power vector at $\epsilon$ = ' + str(
            np.round(np.rad2deg(elevation_angles[plot_index]), 2)) + 'deg', fontsize=15)
        ax.plot(t, W2dB(h_tot[plot_index]), label='std (1 rms): mean: ' + str(
            np.round(W2dB(np.mean(h_tot[plot_index])), 2)), linewidth=2)
        ax.set_ylabel('h tot [dB]')
        ax.legend(fontsize=15)
        ax.set_ylabel('h tot [dB]')
        ax.set_xlabel('Time [s]')

        plt.show()
    def plot_all_losses_pdf():
        # Plot PDF losses
        pdf_h_scint,x  = pdf_function(h_scint, len(ranges), min=0.0, max=2.0, steps=1000)
        pdf_h_bw,   x  = pdf_function(h_bw,    len(ranges), min=0.0, max=2.0, steps=1000)
        pdf_h_aoa,  x  = pdf_function(h_aoa,   len(ranges), min=0.0, max=2.0, steps=1000)
        pdf_h_pj_t, x  = pdf_function(h_pj_t,  1,           min=0.0, max=2.0, steps=1000)
        pdf_h_pj_r, x  = pdf_function(h_pj_r,  1,           min=0.0, max=2.0, steps=1000)

        pdf_h_TX,   x  = pdf_function(h_TX,    len(ranges), min=0.0, max=2.0, steps=1000)
        pdf_h_RX,   x  = pdf_function(h_RX,    len(ranges), min=0.0, max=2.0, steps=1000)

        fig_losses_pdf, ax = plt.subplots(2, 1)
        ax[0].set_title('Micro-scale losses \n '
                        'distributions of all effects at $\epsilon$='+
                        str(np.round(np.rad2deg(elevation_angles[plot_index]),1))+'$\degree$', fontsize=15)
        ax[0].plot(x_h_tot, pdf_h_tot[plot_index], label='combined, $\mu$: ' + str(np.round(np.mean(W2dB(h_tot[plot_index])), 2))+'dB')
        ax[0].plot(x, pdf_h_scint[plot_index],     label='Scintillation, $\mu$: ' + str(np.round(np.mean(W2dB(h_scint[plot_index])), 2))+'dB')
        ax[0].plot(x, pdf_h_bw[plot_index],        label='Beam wander, $\mu$: ' + str(np.round(np.mean(W2dB(h_bw[plot_index])), 2))+'dB')
        ax[0].plot(x, pdf_h_aoa[plot_index],       label='AoA, $\mu$: ' + str(np.round(np.mean(W2dB(h_aoa[plot_index])), 2))+'dB')
        ax[0].plot(x, pdf_h_pj_t,                  label='TX jitter, $\mu$: ' + str(np.round(np.mean(W2dB(h_pj_t)), 2))+'dB')
        ax[0].plot(x, pdf_h_pj_r,                  label='RX jitter, $\mu$: ' + str(np.round(np.mean(W2dB(h_pj_r)), 2))+'dB')

        ax[1].plot(x_P_r, pdf_P_r[plot_index], label='$P_{RX}$, mean: ' + str(np.round(np.mean(W2dBm(P_r[plot_index])), 3))+'dBm')

        ax[0].set_xlabel('Power loss fraction (P/P0)', fontsize=10)
        ax[1].set_xlabel('Power at RX (dBm)', fontsize=10)
        ax[0].set_ylabel('Probability density', fontsize=10)
        ax[1].set_ylabel('Probability density', fontsize=10)

        ax[0].legend(fontsize=10)
        ax[0].grid()
        ax[1].legend(fontsize=10)
        ax[1].grid()

        # fig_losses_pdf1, ax1 = plt.subplots(1, 1)
        # ax1.set_title('Micro-scale losses \n '
        #                 'distributions of TX jitter, RX jitter, scintillation and combined at $\epsilon$=' + str(
        #     np.round(np.rad2deg(elevation_angles[plot_index]),1)) + '$\degree$', fontsize=15)
        # ax1.plot(x_h_tot, pdf_h_tot[plot_index], label='Combined loss, $\mu$: ' + str(
        #     np.round(np.mean(W2dB(h_tot[plot_index])), 2)) + 'dB)')
        # ax1.plot(x, pdf_h_scint[plot_index], label='Scintillation, $\mu$: ' + str(
        #     np.round(np.mean(W2dB(h_scint[plot_index])), 2)) + 'dB)')
        #
        # ax1.plot(x, pdf_h_TX[plot_index], label='Combined TX jitter, $\mu$: ' + str(
        #     np.round(np.mean(W2dB(h_TX[plot_index])), 2)) + 'dB)')
        # ax1.plot(x, pdf_h_RX[plot_index], label='Combined RX jitter, $\mu$: ' + str(
        #     np.round(np.mean(W2dB(h_RX[plot_index])), 2)) + 'dB)')
        #
        # ax1.set_xlabel('Power loss fraction (P/P0)', fontsize=15)
        # ax1.set_ylabel('Probability density', fontsize=15)
        #
        # ax1.legend(fontsize=10)
        # ax1.grid()

        print('mean proof')
        mean_TX    = np.mean(h_TX[plot_index])
        mean_RX    = np.mean(h_RX[plot_index])
        mean_scint = np.mean(h_scint[plot_index])
        mean_tot   = np.mean(h_tot[plot_index])
        print(mean_TX, mean_RX, mean_scint,  mean_tot)
        print(mean_TX * mean_RX * mean_scint, mean_tot)
        print()
        print('variance proof')
        var_TX    = np.var(h_TX[plot_index])
        var_RX    = np.var(h_RX[plot_index])
        var_scint = np.var(h_scint[plot_index])
        var_tot   = np.var(h_tot[plot_index])
        print(var_TX, var_RX, var_scint, var_tot)
        print((mean_TX**2 + var_TX) * (mean_RX**2 + var_RX) * (mean_scint**2 + var_scint) - mean_TX**2 * mean_RX**2 * mean_scint**2, var_tot)
        plt.show()

    def plot_pointing_TX():
        fig_point, ax = plt.subplots(3, 1)
        ax[0].plot(t, LCT.angle_pe_t_X*1.0E6, label='X-component (gaussian)')
        ax[0].plot(t, LCT.angle_pe_t_Y*1.0E6, label='Y-component (gaussian)')
        ax[0].plot(t, LCT.angle_pe_t_R*1.0E6, label='R-component (rayleigh)')
        ax[0].set_title(f'Mechanical TX jitter angle')
        ax[0].set_ylabel('Mechanical TX \n jitter angle [$\mu$rad]')

        ax[1].plot(t, turb.angle_bw_X[plot_index] * 1.0E6, label='X-component (gaussian)')
        ax[1].plot(t, turb.angle_bw_Y[plot_index] * 1.0E6, label='Y-component (gaussian)')
        ax[1].plot(t, turb.angle_bw_R[plot_index] * 1.0E6, label='R-component (rayleigh)')
        ax[1].set_ylabel('Beam wander \n jitter angle [$\mu$rad]')

        ax[2].plot(t, angle_TX[plot_index] * 1.0E6)
        ax[2].set_ylabel('Combined TX \n jitter angle [$\mu$rad]')
        ax[2].set_xlabel('Time [s]')

        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        ax[2].legend()
        ax[2].grid()
        plt.show()

    def plot_airy_disk():
        angle = np.linspace(1.0E-6, 100.0E-6, 1000)
        I_norm_gaussian_approx, I_norm_airy, P_norm_airy, = h_p_airy(angle, D_r, focal_length)
        r = focal_length * np.sin(angle)
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Airy disk, focal length=' + str(focal_length) + 'm, Dr=' + str(np.round(D_r, 3)) + 'm')
        ax.plot(angle * 1.0E6, I_norm_airy, label='Airy disk, I(r)/$I_0$')
        # ax.plot(angle * 1.0E6, I_norm_gaussian_approx, label='Gaussian approx. I(r)/$I_0$')
        ax.plot(angle * 1.0E6, P_norm_airy, label='Airy disk, , P(r)/$P_0$')
        ax.set_xlabel('Angular displacement ($\mu$rad)', fontsize=12)
        ax.set_ylabel('Normalized intensity at =0.0 $\mu$m', fontsize=12)
        ax.legend(fontsize=15)
        ax.grid()
        plt.show()

    def plot_temporal_behaviour(effect0='$h_{pj,TX}$ (platform)', effect1='$h_{bw}$', effect2='$h_{pj,TX}$ (combined)',
                                effect3='$h_{pj,RX}$ (combined)', effect4='$h_{scint}$', effect_tot='$h_{total}$'):

        fig_psd, ax   = plt.subplots(1, 2)
        # Plot PSD over frequency domain
        f0, psd0 = welch(h_pj_t,              sampling_frequency, nperseg=1024)
        f1, psd1 = welch(h_bw[plot_index],    sampling_frequency, nperseg=1024)
        f2, psd2 = welch(h_TX[plot_index],    sampling_frequency, nperseg=1024)
        f3, psd3 = welch(h_RX[plot_index],    sampling_frequency, nperseg=1024)
        f4, psd4 = welch(h_scint[plot_index], sampling_frequency, nperseg=1024)
        f5, psd5 = welch(h_tot[plot_index],   sampling_frequency, nperseg=1024)

        ax[0].semilogy(f0, W2dB(psd0), label=effect0)
        ax[0].semilogy(f1, W2dB(psd1), label=effect1)
        ax[0].semilogy(f2, W2dB(psd2), label=effect2)

        ax[1].semilogy(f2, W2dB(psd2), label=effect2)
        ax[1].semilogy(f3, W2dB(psd3), label=effect3)
        ax[1].semilogy(f4, W2dB(psd4), label=effect4)
        ax[1].semilogy(f5, W2dB(psd5), label=effect_tot)

        ax[0].set_ylabel('PSD [dBW/Hz]')
        ax[0].set_yscale('linear')
        ax[0].set_ylim(-100.0, 0.0)
        ax[0].set_xscale('log')
        ax[0].set_xlim(1.0E0, 1.2E3)
        ax[0].set_xlabel('frequency [Hz]')

        ax[1].set_yscale('linear')
        ax[1].set_ylim(-100.0, 0.0)
        ax[1].set_xscale('log')
        ax[1].set_xlim(1.0E0, 1.2E3)
        ax[1].set_xlabel('frequency [Hz]')

        ax[0].grid()
        ax[0].legend(fontsize = 13, loc='lower left')
        ax[1].grid()
        ax[1].legend(fontsize = 13, loc='lower left')


        fig_psd1, ax1 = plt.subplots(4, 1)
        # for i in plot_indices:
        #     f3, psd3 = welch(h_scint[i], sampling_frequency, nperseg=1024)
        #     ax1.semilogy(f3, W2dB(psd3), label='Scint, $\epsilon$='   +str(np.round(np.rad2deg(elevation_angles[i]), 0)) + '$\degree$, turb. freq.='+str(np.round(turb.freq[i],0))+'Hz')

        for i in plot_indices:
            f4, psd4 = welch(P_r[i],   sampling_frequency, nperseg=1024)
            ax1[0].semilogy(f4, W2dB(psd4), label='$\epsilon$='+str(np.round(np.rad2deg(elevation_angles[i]), 0)) + '$\degree$, turb. freq.='+str(np.round(turb.freq[i],0))+'Hz')

            f4, psd4 = welch(h_tot[i], sampling_frequency, nperseg=1024)
            ax1[1].semilogy(f4, W2dB(psd4))

            f4, psd4 = welch(h_scint[i], sampling_frequency, nperseg=1024)
            ax1[2].semilogy(f4, W2dB(psd4))

        f4, psd4 = welch(h_pj_t, sampling_frequency, nperseg=1024)
        ax1[3].semilogy(f4, W2dB(psd4))

        ax1[0].legend(fontsize=10)

        ax1[0].set_ylabel('$P_{RX}$ PSD [dBW/Hz]')
        ax1[0].set_yscale('linear')
        ax1[0].set_ylim(-200.0, -100.0)
        ax1[0].set_xscale('log')
        ax1[0].set_xlim(1.0E0, 1.2E3)
        ax1[0].set_xlabel('frequency [Hz]')
        ax1[0].grid()

        ax1[1].set_ylabel('$h_{tot}$ PSD [dBW/Hz]')
        ax1[1].set_yscale('linear')
        ax1[1].set_ylim(-100.0, 0.0)
        ax1[1].set_xscale('log')
        ax1[1].set_xlim(1.0E0, 1.2E3)
        ax1[1].set_xlabel('frequency [Hz]')
        ax1[1].grid()

        ax1[2].set_ylabel('$h_{scint}$ PSD [dBW/Hz]')
        ax1[2].set_yscale('linear')
        ax1[2].set_ylim(-100.0, 0.0)
        ax1[2].set_xscale('log')
        ax1[2].set_xlim(1.0E0, 1.2E3)
        ax1[2].set_xlabel('frequency [Hz]')
        ax1[2].grid()

        ax1[3].set_ylabel('$h_{pj,t}$ PSD [dBW/Hz]')
        ax1[3].set_yscale('linear')
        ax1[3].set_ylim(-100.0, 0.0)
        ax1[3].set_xscale('log')
        ax1[3].set_xlim(1.0E0, 1.2E3)
        ax1[3].set_xlabel('frequency [Hz]')
        ax1[3].grid()

        plt.show()

        fig, ax = plt.subplots(1,1)
        for i in plot_indices:
            f4, psd4 = welch(h_scint[i], sampling_frequency, nperseg=1024)
            ax.semilogy(f4, W2dB(psd4), label='$\epsilon$='+str(np.round(np.rad2deg(elevation_angles[i]), 0)) + '$\degree$, turb. freq.='+str(np.round(turb.freq[i],0))+'Hz')
            ax.set_ylabel('$h_{scint}$ PSD [dBW/Hz]')
            ax.set_yscale('linear')
            ax.set_ylim(-100.0, 0.0)
            ax.set_xscale('log')
            ax.set_xlim(1.0E0, 1.2E3)
            ax.set_xlabel('frequency [Hz]')
            ax.grid()
        plt.show()



    # plot_turbulence_data()
    # link_budget.plot(indices=plot_indices, P_r=P_r, elevation=elevation_angles, displacements=angle_TX*ranges[:,None], type="gaussian beam profile")
    # plot_TX_losses()
    # plot_RX_losses()
    # plot_all_losses_time_series()
    # plot_all_losses_pdf()
    # plot_temporal_behaviour()


    return P_r, P_r_no_pointing_errors, PPB, elevation_angles, losses, angles