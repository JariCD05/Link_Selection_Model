import time

import numpy as np
import csv
import sqlite3


from scipy.stats import norm, genexpon, lognorm, rice, rayleigh, chisquare, rv_histogram, beta
from input import *
from helper_functions import *
from Atmosphere import turbulence, attenuation
from LCT import terminal_properties
from Link_budget import link_budget
from PDF import dist
# from mission_level import time, geometrical_output
# elevation = np.rad2deg(geometrical_output['elevation'])

def channel_level(LCT,
                  turb,
                  plot_index: int,
                  ranges: np.array,
                  P_r_0: np.array,
                  elevation_angles: np.array,
                  zenith_angles: np.array,
                  samples):
    print('')
    print('----------CHANNEL LEVEL-(Monte-Carlo-simulations-of-signal-fluctuations)-----------------')
    print('')
    cur_time = time.process_time()

    #------------------------------------------------------------------------
    #-------------------------------TURBULENCE-------------------------------
    #------------------------------------------------------------------------
    # Here, the turbulence parameters are updated with the smaller vectors that are defined by DIVIDE_INDEX
    turb.ranges = ranges
    turb.r0_func(zenith_angles)
    turb.var_rytov_func(zenith_angles)
    turb.var_scint_func(zenith_angles)
    turb.var_bw_func(zenith_angles)
    turb.var_aoa_func(zenith_angles)
    turb.beam_spread()
    turb.Strehl_ratio_func()

    # turb.print(index = plot_index, elevation=np.rad2deg(elevation_angles), ranges=ranges)
    # ------------------------------------------------------------------------
    # ------------------------INITIALIZING-ALL-VECTORS------------------------
    # ----------------------START-MONTE-CARLO-SIMULATIONS---------------------
    # ------------------------------------------------------------------------
    # For each fluctuating variable, a vector is initialized with a standard normal distribution (std=1, mean=0)
    # For all jitter related vectors (beam wander, angle-of-arrival, mechanical TX jitter, mechanical RX jitter), two variables are initialized for both X- and Y-components

    angle_pj_t_X = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_t_Y = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_r_X = norm.rvs(scale=1, loc=0, size=samples)
    angle_pj_r_Y = norm.rvs(scale=1, loc=0, size=samples)

    # The turbulence vectors (beam wander and angle-of-arrival) differ for each ranges.
    # Hence they are initialized seperately for each ranges and stored in one 2D array

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
    # -----------------------FILTERING-&-NORMALIZATION-OF-FLUCTUATION-VECTORS------------------------
    # -----------------------------------------------------------------------------------------------

    # All vectors are filtered and normalized, such that we end up with a standard normal distribution again, but now with a defined spectrum.
    # The turbulence vectors are filtered with a low-pass filter with a default cut-off frequency of 1 kHz.
    # The turbulence vectors are filtered with a band-pass filter with a default cut-off frequency ranges of [0.1- 0.2] Hz, [1.0- 1.1] Hz.
    order = 2
    h_scint     = filtering(effect='scintillation', order=order, data=h_scint, f_cutoff_low=turbulence_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_bw_X  = filtering(effect='beam wander', order=order, data=angle_bw_X, f_cutoff_low=turbulence_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_bw_Y  = filtering(effect='beam wander', order=order, data=angle_bw_Y, f_cutoff_low=turbulence_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_aoa_X = filtering(effect='angle of arrival', order=order, data=angle_aoa_X, f_cutoff_low=turbulence_frequency,
                            filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_aoa_Y = filtering(effect='angle of arrival', order=order, data=angle_aoa_Y, f_cutoff_low=turbulence_frequency,
                            filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
    angle_pj_t_X = filtering(effect='TX jitter', order=order, data=angle_pj_t_X, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_t_Y = filtering(effect='TX jitter', order=order, data=angle_pj_t_Y, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_r_X = filtering(effect='RX jitter', order=order, data=angle_pj_r_X, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')
    angle_pj_r_Y = filtering(effect='RX jitter', order=order, data=angle_pj_r_Y, f_cutoff_low=jitter_freq_lowpass, f_cutoff_band=jitter_freq1, f_cutoff_band1=jitter_freq2,
                        filter_type='multi', f_sampling=sampling_frequency, plot='no')

    # -----------------------------------------------------------------------------------------------
    # -------------------------------REDISTRIBUTE-FLUCTUATION-VECTORS--------------------------------
    # -----------------------------------------------------------------------------------------------

    # Each vector is redistribution to a defined distribution. All distributions are defined in input.py
    # For scintillation vector, the default distribution is LOGNORMAL.
    # For the beam wander vectors (X- and Y-comp.) and the angle-of-arrival vectors (X- and Y-comp.), the default distribution is RAYLEIGH.
    # For the TX jitter vectors (X- and Y-comp.) and the TX jitter vectors (X- and Y-comp.), the default distribution is RICE.

    # The standard normal distribution of scintillation samples are converted to
    h_scint = turb.create_turb_distributions(data=h_scint,
                                              steps=samples,
                                              effect="scintillation")

    angle_bw_R = turb.create_turb_distributions(data=[angle_bw_X, angle_bw_Y],
                                               steps=samples,
                                               effect="beam wander")
    angle_bw_X = angle_bw_Y = []

    angle_aoa_R = turb.create_turb_distributions(data=[angle_aoa_X, angle_aoa_Y],
                                                steps=samples,
                                                effect="angle of arrival")
    angle_aoa_X = angle_aoa_Y = []

    # First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
    angle_pj_t_R = LCT.create_pointing_distributions(data=[angle_pj_t_X, angle_pj_t_Y],
                                                  steps=samples,
                                                  effect='TX jitter')
    angle_pj_t_X = angle_pj_t_Y = []

    # First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
    angle_pj_r_R = LCT.create_pointing_distributions(data=[angle_pj_r_X, angle_pj_r_Y],
                                                  steps=samples,
                                                  effect='RX jitter')
    angle_pj_r_X = angle_pj_r_Y = []

    # filter_PSD(angle_pj_t, f_sampling=sampling_frequency, order=2)

    # -----------------------------------------------------------------------------------------------
    # ---------------------------------COMBINE-FLUCTUATION-VECTORS-----------------------------------
    # -----------------------------------------------------------------------------------------------
    # The radial angle fluctuations for TX (mech. TX jitter and beam wander) are super-positioned
    # The radial angle fluctuations for RX (mech. RX jitter and angle-of-arrival) are super-positioned
    angle_TX = angle_pj_t_R + angle_bw_R
    # --------------------------
    mean_TX = np.mean(angle_TX, axis=1)
    var_TX = np.sum((angle_TX - mean_TX[:, None])**2, axis=1) / samples
    # --------------------------
    angle_RX = angle_pj_r_R + angle_aoa_R
    # --------------------------
    mean_RX = np.mean(angle_RX, axis=1)
    var_RX = np.sum((angle_RX - mean_RX[:, None])**2, axis=1) / samples
    # --------------------------
    # The super-positioned vectors for TX are projected over a Gaussian beam profile to obtain the loss fraction h_TX(t)
    # The super-positioned vectors for RX are projected over a Airy disk profile to obtain the loss fraction h_RX(t)
    h_TX, h_TX_I = h_p_gaussian(angle_TX, ranges, turb.w_ST)
    h_RX = h_p_airy(angle_RX, D_r, focal_length)
    # The combined power vector is obtained by multiplying all three power vectors with each other (under the assumption of statistical independence between the three vectors)
    # REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
    h_tot = h_scint * h_TX * h_RX
    x_h_tot = np.linspace(0, 2, 10000)
    pdf_h_tot = pdf_function(h_tot, len(ranges), x_h_tot)

    # For each jitter related vector, the power vector is also computed separately for analysis of the separate contributions
    # h_bw = gaussian_profile(turbulence_dim1.angle_bw_R, angle_div)                    # Should have 2D with size ( len of ranges list, # of samples )
    # h_pj_t = gaussian_profile(LCT.angle_pe_t_R, angle_div)                            # Should have 2D with size ( len of ranges list, # of samples )
    # h_pj_r = airy_profile(LCT.angle_pe_r_R, D_r, focal_length)                        # Should have 1D with size ( # of samples )
    # h_aoa = airy_profile( turbulence_dim1.angle_aoa_R, D_r, focal_length)             # Should have 2D with size ( len of ranges list, # of samples )

    #------------------------------------------------------------------------
    #------------------------------COMPUTING-P_r-----------------------------
    #------------------------------------------------------------------------
    # The fluctuating signal power at the receiver is computed by multiplying the static power (P_r_0) with the combined power vector (h_tot)
    # The PPB is computed with P_r and data_rate, according to (A.MAJUMDAR, 2008, EQ.3.29, H.HEMMATI, 2004, EQ.4.1-1)
    P_r = (h_tot.transpose() * P_r_0).transpose()
    PPB = PPB_func(P_r, data_rate)

    x_P_r = np.linspace(W2dBm(P_r.min()), W2dBm(P_r.max()), 10000)
    pdf_P_r = pdf_function(W2dBm(P_r), len(P_r_0), x_P_r)

    #------------------------------------------------------------------------
    #-----------------------------PDF-VERIFICATION---------------------------
    #------------------------------------------------------------------------

    # Verification of the sampled data (comparison with theoretical PDFs)
    test_index = [0, 2, 3]
    # turbulence_dim1.test_PDF(effect = "scintillation", index = test_index)
    # turbulence_dim1.test_PDF(effect = 'beam wander', index = test_index)
    # turbulence_dim1.test_PDF(effect = 'angle of arrival', index = test_index)
    # LCT.test_PDF(effect = 'TX jitter', index = test_index)
    # LCT.test_PDF(effect = 'RX jitter', index = test_index)

    # test_index = 2
    # fig_test_pdf_TX, ax = plt.subplots(1, 1)
    # x_TX, pdf_TX = dist.beta_pdf(sigma=np.std(h_TX[test_index]), steps=samples)
    # dist.plot(ax=ax,
    #           sigma=np.std(h_TX[test_index]),
    #           mean=np.mean(h_TX[test_index]),
    #           x=x_TX,
    #           pdf=pdf_TX,
    #           data=h_TX[test_index],
    #           index=test_index,
    #           effect="combined",
    #           name="beta")

    #------------------------------------------------------------------------
    #-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
    #------------------------------------------------------------------------

    def plot_turbulence_data():
        fig_cn, axs = plt.subplots(1, 2, dpi=125)
        axs[0].set_title(f'$C_n^2$ vs  heights')
        axs[0].plot(turb.heights, turb.Cn, label='V wind (rms) = ' + str(turb.windspeed_rms))
        axs[0].set_yscale('log')
        axs[0].set_xscale('log')
        axs[0].set_ylabel('$C_n^2$ (m)')
        axs[0].set_xlabel('heights (m)')
        axs[0].set_ylim(1.0E-12, 1.0E-20)
        axs[0].set_xlim(h_AC, 1.0E5)
        axs[0].legend()
        axs[0].grid()

        axs[1].set_title(f'Wind speed  vs  heights')
        # axs[1].plot(turb.windspeed, turb.heights)
        axs[1].plot(turb.windspeed, turb.heights)
        axs[1].set_ylabel('heights (m)')
        axs[1].set_xlabel('wind speed (m/s)')
        axs[1].set_ylim(h_AC, 1.0E5)
        axs[1].legend()
        axs[1].grid()

        fig_scint, axs_scint = plt.subplots(3, 1, dpi=125)
        axs_scint[0].plot(np.rad2deg(elevation_angles), turb.var_rytov, label='$\sigma_{rytov}^2$')
        axs_scint[0].set_title(f'Scintillation variance (Intensity and Power)')

        axs_scint[0].plot(np.rad2deg(elevation_angles), turb.var_scint_gg, linestyle="--", label='$\sigma_{I}^2$')
        axs_scint[0].set_ylabel('$\sigma_{I}^2$')
        axs_scint[0].legend()

        axs_scint[1].plot(np.rad2deg(elevation_angles), turb.Af, linestyle="--", label='Dr = ' + str(D_r))
        axs_scint[1].set_ylabel('Aperture averaging factor \n [$A_f = \sigma_P / \sigma_I$]')
        axs_scint[1].set_xlabel('Elevation [deg]')

        axs_scint[2].plot(turb.var_rytov, turb.var_scint, linestyle="--", label='Dr = ' + str(D_r))
        axs_scint[2].set_ylabel('$\sigma_{P}^2$')
        axs_scint[2].set_xlabel('$\sigma_{I}^2$')
        axs_scint[2].legend()

        plt.show()
    def plot_channel_output():
        fig, ax = plt.subplots(3, 1)
        ax[0].set_title('Normalized power vector $h_{tot}$, static power $P_{r0}$ and dynamic power $P_r=P_{r0} h_{tot}$ \n'
                        'for elevation angle $\epsilon$='+str(np.round(np.rad2deg(elevation_angles[plot_index]), 2))+'deg',
                        fontsize=15)
        ax[0].set_ylabel('Normalized intensity [-]')
        # ax[0].hist(h_tot[plot_index], density=True, bins=100, range=(0, 2.0))
        ax[0].plot(x_h_tot, pdf_h_tot[plot_index], label='PDF of total dynamic loss $h_{tot}$')
        ax[0].set_ylabel('Probability density')
        ax[0].set_xlabel('$h_{tot}$ [-]')

        ax[1].plot(t, h_tot[plot_index], label='Time series of total dynamic loss $h_{tot}$ \n' 
                                                         'mean: ' + str(np.round(W2dB(np.mean(h_tot[plot_index])), 2)))
        ax[1].set_ylabel('$h_{tot}$ [dBm]')
        ax[2].plot(t[10:], W2dBm(P_r[plot_index,10:]),
                   label='Dynamic power $P_{r}$, mean: ' + str(np.round(W2dBm(np.mean(P_r[plot_index])), 2)) + ' dBm')
        ax[2].plot(t[10:], (np.ones(len(t))*W2dBm(P_r_0[plot_index]))[10:], label='Static power $P_{r0}$, mean: ' + str(np.round(W2dBm(np.mean(P_r_0[plot_index])),2)) + ' dBm',
                   linewidth=2.0)
        ax[2].set_ylabel('P at RX [dBm]')
        ax[2].set_xlabel('Time [s]')

        ax[0].legend(fontsize=15)
        ax[1].legend(fontsize=15)
        ax[2].legend(fontsize=15)
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        plt.show()
    def plot_TX_losses_time_series():
        fig_TX, ax = plt.subplots(4, 1)
        ax[0].plot(t, turb.angle_bw_R[plot_index] * 1.0E6, label='std (1 rms): '+str(np.round(np.var(turb.angle_bw_R[plot_index])*1.0E9,5))+'nrad, '+'mean: '+str(np.round(np.mean(turb.angle_bw_R[plot_index]*1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[0].set_ylabel('angle bw [urad]')
        ax[1].plot(t, LCT.angle_pe_t_R * 1.0E6, label='std (1 rms): '+str(np.round(np.var(LCT.angle_pe_t_R)*1.0E9,5))+ 'nrad, '+'mean: '+str(np.round(np.mean(LCT.angle_pe_t_R * 1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[1].set_ylabel('angle pj t [urad]')
        ax[2].plot(t, angle_TX[plot_index] * 1.0E6, label='std (1 rms): '+str(np.round(np.var(angle_TX[plot_index])*1.0E9,5))+'nrad, '+'mean: '+str(np.round(np.mean(angle_TX[plot_index]*1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[2].set_ylabel('angle TX combined [urad]')

        ax[3].plot(t, h_TX[plot_index], label='Power, mean: '+str(np.round(np.mean(h_TX[plot_index]), 2)),
                   linewidth='0.5')
        ax[3].plot(t, h_TX_I[plot_index], label='Intensity, mean: ' + str(np.round(np.mean(h_TX_I[plot_index]), 2)),
                   linewidth='0.5')

        ax[2].set_ylabel('TX loss combined [-]')
        ax[3].set_xlabel('Time (s)')

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        plt.show()
    def plot_RX_losses_time_series():
        fig_RX, ax = plt.subplots(3, 1)
        ax[0].plot(t, turb.angle_aoa_R[plot_index] * 1.0E6, label='std (1 rms): '+str(np.round(np.var(turb.angle_aoa_R[plot_index])*1.0E9,5))+'nrad, '+'mean: '+str(np.round(np.mean(turb.angle_aoa_R[plot_index]*1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[0].set_ylabel('angle bw [urad]')
        ax[1].plot(t, LCT.angle_pe_r_R * 1.0E6, label='std (1 rms): '+str(np.round(np.var(LCT.angle_pe_r_R)*1.0E9,5))+ 'nrad, '+'mean: '+str(np.round(np.mean(LCT.angle_pe_r_R * 1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[1].set_ylabel('angle pj t [urad]')
        ax[2].plot(t, angle_RX[plot_index] * 1.0E6, label='std (1 rms): '+str(np.round(np.var(angle_RX[plot_index])*1.0E9,5))+'nrad, '+'mean: '+str(np.round(np.mean(angle_RX[plot_index]*1.0E6), 2))+'urad',
                   linewidth='0.5')
        ax[2].set_ylabel('angle TX combined [urad]')
        ax[2].set_xlabel('Time (s)')

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()
    def plot_all_losses_time_series():
        # plot time series radials
        # fig_losses, ax_losses = plt.subplots(5, 1)
        # ax_losses[0].set_title('Dimension 1 Fluctuating Vectors for $\epsilon$ = ' + str(
        #     np.rad2deg(np.round(elevation_angles[plot_index], 2))))
        # ax_losses[0].plot(t, W2dB(h_scint[plot_index]), label='std (1 rms): ' + str(
        #     np.round(turbulence_dim1.std_scint[plot_index], 2)) + ', ' + 'mean: ' + str(
        #     np.round(np.mean(h_scint[plot_index]), 2)),
        #                   linewidth='0.5')
        # ax_losses[0].set_ylabel('h scint [dB]')
        # ax_losses[1].plot(t, angle_bw[plot_index] * 1.0E6, label='std (1 rms): ' + str(
        #     np.round(turbulence_dim1.std_bw[plot_index] * 1.0E6, 2)) + ' urad, ' + 'mean: ' + str(
        #     np.round(np.mean(angle_bw[plot_index] * 1.0E6), 2)) + ' urad',
        #                   linewidth='0.5')
        # ax_losses[1].set_ylabel('angle bw [urad]')
        # ax_losses[2].plot(t, angle_pj_t * 1.0E6,
        #                   label='std (1 rms): ' + str(np.round(LCT.std_pj_t * 1.0E6, 2)) + ' urad, ' + 'mean: ' + str(
        #                       np.round(np.mean(angle_pj_t * 1.0E6), 2)) + ' urad',
        #                   linewidth='0.5')
        # ax_losses[2].set_ylabel('angle pj t [urad]')
        # ax_losses[3].plot(t, angle_pj_r * 1.0E6,
        #                   label='std (1 rms): ' + str(np.round(LCT.std_pj_r * 1.0E6, 2)) + ' urad, ' + 'mean: ' + str(
        #                       np.round(np.mean(angle_pj_r * 1.0E6), 2)) + ' urad',
        #                   linewidth='0.5')
        # ax_losses[3].set_ylabel('angle pj r [urad]')
        # ax_losses[4].plot(t, angle_aoa[plot_index] * 1.0E6, label='std (1 rms): ' + str(
        #     np.round(turbulence_dim1.std_aoa[plot_index] * 1.0E6, 2)) + ' urad, ' + 'mean: ' + str(
        #     np.round(np.mean(angle_aoa[plot_index] * 1.0E6), 2)) + ' urad',
        #                   linewidth='0.5')
        # ax_losses[4].set_ylabel('angle AoA [urad]')
        # ax_losses[4].set_xlabel('Time [s]')
        #
        # ax_losses[0].legend()
        # ax_losses[1].legend()
        # ax_losses[2].legend()
        # ax_losses[3].legend()
        # ax_losses[4].legend()

        # Plot time series losses
        fig_losses_tot, ax_losses_tot = plt.subplots(4, 1)
        ax_losses_tot[0].set_title('Channel level Power Vectors (Scint, TX & RX) for 1 timestep ($\epsilon$ = ' + str(
            np.rad2deg(np.round(elevation_angles[plot_index], 2)))+ ')')
        ax_losses_tot[0].plot(t[10:], W2dB(h_tot[plot_index])[10:], label='std (1 rms): ' + str(
            np.round(np.var(h_tot[plot_index]), 2)) + ', ' + 'mean: ' + str(
            np.round(W2dB(np.mean(h_tot[plot_index])), 2)),
                              linewidth='0.5')
        ax_losses_tot[0].set_ylabel('h tot [dB]')
        ax_losses_tot[1].plot(t[10:], W2dB(turb.h_scint[plot_index])[10:],
                              label='std (1 rms): ' + str( np.round(turb.std_scint[plot_index], 2)) +
                                    ', mean: ' + str(np.round(W2dB(np.mean(turb.h_scint[plot_index])), 2)), linewidth='0.5')
        ax_losses_tot[1].set_ylabel('h scint [dB]')
        ax_losses_tot[2].plot(t[10:], W2dB(h_TX[plot_index])[10:], label='var (1 rms): ' + str(
            np.round(np.var(angle_TX[plot_index]) * 1.0E6, 2)) + ' urad, ' + ' mean: ' + str(
            np.round(W2dB(np.mean(h_TX[plot_index])), 2)),
                              linewidth='0.5')
        ax_losses_tot[2].set_ylabel('h TX [dB] \n (BW + TX jitter)')
        ax_losses_tot[3].plot(t[10:], W2dB(h_RX[plot_index])[10:], label='var (1 rms): ' + str(
            np.round(np.var(angle_RX[plot_index]) * 1.0E6, 2)) + ' urad, ' + ' mean: ' + str(
            np.round(W2dB(np.mean(h_RX[plot_index])), 2)),
                              linewidth='0.5')
        ax_losses_tot[3].set_ylabel('h RX [dB]  \n (AoA + RX jitter)')
        ax_losses_tot[3].set_xlabel('Time [s]')

        ax_losses_tot[0].legend()
        ax_losses_tot[1].legend()
        ax_losses_tot[2].legend()
        ax_losses_tot[3].legend()

        plt.show()
    def plot_all_losses_pdf():
        x_h_tot = np.linspace(0, 2, 1000)
        pdf_h_tot = np.empty((len(P_r_0), len(x_h_tot)))
        for i in range(len(P_r_0)):
            hist = np.histogram(h_tot[i], bins=1000)
            rv_h_tot = rv_histogram(hist, density=False)
            pdf_h_tot[i] = rv_h_tot.pdf(x_h_tot)
        # Plot PDF losses
        x = np.linspace(0, 2, 1000)
        for i in range(len(P_r_0)):
            hist = np.histogram(h_tot[i], bins=1000)
            rv_h_tot = rv_histogram(hist, density=False)
            pdf_h_tot = rv_h_tot.pdf(x)

            shape, loc, scale = lognorm.fit(turb.h_scint[i])
            pdf_scint = lognorm.pdf(x=x, s=shape, loc=loc, scale=scale)

            # hist = np.histogram(h_scint[i], bins=1000)
            # rv_h_scint = rv_histogram(hist, density=False)
            # pdf_scint = rv_h_scint.pdf(x)

            a, b, loc, scale = beta.fit(h_TX[i])
            pdf_h_TX = beta.pdf(x=x, a=a, b=b, loc=loc, scale=scale)

            # hist = np.histogram(h_TX[i], bins=1000)
            # rv_h_TX = rv_histogram(hist, density=False)
            # pdf_h_TX = rv_h_TX.pdf(x)

            a, b, loc, scale = beta.fit(h_RX[i])
            pdf_h_RX = beta.pdf(x=x, a=a, b=b, loc=loc, scale=scale)

            # hist = np.histogram(h_RX[i], bins=1000)
            # rv_h_RX = rv_histogram(hist, density=False)
            # pdf_h_RX = rv_h_RX.pdf(x)

        fig_losses_pdf, ax = plt.subplots(1, 1)
        ax.plot(x, pdf_h_tot, label='combined power vector')
        ax.plot(x, pdf_scint, label='scint. power vector')
        ax.plot(x, pdf_h_TX, label='TX vector')
        ax.plot(x, pdf_h_RX, label='RX vector')

        ax.set_xlabel('Normalized intensity (I/I0)')
        ax.set_ylabel('Probability density')

        ax.legend()
        ax.grid()

        plt.show()
    def plot_PSD(data_filt, f_sampling, order, effect='combined'):
        fig_psd, ax = plt.subplots(1, 1)
        # Create PSD of the filtered signal with the defined sampling frequency
        f1, psd_data1 = welch(data_filt[0], f_sampling)
        f2, psd_data2 = welch(data_filt[1], f_sampling)
        f3, psd_data3 = welch(data_filt[2], f_sampling)

        if effect == 'combined TX':
            effect1 = 'beam wander'
            effect2 = 'TX jitter'
        elif effect == 'combined RX':
            effect1 = 'angle of arrival'
            effect2 = 'RX jitter'

        # Plot PSD over frequency domain
        ax.set_title('Power Spectral Density (PSD) over Frequency domain, order=' + str(order))
        ax.semilogy(f1, psd_data1, label=effect1)
        ax.semilogy(f2, psd_data2, label=effect2)
        ax.semilogy(f3, psd_data3, label=effect)
        ax.set_xscale('log')
        ax.set_xlim(1.0E0, 1.0E4)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD [rad**2/Hz]')

        ax.grid()
        ax.legend()

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

    # plot_turbulence_data()
    # plot_all_losses_time_series()
    # plot_channel_output()
    # plot_TX_losses_time_series()
    # plot_RX_losses_time_series()
    # plot_all_losses_time_series()
    # plot_all_losses_pdf()
    # plot_PSD([angle_bw[plot_index],  angle_pj_t, angle_TX[plot_index]], f_sampling=sampling_frequency, order=order, effect='combined TX')
    # plot_PSD([angle_aoa[plot_index], angle_pj_r, angle_RX[plot_index]], f_sampling=sampling_frequency, order=order, effect='combined RX')
    # plot_pointing_TX()

    print('')
    print('----------CHANNEL-LEVEL-done------------------')
    cur_time = time.process_time() - cur_time
    print('%s seconds-----------------------' % cur_time)
    print('')
    return P_r, PPB, elevation_angles, [pdf_h_tot, x_h_tot], [pdf_P_r, x_P_r], h_tot, h_scint, h_RX, h_TX