import time

import matplotlib.pyplot as plt
import numpy as np
import csv
import sqlite3
from scipy.special import binom

from input import *
from LCT import terminal_properties
from helper_functions import *
from channel_level import channel_level

def bit_level(LCT,
              link_budget,
              plot_index: float,
              divide_index: float,
              P_r_0, P_r, PPB, elevation_angles, pdf_h_tot, h_tot, h_scint, h_RX, h_TX):
    print('')
    print('----------BIT LEVEL-(Monte-Carlo-simulations-of-signal-detection-&-bit-output)-----------------')
    print('')
    cur_time = time.process_time()
    #------------------------------------------------------------------------
    #-------------------------------VERIFICATION-----------------------------
    #------------------------------------------------------------------------
    verification = 'no'
    if verification == 'yes':
        from input import data_rate
        SNR_dict = dict()
        BER_dict = dict()
        BER_coded_interleaved_dict = dict()
        errors_dict = dict()
        errors_coded_interleaved_dict = dict()
        axis = 0
        P_r = np.linspace(dBm2W(-40), dBm2W(-30), 100)
        P_r_length = len(P_r)
        PPB = PPB_func(P_r, data_rate)
        detection_list = ['PIN', 'APD']
        modulation_list = ['OOK-NRZ', 'BPSK', '2-PPM']

    else:
        axis = 1
        detection_list = [LCT.detection]
        modulation_list = [LCT.modulation]

    for detection in detection_list:
        for modulation in modulation_list:
            print('Start computing noise, SNR and BER')
            #------------------------------------------------------------------------
            #--------------------------COMPUTING-SNR-&-BER---------------------------
            #------------------------------------------------------------------------
            # All relevant noise types are computed with analytical equations.
            # These equations are approximations, based on the assumption of a gaussian distribution for each noise type.
            # The noise without any incoming signal is equal to
            noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun)  # Should have 2D with size ( len of iterable list, # of samples )
                                                                                        # Except for noise_th: Should be a scalar

            # The actually received SNR is computed for each sample, as it directly depends on the noise and the received power P_r.
            # The BER is then computed with the analytical relationship between SNR and BER, based on the modulation scheme.
            SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
            BER = LCT.BER_func(Q=Q, modulation=modulation)

            print('Computing noise, SNR and BER done')
            cur_time = time.process_time() - cur_time
            print('%s seconds-----------------------' % cur_time)

            print(np.shape(SNR), np.shape(BER))

            #------------------------------------------------------------------------
            #---------------------------COMPUTING-THRESHOLD--------------------------
            #------------------------------------------------------------------------
            print('start computing errors')
            # The threshold SNR is computed with the analytical relationship between SNR and BER, based on the modulation scheme.
            # The threshold P_r is then computed from the threshold SNR, the noise and the detection technique.
            # The PPB is computed with the threshold P_r and data_rate, according to (A.MAJUMDAR, 2008, EQ.3.29, H.HEMMATI, 2004, EQ.4.1-1)
            LCT.threshold(BER_thres=BER_thres, detection=detection, modulation=modulation)

            data_rate_var_BER9 = data_rate_func(P_r, LCT.PPB_thres[0])
            data_rate_var_BER6 = data_rate_func(P_r, LCT.PPB_thres[1])
            data_rate_var_BER3 = data_rate_func(P_r, LCT.PPB_thres[2])
            P_r_BER9 = h * v * PPB * data_rate_var_BER9
            P_r_BER6 = h * v * PPB * data_rate_var_BER6
            P_r_BER3 = h * v * PPB * data_rate_var_BER3

            if variable_data_rate == "BER9":
                LCT.data_rate = data_rate_var_BER9
                P_r = P_r_BER9
            # ------------------------------------------------------------------------
            # -----------------------------TOTAL-BITS--&------------------------------
            # ------------------------------ERROR-BITS--------------------------------
            # ------------------------------------------------------------------------
            # The amount of erroneous bits is computed and stored as a vector.
            # The total number of errors is computed by summing this vector
            total_bits = LCT.data_rate * interval_channel_level                         # Should be a scalar
            total_bits_per_sample = total_bits / samples_channel_level                  # Should be a scalar
            errors = total_bits_per_sample * BER                                        # Should have 2D with size ( len of iterable list, # of samples )
            print('Computing errors (uncoded) done')
            cur_time = time.process_time() - cur_time
            print('%s seconds-----------------------' % cur_time)
            #------------------------------------------------------------------------
            #------------------------------INTERLEAVING------------------------------
            #---------------------------------CODING---------------------------------
            #------------------------------------------------------------------------
            # print('start computing errors (coding only)')
            # # The coding method in the LCT class simulates the coding scheme and computes the coded SER and BER from the channel BER and the coding input parameters
            # BER_coded = LCT.coding(K=K,
            #                        N=N,
            #                        BER=BER)
            # # The amount of erroneous bits is computed and stored as a vector.
            # # The total number of errors is computed by summing this vector
            # errors_coded = total_bits_per_sample * BER_coded
            # print('computing errors (coding only) done')
            # cur_time = time.process_time() - cur_time
            # print('%s seconds-----------------------' % cur_time)

            # All bits are spread out over a specified interleaving latency (default is 1 ms). The method LCT.interleaving takes only
            # The erroneous bits, spreads these out and stores this in a new vector. Then a new BER is computed by dividing these interleaved errors by
            # The total bits per sample
            print('start computing errors (coding + interleaving)')
            errors_interleaved = LCT.interleaving(errors=errors)
            BER_interleaved = errors_interleaved / total_bits_per_sample
            print('computing errors (interleaving) done')
            cur_time = time.process_time() - cur_time
            print('%s seconds-----------------------' % cur_time)
            # Coding implementation is based on an analytical Reed-Solomon approximation, taken from (CCSDS Historical Document, 2006, CH.5.5, EQ.3-4)
            # Input parameters for the coding scheme are total symbols per codeword (N), number of information symbols per code word (K) and symbol length
            BER_coded_interleaved = LCT.coding(K=K,
                                               N=N,
                                               BER=BER_interleaved)
            errors_coded_interleaved = total_bits_per_sample * BER_coded_interleaved

            if verification == 'no':
                errors_acc = np.cumsum(errors, axis=1)                                     # Should have 2D with size ( len of iterable list, # of samples )
            #     errors_coded_acc = np.cumsum(errors_coded, axis=1)                         # Should have 2D with size ( len of iterable list, # of samples )
            #     errors_interleaved_acc = np.cumsum(errors_interleaved, axis=1)             # Should have 2D with size ( len of iterable list, # of samples )
                errors_coded_interleaved_acc = np.cumsum(errors_coded_interleaved, axis=1) # Should have 2D with size ( len of iterable list, # of samples )
                total_errors = errors_acc[:, -1]                                           # Should have 1D with size ( len of iterable list )
            #     # total_errors_coded = errors_coded_acc[:, -1]                               # Should have 1D with size ( len of iterable list )
            #     total_errors_interleaved = errors_interleaved_acc[:, -1]                   # Should have 1D with size ( len of iterable list )
                total_errors_coded_interleaved = errors_coded_interleaved_acc[:, -1]       # Should have 1D with size ( len of iterable list )

            # print('computing errors (coding + interleaving) done')
            # cur_time = time.process_time() - cur_time
            # print('%s seconds-----------------------' % cur_time)
            #------------------------------------------------------------------------
            #-----------------------------FADE-STATISTICS----------------------------
            #------------------------------------------------------------------------
            # Here, the fading statistics are computed numerically.
            # For each sample, the received power is evaluated and compared with the threshold power for all 3 thresholds (BER = 1.0E-9, 1.0E-6, 1.0E-3)
            # When P_r is lower than P_r_thres, it is counted as a fade. One fade time is equal to the time step of 0.1 ms.

            fades  = np.empty((len(BER_thres), len(P_r)))
            # Fade times for the BER threshold of 1.0E-9, 1.0E-6 and 1.0E-3
            fades[0]  = np.count_nonzero((P_r < LCT.P_r_thres[0]), axis=axis) / samples_channel_level
            fades[1]  = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=axis) / samples_channel_level
            fades[2]  = np.count_nonzero((P_r < LCT.P_r_thres[2]), axis=axis) / samples_channel_level

            # ------------------------------------------------------------------------
            # ---------------UPDATE-LINK-BUDGET-WITH-FLUCTUATIONS-AND-Pr--------------
            # ------------------------------------------------------------------------
            # Here, the link budget parameters are updated with the smaller vectors that are defined by DIVIDE_INDEX
            link_budget.h_ext = link_budget.h_ext[::divide_index]
            link_budget.h_strehl = link_budget.h_strehl[::divide_index]
            link_budget.h_beamspread = link_budget.h_beamspread[::divide_index]
            link_budget.w_r = link_budget.w_r[::divide_index]
            link_budget.P_r_0 = link_budget.P_r_0[::divide_index]

            link_budget.dynamic_contributions(P_r=P_r.mean(axis=1),
                                                   PPB=PPB.mean(axis=1),
                                                   T_dyn_tot=h_tot.mean(axis=1),
                                                   T_scint=h_scint.mean(axis=1),
                                                   T_TX=h_TX.mean(axis=1),
                                                   T_RX=h_RX.mean(axis=1))
            # Initiate tracking. This method makes sure that a part of the incoming light is reserved for the tracking system.
            # In the link budget this is translated to a fraction (standard is set to 0.9) of the light is subtracted from communication budget and used for tracking budget
            link_budget.P_r_tracking_func()
            margin = link_budget.link_margin(P_r_thres=LCT.P_r_thres, PPB_thres=LCT.PPB_thres)
            print('bit level margins')
            print(margin[0])
            # ------------------------------------------------------------------------
            # ---------------------------SAVE-TO-DATABASE-----------------------------
            # ------------------------------------------------------------------------
            # if verification == 'yes':
            #     errors_dict[detection+'_'+modulation] = errors
            #     errors_coded_interleaved_dict[detection + '_' + modulation] = errors_coded_interleaved
            #     SNR_dict[detection+'_'+modulation] = SNR
            #     BER_dict[detection + '_'+modulation] = BER
            #     BER_coded_interleaved_dict[detection + '_' + modulation] = BER_coded_interleaved
            #
            # if verification == 'no':
            #     print('saving data')
            #     # Here, a selection is made of relevant output data, which is first written as a list of strings, then stored in a array with name 'data'
            #     # Then, one row is added to the top of this array to account for a zero value (needed when mapping to dim 2) and finally data is saved to sqlite3 database
            #     # This is a list of the performance variables (metrics) that are saved in a database
            #     data_metrics = ['elevation',
            #                     'P_r',
            #                     'PPB',
            #                     'h_tot',
            #                     # 'fade time (BER=1.0E-3, 1.0E-6, 1.0E-9)',
            #                     'P_margin req. (BER=1.0E-3)',
            #                     'P_margin req. (BER=1.0E-6)',
            #                     'P_margin req. (BER=1.0E-9)',
            #                     'Number of error bits (uncoded)',
            #                     'Number of error bits (coded)']
            #     number_of_metrics = len(data_metrics)
            #
            #     # A data dictionary is created here and filled with all the relevant performance output variables
            #
            #     data = dict()
            #     data[data_metrics[0]] = np.rad2deg(elevation_angles)
            #     data[data_metrics[1]] = P_r.mean(axis=1)
            #     data[data_metrics[2]] = PPB.mean(axis=1)
            #     data[data_metrics[3]] = h_tot.mean(axis=1)
            #     # data[data_metrics[4]] = [fades[0], fades[1], fades[2]]
            #     data[data_metrics[4]] = margin[0]
            #     data[data_metrics[5]] = margin[1]
            #     data[data_metrics[6]] = margin[2]
            #     data[data_metrics[7]] = total_errors
            #     data[data_metrics[8]] = total_errors_coded_interleaved
            #
            #     # Save data to sqlite3 database, 9 metrics are saved for each elevation angle (8+1 columns and N rows)
            #     # UNCOMMENT WHEN YOU WANT TO SAVE DATA TO DATABASE
            #     # save_to_database(data, data_metrics, elevation_angles)
            #     print('saving done')
            #     cur_time = time.process_time() - cur_time
            #     print('%s seconds-----------------------' % cur_time)

    #------------------------------------------------------------------------
    #----------------------------SENSITIVTY-ANALYSIS-------------------------
    #------------------------------------------------------------------------

    # Sensitivity of output w.r.t. elevation angle input
    # sensitivity_dim0(elevation_angles, P_r, total_errors, total_bits)
    # sensitivity_dim0(elevation_angles, P_r, total_errors_coded, ranges)
    # sensitivity_dim0(elevation_angles, P_r, total_errors_coded_interleaved, ranges)
    #------------------------------------------------------------------------
    #-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
    #------------------------------------------------------------------------


    def plot_Pr_SNR_BER_verification():
        if verification == 'yes':
            fig, ax = plt.subplots(nrows=1,ncols=2)
            for detection in detection_list:
                for modulation in modulation_list:
                    label = detection+'_'+modulation
                    if label == 'PIN_OOK-NRZ' or label == 'APD_OOK-NRZ':
                        ax[0].plot(W2dBm(P_r), W2dB(SNR_dict[label]), label=str(detection))

                    # ax[0,0].plot(SNR_dict[label], errors_dict[label], label='detection:' + str(detection) + ', modulation:' + str(modulation),
                    #               linewidth='1.0')
                    # ax[0,0].plot(SNR, errors_coded_interleaved[plot_index],
                    #               label='detection:'+str(detection)+', modulation:'+str(modulation)+', RS coded ('+str(N,K)+'), interleaved '+str(latency_interleaving)+'s)',
                    #               linewidth='1.0')
                    ax[1].plot(W2dBm(P_r), BER_dict[label], label=str(detection)+', '+str(modulation))
                    # ax[0,1].plot(SNR_dict[label], BER_coded_interleaved_dict[label], label=str(detection)+', '+str(modulation)+
                    #                                                                                  ', RS coded ('+str(N)+','+str(K)+'), interleaved '+str(latency_interleaving)+'s)')
                    ax[1].set_yscale('log')
                    # ax[0,1].set_ylim(1.0E-10, 0.5)

                    ax[0].set_ylabel('Signal-to-Noise ratio \n [$i_r$ / $\sigma_n$] (dB)', fontsize=15)
                    ax[1].set_ylabel('Error probability \n [# Error bits / total bits]', fontsize=15)
                    ax[0].set_xlabel('Received power (dBm)', fontsize=15)
                    ax[1].set_xlabel('Received power (dBm)', fontsize=15)

            ax[0].legend(fontsize=20)
            ax[1].legend(fontsize=20)
            ax[0].grid()
            ax[1].grid()
            plt.show()
    def plot_Pr_SNR_BER(SNR=None, BER=None, BER_coded=None):
        parity_bits = int((N - K) / 2)
        fig_ber, axs_ber = plt.subplots(1, 1)
        axs_ber.scatter(W2dB(SNR[plot_index]), BER[plot_index], label='Monte Carlo results: Uncoded (Channel BER)')
        axs_ber.scatter(W2dB(SNR[plot_index]), BER_coded[plot_index], label='Monte Carlo results: (255,223) RS coded')
        axs_ber.set_title(f'BER vs SNR')
        axs_ber.set_ylabel('Bit Error Rate (BER)')
        axs_ber.set_yscale('log')
        axs_ber.set_xlabel('SNR (dB)')
        axs_ber.grid()

        SNR = np.linspace(0, 50, 200)
        Vb = 1 / 2 * erfc(np.sqrt(SNR) / np.sqrt(2))
        Vs = 1 - (1 - Vb) ** symbol_length
        SER_coded = np.zeros(np.shape(Vs))
        for i in range(len(Vs)):
            SER_coded[i] = Vs[i] * sum(
                binom(N - 1, k) * Vs[i] ** k * (1 - Vs[i]) ** (N - k - 1) for k in range(parity_bits, N - 1))
        BER_coded = 2 ** (symbol_length - 1) / N * SER_coded
        axs_ber.plot(W2dB(SNR), Vb, label='Theory: Uncoded (Channel BER), OOK-NRZ', color='orange')
        axs_ber.plot(W2dB(SNR), BER_coded, label='Theory: (255,223) RS coded, OOK-NRZ', color='green')
        axs_ber.legend()
        plt.show()
    def plot_bit_level_time_series():
        # plot performance parameters LONG
        fig_dim0, ax_dim0 = plt.subplots(4,1)
        ax_dim0[0].set_title('Bit level output \n for elevation angle $\epsilon$ = '+str(np.round(np.rad2deg(elevation_angles[plot_index]), 2))+'deg')

        ax_dim0[0].set_ylabel('Pr [dBm]')
        ax_dim0[0].plot(t[10:], W2dBm(P_r[plot_index])[10:], label='mean: '+str(W2dBm(np.mean(P_r[plot_index])))+' dBm')
        ax_dim0[0].plot(t[10:], np.ones(len(t[10:])) * W2dBm(LCT.P_r_thres[0]), label='mean: ' + str(np.round(W2dBm(LCT.P_r_thres[0]),2)) + ' dBm')
        ax_dim0[0].plot(t[10:], np.ones(len(t[10:])) * W2dBm(LCT.P_r_thres[1]), label='mean: ' + str(np.round(W2dBm(LCT.P_r_thres[1]),2)) + ' dBm')
        ax_dim0[0].plot(t[10:], np.ones(len(t[10:])) * W2dBm(LCT.P_r_thres[2]), label='mean: ' + str(np.round(W2dBm(LCT.P_r_thres[2]),2)) + ' dBm',
                        linewidth = '0.5')

        ax_dim0[1].set_ylabel('SNR [dB]')
        ax_dim0[1].plot(t[10:], W2dB(SNR[plot_index])[10:], label='mean: '+str(W2dB(np.mean(SNR[plot_index])))+' dB \n'
                                                        'SNR for $BER_{thres}$=1.0E-3: '+str(np.round(W2dB(LCT.SNR_thres[0]),2))+' dB \n'
                                                        'SNR for $BER_{thres}$=1.0E-6: '+str(np.round(W2dB(LCT.SNR_thres[1]),2))+' dB \n'
                                                        'SNR for $BER_{thres}$=1.0E-9: '+str(np.round(W2dB(LCT.SNR_thres[2]),2))+' dB \n',
                                                  linewidth = '0.5')
        ax_dim0[2].set_ylabel('BER')
        ax_dim0[2].plot(t, BER[plot_index], label='max: ' + str(np.max(BER[plot_index])))
        # ax_dim0[3].plot(t, BER_interleaved[plot_index],
        #                 label='max: ' + str(np.max(BER_interleaved[plot_index])))
        # ax_dim0[3].plot(t, BER_coded_interleaved[plot_index], label='max: ' + str(np.max(BER_coded_interleaved[plot_index])))
        ax_dim0[2].plot(t, np.ones(len(t)) * BER_thres[0],
                        label='mean: ' + str(BER_thres[0]))
        ax_dim0[2].plot(t, np.ones(len(t)) * BER_thres[1],
                        label='mean: ' + str(BER_thres[1]))
        ax_dim0[2].plot(t, np.ones(len(t)) * BER_thres[2],
                        label='mean: ' + str(BER_thres[2]))
        ax_dim0[2].set_yscale('log')
        ax_dim0[2].set_ylim(0.5, 1.0E-10)


        ax_dim0[3].plot(t, errors_acc[plot_index]/1.0E6, label='Uncoded,'
                                                               'total='+str(np.round(total_errors[plot_index]/1.0E6,2))+'Mb of '+str(np.sum(total_bits)/1.0E9)+'Gb',
                        linewidth='2')
        # ax_dim0[2].plot(t, errors_coded_acc[plot_index] / 1.0E6, label='Interleaved '+str(latency_interleaving)+'s)'
        #                                                              'total='+str(np.round(total_errors_coded[plot_index]/1.0E6,2))+'Mb of '+str(np.sum(total_bits)/1.0E9)+'Gb',
        #                 linewidth='2')
        # ax_dim0[2].plot(t, errors_coded_interleaved_acc[plot_index] / 1.0E6, label='RS coded ('+str(N)+','+str(K)+'), interleaved '+str(latency_interleaving)+'s)'
        #                                           'total=' + str(np.round(total_errors_coded_interleaved[plot_index] / 1.0E6, 8)) + 'Mb of ' + str(np.sum(total_bits) / 1.0E9) + 'Gb',linewidth='2')

        ax_dim0[3].set_ylim(errors_acc[plot_index, 10:].min() / 1.0E6, total_errors[plot_index] / 1.0E6)
        ax_dim0[3].set_ylabel('Number of cum. \n Error bits [Mb]')
        ax_dim0[3].set_yscale('log')

        ax_dim0[3].set_xlabel('Time [s]')


        ax_dim0[0].legend(fontsize=10)
        ax_dim0[1].legend(fontsize=10)
        ax_dim0[2].legend(fontsize=10)
        ax_dim0[3].legend(fontsize=10)
        # ax_dim0[4].legend(fontsize=10)
        ax_dim0[0].grid()
        ax_dim0[1].grid()
        ax_dim0[2].grid()
        ax_dim0[3].grid()
        # ax_dim0[4].grid()
        plt.show()
    def plot_coding_errors():
        h_x_pdf = pdf_h_tot[1]
        # Here, the average BER is computed by using the equation for the uncoditional BER.
        P_r_pdf = (h_x_pdf[:, None] * P_r_0).transpose()
        noise_sh_pdf, noise_th_pdf, noise_bg_pdf, noise_beat_pdf = LCT.noise(P_r=P_r_pdf, I_sun=I_sun)
        SNR_pdf, Q_pdf = LCT.SNR_func(P_r=P_r_pdf, detection=detection,
                                      noise_sh=noise_sh_pdf, noise_th=noise_th_pdf, noise_bg=noise_bg_pdf,
                                      noise_beat=noise_beat_pdf)
        BER_pdf = LCT.BER_func(Q=Q_pdf, modulation=modulation)
        BER_avg = np.trapz(pdf_h_tot[0] * BER_pdf, x=h_x_pdf, axis=1)

        BER[BER < BER_thres[0]] = 0

        fig_ber, ax_ber = plt.subplots(1, 1)
        ax_ber.set_xlabel('Elevation angles (deg')
        ax_ber.set_ylabel('Average error probability \n'
                          'Uncoditional (BER)')
        ax_ber.scatter(np.rad2deg(elevation_angles), BER.mean(axis=1), label='Mean BER over channel simulation with 1 million samples')
        ax_ber.scatter(np.rad2deg(elevation_angles), BER_avg, label='Unconditional BER average, from theory')
        ax_ber.set_yscale('log')
        ax_ber.legend(fontsize=15)
        ax_ber.grid()

        fig_coding, ax_coding = plt.subplots(1,2)
        ax_coding[0].set_title('Bit-level error performance for $\epsilon$ = '+str(np.round(np.rad2deg(elevation_angles[plot_index]),2))+'$\degree$ \n'
                            'Interleaver latency = '+ str(latency_interleaving)+'s')

        ax_coding[0].plot(t, BER[plot_index], label='Uncoded')
        ax_coding[0].plot(t, BER_coded[plot_index], label='RS ('+str(N)+', '+str(K)+') coded only')
        ax_coding[1].plot(t, BER_interleaved[plot_index], label='Uncoded, interleaved')
        ax_coding[1].plot(t, BER_coded_interleaved[plot_index], label='RS coded + interleaved')
        ax_coding[0].plot(t, np.ones(len(t)) * BER[plot_index].mean(), color='blue')
        ax_coding[0].plot(t, np.ones(len(t)) * BER_coded[plot_index].mean(), color='orange')
        ax_coding[1].plot(t, np.ones(len(t)) * BER_interleaved[plot_index].mean(), color='blue')
        ax_coding[1].plot(t, np.ones(len(t)) * BER_coded_interleaved[plot_index].mean(), color='orange')
        ax_coding[0].set_yscale('log')
        ax_coding[0].set_ylim(1.0E-15, 0.5)
        ax_coding[1].set_yscale('log')
        ax_coding[1].set_ylim(1.0E-15, 0.5)
        ax_coding[0].set_ylabel('Error probability \n [# Error bits / total bits]')

        ax_coding[0].legend(fontsize=15)
        ax_coding[0].grid()
        ax_coding[1].legend(fontsize=15)
        ax_coding[1].grid()
        plt.show()
    def plot_variable_data_rate():
        plot_index = [0, 2, 4]
        fig_datarate, ax_datarate = plt.subplots(3, 1)
        ax_datarate[0].set_title('Data rates for $\epsilon$ = ' + str(np.round(np.rad2deg(elevation_angles[plot_index]), 2)))

        ax_datarate[0].plot(t, data_rate_var_BER3[plot_index[0]] / 1.0E9, label='Variable DR for BER=1E-3, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER3[plot_index[0]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[0].plot(t, data_rate_var_BER6[plot_index[0]] / 1.0E9, label='Variable DR for BER=1E-6, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER6[plot_index[0]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[0].plot(t, data_rate_var_BER9[plot_index[0]] / 1.0E9, label='Variable DR for BER=1E-9, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER9[plot_index[0]] / 1.0E9), 3))) + ' Gb/s')

        ax_datarate[1].plot(t, data_rate_var_BER3[plot_index[1]] / 1.0E9, label='Variable DR for BER=1E-3, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER3[plot_index[1]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[1].plot(t, data_rate_var_BER6[plot_index[1]] / 1.0E9, label='Variable DR for BER=1E-6, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER6[plot_index[1]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[1].plot(t, data_rate_var_BER9[plot_index[1]] / 1.0E9, label='Variable DR for BER=1E-9, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER9[plot_index[1]] / 1.0E9), 3))) + ' Gb/s')

        ax_datarate[2].plot(t, data_rate_var_BER3[plot_index[2]] / 1.0E9, label='Variable DR for BER=1E-3, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER3[plot_index[2]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[2].plot(t, data_rate_var_BER6[plot_index[2]] / 1.0E9, label='Variable DR for BER=1E-6, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER6[plot_index[2]] / 1.0E9), 3))) + ' Gb/s')
        ax_datarate[2].plot(t, data_rate_var_BER9[plot_index[2]] / 1.0E9, label='Variable DR for BER=1E-9, mean: ' + str(
            (np.round(np.mean(data_rate_var_BER9[plot_index[2]] / 1.0E9), 3))) + ' Gb/s')

        # ax_datarate[0].set_ylim(1.0, 100.0)
        # ax_dim1[5].set_ylim(1.0, 100.0)
        ax_datarate[0].set_ylabel('Data rate [Gb/s]')
        ax_datarate[1].set_ylabel('Data rate [Gb/s]')
        ax_datarate[2].set_ylabel('Data rate [Gb/s]')
        ax_datarate[2].set_xlabel('Time [s]')

        ax_datarate[0].legend()
        ax_datarate[1].legend()
        ax_datarate[2].legend()
        ax_datarate[0].grid()
        ax_datarate[1].grid()
        ax_datarate[2].grid()

        plt.show()

    # plot_Pr_SNR_BER_verification()
    # plot_Pr_SNR_BER(SNR=SNR, BER=BER, BER_coded=BER_coded_interleaved)
    plot_bit_level_time_series()
    # plot_coding_errors()
    # plot_variable_data_rate()

    print('')
    print('----------BIT-LEVEL-done------------------')
    cur_time = time.process_time() - cur_time
    print('%s seconds-----------------------' % cur_time)
    print('')
    return P_r.mean(axis=1), BER.mean(axis=1), BER_coded_interleaved.mean(axis=1), total_errors, total_errors_coded_interleaved, margin