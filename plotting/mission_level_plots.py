#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------
# The following output variables are plotted:
#   (1) Performance metrics
#   (2) Pr and BER: Distribution (global or local)
#   (3) Pointing error losses
#   (5) Fade statistics: versus elevation
#   (6) Temporal behaviour: PSD and auto-correlation
#   (7) Link budget: Cross-section of the macro-scale simulation
#   (8) Geometric plots



def plot_performance_metrics():
    # Plotting performance metrics:
    # 1) Availability
    # 2) Reliability
    # 3) Capacity

    # Print availability
    fig,ax=plt.subplots(1,1)
    ax.plot(time/3600,availability_vector, label='Tracking')
    ax.plot(time/3600,availability_vector, label='Communication')
    ax.set_ylabel('Availability (On/Off)')
    ax.set_xlabel('Time (hrs)')

    ax1 = ax.twinx()
    ax1.plot(time/3600,np.cumsum(availability_vector)/len(time)*100, color='red')
    ax1.set_ylabel('Accumulated availability (%)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax.fill_between(time/3600, y1=1.1,y2=-0.1, where=availability_vector == 0, facecolor='grey', alpha=.25)

    ax.legend()
    ax1.grid()
    plt.show()

    # Print reliability
    fig0, ax = plt.subplots(1,1)
    ax.plot(time_links/3600,BER.mean(axis=1), label='BER')
    ax.set_yscale('log')
    ax.plot(time_links/3600,fractional_fade_time, label='fractional fade time')
    ax.set_yscale('log')

    ax.set_ylabel('Error bits (normalized)')

    ax1 = ax.twinx()
    ax1.plot(time_links/3600,np.cumsum(reliability_BER*data_rate*step_size_link)/(data_rate*mission_duration), color='red')
    ax1.set_ylabel('Accumulated error bits (normalized)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax.fill_between(time/3600, y1=1.1,y2=-0.1, where=availability_vector == 0, facecolor='grey', alpha=.25)

    ax.set_xlabel('time (hrs)')
    ax.grid()
    ax.legend()
    plt.show()

    # Print capacity
    fig1,ax1 = plt.subplots(1,1)

    ax1.plot(time_links/3600, throughput/1E9, label='Actual throughput')
    ax1.plot(time_links/3600, C/1E9, label='Potential throughput')
    ax1.set_ylabel('Throughput (Gb)')

    ax2 = ax1.twinx()
    ax2.plot(time_links/3600, np.cumsum(throughput)/1E12)
    ax2.plot(time_links/3600, np.cumsum(C)/1E12)
    ax2.set_ylabel('Accumulated throughput (Tb)')
    ax1.set_xlabel('time (hrs)')

    ax1.fill_between(time/3600, y1=C.max()/1E9, y2=-5, where=availability_vector == 0, facecolor='grey', alpha=.25)

    ax1.grid()
    ax1.legend()
    plt.show()

def plot_distribution_Pr_BER():
    # Pr and BER output (distribution domain)
    # Output can be:
        # 1) Distribution over total mission interval, where all microscopic evaluations are averaged
        # 2) Distribution for specific time steps, without any averaging
    fig_T, ax_output = plt.subplots(1, 2)

    if analysis == 'total':
        P_r_pdf_total1, P_r_cdf_total1, x_P_r_total1, std_P_r_total1, mean_P_r_total1 = \
    distribution_function(data=W2dBm(P_r.mean(axis=1)), length=1, min=-60.0, max=0.0, steps=1000)

        ax_output[0].plot(x_P_r_total, P_r_cdf_total)
        ax_output[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [0, 1], c='black',
                             linewidth=3, label='thres BER=1.0E-6')

        ax_output[1].plot(x_BER_total, BER_cdf_total)
        ax_output[1].plot(np.ones(2) * np.log10(BER_thres[1]), [0, 1], c='black',
                             linewidth=3, label='thres BER=1.0E-6')

        if coding == 'yes':
            ax_output[1].plot(x_BER_total, BER_coded_cdf_total,
                                 label='Coded')

    elif analysis == 'time step specific':
        for i in indices:
            ax_output[0].plot(x_P_r, cdf_P_r[i], label='$\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '$\degree$, outage fraction='+str(fractional_BER_fade[i]))
            ax_output[1].plot(x_BER, cdf_BER[i], label='$\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '$\degree$, outage fraction='+str(fractional_fade_time[i]))

        ax_output[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [0, 1], c='black', linewidth=3, label='thres')
        ax_output[1].plot(np.ones(2) * np.log10(BER_thres[1]), [0, 1], c='black', linewidth=3, label='thres')

        if coding == 'yes':
            for i in indices:
                ax_output[1].plot(x_BER_coded, pdf_BER_coded[i],
                                label='Coded, $\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '\n % '
                                      'BER over threshold: ' + str(np.round(fractional_BER_coded_fade[i] * 100, 2)))
               
    ax_output[0].set_ylabel('CDF of $P_{RX}$',fontsize=10)
    ax_output[0].set_xlabel('$P_{RX}$ (dBm)',fontsize=10)
    ax_output[0].set_yscale('log')

    ax_output[1].set_ylabel('CDF of BER ',fontsize=10)
    ax_output[1].yaxis.set_label_position("right")
    ax_output[1].yaxis.tick_right()
    ax_output[1].set_xlabel('Error probability ($log_{10}$(BER))',fontsize=10)
    ax_output[1].set_yscale('log')

    ax_output[0].grid(True, which="both")
    ax_output[1].grid(True, which="both")
    ax_output[0].legend(fontsize=10)
    ax_output[1].legend(fontsize=10)

    plt.show()

def plot_mission_performance_pointing():
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Averaged $P_{RX}$ vs elevation')

    if link_number == 'all':
        for i in range(len(routing_output['link number'])):
            ax.plot(np.rad2deg(elevation), np.ones(elevation.shape) * W2dBm(LCT.P_r_thres[1]),     label='thres', color='black')
            ax.plot(np.rad2deg(elevation_per_link[i]), W2dBm(performance_output['Pr 0'][i]),    label='$P_{RX,0}$')
            ax.plot(np.rad2deg(elevation_per_link[i]), W2dBm(performance_output['Pr mean'][i]),    label='$P_{RX,1}$ mean')
            ax.plot(np.rad2deg(elevation_per_link[i]), W2dBm(performance_output['Pr penalty'][i]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')

            ax.plot(np.rad2deg(elevation_per_link[i]), W2dBm(performance_output['Pr mean (perfect pointing)'][i]),    label='$P_{RX,1}$ mean')
            ax.plot(np.rad2deg(elevation_per_link[i]), W2dBm(performance_output['Pr penalty (perfect pointing)'][i]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')

    else:
        ax.plot(np.rad2deg(elevation), np.ones(elevation.shape) * W2dBm(LCT.P_r_thres[1]), label='thres', color='black')
        ax.plot(np.rad2deg(elevation), W2dBm(performance_output['Pr 0']), label='$P_{RX,0}$')
        ax.plot(np.rad2deg(elevation), W2dBm(performance_output['Pr mean']), label='$P_{RX,1}$ mean')
        ax.plot(np.rad2deg(elevation), W2dBm(performance_output['Pr penalty']), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')

        ax.plot(np.rad2deg(elevation), W2dBm(performance_output['Pr mean (perfect pointing)']), label='$P_{RX}$ mean (perfect pointing)')
        ax.plot(np.rad2deg(elevation), W2dBm(performance_output['Pr penalty (perfect pointing)']), label='$P_{RX}$ '+ str(desired_frac_fade_time)+' outage frac (perfect pointing)')

    ax.set_ylabel('$P_{RX}$ (dBm)')
    ax.set_xlabel('Elevation ($\degree$)')
    ax.grid()
    ax.legend(fontsize=10)
    plt.show()

def plot_fades():
    # Fade statistics output (distribution domain)
    # The variables are computed for each microscopic evaluation and plotted over the total mission interval
    # Elevation angles are also plotted to see the relation between elevation and fading.
    # Output consists of:
        # 1) Outage fraction or fractional fade time
        # 2) Mean fade time
        # 3) Number of fades
    fig, ax = plt.subplots(1,2)
    ax2 = ax[0].twinx()

    if link_number == 'all':
        for i in range(len(routing_output['link number'])):
            ax[0].plot(np.rad2deg(elevation_per_link[i]), performance_output['fractional fade time'][i],  color='red', linewidth=1)
            ax[1].plot(np.rad2deg(elevation_per_link[i]), performance_output['mean fade time'][i] * 1000, color='royalblue', linewidth=1)
            ax2.plot(np.rad2deg(elevation_per_link[i]), performance_output['number of fades'][i],         color='royalblue', linewidth=1)
    else:
        ax[0].plot(np.rad2deg(elevation), performance_output['fractional fade time'], color='red', linewidth=1)
        ax[1].plot(np.rad2deg(elevation), performance_output['mean fade time']*1000, color='royalblue',linewidth=1)
        ax2.plot(np.rad2deg(elevation),   performance_output['number of fades'], color='royalblue',linewidth=1)

    ax[0].set_title('$50^3$ samples per micro evaluation')
    ax[0].set_ylabel('Fractional fade time (-)', color='red')
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('Number of fades (-)', color='royalblue')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax[1].set_ylabel('Mean fade time (ms)')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax[0].set_xlabel('Elevation (deg)')
    ax[1].set_xlabel('Elevation (deg)')
    ax[0].grid(True)
    ax[1].grid()

    fig, ax = plt.subplots(1, 1)
    ax.plot(turb.var_scint_I[:-10], W2dB(h_penalty[:-10]))
    ax.set_ylabel('Power penalty (dB)')
    ax.set_xlabel('Power scintillation index (-)')
    ax.grid()

    plt.show()

def plot_temporal_behaviour(data_TX_jitter, data_bw, data_TX, data_RX, data_scint, data_h_total, f_sampling,
             effect0='$h_{pj,TX}$ (platform)', effect1='$h_{bw}$', effect2='$h_{pj,TX}$ (combined)',
             effect3='$h_{pj,RX}$ (combined)', effect4='$h_{scint}$', effect_tot='$h_{total}$'):
    fig_psd,  ax      = plt.subplots(1, 2)
    fig_auto, ax_auto = plt.subplots(1, 2)

    # Plot PSD over frequency domain
    f0, psd0 = welch(data_TX_jitter,    f_sampling, nperseg=1024)
    f1, psd1 = welch(data_bw,           f_sampling, nperseg=1024)
    f2, psd2 = welch(data_TX,           f_sampling, nperseg=1024)
    f3, psd3 = welch(data_RX,           f_sampling, nperseg=1024)
    f4, psd4 = welch(data_scint,        f_sampling, nperseg=1024)
    f5, psd5 = welch(data_h_total,      f_sampling, nperseg=1024)

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
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()

    # Plot auto-correlation function over time shift
    for index in indices:
        auto_corr, lags = autocovariance(x=P_r[index], scale='micro')
        ax_auto[0].plot(lags[int(len(lags) / 2):int(len(lags) / 2)+int(0.02/step_size_channel_level)], auto_corr[int(len(lags) / 2):int(len(lags) / 2)+int(0.02/step_size_channel_level)],
                        label='$\epsilon$='+str(np.round(np.rad2deg(elevation[index]),0))+'$\degree$')

        f, psd = welch(P_r[index], f_sampling, nperseg=1024)
        ax_auto[1].semilogy(f, W2dB(psd), label='turb. freq.=' + str(np.round(turb.freq[index], 0)) + 'Hz')

    # auto_corr, lags = autocovariance(x=P_r.mean(axis=1), scale='macro')
    # ax_auto[1].plot(lags[int(len(lags) / 2):], auto_corr[int(len(lags) / 2):])

    #
    # ax_auto[0].set_title('Micro')
    # ax_auto[1].set_title('Macro')
    ax_auto[0].set_ylabel('Normalized \n auto-correlation (-)')
    ax_auto[1].set_ylabel('PSD (dBW/Hz)')
    ax_auto[1].yaxis.tick_right()
    ax_auto[1].yaxis.set_label_position("right")

    ax_auto[0].set_xlabel('lag (ms)')
    ax_auto[1].set_xlabel('frequency (Hz)')
    ax_auto[1].set_yscale('linear')
    ax_auto[1].set_ylim(-200.0, -100.0)
    ax_auto[1].set_xscale('log')
    ax_auto[1].set_xlim(1.0E0, 1.2E3)

    ax_auto[0].legend(fontsize=10)
    ax_auto[1].legend(fontsize=10)
    ax_auto[0].grid()
    ax_auto[1].grid()

    plt.show()

def plot_mission_geometrical_output_coverage():
    if link_number == 'all':
        pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=elevation, length=1, min=elevation.min(), max=elevation.max(), steps=1000)

        fig, ax = plt.subplots(1, 1)

        for e in range(len(routing_output['elevation'])):
            if np.any(np.isnan(routing_output['elevation'][e])) == False:
                pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=routing_output['elevation'][e],
                                                                                        length=1,
                                                                                        min=routing_output['elevation'][e].min(),
                                                                                        max=routing_output['elevation'][e].max(),
                                                                                        steps=1000)
                ax.plot(np.rad2deg(x_elev), cdf_elev, label='link '+str(routing_output['link number'][e]))

        ax.set_ylabel('Prob. density \n for each link', fontsize=13)
        ax.set_xlabel('Elevation (rad)', fontsize=13)

        ax.grid()
        ax.legend(fontsize=10)

    else:
        fig, ax = plt.subplots(1, 1)
        for e in range(len(routing_output['elevation'])):
            if np.any(np.isnan(routing_output['elevation'][e])) == False:
                pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=routing_output['elevation'][e],
                                                                                        length=1,
                                                                                        min=routing_output['elevation'][e].min(),
                                                                                        max=routing_output['elevation'][e].max(),
                                                                                        steps=1000)
                ax.plot(np.rad2deg(x_elev), cdf_elev, label='link ' + str(routing_output['link number'][e]))

        ax.set_ylabel('Ratio of occurence \n (normalized)', fontsize=12)
        ax.set_xlabel('Elevation (rad)', fontsize=12)
        ax.grid()
        ax.legend(fontsize=15)
    plt.show()

def plot_mission_geometrical_output_slew_rates():
    if link_number == 'all':
        pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(data=routing_total_output['slew rates'],
                                                                                length=1,
                                                                                min=routing_total_output['slew rates'].min(),
                                                                                max=routing_total_output['slew rates'].max(),
                                                                                steps=1000)

        fig, ax = plt.subplots(1, 1)

        for i in range(len(routing_output['link number'])):
            if np.any(np.isnan(routing_output['slew rates'][i])) == False:
                pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                    data=routing_output['slew rates'][i],
                    length=1,
                    min=routing_output['slew rates'][i].min(),
                    max=routing_output['slew rates'][i].max(),
                    steps=1000)
                ax.plot(np.rad2deg(x_slew), cdf_slew)

        ax.set_ylabel('Ratio of occurence \n (normalized)', fontsize=12)
        ax.set_xlabel('Slew rate (deg/sec)', fontsize=12)
        ax.grid()
    plt.show()




#---------------------------------
# Plot mission output
#---------------------------------
plot_performance_metrics()
#plot_distribution_Pr_BER()
plot_mission_performance_pointing()
#plot_fades()
# plot_temporal_behaviour(data_TX_jitter=h_pj_t, data_bw=h_bw[indices[index_elevation]], data_TX=h_TX[indices[index_elevation]], data_RX=h_RX[indices[index_elevation]],
#          data_scint=h_scint[indices[index_elevation]], data_h_total=h_tot[indices[index_elevation]], f_sampling=1/step_size_channel_level)
#---------------------------------
# Plot/print link budget
#---------------------------------
# link.print(index=index_elevation, elevation=elevation, static=False)
# link.plot(P_r=P_r, displacements=None, indices=indices, elevation=elevation, type='table')
#---------------------------------
# Plot geometric output
#---------------------------------
# link_geometry.plot(type='trajectories', time=time)
link_geometry.plot(type='AC flight profile', routing_output=routing_output)
link_geometry.plot(type = 'satellite sequence', routing_output=routing_output)
# link_geometry.plot(type='longitude-latitude')
# link_geometry.plot(type='angles', routing_output=routing_output)
#plot_mission_geometrical_output_coverage()
# plot_mission_geometrical_output_slew_rates()
