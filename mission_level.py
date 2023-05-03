from Routing_network import routing_network
from input import *
from helper_functions import *
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

from matplotlib import pyplot as plt
from scipy.stats import norm, lognorm, rayleigh, rice, rv_histogram
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

print('')
print('-----------MISSION-LEVEL-----------------')
print('')
plot_index = 20

#------------------------------------------------------------------------
#------------------------------------LCT---------------------------------
#------------------------------------------------------------------------
LCT = terminal_properties()
LCT.threshold(BER_thres = BER_thres,
               modulation = "OOK-NRZ",
               detection = "APD")

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
link_geometry = link_geometry()
link_geometry.propagate(stepsize_AC=step_size_AC, stepsize_SC=step_size_SC)
link_geometry.geometrical_outputs()
# Initiate time vector for entire communication period. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
# Total number of samples for all links combined
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])
#------------------------------------------------------------------------
#---------------------------ROUTING-OF-LINKS-----------------------------
#------------------------------------------------------------------------
# The routing_network class is initiated here.
# This class takes the geometrical output of the link_geometry class (which is the output of the aircraft and all satellites)
routing_network = routing_network(time)

# The hand_over_strategy function computes a sequence of links between the aircraft and satellites.
# One method is based on minimum required number of handovers. Here, the satellite with the lowest, increasing, elevation angle is found and
# acquisition & communication is started, until the elevation angle decreases below the minimum elevation angle (20 degrees).
# Then a new satellite is found again, until termination time.
# This function thus filters the input and outputs arrays of all geometrical variables over the time interval of dimension 2.
geometrical_output, mask = routing_network.routing(link_geometry.geometrical_output, time)
time_hrs = time[mask] / 3600
# PLOT MISSION GEOMETRIC OUTPUT
# link_geometry.plot()
link_geometry.plot(type='angles')
link_geometry.plot(type = 'satellite sequence', sequence=geometrical_output['pos SC'], links= routing_network.number_of_links)
plt.show()

#------------------------------------------------------------------------
#-------------------------------ATTENUATION------------------------------
att = attenuation()
att.h_ext_func(range_link=geometrical_output['ranges'], zenith_angles=geometrical_output['zenith'], method="standard_atmosphere")
sampling_frequency = 1 / step_size_link
ext_frequency = 1/600 # 10 minute frequency of the transmission due to clouds

# For h_clouds, first generate random values for cloud transmission loss. Then, use a lowpass filter to implement a cut-off frequency of 10 min.
# Then convert to lognormal distribution (IVANOV ET AL. 2022, EQ.18). Then use the mask from routing model to filter.
# For h_ext, use a model that computes molecule/aerosol transmission loss.
samples_link_level = len(time)
h_clouds = norm.rvs(scale=1, loc=0, size=samples_link_level)
order = 2
h_clouds = filtering(effect='extinction', order=order, data=h_clouds, f_cutoff_low=ext_frequency,
                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
att.h_clouds_func(data=h_clouds)
att.h_clouds = att.h_clouds[mask]
h_ext = att.h_ext * att.h_clouds

# x_att = np.linspace(0, 2, 10000)
# hist_att = np.histogram(h_ext, bins=10000)
# rv_att = rv_histogram(hist_att, density=False)
# pdf_att = rv_att.pdf(x_att)

#------------------------------------------------------------------------
#-------------------------------TURBULENCE-------------------------------
# The turbulence class is initiated here. Inside the turbulence class, there are multiple methods that are run directly.
# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
# With Cn^2 and r0, the variances for scintillation and beam wander are computed

slew_rate = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)
turb = turbulence(ranges=geometrical_output['ranges'], link=link)
turb.windspeed_func(slew=slew_rate, Vg=link_geometry.speed_AC.mean(), wind_model_type=wind_model_type)
turb.Cn_func(turbulence_model=turbulence_model)
r0 = turb.r0_func(zenith_angles=geometrical_output['zenith'])
turb.var_rytov_func(zenith_angles=geometrical_output['zenith'])
turb.var_scint_func(zenith_angles=geometrical_output['zenith'])
turb.Strehl_ratio_func(tip_tilt="YES")
turb.beam_spread()
turb.var_bw_func(zenith_angles=geometrical_output['zenith'])
turb.var_aoa_func(zenith_angles=geometrical_output['zenith'])
turb.print(index = plot_index, elevation=np.rad2deg(geometrical_output['elevation']), ranges=geometrical_output['ranges'])
# ------------------------------------------------------------------------
# -----------------------------LINK-BUDGET--------------------------------
# The link budget class is initiated here.
link = link_budget(ranges=geometrical_output['ranges'], h_strehl=turb.h_strehl, w_ST=turb.w_ST, h_beamspread=turb.h_beamspread, h_ext=h_ext)
# The power at the receiver is computed from the link budget.
P_r_0 = link.P_r_0_func()

# ------------------------------------------------------------------------
# ----------------------------COARSE-SOLVER-------------------------------
# COARSE SOLVER (for 1s timesteps)
noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun)
SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)
margin_0 = np.zeros((3, len(time[mask])))
margin_0[0] =  P_r_0 / LCT.P_r_thres[0]
margin_0[1] =  P_r_0 / LCT.P_r_thres[1]
margin_0[2] =  P_r_0 / LCT.P_r_thres[2]
errors_0 = BER_0 * data_rate

# ------------------------------------------------------------------------
# ---------------------FINE SOLVER-(interpolated)-------------------------
divide_index = 1
P_r_0_fine = P_r_0[::divide_index]
ranges_fine = geometrical_output['ranges'][::divide_index]
elevation_angles_fine = geometrical_output['elevation'][::divide_index]
zenith_angles_fine = geometrical_output['zenith'][::divide_index]

P_r, PPB, elevation_angles, pdf_h_tot, pdf_P_r, h_tot, h_scint, h_RX, h_TX = \
    channel_level(plot_index=int(plot_index/divide_index),
                  LCT=LCT, turb=turb,
                  P_r_0=P_r_0_fine,
                  ranges=ranges_fine,
                  elevation_angles=elevation_angles_fine,
                  zenith_angles=zenith_angles_fine,
                  samples=samples_channel_level)

# P_r, BER, pdf_BER, BER_coded, pdf_BER_coded, errors, errors_coded, link = \
P_r, BER, pdf_BER, errors, link = \
    bit_level(LCT=LCT,
              link_budget=link,
              plot_index=int(plot_index/divide_index),
              divide_index=divide_index,
              samples=samples_channel_level,
              P_r_0=P_r_0_fine,
              pdf_P_r=pdf_P_r,
              P_r=P_r,
              PPB=PPB,
              elevation_angles=elevation_angles_fine,
              pdf_h_tot=pdf_h_tot,
              h_tot=h_tot,
              h_scint=h_scint,
              h_RX=h_RX,
              h_TX=h_TX,
              coding='no')

# ------------------------------------------------------------------------
# ---------------------------------OUTPUT---------------------------------

# In case of SEQUENTIAL COUPLING of micro-model and macro-model (with the use of a database): the geometrical output of macro-model is mapped with the database.
# The output is an array of performance values for each timestep of the macro-model
# performance_output = routing_network.performance_output_database(geometrical_output['elevation'])

# In case of CONCURRENT COUPLING: The fine solver output from above code is used.
# The output is an array of performance values for each timestep of the macro-model

P_r_avg = P_r.mean(axis=1)
BER_avg = BER.mean(axis=1)
# BER_coded_avg = BER_coded.mean(axis=1)
errors = errors * step_size_link / interval_channel_level
# errors_coded = errors_coded * step_size_link / interval_channel_level

fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
fractional_BER_fade = np.count_nonzero((BER > BER_thres[1]), axis=1) / samples_channel_level
# fractional_BER_coded_fade = np.count_nonzero((BER_coded > BER_thres[1]), axis=1) / samples_channel_level

margin = link.link_margin(P_r_thres=LCT.P_r_thres, PPB_thres=LCT.PPB_thres)

# Here, the latency is computed over the whole time series.
# The only assumed contributions are geometrical latency and interleaving latency.
# Latency due to coding/detection/modulation/data processing can be optionally added.
latency_propagation = geometrical_output['ranges'] / speed_of_light
latency_data_rate = 1 / data_rate
latency = latency_propagation + latency_data_rate + latency_interleaving

# Here, the total throughput is computed
throughput = data_rate * step_size_link - errors
# throughput_coded = data_rate * step_size_link - errors_coded


margin1_interpolated = interpolator(x=time[mask][::divide_index], y=margin[0], x_interpolate=time[mask], interpolation_type='cubic spline')
margin2_interpolated = interpolator(x=time[mask][::divide_index], y=margin[1], x_interpolate=time[mask], interpolation_type='cubic spline')
margin3_interpolated = interpolator(x=time[mask][::divide_index], y=margin[2], x_interpolate=time[mask], interpolation_type='cubic spline')

errors_interpolated = interpolator(x=time[mask][::divide_index], y=errors, x_interpolate=time[mask], interpolation_type='cubic spline')
# errors_coded_interpolated = interpolator(x=time[mask][::divide_index], y=errors_coded, x_interpolate=time[mask], interpolation_type='cubic spline')
BER_interpolated = interpolator(x=time[mask][::divide_index], y=BER_avg, x_interpolate=time[mask], interpolation_type='cubic spline')
# BER_coded_interpolated = interpolator(x=time[mask][::divide_index], y=BER_coded_avg, x_interpolate=time[mask], interpolation_type='cubic spline')

# performance_output = routing_network.performance_output(margin=[margin1_interpolated, margin2_interpolated, margin3_interpolated],
#                                                         errors=errors_interpolated,
#                                                         errors_coded=errors_coded_interpolated)


# Here, the data rate is converted from a constant value to a variable value
# This is done by setting the fluctuating received power equal to the threshold power and
# computing the corresponding data rate that is needed to increase/decrease this power.
# data_rate_var = routing_network.variable_data_rate(P_r_threshold=LCT.P_r_thres, PPB_threshold=LCT.PPB_thres)


# Save all data to csv file: First merge geometrical output and performance output dictionaries. Then save to csv file.
# save_to_file([geometrical_output, performance_output])

#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------
time_cross_section = [0.52, 0.545]
indices = []
for t in time_cross_section:
    t = t * 3600
    index = np.argmin(abs(time[mask][::divide_index] - t))
    indices.append(index)
link.plot(time=time[mask], indices=indices, elevation=routing_network.geometrical_output['elevation'][::divide_index], type='table')

def plot_slow_variables():
    fig, ax_macro = plt.subplots(4, 1)
    ax_macro[0].scatter(time_hrs, routing_network.geometrical_output['link number'],
                       c=routing_network.geometrical_output['link number'], cmap='Set1', s=5)
    ax_macro[0].set_ylabel('Link number')
    ax_macro[1].set_ylabel('Elevation (deg)')

    ax_macro[1].scatter(time_hrs, np.rad2deg(geometrical_output['elevation']), s=10)
    ax_macro[2].set_ylabel('$h_{ext}$ (dB)')
    ax_macro[2].scatter(time_hrs, W2dB(h_ext), s=10)
    ax_macro[3].set_ylabel('P_{RX,0}$ (dBm)')
    # ax_macro[1].scatter(time_hr, W2dB(h_ext), s=10)
    ax_macro[3].scatter(time_hrs, W2dBm(P_r_0), s=10)

    ax_macro[3].set_xlabel('Time (hours)')

    ax_macro[0].legend()
    ax_macro[1].legend()
    ax_macro[2].legend()
    ax_macro[0].grid()
    ax_macro[1].grid()
    ax_macro[2].grid()
    plt.show()

def plot_mission_performance_output_coarse_fine():
    fig, ax = plt.subplots(4, 1)
    # Plot coarse response
    ax[0].scatter(time_hrs, routing_network.geometrical_output['link number'],
                       c=routing_network.geometrical_output['link number'], cmap='Set1', s=10)
    ax[1].plot(np.ones(2) * time_cross_section[0], [0, 15], color='black')
    ax[1].plot(np.ones(2) * time_cross_section[1], [0, 15], color='black')
    ax[1].scatter(time_hrs, W2dB(margin[1]), s=15, label='Coarse margin - $BER_{min}$=1.0E-6')
    ax[2].scatter(time_hrs, BER, s=15, label='Coarse BER')
    # Plot fine response
    ax[1].scatter(time_hrs, W2dB(margin2_interpolated), s=5,
                  label='Interpolated fine margin - $BER_{min}$=1.0E-6')
    print(np.shape(time_hrs[::1]), np.shape(W2dB(margin[1])))
    ax[1].scatter(time_hrs[::divide_index], W2dB(margin[1]), s=15, label='Fine margin - $BER_{min}$=1.0E-6')
    ax[2].scatter(time_hrs, BER_interpolated, s=5, label='Interpolated fine BER')
    ax[2].scatter(time_hrs[::divide_index], BER, s=15, label='Fine BER')
    # ax[2].scatter(time_hr[::divide_index], BER_coded_micro, s=15, label='Fine BER coded')
    # ax[2].scatter(time_hr, BER_coded_interpolated, s=5, label='Interpolated fine BER coded')

    ax[3].scatter(time_hrs, fractional_BER_fade, s=15, label='Coarse BER fade')

    ax[0].set_ylabel('Link number')
    ax[1].set_ylabel('Link margin (dB)')
    ax[2].set_ylabel('BER')
    ax[2].set_yscale('log')
    ax[2].set_ylim(1.0E-20, 0.5)
    ax[3].set_ylabel('BER fractional fade')
    ax[3].set_xlabel('Time (s)')

    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()

    plt.show()
def plot_mission_performance_output_parameters():
    time_hr = time[mask] / 3600.0

    fig_dim2, ax_dim2 = plt.subplots(4,1)
    ax_dim2[0].set_title('Mission level performance output time-series')
    ax_dim2[0].scatter(time_hr, routing_network.geometrical_output['link number'], c=routing_network.geometrical_output['link number'], cmap='Set1')
    ax_dim2[0].set_ylabel('Link number')

    # ax_dim2[1].plot(np.ones(2) * time_cross_section[0], [0, 15], color='black')
    # ax_dim2[1].plot(np.ones(2) * time_cross_section[1], [0, 15], color='black')
    # ax_dim2[1].plot(np.ones(2) * time_cross_section[2], [0, 15], color='black')
    ax_dim2[1].scatter(time_hr, W2dB(margin[0]), label='Link margin - $BER_{min}$=1.0E-3')
    ax_dim2[1].scatter(time_hr, W2dB(margin[1]), label='Link margin - $BER_{min}$=1.0E-6')
    ax_dim2[1].scatter(time_hr, W2dB(margin[2]), label='Link margin - $BER_{min}$=1.0E-9')
    ax_dim2[1].set_ylabel('Comm \n Link margin (dB)')

    # ax_dim2[2].scatter(time_hr, errors/1.0E6, s=1,
    #                 label='Uncoded, total throughput='+str(throughput.sum()/1.0E12)+'Tb')
    # ax_dim2[2].scatter(time_hr, errors_coded / 1.0E6, s=1,
    #                 label='RS coded ('+str(N)+','+str(K)+'), interleaving='+str(latency_interleaving)+'s, \n'
    #                       'total throughput='+str(throughput_coded.sum()/1.0E12)+'Tb')
    # ax_dim2[2].set_ylabel('Error bits \n per stepsize (Mb/s)')
    # ax_dim2[2].set_yscale('log')
    # ax_dim2[2].set_ylim(performance_output['errors'].min() / 1.0E6, performance_output['errors'].max() / 1.0E6)

    ax_dim2[2].scatter(time_hr, fractional_BER_fade , s=1, label='Uncoded')
    # ax_dim2[2].scatter(time_hr, errors_coded / 1.0E6, s=1, label='RS coded (' + str(N) + ',' + str(K) + '), interleaving=' + str(
    #                        latency_interleaving) + 's')
    ax_dim2[2].set_ylabel('Fractional BER \n above threshold (BER=1.0E-6)')

    ax_dim2[3].scatter(time_hr, latency, s=1, label='interleaving='+str(latency_interleaving)+'s')
    ax_dim2[3].set_ylabel('Latency (s)')
    ax_dim2[3].set_xlabel('Time (hrs)')

    ax_dim2[0].legend(fontsize=10)
    ax_dim2[1].legend(fontsize=10)
    ax_dim2[2].legend(fontsize=10)
    ax_dim2[3].legend(fontsize=10)
    ax_dim2[0].grid()
    ax_dim2[1].grid()
    ax_dim2[2].grid()
    ax_dim2[3].grid()
    plt.show()

def plot_range_elevation_distribution(elevation, ranges, time, bins):
    ranges = ranges / 1000
    # CREATE DISTRIBUTION (HISTOGRAM) OF GEOMETRICAL DATA
    time_hr = time / 3600.0
    x_elev = np.linspace(elevation.min(), elevation.max(), bins)
    hist_e = np.histogram(elevation, bins=bins)
    rv = rv_histogram(hist_e, density=False)
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

    fig, ax = plt.subplots(2,2)
    ax[0,0].set_title('bins: ' + str(bins) + '\n'
                      '$\Delta L$: '+ str(np.round(np.diff(hist_R_midpoints)[0]/1000,2)) +
                      'km, $T_{fs}$='+str(np.round(np.diff(T_fs)[0],2)))

    ax[0,1].set_title('Samples before routing: '+str(samples_mission_level) + '\n'
                      'Samples unfiltered: '+str(samples_link_level) + '\n'
                      'Samples filtered: '+str(len(time)))
    ax[0,0].plot(x_R, pdf_R)
    ax[0,0].plot(x_R, cdf_R)
    ax[0,0].hist(ranges, density=True, bins=bins)
    ax[0,0].set_ylabel('Prob. density')
    ax[0,0].set_xlabel('Range (km)')
    ax[0,1].scatter(time_hr, ranges, s=0.5)
    ax[0,1].set_ylabel('Range (km)')
    ax[0,1].set_xlabel('Time (hours)')

    ax[1, 0].set_title('$\Delta \epsilon$: ' + str(np.round(np.diff(hist_e_midpoints)[0], 2)) + '$\degree$')
    ax[1, 0].plot(x_elev, pdf_elev)
    ax[1, 0].plot(x_elev, cdf_elev)
    ax[1, 0].hist(elevation, density=True, bins=bins)
    ax[1, 0].set_ylabel('Prob. density')
    ax[1, 0].set_xlabel('Elevation (rad)')
    ax[1, 1].scatter(time_hr, np.rad2deg(elevation), s=0.5)
    ax[1, 1].set_ylabel('Elevation (degrees)')
    ax[1, 1].set_xlabel('Time (hours)')

    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()
    plt.show()

plot_slow_variables()
# plot_mission_performance_output_coarse_fine()
plot_mission_performance_output_parameters()
plot_range_elevation_distribution(elevation=geometrical_output['elevation'], ranges=geometrical_output['ranges'], time=time[mask], bins=100)