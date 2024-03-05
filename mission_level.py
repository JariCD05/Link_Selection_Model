import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Routing_network import routing_network
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level


# Process to activate tudatpy 
# 1. within command line "conda activate tudat-space"
# 2. change python version on bottom right to 3.10.13{'tudat-space':conda}

print('')
print('------------------END-TO-END-LASER-SATCOM-MODEL-------------------------')
#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
print('')
print('-----------------------------------MISSION-LEVEL-----------------------------------------')
#------------------------------------------------------------------------
#------------------------------------LCT---------------------------------
#------------------------------------------------------------------------
# Compute the sensitivity and compute the threshold
LCT = terminal_properties()
LCT.BER_to_P_r(BER = BER_thres,
               modulation = modulation,
               detection = detection,
               threshold = True)
PPB_thres = PPB_func(LCT.P_r_thres, data_rate)

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
# First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
# Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
link_geometry.geometrical_outputs()
# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
mission_duration = time[-1] - time[0]
# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])

#------------------------------------------------------------------------
#---------------------------ROUTING-OF-LINKS-----------------------------
#------------------------------------------------------------------------
# The routing_network class takes the relative geometrical state between AIRCRAFT and all SATELLITES and
# Performs optimization to select a number of links, with 'routing_network.routing'
# Cost variables are:
#   (1) Maximum link time
#   (2) maximum elevation during one link
# Constraints are:
#   (1) Minimum elevation angle: 10 degrees
#   (2) Positive elevation rate at start of link
routing_network = routing_network(time=time)
routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_output, time, step_size_link)

total_time = len(time)*step_size_link
comm_time = len(flatten(routing_output['time']))*step_size_link
acq_time = routing_network.total_acquisition_time



# Options are to analyse 1 link or analyse 'all' links
#   (1) link_number == 'all'   : Creates 1 vector for each geometric variable for each selected link & creates a flat vector
#   (2) link_number == 1 number: Creates 1 vector for each geometric variable
if link_number == 'all':
    time_links  = flatten(routing_output['time'     ])
    time_links_hrs = time_links / 3600.0
    ranges     = flatten(routing_output['ranges'    ])
    elevation  = flatten(routing_output['elevation' ])
    zenith     = flatten(routing_output['zenith'    ])
    slew_rates = flatten(routing_output['slew rates'])
    heights_SC = flatten(routing_output['heights SC'])
    heights_AC = flatten(routing_output['heights AC'])
    speeds_AC  = flatten(routing_output['speeds AC'])

    time_per_link       = routing_output['time'      ]
    time_per_link_hrs   = time_links / 3600.0
    ranges_per_link     = routing_output['ranges'    ]
    elevation_per_link  = routing_output['elevation' ]
    zenith_per_link     = routing_output['zenith'    ]
    slew_rates_per_link = routing_output['slew rates']
    heights_SC_per_link = routing_output['heights SC']
    heights_AC_per_link = routing_output['heights AC']
    speeds_AC_per_link  = routing_output['speeds AC' ]

else:
    time_links     = routing_output['time'      ][link_number]
    time_links_hrs = time_links / 3600.0
    ranges         = routing_output['ranges'    ][link_number]
    elevation      = routing_output['elevation' ][link_number]
    zenith         = routing_output['zenith'    ][link_number]
    slew_rates     = routing_output['slew rates'][link_number]
    heights_SC     = routing_output['heights SC'][link_number]
    heights_AC     = routing_output['heights AC'][link_number]
    speeds_AC      = routing_output['speeds AC' ][link_number]


# Define cross-section of macro-scale simulation based on the elevation angles.
# These cross-sections are used for micro-scale plots.
elevation_cross_section = [2.0, 20.0, 40.0]
index_elevation = 1
indices, time_cross_section = cross_section(elevation_cross_section, elevation, time_links)

print('')
print('-------------------------------------LINK-LEVEL------------------------------------------')
print('')
#------------------------------------------------------------------------
#-------------------------------ATTENUATION------------------------------

att = attenuation(att_coeff=att_coeff, H_scale=scale_height)
att.h_ext_func(range_link=ranges, zenith_angles=zenith, method=method_att)
att.h_clouds_func(method=method_clouds)
h_ext = att.h_ext * att.h_clouds
# Print attenuation parameters
att.print()
#------------------------------------------------------------------------
#-------------------------------TURBULENCE-------------------------------
# The turbulence class is initiated here. Inside the turbulence class, there are multiple methods that are run directly.
# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
# With Cn^2 and r0, the variances for scintillation and beam wander are computed


turb = turbulence(ranges=ranges,
                  h_AC=heights_AC,
                  h_SC=heights_SC,
                  zenith_angles=zenith,
                  angle_div=angle_div)
turb.windspeed_func(slew=slew_rates,
                    Vg=speeds_AC,
                    wind_model_type=wind_model_type)
turb.Cn_func()
turb.frequencies()
r0 = turb.r0_func()
turb.var_rytov_func()
turb.var_scint_func()
turb.WFE(tip_tilt="YES")
turb.beam_spread()
turb.var_bw_func()
turb.var_aoa_func()

print('')
print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
print('')
print('-----------------------------------CHANNEL-LEVEL-----------------------------------------')
print('')
for i in indices:
    turb.print(index=i, elevation=np.rad2deg(elevation), ranges=ranges, Vg=link_geometry.speed_AC.mean(),slew=slew_rates)
# ------------------------------------------------------------------------
# -----------------------------LINK-BUDGET--------------------------------
# The link budget class computes the static link budget (without any micro-scale effects)
# Then it generates a link margin, based on the sensitivity
link = link_budget(angle_div=angle_div, w0=w0, ranges=ranges, h_WFE=turb.h_WFE, w_ST=turb.w_ST, h_beamspread=turb.h_beamspread, h_ext=h_ext)
link.sensitivity(LCT.P_r_thres, PPB_thres)

# Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
P_r_0, P_r_0_acq = link.P_r_0_func()
link.print(index=indices[index_elevation], elevation=elevation, static=True)

# ------------------------------------------------------------------------
# -------------------------MACRO-SCALE-SOLVER-----------------------------
noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun, index=indices[index_elevation])
SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)

# ------------------------------------------------------------------------
# ----------------------------MICRO-SCALE-MODEL---------------------------
# Here, the channel level is simulated, losses and Pr as output
P_r, P_r_perfect_pointing, PPB, elevation_angles, losses, angles = \
    channel_level(t=t_micro,
                  link_budget=link,
                  plot_indices=indices,
                  LCT=LCT, turb=turb,
                  P_r_0=P_r_0,
                  ranges=ranges,
                  angle_div=link.angle_div,
                  elevation_angles=elevation,
                  samples=samples_channel_level,
                  turb_cutoff_frequency=turbulence_freq_lowpass)
h_tot = losses[0]
h_scint = losses[1]
h_RX    = losses[2]
h_TX    = losses[3]
h_bw    = losses[4]
h_aoa   = losses[5]
h_pj_t  = losses[6]
h_pj_r  = losses[7]
h_tot_no_pointing_errors = losses[-1]
r_TX = angles[0] * ranges[:, None]
r_RX = angles[1] * ranges[:, None]

# Here, the bit level is simulated, SNR, BER and throughput as output
if coding == 'yes':
    SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_coding = \
        bit_level(LCT=LCT,
                  t=t_micro,
                  plot_indices=indices,
                  samples=samples_channel_level,
                  P_r_0=P_r_0,
                  P_r=P_r,
                  elevation_angles=elevation,
                  h_tot=h_tot)

else:
    SNR, BER, throughput = \
        bit_level(LCT=LCT,
                  t=t_micro,
                  plot_indices=indices,
                  samples=samples_channel_level,
                  P_r_0=P_r_0,
                  P_r=P_r,
                  elevation_angles=elevation,
                  h_tot=h_tot)


# ----------------------------FADE-STATISTICS-----------------------------

number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
mean_fade_time = fractional_fade_time / number_of_fades * interval_channel_level

# Power penalty in order to include a required fade fraction.
# REF: Giggenbach (2008), Fading-loss assessment
h_penalty   = penalty(P_r=P_r, desired_frac_fade_time=desired_frac_fade_time)                                           
h_penalty_perfect_pointing   = penalty(P_r=P_r_perfect_pointing, desired_frac_fade_time=desired_frac_fade_time)
P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing

# ---------------------------------LINK-MARGIN--------------------------------
margin     = P_r / LCT.P_r_thres[1]

# -------------------------------DISTRIBUTIONS----------------------------
# Local distributions for each macro-scale time step (over micro-scale interval)
pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
if coding == 'yes':
    pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
        distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)

# Global distributions over macro-scale interval
P_r_total = P_r.flatten()
BER_total = BER.flatten()
P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)

if coding == 'yes':
    BER_coded_total = BER_coded.flatten()
    BER_coded_pdf_total, BER_coded_cdf_total, x_BER_coded_total, std_BER_coded_total, mean_BER_coded_total = \
        distribution_function(data=np.log10(BER_coded_total), length=1, min=-30.0, max=0.0, steps=100)


# ------------------------------------------------------------------------
# -------------------------------AVERAGING--------------------------------
# ------------------------------------------------------------------------

# ---------------------------UPDATE-LINK-BUDGET---------------------------
# All micro-scale losses are averaged and added to the link budget
# Also adds a penalty term to the link budget as a requirement for the desired fade time, defined in input.py
link.dynamic_contributions(PPB=PPB.mean(axis=1),
                           T_dyn_tot=h_tot.mean(axis=1),
                           T_scint=h_scint.mean(axis=1),
                           T_TX=h_TX.mean(axis=1),
                           T_RX=h_RX.mean(axis=1),
                           h_penalty=h_penalty,
                           P_r=P_r.mean(axis=1),
                           BER=BER.mean(axis=1))


if coding == 'yes':
    link.coding(G_coding=G_coding.mean(axis=1),
                BER_coded=BER_coded.mean(axis=1))
    P_r = P_r_coded
# A fraction (0.9) of the light is subtracted from communication budget and used for tracking budget
link.tracking()
link.link_margin()


# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS---------------------------
# ------------------------------------------------------------------------

# Availability
# No availability is assumed below link margin threshold
availability_vector = mask.astype(int)
find_lm = np.where(link.LM_comm_BER6 < 1.0)[0]
time_link_fail = time_links[find_lm]
find_time = np.where(np.in1d(time, time_link_fail))[0]
availability_vector[find_time] = 0.0

# Reliability
# No reliability is assumed below link margin threshold
reliability_BER = BER.mean(axis=1)
reliability_BER[find_lm] = 0.0

# Actual throughput
# No throughput is assumed below link margin threshold
throughput[find_lm] = 0.0
# Potential throughput with the Shannon-Hartley theorem
noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=link.P_r, I_sun=I_sun, index=indices[index_elevation])
SNR_penalty, Q_penalty = LCT.SNR_func(link.P_r, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
C = BW * np.log2(1 + SNR_penalty)

# Latency is computed as a macro-scale time-series
# The only assumed contributions are geometrical latency and interleaving latency.
# Latency due to coding/detection/modulation/data processing can be optionally added.
latency_propagation = ranges / speed_of_light
latency_transmission = 1 / data_rate
latency_qeue = 5.0e-3
latency_processing = 3.0e-3
latency = latency_propagation + latency_transmission + latency_qeue + latency_processing


# ------------------------------------------------------------------------
# ---------------------------------OUTPUT---------------------------------

performance_output = {
        'time'                : [],
        'throughput'          : [],
        'link number'         : [],
        'Pr 0'                : [],
        'Pr mean'             : [],
        'Pr penalty'          : [],
        'BER mean'            : [],
        'fractional fade time': [],
        'mean fade time'      : [],
        'number of fades'     : [],
        'link margin'         : [],
        'latency'             : [],
        'Pr mean (perfect pointing)'   : [],
        'Pr penalty (perfect pointing)': [],
        'Pr coded'            : [],
        'BER coded'           : [],
        'throughput coded'    : [],

    }

if link_number == 'all':
    performance_output['link number'] = routing_output['link number']

    for i in range(len(routing_output['link number'])):
        condition_1 = time[mask] >= routing_output['time'][i][0]
        condition_2 = time[mask] <= routing_output['time'][i][-1]
        conditions = [condition_1, condition_2]
        full_condition = [all(condition) for condition in zip(*conditions)]

        performance_output['time'].append(time_links[full_condition])
        performance_output['throughput'].append(throughput[full_condition])
        performance_output['Pr 0'].append(P_r_0[full_condition])
        performance_output['Pr mean'].append(P_r.mean(axis=1)[full_condition])
        performance_output['Pr penalty'].append(link.P_r[full_condition])
        performance_output['fractional fade time'].append(fractional_fade_time[full_condition])
        performance_output['mean fade time'].append(mean_fade_time[full_condition])
        performance_output['number of fades'].append(number_of_fades[full_condition])
        performance_output['BER mean'].append(BER.mean(axis=1)[full_condition])
        performance_output['link margin'].append(link.LM_comm_BER6[full_condition])
        performance_output['latency'].append(latency[full_condition])
        performance_output['Pr mean (perfect pointing)'   ].append(P_r_perfect_pointing.mean(axis=1)[full_condition])
        performance_output['Pr penalty (perfect pointing)'].append(P_r_penalty_perfect_pointing[full_condition])

        if coding == 'yes':
            performance_output['Pr coded'].append(P_r_coded[full_condition])
            performance_output['BER coded'].append(BER_coded.mean(axis=1)[full_condition])
            performance_output['throughput coded'].append(throughput_coded[full_condition])

else:
    performance_output['time']                 = time_links
    performance_output['throughput']           = throughput
    performance_output['Pr 0']                 = P_r_0
    performance_output['Pr mean']              = P_r.mean(axis=1)
    performance_output['Pr penalty']           = link.P_r
    performance_output['fractional fade time'] = fractional_fade_time
    performance_output['mean fade time']       = mean_fade_time
    performance_output['number of fades']      = number_of_fades
    performance_output['BER mean']             = BER.mean(axis=1)
    performance_output['link margin']          = margin
    performance_output['latency']              = latency
    performance_output['Pr mean (perfect pointing)'] = P_r_perfect_pointing.mean(axis=1)
    performance_output['Pr penalty (perfect pointing)'] = P_r_penalty_perfect_pointing
    if coding == 'yes':
        performance_output['Pr coded'].append(P_r_coded)
        performance_output['BER coded'].append(BER_coded.mean(axis=1))
        performance_output['throughput coded'].append(throughput_coded)


# Save all data to csv file: First merge geometrical output and performance output dictionaries. Then save to csv file.
# save_to_file([geometrical_output, performance_output])






