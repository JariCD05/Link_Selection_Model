# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from JM_Atmosphere import JM_attenuation, JM_turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

#import link applicabality
from JM_applicable_links import applicable_links



#import performance paramaters 
from JM_Perf_Param_Availability import Availability_performance
from JM_Perf_Param_BER import ber_performance
from JM_Perf_Param_Latency import Latency_performance
from JM_Perf_Param_Throughput import Throughput_performance
from JM_Perf_Param_Cost import Cost_performance



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

Links_applicable = applicable_links(time=time)
applicable_output, sats_applicable = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)


# Define cross-section of macro-scale simulation based on the elevation angles.
# These cross-sections are used for micro-scale plots.
elevation_cross_section = [2.0, 20.0, 40.0]
index_elevation = 1

# Retrieve data for all links directly
time_links = flatten(applicable_output['time'])
time_links_hrs = [t / 3600.0 for t in time_links]
ranges = flatten(applicable_output['ranges'])
elevation = flatten(applicable_output['elevation'])
zenith = flatten(applicable_output['zenith'])
slew_rates = flatten(applicable_output['slew rates'])
heights_SC = flatten(applicable_output['heights SC'])
heights_AC = flatten(applicable_output['heights AC'])
speeds_AC = flatten(applicable_output['speeds AC'])


time_per_link       = applicable_output['time']
time_per_link_hrs   = time_links / 3600.0
ranges_per_link     = applicable_output['ranges'    ]
elevation_per_link  = applicable_output['elevation' ]
zenith_per_link     = applicable_output['zenith'    ]
slew_rates_per_link = applicable_output['slew rates']
heights_SC_per_link = applicable_output['heights SC']
heights_AC_per_link = applicable_output['heights AC']
speeds_AC_per_link  = applicable_output['speeds AC' ]

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



#throughput = np.array_split(throughput, num_satellites)
throughput = np.array_split(throughput, num_satellites)


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



# No reliability is assumed below link margin threshold
reliability_BER = BER.mean(axis=1)


#print(reliability_BER)
#print(len(reliability_BER))


# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS---------------------------
# ------------------------------------------------------------------------


availability_vector = (link.LM_comm_BER6 >= 1.0).astype(int)
availability_vector = np.array_split(availability_vector, num_satellites)

#print(availability_vector)

# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS---------------------------
# ------------------------------------------------------------------------

Availability_performance_instance = Availability_performance(time, link_geometry, availability_vector)
BER_performance_instance = ber_performance(time, link_geometry, throughput)
Cost_performance_instance = Cost_performance(time, link_geometry)
Latency_performance_instance = Latency_performance(time, link_geometry)
Throughput_performance_instance = Throughput_performance(time, link_geometry, throughput)


# Now call the method on the instance and initiliaze the four matrices

#Availability
availability_performance = Availability_performance_instance.calculate_availability_performance()
normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance(data = availability_performance, sats_applicable=sats_applicable)
availability_performance_including_penalty = Availability_performance_instance.calculate_availability_performance_including_penalty(T_acq = 20, step_size_link = 5)
normalized_availability_performance_including_penalty = Availability_performance_instance.calculate_normalized_availability_performance_including_penalty(data = availability_performance_including_penalty, sats_applicable=sats_applicable)

#BER
BER_performance = BER_performance_instance.calculate_BER_performance()
normalized_BER_performance = BER_performance_instance.calculate_normalized_BER_performance(data = BER_performance, availability_performance=availability_performance)
BER_performance_including_penalty = BER_performance_instance.calculate_BER_performance_including_penalty(T_acq = 20, step_size_link = 5)
normalized_BER_performance_including_penalty = BER_performance_instance.calculate_normalized_BER_performance_including_penalty(data = BER_performance_including_penalty, sats_applicable=sats_applicable)

# Cost
cost_performance = Cost_performance_instance.calculate_cost_performance()
normalized_cost_performance = Cost_performance_instance.calculate_normalized_cost_performance(cost_performance)
cost_performance_including_penalty = Cost_performance_instance.calculate_cost_performance_including_penalty()
normalized_cost_performance_including_penalty = Cost_performance_instance.calculate_normalized_cost_performance_including_penalty(cost_performance_including_penalty)


# Latency
propagation_latency = Latency_performance_instance.calculate_latency_performance()
normalized_latency_performance = Latency_performance_instance.distance_normalization_min(propagation_latency=propagation_latency)

#Throughput
throughput_performance = Throughput_performance_instance.calculate_throughput_performance()
normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance(data = throughput_performance, data_rate_ac=data_rate_ac)

import matplotlib.pyplot as plt
import numpy as np
import os

def visualization_performance_parameters(data, title):
    """
    Generates and saves plots for each satellite's performance parameter data with dynamic y-axis limits.
    
    Parameters:
    - data: NumPy array with shape (num_satellites, len(time)), performance data for satellites.
    - title: String, the title and filename base for the plot.
    """
    num_satellites, len_time = data.shape
    save_folder = "midterm"  # Adjust as necessary
    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists
    
    # Creating the plot
    fig, axes = plt.subplots(num_satellites, 1, figsize=(10, num_satellites * 2), sharex=True)
    
    for i in range(num_satellites):
        ax = axes[i] if num_satellites > 1 else axes  # Handle case of single subplot
        ax.plot(data[i, :], label=f'Satellite {i+1}')
        
        # Dynamic ylim based on the maximum value of the current dataset
        max_value = np.max(data[i, :])
        ax.set_ylim(max(max_value * 1.1,1))        
        ax.set_ylabel('Performance')
        ax.legend(loc='upper right')
        if i == num_satellites - 1:  # Only label the x-axis on the bottom subplot
            ax.set_xlabel('Time Stamp')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)  # Add title above all subplots
    plt.show()
    
    # Saving the plot
    filename = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
    fig.savefig(filename)
    print(f"Plot saved to {filename}")



def generate_all_performance_plots():
    # Define your datasets and titles here
    datasets = [
        availability_performance,
        BER_performance,
        cost_performance,
        propagation_latency,
        throughput_performance,
        normalized_availability_performance,
        normalized_BER_performance,
        normalized_cost_performance,
        normalized_latency_performance,
        normalized_throughput_performance,
        availability_performance_including_penalty,
        BER_performance_including_penalty,
        cost_performance_including_penalty
    ]
    titles = [
        "availability_performance",
        "BER_performance",
        "Cost_performance",
        "Latency_performance",
        "Throughput_performance",
        "Normalized Availability Performance",
        "Normalized BER Performance",
        "Normalized Cost Performance",
        "Normalized Latency Performance",
        "Normalized Throughput Performance",
        "Availability Performance Including Penalty",
        "BER Performance Including Penalty",
        "Cost Performance Including Penalty"
    ]
    
    # Ensure datasets are NumPy arrays; if not, convert them
    datasets = [np.array(data) for data in datasets]

    # Loop through datasets and titles, calling the visualization function for each
    for data, title in zip(datasets, titles):
        visualization_performance_parameters(data, title)

# Call the function to generate and save all plots
generate_all_performance_plots()