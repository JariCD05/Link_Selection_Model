# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Import input parameters and helper functions
from input import *
from helper_functions import *
from JM_Visualisations_mission_level import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
#from JM_Atmosphere import JM_attenuation, JM_turbulence
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
from JM_Perf_Param_latency_data_transfer import Latency_data_transfer_performance
from throughput_test import SatelliteThroughputCorrected


#import link selection
from JM_link_selection import link_selection

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


# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS-SETUP---------------------------
# ------------------------------------------------------------------------


availability_vector = (link.LM_comm_BER6 >= 1.0).astype(int)
availability_vector = np.array_split(availability_vector, num_satellites)
#print(availability_vector)



###-------------------------Visibility------------------------------------

Applicable_links_instance = applicable_links(time)
applicable_output, sats_applicable = Applicable_links_instance.applicability(link_geometry.geometrical_output, time, step_size_link)



# ------------------------------------------------------------------------
# --------------------------Throughput Test Setup-------------------------
# ------------------------------------------------------------------------

#throughput_test_instance = SatelliteThroughputCorrected(time)
#throughput_test = throughput_test_instance.get_throughput()

# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS---------------------------
# ------------------------------------------------------------------------

Availability_performance_instance = Availability_performance(time, link_geometry, availability_vector) #NORMAL CASE THIS SHOULD BE availability_vector
BER_performance_instance = ber_performance(time, link_geometry, throughput)
Cost_performance_instance = Cost_performance(time, link_geometry)
Latency_performance_instance = Latency_performance(time, link_geometry)
Latency_data_transfer_performance_instance = Latency_data_transfer_performance(time, link_geometry)
Throughput_performance_instance = Throughput_performance(time, link_geometry, throughput) #NORMAL CASE THIS SHOULD BE througphut
 


# Now call the method on the instance and initiliaze the four matrices

#Availability
availability_performance = Availability_performance_instance.calculate_availability_performance()
normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance(data = availability_performance, sats_applicable=sats_applicable)
availability_performance_including_penalty = Availability_performance_instance.calculate_availability_performance_including_penalty(T_acq = 20, step_size_link = 5)
normalized_availability_performance_including_penalty = Availability_performance_instance.calculate_normalized_availability_performance_including_penalty(data = availability_performance_including_penalty, sats_applicable=sats_applicable)


#print(availability_performance)
#print(normalized_availability_performance)


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
propagation_latency = Latency_performance_instance.calculate_latency_performance()                                          # NORMAL EQUATION IS calculate_latency_performance OPTIONAL: calculate_latency_performance_averaged
normalized_latency_performance = Latency_performance_instance.distance_normalization_min(propagation_latency)

# Latency Data transfer
latency_data_transfer = Latency_data_transfer_performance_instance.calculate_latency_data_transfer_performance()                                          
normalized_latency_data_transfer_performance = Latency_data_transfer_performance_instance.distance_normalization_data_transfer(latency_data_transfer)

#Throughput
throughput_performance = Throughput_performance_instance.calculate_throughput_performance_including_decay(decay_rate=decay_rate)             # NORMAL EQUATION IS calculate_throughput_performance OPTIONAL: calculate_throughput_performance_including_decay
normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance(data = throughput_performance, data_rate_ac=data_rate_ac)
#print(throughput_performance)
#print(normalized_throughput_performance)

# ------------------------------------------------------------------------
# --------------------------client Input---------------------------
# ------------------------------------------------------------------------

# Configuration parameters
T_acq = 20  # Acquisition time in seconds
mission_time = 3600  # Total mission time in seconds
Num_opt_head = 1  # Number of optical heads



# define input weights per performance parameter
client_input_availability = 0.2
client_input_BER = 0.2
client_input_cost = 0.1
client_input_latency = 0.1
client_input_latency_data_transfer = 0.1
client_input_throughput = 0.3

weights = [client_input_availability, client_input_BER, client_input_cost, client_input_latency, client_input_latency_data_transfer, client_input_throughput]
performance_matrices = [normalized_availability_performance, normalized_BER_performance, normalized_cost_performance, normalized_latency_performance, normalized_latency_data_transfer_performance, normalized_throughput_performance]
performance_matrices_including_penalty = [normalized_availability_performance_including_penalty, normalized_BER_performance_including_penalty, normalized_cost_performance_including_penalty, normalized_latency_performance, normalized_latency_data_transfer_performance, normalized_throughput_performance]


#call on link selection
link_selection_instance = link_selection()


# Example usage and CSV export
active_satellites, data_frame = link_selection_instance.find_and_track_active_satellites_with_pandas(weights, sats_applicable, performance_matrices, performance_matrices_including_penalty, availability_vector)
data_frame.to_csv('satellite_performance_detailed.csv', index=False)




# Now call the function with the actual parameters
#plot_active_satellites(time_steps, active_satellites, num_satellites)
from matplotlib.animation import FuncAnimation
from PIL import Image
# Configuration and data setup
num_columns = len(active_satellites)  # Based on the length of your 'active_satellites' array
time_steps = np.arange(num_columns)  # Create an array for the time steps

# Load and prepare the satellite image
satellite_img = Image.open("satellite.png")  # Adjust path if necessary
satellite_img = satellite_img.resize((200, 2000))  # Resize image for visibility in plot

# Prepare for plotting
fig, ax = plt.subplots(figsize = (15,10))
ax.set_xlim(0, num_columns)
ax.set_ylim(-1, num_satellites)  # -1 to include "No link" below the first satellite index
line, = ax.plot([], [], lw=2)
satellite_icon = ax.imshow(satellite_img, extent=(0, 1, -1, 0))  # Initial placement

# Function to initialize the animation
def init():
    line.set_data([], [])
    satellite_icon.set_extent((0, 0, -1, -1))  # Hide icon initially
    return line, satellite_icon,

fps = 2.5  # For example, if your animation runs at 2.5 frames per second
shift_per_second = 0.1  # Shift the window 5 timesteps to the right every second

# Determine the number of frames after which to shift the window
frames_per_shift = fps  # Assuming you want to shift every second
window_width = 100

# Function to update the animation at each frame
def update(frame):
    xdata = time_steps[:frame]  # Time steps up to the current frame
    # Ensure ydata contains elements up to the current frame, handle "No link" properly
    ydata = [-1 if i >= len(active_satellites) or active_satellites[i] == "No link" else int(active_satellites[i]) for i in range(frame)]
    # Update the line data
    line.set_data(xdata, ydata)
    # Only update the satellite position if there is corresponding ydata for the frame
    if ydata:  # Check if ydata is not empty
        satellite_position = ydata[-1]  # Get the last item safely since we now know ydata is not empty
        # Update the satellite icon's position, adjusting if frame is within bounds
        if frame < num_columns:
            satellite_icon.set_extent((frame - 0.5, frame + 0.5, satellite_position - 0.5, satellite_position + 0.5))
    else:
        # Optionally, hide the satellite icon if there's no valid position
        satellite_icon.set_extent((0, 0, 0, 0))  # This hides the icon by setting its extent to zero area
    return line, satellite_icon



# Create the animation
ani = FuncAnimation(fig, update, frames=num_columns, init_func=init, blit=True, repeat=False, interval=400)


plt.title('Link selection')
plt.xlabel('Time Step')
plt.ylabel('Link selected')
plt.yticks(range(-1, num_satellites), ['No link'] + [f'Sat {i+1}' for i in range(num_satellites)])
plt.grid(True)
folder_path = 'animations'  # Adjust path if the folder is not in the current working directory
filename = 'link_selection.mp4'
full_path = f"{folder_path}/{filename}"

# Save the animation
ani.save(full_path, writer='ffmpeg', fps=2.5)  # Ensure ffmpeg is installed or use an available writer

# Show the plot
plt.show()


#-------------- visualizations -----------------------------------------------------------


#Links_applicable.plot_satellite_visibility_scatter(time=time)
#Links_applicable.plot_satellite_visibility(time = time)

#plot BER vs availability
#BER_performance_instance.BER_performance_vs_availability_visualization(throughput, availability_vector)

#plot availability
#Availability_performance_instance.availability_visualization(availability_performance, normalized_availability_performance, availability_performance_including_penalty, normalized_availability_performance_including_penalty)


#plot BER
#BER_performance_instance.BER_visualization(BER_performance, normalized_BER_performance, BER_performance_including_penalty, normalized_BER_performance_including_penalty)

#plot Latency
#Latency_performance_instance.latency_visualization(propagation_latency, normalized_latency_performance)

#plot Latency data trasnfer
#Latency_data_transfer_performance_instance.latency_data_transfer_visualization(latency_data_transfer, normalized_latency_data_transfer_performance)

#plot throughput
#Throughput_performance_instance.throughput_visualization(throughput_performance, normalized_throughput_performance)

#plot Cost
#Cost_performance_instance.cost_visualization(cost_performance, normalized_cost_performance, cost_performance_including_penalty, normalized_cost_performance_including_penalty)

#plot performance of each individual satellite, stating which one is active and which one is not
#Applicable_links_instance.plot_satellite_visibility_scatter_update()

#plot_individual_satellite_performance_update(active_satellites, performance_over_time, num_satellites, time_steps, sats_applicable)

#print_link_handovers(active_satellites)



