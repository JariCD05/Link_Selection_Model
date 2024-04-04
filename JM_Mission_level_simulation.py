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
from JM_Perf_Param_BER import BER_performance
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



#print(BER_total)


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
# BER_performance_instance = BER_performance()
Cost_performance_instance = Cost_performance(time, link_geometry)
Latency_performance_instance = Latency_performance(time, link_geometry)
Throughput_performance_instance = Throughput_performance(time, link_geometry, throughput)


# Now call the method on the instance and initiliaze the four matrices

#Availability
availability_performance = Availability_performance_instance.calculate_availability_performance()
normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance(data = availability_performance, sats_applicable=sats_applicable)

# Cost
cost_performance = Cost_performance_instance.calculate_cost_performance()
normalized_cost_performance = Cost_performance_instance.calculate_normalized_cost_performance(cost_performance)


# Latency
propagation_latency = Latency_performance_instance.calculate_latency_performance()
normalized_latency_performance = Latency_performance_instance.distance_normalization_min(propagation_latency=propagation_latency)

#Throughput
throughput_performance = Throughput_performance_instance.calculate_throughput_performance()
normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance(data = throughput_performance, availability_performance=availability_performance)


# Configuration parameters
T_acq = 20  # Acquisition time in seconds
mission_time = 3600  # Total mission time in seconds
Num_opt_head = 1  # Number of optical heads
time_step = 100  # Time step in seconds


# define input weights per performance parameter
client_input_availability = 0.2
client_input_BER = 0
client_input_cost = 0.2
client_input_latency = 0.2
client_input_throughput = 0.4

weights = [client_input_availability, client_input_BER, client_input_cost, client_input_latency, client_input_throughput]

# setup mission outlin
num_columns = mission_time // time_step
visibility_duration = 12 * 60   # 12 minutes in seconds
time_steps = np.arange(0, mission_time, time_step)
num_rows = num_satellites


def initialize_performance_matrices_with_ones(normalized_latency):
    """
    Initialize matrices for throughput, BER, and availability with ones,
    matching the size of the normalized_latency matrix.

    Parameters:
    - normalized_latency: A NumPy array representing the normalized latency.

    Returns:
    - A tuple of matrices for throughput, BER, and availability, all filled with ones.
    """
    # Determine the shape of the normalized_latency matrix
    shape = normalized_latency.shape

    # Initialize matrices for throughput, BER, and availability with ones
    
    dummy_BER_performance = np.ones(shape, dtype=float)
    

    return dummy_BER_performance

#initiliaze dummy matrices

dummy_BER_performance = initialize_performance_matrices_with_ones(normalized_latency=normalized_latency_performance)
#print(len(dummy_BER_performance))
#print(len(dummy_BER_performance[5]))
performance_matrices = [normalized_availability_performance, dummy_BER_performance, normalized_cost_performance, normalized_latency_performance, normalized_throughput_performance]

#print(len(performance_matrices))
def find_and_track_active_satellites(weights, sats_applicable, *performance_matrices):
    sats_applicable=np.array(sats_applicable)
    # Ensure the number of weights matches the number of performance matrices
    if len(weights) != len(performance_matrices):
        raise ValueError("Mismatch in number of weights and matrices")

    # Prepare a matrix to store the weighted sum of performance metrics for each satellite at each time step
    weighted_sum = np.zeros_like(sats_applicable, dtype=float)

    # Apply weights to performance matrices
    for weight, matrix in zip(weights, performance_matrices):
        weighted_sum += weight * matrix

    active_satellites = []
    performance_over_time = []
    for time_step in range(weighted_sum.shape[1]):  # Iterate through each time step
        # Filter to consider only applicable (visible) satellites at this time step
        applicable_at_time_step = sats_applicable[:, time_step]
        performance_at_time_step = weighted_sum[:, time_step] * applicable_at_time_step
        performance_over_time.append(performance_at_time_step)

        # Determine the active satellite by checking the highest weighted performance value, if any are applicable
        if np.any(~np.isnan(performance_at_time_step)) and np.nanmax(performance_at_time_step) > 0:
            active_satellite = np.nanargmax(performance_at_time_step)
            active_satellites.append(active_satellite)
            
        else:
            # No satellites are applicable at this time step
            active_satellites.append("No link")
    #print("sats_applicable", sats_applicable)
    #print("performance over time at timestamp 5", performance_over_time[5])
    #print("performance over time", performance_over_time)


    return active_satellites


active_satellites = find_and_track_active_satellites(weights, sats_applicable, *performance_matrices)

#print("normalized Latency", normalized_latency_performance)
#print("normalized cost", normalized_cost_performance)
#print("Active satellites", active_satellites)

def plot_active_satellites(time_steps, active_satellites, num_rows):
    """
    Plot the active satellites over time.
    Parameters:
    time_steps: An array of time steps.
    active_satellites: A list containing the index of the active satellite or "No link" at each time step.
    num_rows: The number of satellites (used to set the y-axis limits).
    """
    # Convert 'active_satellites' entries from 'No link' to -1, and all others to their respective integer indices
    active_satellite_indices = [-1 if sat == "No link" else int(sat) for sat in active_satellites]
    # Now, plot the corrected active satellite indices over time
    plt.figure(figsize=[15,5])
    plt.plot(time_steps, active_satellite_indices, label='Active Satellite', marker='o', linestyle='-', markersize=5)
    plt.title('Active Satellite Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Satellite Index')
    plt.grid(True)
    # Adjust y-ticks to include all satellite indices plus an extra one for "No link"
    # Here, I set 'Sat 0' label to represent 'No link' for clarity
    plt.yticks(range(-1, num_satellites), ['No link'] + [f'Sat {i+1}' for i in range(num_satellites)])  # Note: Satellites indexed from 1 for readability
    plt.legend()
    plt.show()

# Now call the function with the actual parameters
#plot_active_satellites(time_steps, active_satellites, num_rows)
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
ax.set_ylim(-1, num_rows)  # -1 to include "No link" below the first satellite index
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
plt.yticks(range(-1, num_rows), ['No link'] + [f'Sat {i+1}' for i in range(num_rows)])
plt.grid(True)
ani.save('link_selection.mp4', writer='ffmpeg', fps=2.5)
plt.show()


#Links_applicable.plot_satellite_visibility_scatter(time=time)
#Links_applicable.plot_satellite_visibility(time = time)






