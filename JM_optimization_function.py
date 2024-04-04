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


# let's assume that the output from the mission level is four matrices
# the collumns in the matrices are the time indeces
# the rows are the number of satellites
# the names of the matrices are, normalized_latency, throuhgput, availability, Bit_error_rate

# ------------------------------------------------------------------
# ------------------------------------------------------------------
#-------------------------- SETUP ----------------------------
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

print(sats_applicable)

# Create an instance of the performance classes




Availability_performance_instance = Availability_performance(time, link_geometry)
# BER_performance_instance = BER_performance()
Cost_performance_instance = Cost_performance(time, link_geometry)
Latency_performance_instance = Latency_performance(time, link_geometry)
Throughput_performance_instance = Throughput_performance(time, link_geometry)


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

#print(throughput_performance)
#print(normalized_throughput_performance)




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
    print("performance over time at timestamp 5", performance_over_time[5])
    print("performance over time", performance_over_time)


    return active_satellites


active_satellites = find_and_track_active_satellites(weights, sats_applicable, *performance_matrices)

print("normalized Latency", normalized_latency_performance)
print("normalized cost", normalized_cost_performance)
print("Active satellites", active_satellites)

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





