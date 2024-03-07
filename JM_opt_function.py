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

# let's assume that the output from the mission level is four matrices
# the collumns in the matrices are the time indeces
# the rows are the number of satellites
# the names of the matrices are, latency, throuhgput, availability, Bit_error_rate

# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Assuming these values from your scenario
T_acq = 20  # Acquisition time
mission_time = 3600  # Total mission time in seconds
Num_opt_head = 1  # Number of optimal heads
time_step = 10  # Time step in seconds
num_satellites = 5 # Number of satellites

num_columns = mission_time // time_step
visibility_duration = 12 * 60   # 12 minutes in seconds
time_steps = np.arange(0, mission_time, time_step)

# Function to create aligned visibility matrices

num_rows = num_satellites
def create_aligned_visibility_matrices(num_rows, num_columns, visibility_duration, time_step):
    visibility_length = visibility_duration // time_step
    start_time_steps = np.random.randint(0, num_columns - visibility_length, size=num_rows)
    matrices = []
    for _ in range(4):  # For latency, throughput, availability, bit_error_rate
        matrix = np.zeros((num_rows, num_columns))
        for row, start in enumerate(start_time_steps):
            end = start + visibility_length
            matrix[row, start:end] = np.random.rand(visibility_length)
        matrices.append(matrix)
    return matrices

# Re-create the aligned matrices
latency, throughput, availability, bit_error_rate = create_aligned_visibility_matrices(num_rows, num_columns, visibility_duration, time_step)


# Create the matrices with the aligned visibility pattern
latency, throughput, availability, bit_error_rate = create_aligned_visibility_matrices(num_rows, num_columns, visibility_duration, time_step)

#print(latency, throughput, availability, bit_error_rate)  # Return the matrices for inspection or further use

# ------------------------------------------------------------------
# ------------------------------------------------------------------

def apply_custom_penalty(values, active_index, T_acq, mission_time, Num_opt_head):
    """
    Applies a custom penalty to non-active satellites based on T_acq and Num_opt_head.

    Parameters:
    values: An array of weighted values for all satellites at a specific time step.
    active_index: The index of the currently active satellite.
    T_acq: Acquisition time.
    mission_time: Total mission time in seconds.
    Num_opt_head: Number of optimal heads.

    Returns:
    penalized_values: The values after applying the penalty to non-active satellites.
    """
    # Calculate penalty value based on T_acq, mission time, and Num_opt_head
    penalty_value = 0
    if Num_opt_head == 1:
        penalty_value = T_acq / mission_time
    
    # Apply the calculated penalty to non-active satellites
    penalized_values = np.array(values)  # Create a copy to avoid modifying the original array
    for i in range(len(values)):
        if i != active_index:
            penalized_values[i] -= penalty_value  # Apply penalty based on condition
    return penalized_values





#max_satellite_indices = [np.random.randint(0, 10) for _ in range(mission_time // time_step)]  # Replace with actual data

#plot_highest_value_satellites(mission_time, time_step, max_satellite_indices)

def find_and_track_active_satellites(weights, T_acq, mission_time, Num_opt_head, *matrices):
    # Setup as before
    if len(weights) != len(matrices):
        raise ValueError("Mismatch in number of weights and matrices")
    if not all(m.shape == matrices[0].shape for m in matrices):
        raise ValueError("All matrices must have the same shape")

    # Combine weights with matrices
    weighted_matrices = [w * m for w, m in zip(weights, matrices)]
    weighted_sum = sum(weighted_matrices)

    active_satellites = []
    last_active_satellite = -1

    for time_step in range(weighted_sum.shape[1]):  # Iterate through each time step
        values_at_time_step = weighted_sum[:, time_step]

        # Apply penalty only if there was an active satellite before and we are not at the first step
        if last_active_satellite != -1:
            values_at_time_step = apply_custom_penalty(values_at_time_step, last_active_satellite, T_acq, mission_time, Num_opt_head)

        # Determine the active satellite: check the highest value if it's greater than 0
        if np.max(values_at_time_step) > 0:  # There's a visible satellite
            active_satellite = np.argmax(values_at_time_step)
            active_satellites.append(active_satellite)
            last_active_satellite = active_satellite
        else:  # No satellites visible
            active_satellites.append("No link")
            last_active_satellite = -1  # Reset last active satellite since there's no link

    return active_satellites




weights = [0.25, 0.25, 0.25, 0.25]  # Define your weights
active_satellites = find_and_track_active_satellites(weights, T_acq, mission_time, Num_opt_head, latency, throughput, availability, bit_error_rate)
print(active_satellites)





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
    plt.yticks(range(-1, num_rows), ['No link'] + [f'Sat {i+1}' for i in range(num_rows)])  # Note: Satellites indexed from 1 for readability
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
satellite_img = satellite_img.resize((10, 100))  # Resize image for visibility in plot

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

# Function to update the animation at each frame
def update(frame):
    xdata = time_steps[:frame]
    ydata = [-1 if sat == "No link" else int(sat) for sat in active_satellites[:frame]]
    line.set_data(xdata, ydata)
    
    if frame < num_columns:
        satellite_position = ydata[-1]
        satellite_icon.set_extent((frame - 0.5, frame + 0.5, satellite_position - 0.5, satellite_position + 0.5))
    
    return line, satellite_icon,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_columns, init_func=init, blit=True, repeat=False, interval=200)

plt.title('Active Satellite Over Time with Dynamic Image')
plt.xlabel('Time Step')
plt.ylabel('Satellite Index')
plt.yticks(range(-1, num_rows), ['No link'] + [f'Sat {i+1}' for i in range(num_rows)])
plt.grid(True)

#plt.show()



