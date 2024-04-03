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
from JM_applicable_links import applicable_links

from matplotlib.animation import FuncAnimation
from PIL import Image

# Based on the code snippet provided, let's create a simulation where the plane's altitude starts at zero and goes up to 30,000 ft in the first 100 time steps.

# Define the time steps and end time
end_time = 956
time_steps = np.linspace(0, end_time, end_time + 1)

# Define the altitude over time function
def altitude_over_time(t, end_time, max_altitude, ascent_time):
    if t <= ascent_time:
        # Calculate altitude using a simple linear ascent for the first ascent_time steps
        return max_altitude * (t / ascent_time)
    else:
        # Keep the altitude constant after reaching max altitude
        return max_altitude

# The plane's maximum altitude in feet
max_altitude = 30000

# The number of time steps it takes to reach max altitude
ascent_time = 100

# Generate altitude data
altitudes = np.array([altitude_over_time(t, end_time, max_altitude, ascent_time) for t in time_steps])

# Prepare for plotting
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, end_time)
ax.set_ylim(0, max(altitudes) + 1000)  # Add a buffer above the highest altitude for better visualization

# Adding a point to simulate the plane
plane_point, = ax.plot([], [], 'o', color='blue')

# Initialize the animation with a function
def init():
    plane_point.set_data([], [])
    return (plane_point,)

# Update function for the animation
def update(frame):
    # Update the point position to the current altitude
    plane_point.set_data(time_steps[frame], altitudes[frame])
    return (plane_point,)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_steps), init_func=init, blit=True, repeat=False, interval=20)

# Set plot labels and title
ax.set_title('Plane Altitude Simulation')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Altitude (ft)')


plt.show()