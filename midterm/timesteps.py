import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Setup for the animation
total_steps = 96
max_height = 30000  # in feet for the plane
climb_steps = 15

# Satellite heights
start_height = 210000  # Starting height for the satellite
peak_height = 260000  # Peak height for the satellite in the middle

# Generate the x-axis and y-axis values for the plane
x_values_plane = np.arange(total_steps)
y_values_plane = np.concatenate([
    np.linspace(0, max_height, climb_steps),  # Climb
    np.full(total_steps - 2 * climb_steps, max_height),  # Cruise
    np.linspace(max_height, 0, climb_steps)  # Descent
])

# Circular path for the satellite
radius = (peak_height - start_height) / 2
center_y = start_height + radius
theta = np.linspace(np.pi, 0, total_steps)  # Half-circle from pi to 0
x_values_sat = np.arange(total_steps)
y_values_sat = center_y + radius * np.sin(theta)  # Sinusoidal path for the satellite

# Load the plane and satellite images
plane_img = plt.imread("plane.png")  # Adjust path if necessary
satellite_img = plt.imread("satellite.png")  # Adjust path if necessary

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, total_steps - 1)
ax.set_ylim(0, peak_height + 5000)  # Adjust the y-axis limit to include the satellite's path

# Setting the axis titles
ax.set_xlabel("Time Stamp")
ax.set_ylabel("Altitude (ft)")

# Plotting the lines (initially empty) for both plane and satellite
line_plane, = ax.plot([], [], 'b-', linewidth=2)
line_sat, = ax.plot([], [], 'r-', linewidth=2)  # Red line for the satellite's tail

# Adding the plane (initially without a specific position)
plane_imagebox = OffsetImage(plane_img, zoom=0.45)  # Adjust zoom as necessary
plane_ab = AnnotationBbox(plane_imagebox, (0, 0), frameon=False)
ax.add_artist(plane_ab)

# Adding the satellite (initially without a specific position)
satellite_imagebox = OffsetImage(satellite_img, zoom=0.2)  # Adjust zoom as necessary
satellite_ab = AnnotationBbox(satellite_imagebox, (0, 0), frameon=False)
ax.add_artist(satellite_ab)

# Function to update the animation at each frame
def update(frame):
    # Update the plane's line and position
    line_plane.set_data(x_values_plane[:frame+1], y_values_plane[:frame+1])
    plane_ab.xybox = (x_values_plane[frame], y_values_plane[frame])
    
    # Update the satellite's line and position
    line_sat.set_data(x_values_sat[:frame+1], y_values_sat[:frame+1])
    satellite_ab.xybox = (x_values_sat[frame], y_values_sat[frame])
    
    return line_plane, plane_ab, line_sat, satellite_ab,

# Initialize the animation
def init():
    # Initialize the lines
    line_plane.set_data([], [])
    line_sat.set_data([], [])
    return line_plane, plane_ab, line_sat, satellite_ab,

# Create the animation
ani = FuncAnimation(fig, update, frames=total_steps, init_func=init, blit=True, interval=100)

# Save the animation
ani.save('mission_time_with_trail_and_satellite.mp4', writer='ffmpeg', fps=5)  # Adjust fps as necessary

plt.close()  # Close the plot to prevent it from displaying inline if using a Jupyter notebook
