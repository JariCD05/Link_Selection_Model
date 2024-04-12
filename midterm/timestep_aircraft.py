import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Setup for the animation
total_steps = 96
max_height = 30000  # in feet
climb_steps = 10

# Generate the x-axis and y-axis values
x_values = np.arange(total_steps)
y_values = np.concatenate([
    np.linspace(0, max_height, climb_steps),  # Climb
    np.full(total_steps - 2 * climb_steps, max_height),  # Cruise
    np.linspace(max_height, 0, climb_steps)  # Descent
])

# Load the plane image
plane_img = plt.imread("plane.png")  # Adjust path if necessary

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, total_steps - 1)
ax.set_ylim(0, 260000)

# Setting the axis titles
ax.set_xlabel("Time Stamp")
ax.set_ylabel("Altitude (ft)")

# Plotting the line (initially empty)
line, = ax.plot([], [], 'b-', linewidth=2)

# Adding the plane (initially without a specific position)
imagebox = OffsetImage(plane_img, zoom=0.5)  # Adjust zoom as necessary
ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
ax.add_artist(ab)

# Function to update the animation at each frame
def update(frame):
    # Update the line's data to extend to the current frame
    line.set_data(x_values[:frame+1], y_values[:frame+1])
    
    # Update the position of the plane
    ab.xybox = (x_values[frame], y_values[frame])
    
    return line, ab,

# Initialize the animation
def init():
    # Initialize the line
    line.set_data([], [])
    return line, ab,

# Create the animation
ani = FuncAnimation(fig, update, frames=total_steps, init_func=init, blit=True, interval=100)

# Save the animation
ani.save('mission_time_with_trail.mp4', writer='ffmpeg', fps=10)  # Adjust fps as necessary

plt.close()  # Close the plot to prevent it from displaying inline if using a Jupyter notebook
