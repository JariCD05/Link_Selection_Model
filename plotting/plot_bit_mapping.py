import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('CSV/mapping.csv', header=None)
data_array = df.values  # Convert the DataFrame to a numpy array

# Assuming the array has been preprocessed to have the correct dimensions
# If data_array still needs processing, adjust here

# Create a custom colormap: False -> red, True -> green
cmap = ListedColormap(['red', 'green'])

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust figure size as needed
cax = ax.matshow(data_array, cmap=cmap, interpolation='nearest', aspect='auto')

# Calculate tick positions for y-axis: Since we have halved the rows, we have 18 rows to cover -90 to 90
y_tick_positions = np.linspace(0, data_array.shape[0] - 1, 7)  # 7 positions including ends and zero
y_tick_labels = ['90', '60', '30', '0', '-30', '-60', '-90']  # Labels corresponding to calculated positions

# Calculate tick positions for x-axis: Assuming no change in x-axis handling
x_tick_positions = np.linspace(0, data_array.shape[1] - 1, 5)  # Example: 5 positions for x-axis
x_tick_labels = ['180', '90', '0', '-90', '-180']  # Example labels for x-axis, adjust as necessary

# Set ticks and labels
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels)
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(y_tick_labels)

# Axis labels
ax.set_xlabel('Azimuth')
ax.set_ylabel('Zenith')

# Add gridlines for clarity
ax.set_xticks(np.arange(-.5, data_array.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, data_array.shape[0], 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

ax.set_title('Masking Bit Map')
plt.show()
