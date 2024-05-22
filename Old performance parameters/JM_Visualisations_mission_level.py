import sqlite3

import numpy as np
import scipy.signal
from input_old import *
import os

import random
from scipy.special import j0, j1, binom
from scipy.stats import rv_histogram, norm
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq
from scipy.special import erfc, erf, erfinv, erfcinv
from scipy.special import erfc, erfcinv
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.util import result2array
import csv

#Function used in mission level to get visuals


def export_to_csv(filename, data, headers):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for record in data:
            # Prepare each row to be written to CSV
            satellite, timestamp, values = record
            # Flattening the list of weights/values to a single string for easier CSV handling
            values_str = ', '.join(f'{value:.2f}' for value in values)
            writer.writerow([satellite, timestamp, values_str])

def export_dataframe_to_csv(data_frame, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(directory, filename)  # Full file path
    data_frame.to_csv(file_path, index=False)  # Save the DataFrame to CSV
    print(f"Data exported to {file_path}")

def plot_active_satellites(time_steps, active_satellites, num_satellites):
    """
    Plot the active satellites over time.
    Parameters:
    time_steps: An array of time steps.
    active_satellites: A list containing the index of the active satellite or "No link" at each time step.
    num_satellites: The number of satellites (used to set the y-axis limits).
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


def plot_individual_satellite_performance(active_satellites, performance_over_time, num_satellites, time_steps, sats_applicable):
    performance_over_time = np.array(performance_over_time).T  # Transpose for easier plotting
    sats_applicable = np.array(sats_applicable)  # Ensure sats_applicable is a numpy array

    # Calculate the number of subplot rows and columns dynamically
    num_cols = 2  # Set the number of columns for the subplot grid
    num_rows = (num_satellites + num_cols - 1) // num_cols  # Compute rows needed
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the array of axes for easier iteration

    for sat_index in range(num_satellites):
        ax = axs[sat_index]
        sat_performance = performance_over_time[sat_index, :]
        time_indices = np.arange(len(time_steps))  # Use defined time_steps

        # Plotting the performance
        ax.plot(time_indices, sat_performance, label=f'Sat {sat_index + 1}', color='blue', alpha=0.6)
        
        # Highlighting active and inactive satellites
        active_times = [i for i in range(len(time_steps)) if active_satellites[i] == sat_index]
        non_active_times = [i for i in range(len(time_steps)) if sats_applicable[sat_index, i] and active_satellites[i] != sat_index]

        ax.scatter(active_times, sat_performance[active_times], color='green', label='Active', zorder=5)
        ax.scatter(non_active_times, sat_performance[non_active_times], color='red', label='Not Active', zorder=5)

        ax.grid(True)
        ax.legend()

    # Hide empty subplots if any
    for i in range(num_satellites, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


def print_link_handovers(active_satellites):
    # Start from the second element to compare with the previous one
    for i in range(1, len(active_satellites)):
        previous_link = active_satellites[i - 1]
        current_link = active_satellites[i]
        
        # Check if there is a handover (change in active link)
        if previous_link != current_link:
            # Handle the 'No link' case
            if previous_link == 'No link':
                print(f"At timestamp {i} the link is established to sat {current_link+1}.")
            elif current_link == 'No link':
                print(f"At timestamp {i} the link from sat {previous_link+1} is lost.")
            else:
                print(f"At timestamp {i} the link is handed over from sat {previous_link+1} to sat {current_link+1}.")


                import numpy as np


def plot_individual_satellite_performance_update(active_satellites, performance_over_time, num_satellites, time_steps, sats_applicable):
    # Ensure performance_over_time is a numpy array
    performance_over_time = np.array(performance_over_time)

    # Handle case where performance_over_time might be empty
    if performance_over_time.size == 0:
        print("Warning: No performance data available to plot.")
        return

    if performance_over_time.ndim == 1:
        performance_over_time = performance_over_time.reshape((num_satellites, -1)).T

    sats_applicable = np.array(sats_applicable)

    # Calculate the number of subplot rows and columns dynamically
    num_cols = 2
    num_rows = (num_satellites + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the array of axes for easier iteration

    for sat_index in range(num_satellites):
        ax = axs[sat_index]
        sat_performance = performance_over_time[sat_index, :]
        time_indices = np.arange(len(time_steps))

        ax.plot(time_indices, sat_performance, label=f'Sat {sat_index + 1}', color='blue', alpha=0.6)

        active_times = [i for i in time_indices if active_satellites[i] == sat_index]
        non_active_times = [i for i in time_indices if sats_applicable[sat_index, i] and active_satellites[i] != sat_index]

        ax.scatter(active_times, sat_performance[active_times], color='green', label='Active', zorder=5)
        ax.scatter(non_active_times, sat_performance[non_active_times], color='red', label='Not Active', zorder=5)

        ax.grid(True)
        ax.legend()

    for i in range(num_satellites, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
