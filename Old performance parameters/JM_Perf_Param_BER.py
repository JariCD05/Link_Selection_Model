# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input_old import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_Sat_Applicable_links import applicable_satellites



class ber_performance():
    def __init__(self, time, throughput):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.time = time
        self.speed_of_light = speed_of_light
        self.throughput = throughput

    def calculate_BER_performance(self):
        self.BER_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Convert the throughput list for the current satellite to a NumPy array if not already
            throughput_array = np.array(self.throughput[s])

            # Find indices where throughput equals 2,500,000,000
            qualifying_indices = np.where(throughput_array == 2500000000)[0]

            if len(qualifying_indices) == 0:
                # If no instance of 250,000,000 is found, skip this satellite
                continue

            # Get the index of the first occurrence of 2,500,000,000
            first_qualifying_index = qualifying_indices[0]

            # Initialize the running sum to 0 initially
            running_sum = 0

            # Loop through time in reverse order starting from the first qualifying index
            for index in range(len(self.time)-1, first_qualifying_index-1, -1):
                # Increment running_sum if the condition is met
                if throughput_array[index] == 2500000000:
                    running_sum += 1
                # Update throughput performance with the current running sum
                self.BER_performance[s][index] = running_sum

            # Ensure all values before the first qualifying index are set to 0
            for index in range(first_qualifying_index):
                self.BER_performance[s][index] = 0

        return self.BER_performance
    
    def calculate_normalized_BER_performance(self, data, availability_vector):
        max_time = np.nansum(availability_vector, axis=1)
        max_time_max = np.nanmax(max_time)
        self.normalized_BER_performance = data / max_time_max

        return self.normalized_BER_performance
    


    def calculate_BER_performance_including_penalty(self, T_acq, step_size_link):
        # Convert T_acq to the number of time steps
        delay_steps = int(np.ceil(T_acq / step_size_link))

        self.BER_performance_including_penalty = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Convert the throughput list for the current satellite to a NumPy array if not already
            throughput_array = np.array(self.throughput[s])

            # Find indices where throughput equals 2,500,000,000
            qualifying_indices = np.where(throughput_array == 2500000000)[0]

            if len(qualifying_indices) == 0:
                # If no instance of 250,000,000 is found, skip this satellite
                continue

            # Get the index of the first occurrence of 2,500,000,000
            first_qualifying_index = qualifying_indices[0]

            # Adjust the starting index for calculating BER performance by adding the delay
            # Ensure this adjusted index does not exceed the length of the time array
            adjusted_start_index = min(first_qualifying_index + delay_steps, len(self.time))

            # Initialize the running sum to 0 initially
            running_sum = 0

            # Update the BER performance list starting from the adjusted index after considering the delay
            for index in range(len(self.time)-1, adjusted_start_index-1, -1):
                # Increment running_sum if the condition is met
                if throughput_array[index] == 2500000000:
                    running_sum += 1
                self.BER_performance_including_penalty[s][index] = running_sum

            # For indices before the adjusted start index, the performance is considered not applicable due to the delay
            # So, you might want to set a default value that indicates non-applicability or simply leave it as 0
            for index in range(adjusted_start_index):
                self.BER_performance_including_penalty[s][index] = 0  # or another value indicating non-applicability

        return self.BER_performance_including_penalty


    def calculate_normalized_BER_performance_including_penalty(self, data, availability_vector):
        # Calculate the maximum time a satellite is applicable to normalize against it
        max_time_seperate = np.nansum(availability_vector, axis=1)
        max_time_combined = np.nanmax(max_time_seperate)
        print(max_time_combined)
        self.normalized_BER_performance_including_penalty = data / max_time_combined
        

        return self.normalized_BER_performance_including_penalty

    def BER_visualization(self, BER_performance, normalized_BER_performance, BER_performance_including_penalty, normalized_BER_performance_including_penalty):
        # To remove all zero values for the BER visualizations, we'll modify the plotting logic
        # to filter out zero values before plotting

        # Plotting with zero values filtered out
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Filter and Plot 1: BER Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(BER_performance[s]) if v > 0]
            non_zero_values = [v for v in BER_performance[s] if v > 0]
            axs[0, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 0].set_title('BER  Performance $Q_{BER} (t_{j})$ ')
        axs[0, 0].set_xlabel('Time Steps')
        axs[0, 0].set_ylabel('BER Performance [# timesteps]')
        axs[0, 0].legend()

        # Filter and Plot 2: Normalized BER Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_BER_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_BER_performance[s] if v > 0]
            axs[0, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 1].set_title('Normalized BER Performance $\hat{Q}_{BER} (t_{j})$')
        axs[0, 1].set_xlabel('Time Steps')
        axs[0, 1].set_ylabel('Normalized BER Performance [-]')
        axs[0, 1].legend()

        # Filter and Plot 3: BER Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(BER_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in BER_performance_including_penalty[s] if v > 0]
            axs[1, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 0].set_title('BER Performance Including Penalty $Q_{BER} (t_{j})$')
        axs[1, 0].set_xlabel('Time Steps')
        axs[1, 0].set_ylabel('BER Performance w/ Penalty [# timesteps]')
        axs[1, 0].legend()

        # Filter and Plot 4: Normalized BER Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_BER_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in normalized_BER_performance_including_penalty[s] if v > 0]
            axs[1, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 1].set_title('Normalized BER Performance Including Penalty $\hat{Q}_{BER} (t_{j})$')
        axs[1, 1].set_xlabel('Time Steps')
        axs[1, 1].set_ylabel('Normalized BER Performance w/ Penalty [-]')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def BER_performance_vs_availability_visualization(self, throughput, availability_vector):
        plt.figure(figsize=(14, 7))

        # Assuming throughput is a 2D array where each row corresponds to a satellite's throughput data
        # and availability_vector is similarly structured.
        # Convert throughput to Gb/s for visualization
        throughput_gbps = np.array(throughput) / 1e9  # Convert bits per second to Gbps

        # Time points (assuming equal spacing)
        time_points = np.arange(throughput_gbps.shape[1])  # Assuming all satellites have the same number of time points

        # Plot throughput for each satellite
        for s in range(throughput_gbps.shape[0]):  # Loop over satellites
            plt.plot(time_points, throughput_gbps[s, :], label=f'Sat {s+1} Throughput (Gb/s)', marker='o', linestyle='-', alpha=0.7)

        # Since we're plotting availability on the same figure, let's handle that next
        ax2 = plt.gca().twinx()  # Create a twin Axes sharing the xaxis
        # Convert availability to integers (1 or 0), with NaN values handled appropriately
        availability_int = np.where(np.isnan(availability_vector), 0, availability_vector)

        # Plotting availability might be tricky if you want to overlay it directly, 
        # So, an alternative approach would be plotting it separately or using it to shade areas of unavailability.
        # Here, we'll demonstrate a simple approach to visualize availability by highlighting periods of unavailability
        for s in range(availability_int.shape[0]):
            for i in range(availability_int.shape[1]):
                if availability_int[s, i] == 0:
                    plt.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2)  # Shade regions of unavailability

        plt.title('Throughput and Availability Visualization')
        plt.xlabel('Time Point')
        plt.ylabel('Throughput (Gb/s)')
        ax2.set_ylabel('Availability')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()





