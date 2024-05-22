import numpy as np
import matplotlib.pyplot as plt

class SatelliteDataVisualizer:
    def __init__(self, time, activated_satellites, satellite_availability, throughput, step_size_link, sats_applicable, max_num_satellites, list_of_total_scores_all_satellite):
        self.time = np.array(time) / step_size_link  # Adjust time based on step_size_link
        self.activated_satellites = activated_satellites
        self.satellite_availability = satellite_availability
        self.throughput = np.array([max(0, x) for x in throughput])
        self.step_size_link = step_size_link
        self.max_num_satellites = max_num_satellites
        self.sats_applicable = np.array(sats_applicable, dtype=float)
        self.scores = np.array(list_of_total_scores_all_satellite)
        self.mission_throughput = self.calculate_mission_throughput()
        self.cumulative_throughput = self.calculate_cumulative_throughput()
        self.average_throughput = np.mean(self.mission_throughput)  # Average throughput calculation

        # Define a list of unique colors for satellites
        self.satellite_colors = ['blue', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'indigo']

        

    def calculate_mission_throughput(self):
        # Calculate throughput per time instance
        return self.throughput * self.step_size_link

    def calculate_cumulative_throughput(self):
        # Calculate cumulative throughput
        return np.cumsum(self.mission_throughput)

    def plot_cumulative_throughput(self):
        # Plotting the accumulated throughput over time
        plt.figure(figsize=(12, 6))
        plt.plot(self.cumulative_throughput, label='Cumulative Throughput', color='orange')
        
        # Adding a grid
        plt.grid(True)

        # Highlight areas where throughput is not increasing
        for i in range(1, len(self.cumulative_throughput)):
            if self.cumulative_throughput[i] <= self.cumulative_throughput[i - 1]:
                plt.axvspan(i - 1, i, color='lightgrey', alpha=0.5)
        

        converted_troughput = self.average_throughput/(10**9)
        # Adding average throughput line
        plt.axhline(y=self.average_throughput, color='red', linestyle='--', label=f'Average Throughput: {converted_troughput:.2f} Gb/s')

        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Throughput [bits]')
        plt.title('Accumulated Throughput Over Mission Time')
        plt.legend()
        plt.show()



    def plot_satellite_visibility_scatter_update(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_title("Satellite Visibility and Performance Over Time")

        # Plot satellite availability
        for t in range(len(self.time)):
            for s in range(1, self.max_num_satellites + 1):
                if self.sats_applicable[t][s - 1] == 1:
                    color = 'green' if self.activated_satellites[t] == s else 'red'
                    ax1.scatter(t, s, color=color, alpha=0.6)

        ax1.set_xlabel("Time (scaled by step size link)")
        ax1.set_ylabel("Satellite Index")
        ax1.set_yticks(range(1, self.max_num_satellites + 1))
        ax1.grid(True)

        # Secondary axis for performance scores
        ax2 = ax1.twinx()
        for s in range(self.scores.shape[0]):
            if not np.isnan(self.scores[s]).all():
                ax2.plot(self.time, self.scores[s], label=f'Satellite {s+1}', linewidth=2)

                # Check for transitions from a value to NaN and plot a grey scatter point
                for idx in range(1, len(self.scores[s])):
                    if np.isnan(self.scores[s][idx]) and not np.isnan(self.scores[s][idx-1]):
                        ax2.scatter(self.time[idx], self.scores[s][idx-1], color='grey', s=50, zorder=5)  # Place at last valid score

        ax2.set_ylabel('Performance Scores')
        ax2.set_ylim(0, 1)  # Set the scale of the right y-axis from 0 to 1
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


    def plot_satellite_visibility_scatter_only_visibility(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_title("Satellite Visibility and Performance Over Time")
        # Plot satellite availability
        for t in range(len(self.time)):
            for s in range(1, self.max_num_satellites + 1):
                if self.sats_applicable[t][s - 1] == 1:
                    color = 'green' if self.activated_satellites[t] == s else 'red'
                    ax1.scatter(t, s, color=color, alpha=0.6)
        ax1.set_xlabel("Time (scaled by step size link)")
        ax1.set_ylabel("Satellite Index")
        ax1.set_yticks(range(1, self.max_num_satellites + 1))
        ax1.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage of the class
# Assuming you have initialized the class with required data
# visualizer = SatelliteDataVisualizer(activated_satellites, satellite_availability, throughput, step_size_link)
# visualizer.plot_cumulative_throughput()


# Example usage of the class
# Assuming you have initialized the class with required data
# visualizer = SatelliteDataVisualizer(activated_satellites, satellite_availability, throughput, step_size_link)
# visualizer.plot_cumulative_throughput()


# Example data
activated_satellites = ['No Link', 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4]
satellite_availability = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
throughput = [0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, -10000000000.0, -7312768575.714134, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 761230171.0997376, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2499999999.9999986, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, -8422629489.947367, 1185287670.9508812, 2495501185.675274, 2499995675.268521, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, -9999056051.589945, 2500000000.0, 2500000000.0, 2500000000.0, 2497816592.346925, 2499999999.94326, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, 2500000000.0, -9993225625.650894, -5387976229.408816, 1659305662.862266, 2481472370.526987, 2416672837.7194543, -32823555.62930145, -7896038231]
step_size_link = 5
#sats_applicable = [[nan, nan, nan, nan], [nan, 1, n#an, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, nan], [nan, 1, nan, 1], [nan, 1, nan, 1], [nan, 1, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, 1], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, nan, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [1, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan], [nan, nan, 1, nan]]


# Create an instance of the visualizer
#visualizer = SatelliteDataVisualizer(activated_satellites, satellite_availability, throughput, step_size_link)

# Plot the cumulative throughput
#visualizer.plot_cumulative_throughput()
