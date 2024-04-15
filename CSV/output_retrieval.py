
import csv

time_ranges = [(21, 37), (55, 69)] 

with open('performance_parameters_over_time.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Parameter Index', 'Time', 'Performance Value'])  # Header

    # Iterate over each performance parameter and time range
    for param_index, performances in performance_parameters_over_time.items():
        for time_range in time_ranges:
            for time_point in range(time_range[0], time_range[1] + 1):  # Include the end time
                if time_point < len(performances):
                    writer.writerow([param_index, time_point, performances[time_point]])
                else:
                    # Handle the case where the time_point index is out of range
                    print(f"Index {time_point} is out of range for performance parameter index {param_index}.")