class SatelliteLinkMetrics:
    def __init__(self, list_of_activated_sattellite_numbers, list_of_sattellite_availability_while_being_active, step_size_link, acquisition_time_steps, total_mission_time, stored_throughput_active_satellite):
        self.list_of_activated_sattellite_numbers = list_of_activated_sattellite_numbers
        self.list_of_sattellite_availability_while_being_active = list_of_sattellite_availability_while_being_active
        self.step_size_link = step_size_link
        self.acquisition_time_steps = acquisition_time_steps
        self.total_mission_time = total_mission_time
        self.stored_throughput_active_satellite = stored_throughput_active_satellite

    def calculate_metrics(self):
        unique_links = set(x for x in self.list_of_activated_sattellite_numbers if x != 'No Link')
        self.stored_throughput_active_satellite = [x if x >= 0 else 0 for x in self.stored_throughput_active_satellite]
        number_of_links = len(unique_links)

        total_active_time = 0
        total_throughput = 0  # Initialize total throughput

        for i in range(len(self.list_of_sattellite_availability_while_being_active)):
            if self.list_of_sattellite_availability_while_being_active[i] == 1.0:
                total_active_time += self.step_size_link
                total_throughput += self.stored_throughput_active_satellite[i]

        # Calculate average throughput
        if total_active_time > 0:
            average_throughput = total_throughput / total_active_time
        else:
            average_throughput = 0

        # Subtract acquisition times for each transition
        total_service_time = total_active_time  # Adjusted directly in availability vector

        # Calculate average link time and service percentage
        if number_of_links > 0:
            average_link_time = total_service_time / number_of_links  # in sec
        else:
            average_link_time = 0
        service_time_percentage = (total_service_time / self.total_mission_time) * 100 if self.total_mission_time else 0

        return {
            'Number of Different Links': number_of_links,
            'Average Link Time': f"{average_link_time / 60:.2f} min",
            'Total Service Time': f"{total_service_time / 60:.2f} min",
            'Service Time Percentage of Total Mission Time': f"{service_time_percentage:.2f} %"
        }
