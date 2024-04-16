    #def calculate_throughput_performance_including_decay(self, decay_rate=0.0):
    #   # Initialize throughput_performance with zeros
    #   self.throughput_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

    #   for s in range(num_satellites):
    #       for t in range(len(self.time)):
    #           # Only calculate the average for non-zero throughput values
    #           if self.throughput[s][t] > 0:
    #               future_values = [self.throughput[s][index] for index in range(t, len(self.time)) if self.throughput[s][index] > 0]
    #               if future_values:  # Check if there are future non-zero values
    #                   # Calculate weights using exponential decay, which will be 1 if decay_rate is 0
    #                   weights = [math.exp(-decay_rate * (index - t)) for index in range(t, len(self.time)) if self.throughput[s][index] > 0]
    #                   weighted_sum = sum(fv * w for fv, w in zip(future_values, weights))
    #                   total_weight = sum(weights)
    #                   self.throughput_performance[s][t] = weighted_sum / total_weight

    #   return self.throughput_performance




    #
    #def calculate_throughput_performance(self):
    #   # Initialize throughput_performance with zeros
    #   self.throughput_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

    #   for s in range(num_satellites):
    #       for t in range(len(self.time)):
    #           # Only calculate the average for non-zero throughput values
    #           if self.throughput[s][t] > 0:
    #               future_values = [throughput for throughput in self.throughput[s][t:] if throughput > 0]
    #               if future_values:  # Check if there are future non-zero values
    #                   self.throughput_performance[s][t] = sum(future_values) / len(future_values)
    #           # If the current throughput is zero, it remains zero and we don't calculate the average

    #   return self.throughput_performance
