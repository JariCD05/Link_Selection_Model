import numpy as np

class link_selection_no_link():
    def __init__(self, num_satellites, time, normalized_values, normalized_penalized_values, weights, satellite_position_indices, max_satellites, active_satellite):
        self.num_satellites = num_satellites
        self.time = time
        self.normalized_values = normalized_values  # This should be a dict of lists
        self.normalized_penalized_values = normalized_penalized_values  # Penalized values for non-active satellites
        self.weights = weights  # List of weights for each performance metric
        self.satellite_position_indices = satellite_position_indices
        self.max_satellites = max_satellites
        self.active_satellite = active_satellite
        # Initialize the NumPy array to store historical scores and all satellite scores over time
        self.historical_weighted_scores = np.zeros((len(self.time), self.num_satellites))

    def calculate_weighted_performance(self, time_step):
        # Initialize a list to store the weighted score for each satellite at current time step
        weighted_scores = [0] * self.num_satellites

        # Loop through each satellite
        for s in range(self.num_satellites):
            # Determine if current satellite is the active one
            if self.satellite_position_indices[s] == self.active_satellite:
                current_values = self.normalized_values
            else:
                current_values = self.normalized_penalized_values

            # Sum the weighted scores for each metric using the appropriate normalized values
            for metric_index, metric_values in enumerate(current_values):
                weighted_scores[s] += metric_values[s] * self.weights[metric_index]

        # Store scores in the historical scores array
        self.historical_weighted_scores[time_step, :] = weighted_scores

        return weighted_scores

    def select_best_satellite(self, time_step):
        weighted_scores = self.calculate_weighted_performance(time_step)
        max_score = max(weighted_scores)
        best_satellite = weighted_scores.index(max_score)
        activated_satellite_index = self.satellite_position_indices[best_satellite]
        activated_satellite_number = activated_satellite_index + 1

        print(f"All Weighted Scores at time step {time_step}:")
        for index, score in enumerate(weighted_scores):
            print(f"Satellite {self.satellite_position_indices[index] + 1}: Score = {score}{' (Best)' if index == best_satellite else ''}")

        return best_satellite, max_score, activated_satellite_index, activated_satellite_number

    def get_historical_scores(self):
        return self.historical_weighted_scores
