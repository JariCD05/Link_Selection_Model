def calculate_throughput_performance_exp(self, lambda_decay=0.1):
    num_satellites = len(self.throughput)
    self.throughput_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]
    for s in range(num_satellites):
        for t in range(len(self.time)):
            if self.throughput[s][t] > 0:
                future_values = self.throughput[s][t:]
                weights = [math.exp(-lambda_decay * i) for i in range(len(future_values))]
                weighted_future_values = [value * weight for value, weight in zip(future_values, weights)]
                self.weights_record.append((s, t, weights))
                self.weighted_values_record.append((s, t, weighted_future_values))
                if sum(weights) > 0:
                    self.throughput_performance[s][t] = sum(weighted_future_values) / sum(weights)
    #export_to_csv('weights.csv', self.weights_record, ['Satellite', 'Timestamp', 'Weights'])
    #export_to_csv('weighted_values.csv', self.weighted_values_record, ['Satellite', 'Timestamp', 'Weighted_Values'])
