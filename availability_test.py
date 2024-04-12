# Creating the arrays for Satellite 1 and Satellite 2
import numpy as np
import matplotlib.pyplot as plt

# Satellite 1: First 20 instances counting down from 20 to 0, then zeros
sat1 = np.array([20 - i if i < 20 else 0 for i in range(96)])
# Satellite 2: Start with 96 and countdown to 0
sat2 = np.array([96 - i for i in range(96)])


# Normalizing the non-zero values
sat1_nonzero_normalized = sat1 / 20
sat2_nonzero_normalized = sat2 / 96

# Simulated normalized_throughput_performance for demonstration


# Combined normalized values
sat1_combined = sat1_nonzero_normalized + normalized_throughput_performance[0]
sat2_combined = sat2_nonzero_normalized + normalized_throughput_performance[1]

# Adjust indices for plotting all values
indices_all = np.arange(96)  # Since we're plotting all 96 instances now

# Plotting with correct indices and sizes
fig, ax = plt.subplots(2, 1, figsize=(7, 10))

# Original Values (All, including zeros for demonstration)
ax[0].scatter(indices_all, sat1, color='blue', label='Satellite 1', marker='.')
ax[0].scatter(indices_all, sat2, color='orange', label='Satellite 2', marker='.')
ax[0].set_title('Availability Performance $Q_{A} (t_{j})$')
ax[0].legend()

# Combined Normalized Values (All, assuming normalized_throughput_performance is correctly aligned)
ax[1].scatter(indices_all, sat1_combined, color='blue', label='Satellite 1', marker='.')
ax[1].scatter(indices_all, sat2_combined, color='orange', label='Satellite 2', marker='.')
ax[1].set_title('Combined throughput and availability')
ax[1].legend()

plt.tight_layout()
plt.show()