import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
num_sets = 3  # Number of sets in the match
num_points_per_set = 50  # Number of points per set

# Define variational free energy (VFE) model parameters
precision = 0.5  # Controls the agent's confidence in beliefs
learning_rate = 0.1  # Rate at which the agent updates beliefs
uncertainty_decay = 0.95  # Decay rate of uncertainty after each point

# Initialize variables
vfe_values = []  # Store VFE values
uncertainty = 1.0  # Initial uncertainty

# Simulate the match
for set_number in range(1, num_sets + 1):
    print(f"Simulating set {set_number}...")
    for point in range(1, num_points_per_set + 1):
        # Update VFE based on the current uncertainty and precision
        vfe = -precision * np.log(uncertainty + 1e-5) + (1 - precision) * uncertainty
        vfe_values.append(vfe)

        # Update uncertainty based on agent's learning rate
        uncertainty = uncertainty * uncertainty_decay + learning_rate * np.random.random()

# Plotting the VFE fluctuations over the match
plt.figure(figsize=(12, 6))
plt.plot(vfe_values, label="Variational Free Energy", color="blue")
plt.title("Fluctuations in Variational Free Energy During Tennis Match")
plt.xlabel("Points in Match")
plt.ylabel("Variational Free Energy")
plt.axvline(x=num_points_per_set, color="red", linestyle="--", label="End of Set 1")
plt.axvline(x=2 * num_points_per_set, color="green", linestyle="--", label="End of Set 2")
plt.legend()
plt.grid()
plt.show()
