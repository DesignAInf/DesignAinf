
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
NUM_SETS = 3
POINTS_PER_SET = 10  # Number of points in each set

# Initialize variational free energy for each player
player1_vfe = []
player2_vfe = []

# Define the initial state for both players
player1_state = np.random.rand()
player2_state = np.random.rand()

def update_vfe(player_state, action):
    """Simulate the update of variational free energy."""
    # Example: VFE depends on action precision and prior beliefs
    precision = 1.0 - np.abs(action - 0.5)  # Precision based on action
    prior_belief = np.exp(-player_state ** 2)  # Example prior belief
    vfe = -np.log(prior_belief * precision + 1e-5)  # Avoid log(0)
    return vfe

# Simulation loop
for set_number in range(NUM_SETS):
    print(f"Simulating set {set_number + 1}")
    for point in range(POINTS_PER_SET):
        # Random actions for simplicity
        action1 = np.random.rand()
        action2 = np.random.rand()

        # Update states and compute VFE
        player1_state = np.random.rand()  # Update state dynamically
        player2_state = np.random.rand()

        vfe1 = update_vfe(player1_state, action1)
        vfe2 = update_vfe(player2_state, action2)

        player1_vfe.append(vfe1)
        player2_vfe.append(vfe2)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(player1_vfe, label="Player 1 VFE", linestyle='-', marker='o')
plt.plot(player2_vfe, label="Player 2 VFE", linestyle='-', marker='x')
plt.xlabel("Point")
plt.ylabel("Variational Free Energy")
plt.title("Fluctuations of Variational Free Energy During the Match")
plt.legend()
plt.grid()
plt.show()
