
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
NUM_SETS = 3
POINTS_PER_SET = 10  # Number of points in each set

def expected_free_energy(beliefs, observation_likelihood, preferences):
    """Compute Expected Free Energy (EFE) based on beliefs, observation likelihood, and preferences."""
    efe = 0
    for i, belief in enumerate(beliefs):
        for j, obs_prob in enumerate(observation_likelihood[i]):
            efe += belief * obs_prob * (np.log(obs_prob + 1e-5) - np.log(preferences[j] + 1e-5))
    return efe

# Define state-space, actions, and observations
hidden_states = ["aggressive", "defensive", "neutral"]  # Example hidden states
actions = ["serve", "return", "smash"]  # Possible actions
observations = ["win", "lose"]  # Observable outcomes

# Transition probabilities
def state_transition(current_state, action):
    """Define state transitions based on current state and action."""
    if current_state == "aggressive":
        if action == "serve":
            return np.random.choice(hidden_states, p=[0.6, 0.2, 0.2])
        elif action == "return":
            return np.random.choice(hidden_states, p=[0.4, 0.4, 0.2])
        else:  # smash
            return np.random.choice(hidden_states, p=[0.7, 0.2, 0.1])
    elif current_state == "defensive":
        if action == "serve":
            return np.random.choice(hidden_states, p=[0.2, 0.6, 0.2])
        elif action == "return":
            return np.random.choice(hidden_states, p=[0.1, 0.8, 0.1])
        else:  # smash
            return np.random.choice(hidden_states, p=[0.3, 0.6, 0.1])
    else:  # neutral
        return np.random.choice(hidden_states, p=[0.3, 0.3, 0.4])

# Observation probabilities
def generate_observation(state):
    """Generate observations based on the hidden state."""
    if state == "aggressive":
        return np.random.choice(observations, p=[0.7, 0.3])
    elif state == "defensive":
        return np.random.choice(observations, p=[0.5, 0.5])
    else:  # neutral
        return np.random.choice(observations, p=[0.6, 0.4])

# Variational Free Energy update
def compute_vfe(beliefs, observation, observation_likelihood):
    """Compute variational free energy based on beliefs and observation."""
    likelihood = observation_likelihood[:, observations.index(observation)]
    posterior = likelihood * beliefs / np.sum(likelihood * beliefs)
    vfe = -np.sum(posterior * np.log(likelihood + 1e-5))  # Avoid log(0)
    return vfe, posterior

# Initialize variables
player1_beliefs = np.array([1/3, 1/3, 1/3])  # Uniform prior
player2_beliefs = np.array([1/3, 1/3, 1/3])
player1_vfe = []
player2_vfe = []
preferences = np.array([0.8, 0.2])  # Preference for "win" over "lose"
observation_likelihood = np.array([
    [0.7, 0.3],  # Aggressive
    [0.5, 0.5],  # Defensive
    [0.6, 0.4]   # Neutral
])

# Simulation loop
for set_number in range(NUM_SETS):
    print(f"Simulating set {set_number + 1}")
    player1_state = np.random.choice(hidden_states)
    player2_state = np.random.choice(hidden_states)

    for point in range(POINTS_PER_SET):
        # Players choose actions based on EFE (simplified random for demo)
        action1 = np.random.choice(actions)
        action2 = np.random.choice(actions)

        # State transitions
        player1_state = state_transition(player1_state, action1)
        player2_state = state_transition(player2_state, action2)

        # Generate observations
        obs1 = generate_observation(player1_state)
        obs2 = generate_observation(player2_state)

        # Update VFE and beliefs
        vfe1, player1_beliefs = compute_vfe(player1_beliefs, obs1, observation_likelihood)
        vfe2, player2_beliefs = compute_vfe(player2_beliefs, obs2, observation_likelihood)

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
