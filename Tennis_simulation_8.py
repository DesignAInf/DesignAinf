import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
NUM_SETS = 3
AVG_MINUTES_PER_SET = 40  # Average duration of a set in minutes
MINUTES_PER_POINT = 1  # Average duration of a point in minutes
TOTAL_POINTS = AVG_MINUTES_PER_SET * NUM_SETS // MINUTES_PER_POINT

# Define state-space, actions, and observations
hidden_states = ["aggressive", "defensive", "neutral", "tired", "focused"]  # Expanded hidden states
actions = ["serve", "return", "smash", "drop shot", "lob", "volley"]  # Expanded actions
observations = ["win", "lose", "forced error", "unforced error"]  # Expanded observations

# Dynamic Observation Likelihoods
def update_observation_likelihoods(match_progress, player_state):
    """Dynamically adjust observation likelihoods based on match progression and player states."""
    base_probs = {
        "aggressive": [0.6, 0.2, 0.1, 0.1],
        "defensive": [0.4, 0.3, 0.2, 0.1],
        "neutral": [0.5, 0.2, 0.2, 0.1],
        "tired": [0.3, 0.4, 0.2, 0.1],
        "focused": [0.7, 0.1, 0.1, 0.1]
    }
    adjustment = 0.1 * (match_progress / TOTAL_POINTS)
    state_factor = 0.1 if player_state == "tired" else 0.05
    adjusted_probs = {
        state: [max(0, prob + adjustment * (0.5 - prob) - state_factor) for prob in probs]
        for state, probs in base_probs.items()
    }
    # Normalize probabilities to ensure they sum to 1
    normalized_probs = {
        state: np.array(probs) / np.sum(probs) for state, probs in adjusted_probs.items()
    }
    return normalized_probs

# State Transition probabilities
def state_transition(current_state, action, fatigue, opponent_action, match_context):
    """Define state transitions based on current state, action, fatigue level, opponent influence, and match context."""
    base_probs = {
        "aggressive": [0.5, 0.2, 0.1, 0.1, 0.1],
        "defensive": [0.2, 0.5, 0.2, 0.1, 0.0],
        "neutral": [0.3, 0.3, 0.2, 0.1, 0.1],
        "tired": [0.1, 0.2, 0.2, 0.4, 0.1],
        "focused": [0.4, 0.2, 0.2, 0.1, 0.1]
    }
    interaction_factor = 0.1 if opponent_action in ["smash", "serve"] else 0.05
    context_factor = 0.1 if match_context == "high pressure" else 0.05
    probs = np.array(base_probs[current_state])
    adjusted_probs = probs * (1 - fatigue) + interaction_factor * fatigue / len(probs) + context_factor
    adjusted_probs /= np.sum(adjusted_probs)  # Normalize probabilities
    return np.random.choice(hidden_states, p=adjusted_probs)

# Fatigue Adaptation
def update_fatigue(fatigue, action, match_context):
    """Update fatigue level based on the player's action and match context."""
    fatigue_increase = 0.01
    if action in ["smash", "serve"]:
        fatigue_increase += 0.02  # Aggressive actions increase fatigue
    if match_context == "high pressure":
        fatigue_increase += 0.01  # High-pressure moments add to fatigue
    return min(1.0, fatigue + fatigue_increase)

# Observation Likelihoods
def generate_observation(state, observation_likelihood):
    """Generate observations based on the hidden state and dynamic observation likelihoods."""
    probs = observation_likelihood[state]
    probs = np.array(probs) / np.sum(probs)  # Normalize probabilities
    return np.random.choice(observations, p=probs)

# Variational Free Energy update
def compute_vfe(beliefs, observation, observation_likelihood):
    """Compute variational free energy based on beliefs and observation."""
    likelihood = observation_likelihood[:, observations.index(observation)]
    posterior = likelihood * beliefs / np.sum(likelihood * beliefs)
    vfe = -np.sum(posterior * np.log(likelihood + 1e-5))  # Avoid log(0)
    return vfe, posterior

# Expected Free Energy calculation
def compute_efe(beliefs, observation_likelihood, preferences):
    """Compute expected free energy for each action."""
    efe = 0
    for i, belief in enumerate(beliefs):
        for j, obs_prob in enumerate(observation_likelihood[i]):
            efe += belief * obs_prob * (np.log(obs_prob + 1e-5) - np.log(preferences[j] + 1e-5))
    return efe

# Joint Multi-Step Planning
def multi_step_planning(beliefs, opponent_beliefs, observation_likelihood, preferences, horizon=3):
    """Plan actions over multiple steps to optimize long-term outcomes, considering opponent strategy."""
    best_action = None
    best_expected_vfe = float('inf')

    for action in actions:
        cumulative_vfe = 0
        simulated_beliefs = beliefs.copy()
        simulated_opponent_beliefs = opponent_beliefs.copy()

        for _ in range(horizon):
            # Simulate an observation based on the current belief
            simulated_observation_probs = np.sum(observation_likelihood * simulated_beliefs[:, None], axis=0)
            simulated_observation_probs /= np.sum(simulated_observation_probs)  # Normalize probabilities
            simulated_observation = np.random.choice(observations, p=simulated_observation_probs)

            # Update simulated beliefs using the observation
            _, simulated_beliefs = compute_vfe(simulated_beliefs, simulated_observation, observation_likelihood)

            # Simulate opponent action and its influence
            opponent_action = np.random.choice(actions)
            _, simulated_opponent_beliefs = compute_vfe(simulated_opponent_beliefs, simulated_observation, observation_likelihood)

            # Compute EFE as the sum of simulated beliefs weighted by preferences over observations
            efe = compute_efe(simulated_beliefs, observation_likelihood, preferences)
            cumulative_vfe += efe

        if cumulative_vfe < best_expected_vfe:
            best_expected_vfe = cumulative_vfe
            best_action = action

    return best_action

# Initialize variables
player1_beliefs = np.array([1 / len(hidden_states)] * len(hidden_states))  # Uniform prior
player2_beliefs = np.array([1 / len(hidden_states)] * len(hidden_states))
player1_vfe = []
player2_vfe = []
player1_efe = []
player2_efe = []
preferences = np.array([0.5, 0.2, 0.2, 0.1])  # Preferences for outcomes
fatigue_level_player1 = 0.1  # Initial fatigue level for Player 1
fatigue_level_player2 = 0.1  # Initial fatigue level for Player 2

for point in range(TOTAL_POINTS):
    # Dynamic observation likelihoods
    match_progress = point / TOTAL_POINTS
    dynamic_likelihoods_player1 = update_observation_likelihoods(match_progress, "neutral")
    dynamic_likelihoods_player2 = update_observation_likelihoods(match_progress, "neutral")

    # Multi-step planning for actions
    action1 = multi_step_planning(player1_beliefs, player2_beliefs, np.array(list(dynamic_likelihoods_player1.values())), preferences)
    action2 = multi_step_planning(player2_beliefs, player1_beliefs, np.array(list(dynamic_likelihoods_player2.values())), preferences)

    # State transitions with opponent modeling
    player1_state = state_transition(np.random.choice(hidden_states), action1, fatigue_level_player1, action2, "normal")
    player2_state = state_transition(np.random.choice(hidden_states), action2, fatigue_level_player2, action1, "normal")

    # Generate observations
    obs1 = generate_observation(player1_state, dynamic_likelihoods_player1)
    obs2 = generate_observation(player2_state, dynamic_likelihoods_player2)

    # Update VFE and beliefs
    vfe1, player1_beliefs = compute_vfe(player1_beliefs, obs1, np.array(list(dynamic_likelihoods_player1.values())))
    vfe2, player2_beliefs = compute_vfe(player2_beliefs, obs2, np.array(list(dynamic_likelihoods_player2.values())))

    # Compute EFE
    efe1 = compute_efe(player1_beliefs, np.array(list(dynamic_likelihoods_player1.values())), preferences)
    efe2 = compute_efe(player2_beliefs, np.array(list(dynamic_likelihoods_player2.values())), preferences)

    player1_vfe.append(vfe1)
    player2_vfe.append(vfe2)
    player1_efe.append(efe1)
    player2_efe.append(efe2)

    # Update fatigue
    fatigue_level_player1 = update_fatigue(fatigue_level_player1, action1, "normal")
    fatigue_level_player2 = update_fatigue(fatigue_level_player2, action2, "normal")

# Visualization
# Variational Free Energy Plot
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, TOTAL_POINTS, len(player1_vfe)), np.convolve(player1_vfe, np.ones(10)/10, mode='same'), label="Player 1 VFE (Smoothed)", color="black")
plt.plot(np.linspace(0, TOTAL_POINTS, len(player2_vfe)), np.convolve(player2_vfe, np.ones(10)/10, mode='same'), label="Player 2 VFE (Smoothed)", color="gray")
plt.axhline(y=0, color="red", linestyle="--", label="Baseline")
plt.xlabel("Minute")
plt.ylabel("Variational Free Energy")
plt.title("Fluctuations of Variational Free Energy During the Match (Smoothed)")
plt.legend()
plt.grid()
plt.show()

# Expected Free Energy Plot
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, TOTAL_POINTS, len(player1_efe)), np.convolve(player1_efe, np.ones(10)/10, mode='same'), label="Player 1 EFE (Smoothed)", color="blue")
plt.plot(np.linspace(0, TOTAL_POINTS, len(player2_efe)), np.convolve(player2_efe, np.ones(10)/10, mode='same'), label="Player 2 EFE (Smoothed)", color="green")
plt.axhline(y=0, color="red", linestyle="--", label="Baseline")
plt.xlabel("Minute")
plt.ylabel("Expected Free Energy")
plt.title("Fluctuations of Expected Free Energy During the Match (Smoothed)")
plt.legend()
plt.grid()
plt.show()
