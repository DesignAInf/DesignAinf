
import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
T = 100  # Number of time steps
n_states = [3, 3, 3]  # States for Designer, Artifact, User
n_actions = [2, 2, 2]  # Actions for Designer, Artifact, User
n_obs = [3, 3, 3]  # Observations for Designer, Artifact, User

# Initialize Transition Matrices
def initialize_transition_matrices(n_states, n_actions):
    T = np.random.rand(n_states, n_states, n_actions)
    for i in range(n_actions):
        T[:, :, i] = normalize(T[:, :, i])
    return T

T_designer = initialize_transition_matrices(n_states[0], n_actions[0])
T_artifact = initialize_transition_matrices(n_states[1], n_actions[1])
T_user = initialize_transition_matrices(n_states[2], n_actions[2])

# Normalize Function
def normalize(P):
    return P / P.sum(axis=1, keepdims=True)

# Initialize Observation Models
def initialize_observation_models(n_obs, n_states):
    O = np.random.rand(n_obs, n_states)
    return normalize(O.T).T

O_designer = initialize_observation_models(n_obs[0], n_states[0])
O_artifact = initialize_observation_models(n_obs[1], n_states[1])
O_user = initialize_observation_models(n_obs[2], n_states[2])

# Precision Parameters (Adjustable by Designer)
precision_designer = 1.0
precision_artifact = 1.0
precision_user = 1.0

# Curiosity Parameters (Adjustable by Designer)
curiosity_designer = 0.1
curiosity_artifact = 0.1
curiosity_user = 0.1

# Prediction Priors (Adjustable by Designer)
prior_designer = np.array([0.5, 0.3, 0.2])
prior_artifact = np.array([0.4, 0.4, 0.2])
prior_user = np.array([0.6, 0.2, 0.2])

# Initialize Beliefs
belief_designer = np.ones(n_states[0]) / n_states[0]
belief_artifact = np.ones(n_states[1]) / n_states[1]
belief_user = np.ones(n_states[2]) / n_states[2]

# Initialize States
state_designer = np.random.choice(n_states[0])
state_artifact = np.random.choice(n_states[1])
state_user = np.random.choice(n_states[2])

# Record States, Beliefs, and Free Energy
states_over_time = np.zeros((T, 3))
beliefs_over_time = np.zeros((T, sum(n_states)))
free_energy_over_time = np.zeros((T, 3))

# Free Energy Function
def calculate_free_energy(belief, observation, O, T, precision):
    predicted = O[observation, :] @ T @ belief
    return -precision * np.sum(belief * np.log(predicted + 1e-8))

# Simulation Loop
for t in range(T):
    # Designer's Belief Update
    action_designer = np.random.choice(n_actions[0])
    state_designer = np.random.choice(n_states[0], p=T_designer[state_designer, :, action_designer])
    observation_designer = np.random.choice(n_obs[0], p=O_designer[:, state_designer])
    free_energy_designer = calculate_free_energy(belief_designer, observation_designer, O_designer, T_designer[:, :, action_designer], precision_designer)
    belief_designer = O_designer[observation_designer, :] * (T_designer[:, :, action_designer] @ belief_designer) + curiosity_designer
    belief_designer /= belief_designer.sum()
    belief_designer = belief_designer * (1 - prior_designer) + prior_designer

    # Artifact's Belief Update
    action_artifact = np.random.choice(n_actions[1])
    state_artifact = np.random.choice(n_states[1], p=T_artifact[state_artifact, :, action_artifact])
    observation_artifact = np.random.choice(n_obs[1], p=O_artifact[:, state_artifact])
    free_energy_artifact = calculate_free_energy(belief_artifact, observation_artifact, O_artifact, T_artifact[:, :, action_artifact], precision_artifact)
    belief_artifact = O_artifact[observation_artifact, :] * (T_artifact[:, :, action_artifact] @ belief_artifact) + curiosity_artifact
    belief_artifact /= belief_artifact.sum()
    belief_artifact = belief_artifact * (1 - prior_artifact) + prior_artifact

    # User's Belief Update
    action_user = np.random.choice(n_actions[2])
    state_user = np.random.choice(n_states[2], p=T_user[state_user, :, action_user])
    observation_user = np.random.choice(n_obs[2], p=O_user[:, state_user])
    free_energy_user = calculate_free_energy(belief_user, observation_user, O_user, T_user[:, :, action_user], precision_user)
    belief_user = O_user[observation_user, :] * (T_user[:, :, action_user] @ belief_user) + curiosity_user
    belief_user /= belief_user.sum()
    belief_user = belief_user * (1 - prior_user) + prior_user

    # Record States, Beliefs, and Free Energy
    states_over_time[t, :] = [state_designer, state_artifact, state_user]
    beliefs_over_time[t, :] = np.concatenate([belief_designer, belief_artifact, belief_user])
    free_energy_over_time[t, :] = [free_energy_designer, free_energy_artifact, free_energy_user]

# Visualizations

# States Over Time
plt.figure(figsize=(10, 6))
plt.plot(range(T), states_over_time)
plt.legend(['Designer', 'Artifact', 'User'])
plt.xlabel('Time Steps')
plt.ylabel('State Index')
plt.title('States of Designer, Artifact, and User Over Time')
plt.grid(True)
plt.show()

# Belief Dynamics
plt.figure(figsize=(10, 6))
plt.imshow(beliefs_over_time.T, aspect='auto', cmap='Greys')
plt.colorbar(label='Belief Value')
plt.xlabel('Time Steps')
plt.ylabel('Belief Components')
plt.title('Belief Dynamics Over Time')
plt.grid(True)
plt.show()

# Free Energy Over Time
plt.figure(figsize=(10, 6))
plt.plot(range(T), free_energy_over_time)
plt.legend(['Designer', 'Artifact', 'User'])
plt.xlabel('Time Steps')
plt.ylabel('Free Energy')
plt.title('Free Energy Minimization Over Time')
plt.grid(True)
plt.show()
