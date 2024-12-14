
import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
T = 100  # Number of time steps
n_states = [3, 3, 3]  # States for Designer, Artifact, User
n_actions = [2, 2, 2]  # Actions for Designer, Artifact, User
n_obs = [3, 3, 3]  # Observations for Designer, Artifact, User

# Initialize Transition Matrices
T_designer = np.random.rand(n_states[0], n_states[0], n_actions[0])
T_artifact = np.random.rand(n_states[1], n_states[1], n_actions[1])
T_user = np.random.rand(n_states[2], n_states[2], n_actions[2])

# Normalize Transition Matrices
def normalize(P):
    return P / P.sum(axis=1, keepdims=True)

for i in range(n_actions[0]):
    T_designer[:, :, i] = normalize(T_designer[:, :, i])
for i in range(n_actions[1]):
    T_artifact[:, :, i] = normalize(T_artifact[:, :, i])
for i in range(n_actions[2]):
    T_user[:, :, i] = normalize(T_user[:, :, i])

# Initialize Observation Models
O_designer = np.random.rand(n_obs[0], n_states[0])
O_artifact = np.random.rand(n_obs[1], n_states[1])
O_user = np.random.rand(n_obs[2], n_states[2])

# Normalize Observation Models
O_designer = normalize(O_designer.T).T
O_artifact = normalize(O_artifact.T).T
O_user = normalize(O_user.T).T

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
def calculate_free_energy(belief, observation, O, T):
    predicted = O[observation, :] @ T @ belief
    return -np.sum(belief * np.log(predicted + 1e-8))

# Simulation Loop
for t in range(T):
    # Designer's Belief Update
    action_designer = np.random.choice(n_actions[0])
    state_designer = np.random.choice(n_states[0], p=T_designer[state_designer, :, action_designer])
    observation_designer = np.random.choice(n_obs[0], p=O_designer[:, state_designer])
    free_energy_designer = calculate_free_energy(belief_designer, observation_designer, O_designer, T_designer[:, :, action_designer])
    belief_designer = O_designer[observation_designer, :] * (T_designer[:, :, action_designer] @ belief_designer)
    belief_designer /= belief_designer.sum()

    # Artifact's Belief Update
    action_artifact = np.random.choice(n_actions[1])
    state_artifact = np.random.choice(n_states[1], p=T_artifact[state_artifact, :, action_artifact])
    observation_artifact = np.random.choice(n_obs[1], p=O_artifact[:, state_artifact])
    free_energy_artifact = calculate_free_energy(belief_artifact, observation_artifact, O_artifact, T_artifact[:, :, action_artifact])
    belief_artifact = O_artifact[observation_artifact, :] * (T_artifact[:, :, action_artifact] @ belief_artifact)
    belief_artifact /= belief_artifact.sum()

    # User's Belief Update
    action_user = np.random.choice(n_actions[2])
    state_user = np.random.choice(n_states[2], p=T_user[state_user, :, action_user])
    observation_user = np.random.choice(n_obs[2], p=O_user[:, state_user])
    free_energy_user = calculate_free_energy(belief_user, observation_user, O_user, T_user[:, :, action_user])
    belief_user = O_user[observation_user, :] * (T_user[:, :, action_user] @ belief_user)
    belief_user /= belief_user.sum()

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
