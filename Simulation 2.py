
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Designer
Designer = {
    'latent_states': ['Objectives'],
    'sensory_states': ['UserFeedback', 'RobotPerformance'],
    'actions': ['ModifyDesign'],
    'transition_model': np.array([[0.8, 0.2], [0.4, 0.6]]),
    'observation_model': np.array([[0.9, 0.1], [0.2, 0.8]])
}

# Parameters for Robot
Robot = {
    'latent_states': ['BehavioralModels'],
    'sensory_states': ['UserInput', 'EnvironmentData'],
    'actions': ['BehavioralOutputs'],
    'transition_model': np.array([[0.7, 0.3], [0.3, 0.7]]),
    'observation_model': np.array([[0.85, 0.15], [0.25, 0.75]])
}

# Parameters for User
User = {
    'latent_states': ['EmotionalStates'],
    'sensory_states': ['PerceivedRobotBehavior'],
    'actions': ['Feedback'],
    'transition_model': np.array([[0.9, 0.1], [0.3, 0.7]]),
    'observation_model': np.array([[0.8, 0.2], [0.3, 0.7]])
}

# Shared Variables
Shared = {
    'engagement_metrics': np.array([0.5, 0.5]),
    'task_success': np.array([0.6, 0.4])
}

# Simulation Parameters
T = 50  # Simulation time steps
Designer_belief = np.zeros((T, 2))
Robot_belief = np.zeros((T, 2))
User_belief = np.zeros((T, 2))

# Initialize beliefs
Designer_belief[0, :] = [0.7, 0.3]
Robot_belief[0, :] = [0.6, 0.4]
User_belief[0, :] = [0.8, 0.2]

for t in range(1, T):
    # Update Designer's belief
    Designer_belief[t, :] = Designer_belief[t-1, :].dot(Designer['transition_model'])
    Designer_observation = Designer_belief[t, :].dot(Designer['observation_model'])

    # Update Robot's belief
    Robot_belief[t, :] = Robot_belief[t-1, :].dot(Robot['transition_model'])
    Robot_observation = Robot_belief[t, :].dot(Robot['observation_model'])

    # Update User's belief
    User_belief[t, :] = User_belief[t-1, :].dot(User['transition_model'])
    User_observation = User_belief[t, :].dot(User['observation_model'])

    # Influence via shared variables
    Shared['engagement_metrics'] = 0.5 * (Designer_observation + User_observation)
    Shared['task_success'] = 0.5 * (Robot_observation + Designer_observation)

# Visualization
plt.figure(figsize=(10, 6))

# Designer Beliefs
plt.subplot(3, 1, 1)
plt.plot(range(T), Designer_belief[:, 0], '-r', label='Objective State 1')
plt.plot(range(T), Designer_belief[:, 1], '-b', label='Objective State 2')
plt.title('Designer Beliefs Over Time')
plt.xlabel('Time')
plt.ylabel('Belief States')
plt.legend()

# Robot Beliefs
plt.subplot(3, 1, 2)
plt.plot(range(T), Robot_belief[:, 0], '-g', label='Behavior Model 1')
plt.plot(range(T), Robot_belief[:, 1], '-k', label='Behavior Model 2')
plt.title('Robot Beliefs Over Time')
plt.xlabel('Time')
plt.ylabel('Belief States')
plt.legend()

# User Beliefs
plt.subplot(3, 1, 3)
plt.plot(range(T), User_belief[:, 0], '-m', label='Emotion State 1')
plt.plot(range(T), User_belief[:, 1], '-c', label='Emotion State 2')
plt.title('User Beliefs Over Time')
plt.xlabel('Time')
plt.ylabel('Belief States')
plt.legend()

plt.tight_layout()
plt.show()

# Shared Variables
plt.figure(figsize=(8, 4))
plt.plot(range(T), [Shared['engagement_metrics'][0]]*T, '-o', label='Engagement Metrics')
plt.plot(range(T), [Shared['task_success'][0]]*T, '-x', label='Task Success')
plt.title('Shared Variables Over Time')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
