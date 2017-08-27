import gym
import numpy as np
import simple_dqn_agent as simple_dqn
import results_plots as plots
import matplotlib.pyplot as plt

# Choose environment and run settings
env = gym.make('CartPole-v0')
render = False
max_episodes = 600
max_time_steps = 2000
n_learning_repeats = 1
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialise empty arrays to store rewards and explore rates per learning repeat
reward_array = np.zeros((n_learning_repeats, max_episodes))
explore_rate_array = np.zeros((n_learning_repeats, max_episodes))

# Loop over repeats
for repeat in range(n_learning_repeats):
    # Initialise agent
    agent = simple_dqn.Agent(n_states, n_actions, decay_explore=0.0001)
    # Loop over episodes
    for episode in range(max_episodes):
        episode_reward = agent.execute_episode(max_time_steps, render, env)
        explore_rate = agent.get_explore_rate()
        reward_array[repeat, episode] = episode_reward
        explore_rate_array[repeat, episode] = explore_rate
        if episode >= 10 and np.mod(episode, 10) == 0:
            print("Mean score over episodes " + str(episode - 10) + " to " + str(episode) + " = " +
                  str(np.mean(reward_array[repeat, episode-10:episode])))

plots.plot_reward_explore_min_mean_max(reward_array, explore_rate_array)
plots.plot_reward_explore_all(reward_array, explore_rate_array)
plt.show()

