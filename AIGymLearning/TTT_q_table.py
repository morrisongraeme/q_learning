import gym
import numpy as np
import TTT_agent as q_table
import results_plots as plots
import matplotlib.pyplot as plt

# Choose environment and run settings
render = False
max_episodes = 10000
max_time_steps = 10

# Agent hyper-parameters
n_states = (3, 3, 3, 3, 3, 3, 3, 3, 3)

# Initialise agent
agent = q_table.Agent(n_states)

# Initialise empty list to store rewards per episode
reward_list = []
learning_rate_list = []
explore_rate_list = []

for episode in range(max_episodes):
    episode_reward = agent.execute_episode(max_time_steps, render)
    explore_rate = agent.get_explore_rate()
    learning_rate = agent.get_learning_rate()
    reward_list.append(episode_reward)
    learning_rate_list.append(learning_rate)
    explore_rate_list.append(explore_rate)
    if episode >= 10 and np.mod(episode, 10) == 0:
            print("Mean score over episodes " + str(episode - 10) + " to " + str(episode) + " = " +
                  str(np.mean(reward_list[episode-10:episode])))

plots.plot_reward_explore_learning(reward_list, explore_rate_list, learning_rate_list)
plt.show()