import gym
import numpy as np
import simple_dqn_agent as simple_dqn
import results_plots as plots

# Choose environment and run settings
env = gym.make('CartPole-v0')
render = False
max_episodes = 2000
max_time_steps = 200

# Initialise agent
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = simple_dqn.Agent(n_states,n_actions)

# Initialise empty list to store rewards per episode
reward_list = []
explore_rate_list = []

# Loop over episodes
for episode in range(max_episodes):
    episode_reward = agent.execute_episode(max_time_steps, render, env)
    explore_rate = agent.get_explore_rate()
    reward_list.append(episode_reward)
    explore_rate_list.append(explore_rate)
    if episode >= 10 and np.mod(episode, 10) == 0:
            print("Mean score over episodes " + str(episode - 10) + " to " + str(episode) + " = " +
                  str(np.mean(reward_list[episode-10:episode])))

plots.plot_reward_explore_learning(reward_list, explore_rate_list, None)
