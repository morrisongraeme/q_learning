import gym
from gym import wrappers
import numpy as np
import agents.q_network_convolutional as dqn
import utils.results_plots as plots
import matplotlib.pyplot as plt

# Generic settings
render_interval = 10000
max_episodes = 1000
n_learning_repeats = 1
start_with_filled_memory = True
monitor = False

# Environment selection
env_name = 'CartPole-v0'

# Make selected environment
env = gym.make(env_name)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
options = {'loss_type': 'mse', 'double_dqn': True}

# Initialise empty arrays to store rewards and explore rates per learning repeat
reward_array = np.zeros((n_learning_repeats, max_episodes))
explore_rate_array = np.zeros((n_learning_repeats, max_episodes))
save_paths = []

# Loop over repeats
for repeat in range(n_learning_repeats):

    # Initialise agent
    agent = dqn.Agent(n_states, n_actions, **options)
    if start_with_filled_memory:
        print('Allowing agent to randomly explore environment '
              'until memory capacity is filled before commencing learning')
        while agent.memory.get_memory_size() < agent.memory.capacity:
            agent.execute_episode(False, env, explore_override=1, learn=False)

    # Loop over episodes
    if monitor:
        env = wrappers.Monitor(env, '/monitors/' + env_name, force=True)
    for episode in range(max_episodes):

        # Render one in every render_interval episodes
        if np.mod(episode, render_interval) == 0 and episode > 0:
            render = True
        else:
            render = False

        # Execute the episode and store the reward, explore rate at finish and memory size at finish
        reward_array[repeat, episode] = agent.execute_episode(render, env)
        explore_rate_array[repeat, episode] = agent.get_explore_rate()
        memory_size = agent.memory.get_memory_size()

        # Print some aggregated performance information every 10 episodes
        if episode > 0 and np.mod(episode, 10) == 0:
            print("Mean score over episodes " + str(episode - 10) + " to " + str(episode) + " = " +
                  str(np.mean(reward_array[repeat, episode-10:episode])) + ", Memory Size = " + str(memory_size) +
                  ", Explore Rate = " + str(explore_rate_array[repeat, episode]))

plots.plot_reward_explore_min_mean_max(reward_array, explore_rate_array)
plots.plot_reward_explore_all(reward_array, explore_rate_array)
plt.show()
