import gym
import numpy as np
import agents.q_network as dqn
import utils.results_plots as plots
import matplotlib.pyplot as plt

# Generic settings
render_interval = 100
max_episodes = 1000
n_learning_repeats = 1
start_with_filled_memory = True

# Environment selection: 1 for MountainCar-v0, 2 for CartPole-v0, 3 for LunarLander-v2
# env_name = 'MountainCar-v0'
# env_name = 'CartPole-v0'
env_name = 'LunarLander-v2'

# Make selected environment
env = gym.make(env_name)

# Environment-specific settings
if env_name is 'MountainCar-v0':
    env._max_episode_steps = 100000  # Over-ride default maximum number of time steps allowed per episode
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n-1  # Set this to n-1 because we are remapping the action space from size 3 to 2
    options = {'min_explore': 0.1, 'action_remap_gain': 2, 'separate_target': True, 'update_target_interval': 5000}
elif env_name is 'CartPole-v0':
    # env._max_episode_steps = 200  # Over-ride default maximum number of time steps allowed per episode
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    options = {'separate_target': True}
elif env_name is 'LunarLander-v2':
    # env._max_episode_steps = 200  # Over-ride default maximum number of time steps allowed per episode
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    options = {'min_explore': 0.01, 'decay_explore': 0.0001, 'n_hidden_neurons': 256, 'separate_target': True,
               'update_target_interval': 5000, 'loss_type': 'huber'}

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

    #save_paths.append(agent.brain.save_network("SimpleDQNs/Agent" + str(repeat)))  # Uncomment to save trained agents

plots.plot_reward_explore_min_mean_max(reward_array, explore_rate_array)
plots.plot_reward_explore_all(reward_array, explore_rate_array)
plt.show()

# Uncomment this section to replay ten episodes with each trained agent
#agent = dqn.Agent(n_states, n_actions, max_explore=0, min_explore=0)
#for save_path in save_paths:
#    agent.brain.restore_network(save_path)
#    replay_rewards = np.zeros((10, 1))
#    for episode in range(10):
#        replay_rewards[episode] = agent.replay_episode(max_time_steps, True, env)
#    print("Trained agent " + save_path + " achieved average reward " +
#          str(np.mean(replay_rewards)) + " over 10 episodes")
