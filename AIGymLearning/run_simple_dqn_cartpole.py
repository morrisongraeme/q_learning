import gym
import numpy as np
import simple_dqn_agent as simple_dqn
import results_plots as plots
import matplotlib.pyplot as plt


# Choose environment and run settings
env = gym.make('CartPole-v0')
max_time_steps = 200  # Maximum number of time steps allowed per episode
env._max_episode_steps = max_time_steps
render = False
max_episodes = 1000

n_learning_repeats = 1
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialise empty arrays to store rewards and explore rates per learning repeat
reward_array = np.zeros((n_learning_repeats, max_episodes))
explore_rate_array = np.zeros((n_learning_repeats, max_episodes))
save_paths = []
# Loop over repeats
for repeat in range(n_learning_repeats):
    # Initialise agent
    agent = simple_dqn.Agent(n_states, n_actions)
    # Loop over episodes
    for episode in range(max_episodes):
        if np.mod(episode, 100) == 0 and episode > 0:  # Render one in every 100 episodes
            render = True
        else:
            render = False
        episode_reward = agent.execute_episode(max_time_steps, render, env)
        explore_rate = agent.get_explore_rate()
        memory_size = agent.memory.get_memory_size()
        reward_array[repeat, episode] = episode_reward
        explore_rate_array[repeat, episode] = explore_rate
        if episode >= 10 and np.mod(episode, 10) == 0:
            print("Mean score over episodes " + str(episode - 10) + " to " + str(episode) + " = " +
                  str(np.mean(reward_array[repeat, episode-10:episode])) + ", Memory Size = " + str(memory_size) +
                  ", Explore Rate = " + str(explore_rate))

    save_paths.append(agent.brain.save_network("SimpleDQNs/Agent" + str(repeat)))  # Uncomment to save trained agents

print(save_paths)

plots.plot_reward_explore_min_mean_max(reward_array, explore_rate_array)
plots.plot_reward_explore_all(reward_array, explore_rate_array)
plt.show()

# Uncomment this section to replay ten episodes with each trained agent
agent = simple_dqn.Agent(n_states, n_actions, max_explore=0, min_explore=0)
for save_path in save_paths:
    agent.brain.restore_network(save_path)
    replay_rewards = np.zeros((10, 1))
    for episode in range(10):
        replay_rewards[episode] = agent.replay_episode(max_time_steps, True, env)
    print("Trained agent " + save_path + " achieved average reward " +
          str(np.mean(replay_rewards)) + " over 10 episodes")
