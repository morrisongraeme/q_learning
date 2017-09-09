import gym
import numpy as np
import simple_dqn_agent as simple_dqn

env = gym.make('MountainCar-v0')
max_time_steps = 1000000  # Maximum number of time steps allowed per episode
env._max_episode_steps = max_time_steps
render = False
max_episodes = 1000

n_learning_repeats = 1
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n-1

save_paths = ['.SimpleDQNs/Agent0']

# Uncomment this section to replay ten episodes with each trained agent
agent = simple_dqn.Agent(n_states, n_actions, max_explore=0, min_explore=0)
for save_path in save_paths:
    agent.brain.restore_network(save_path)
    replay_rewards = np.zeros((10, 1))
    for episode in range(10):
        replay_rewards[episode] = agent.replay_episode(max_time_steps, True, env)
    print("Trained agent " + save_path + " achieved average reward " +
          str(np.mean(replay_rewards)) + " over 10 episodes")