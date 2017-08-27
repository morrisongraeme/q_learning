import matplotlib.pyplot as plt
import numpy as np


# Plots reward averaged every N episodes, explore rate and learning rate
def plot_reward_explore_learning(reward_list, explore_rate_list, learning_rate_list, stride=10):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(moving_average(reward_list, stride))
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode reward')

    ax2.plot(explore_rate_list)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Explore rate')

    if learning_rate_list is not None:
        ax3.plot(learning_rate_list)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Learning rate')


# Plots min, mean and max reward and explore rate from learning repeats, averaged every N episodes
def plot_reward_explore_min_mean_max(reward_array, explore_rate_array, stride=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    mean_reward = np.mean(reward_array, 0)
    min_reward = np.min(reward_array, 0)
    max_reward = np.max(reward_array, 0)
    ax1.plot(moving_average(min_reward, stride), 'r--', moving_average(mean_reward, stride), 'b',
             moving_average(max_reward, stride), 'r--')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode reward')

    mean_explore = np.mean(explore_rate_array, 0)
    min_explore = np.min(explore_rate_array, 0)
    max_explore = np.max(explore_rate_array, 0)
    ax2.plot(moving_average(min_explore, stride), 'r--', moving_average(mean_explore, stride), 'b',
             moving_average(max_explore, stride), 'r--')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Explore rate')


# Plots all reward and explore rate from learning repeats, averaged every N episodes
def plot_reward_explore_all(reward_array, explore_rate_array, stride=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i in range(reward_array.shape[0]):
        print(i)
        ax1.plot(moving_average(reward_array[i, :], stride), '-')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode reward')

    for i in range(explore_rate_array.shape[0]):
        ax2.plot(moving_average(explore_rate_array[i, :], stride), '-')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Explore rate')


# Take un-padded moving average of a data list
def moving_average(data, n_points=3):
    ret = np.cumsum(data, dtype=float)
    ret[n_points:] = ret[n_points:] - ret[:-n_points]
    return ret[n_points - 1:] / n_points



