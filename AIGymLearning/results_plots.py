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

    plt.show()


# Take un-padded moving average of a data list
def moving_average(data, n_points=3):
    ret = np.cumsum(data, dtype=float)
    ret[n_points:] = ret[n_points:] - ret[:-n_points]
    return ret[n_points - 1:] / n_points



