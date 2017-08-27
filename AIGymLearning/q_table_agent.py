import numpy as np
import random
import math


class Agent:
    def __init__(self, n_states, state_box_low, state_box_high, env, discount=0.99, max_explore=1,
                 min_explore=0.01, decay_explore=0.0001, max_learn=0.5, min_learn=0.1, decay_learn=0.0001):
        # Initialise with zero steps taken
        self.steps = 0
        # Define explore and learning rate parameters
        self.max_explore_rate = max_explore
        self.min_explore_rate = min_explore
        self.explore_decay_rate = decay_explore
        self.max_learning_rate = max_learn
        self.min_learning_rate = min_learn
        self.learning_decay_rate = decay_learn
        self.discount_factor = discount
        # Define size and discretisation of Q_Table
        self.state_box_low = state_box_low  # Lower limits of discretised state in each dimension
        self.state_box_high = state_box_high  # Upper limits of discretised state in each dimension
        self.n_states = n_states  # N x 1 vector, no. of discrete states in each state-space dimension
        self.n_actions = env.action_space.n  # Scalar defining the number of possible actions
        self.n_observations = len(n_states)+1  # Dimension of the state space (or number of continuous observations)
        # Initialise empty Q_Table
        self.q_table = np.zeros(self.n_states + (env.action_space.n,))

    def act(self, states):
        # Compute explore rate, which decays with agent's experience
        explore_rate = self.get_explore_rate()
        # Either select a random action or best action from table, depending on current explore rate
        if random.random() <= explore_rate:
            action = random.randint(0, self.n_actions-1)
        else:
            action = np.argmax(self.q_table[tuple(states)])
        self.steps += 1
        return action

    def discretise(self, observations):
        states = []
        for i in range(len(observations)):
            if self.n_states[i] == 1:
                state = 0
            elif observations[i] <= self.state_box_low[i]:
                state = 0
            elif observations[i] > self.state_box_high[i]:
                state = self.n_states[i]-1
            else:
                d = (self.state_box_high[i] - self.state_box_low[i])/self.n_states[i]  # Width of each state bucket
                state = math.floor((observations[i]-self.state_box_low[i])/d)
            states.append(state)
        return states

    def update(self, states, action, next_states, reward):
        learning_rate = self.get_learning_rate()
        self.q_table[tuple(states + [action])] += learning_rate * \
            (reward + self.discount_factor*np.max(self.q_table[tuple(next_states)]) -
             self.q_table[tuple(states + [action])])

    def get_learning_rate(self):
        learning_rate = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * \
                                                 np.exp(-self.learning_decay_rate * self.steps)
        return learning_rate

    def get_explore_rate(self):
        explore_rate = self.min_explore_rate + (self.max_explore_rate - self.min_explore_rate) * \
                                                 np.exp(-self.explore_decay_rate * self.steps)
        return explore_rate

    def execute_episode(self, max_time_steps, render, env):
        episode_reward = 0
        observations = env.reset()
        states = self.discretise(observations)
        for time_step in range(max_time_steps):
            if render:
                env.render()
            action = self.act(states)
            new_observations, reward, done, _ = env.step(action)
            new_states = self.discretise(new_observations)
            self.update(states, action, new_states, reward)
            states = new_states
            episode_reward += reward
            if done:
                break
        return episode_reward
