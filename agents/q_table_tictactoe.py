import numpy as np
import random
import math

# Q-table learning agent and environment for tic-tac-toe, created by Chris Nash

class Agent:
    def __init__(self, n_states, discount=0.99, max_explore=1,
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
        self.n_states = n_states  # N x 1 vector, no. of discrete states in each state-space dimension
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # N x 1 vector, no. of discrete states in each state-space dimension
        self.n_actions = 9  # Scalar defining the number of possible actions
        self.n_observations = len(n_states)+1  # Dimension of the state space (or number of continuous observations)
        # Initialise empty Q_Table
        self.q_table = np.zeros(self.n_states + (self.n_actions,))

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

    def execute_episode(self, max_time_steps, render):
        # initialise empty board
        episode_reward = 0
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        for time_step in range(max_time_steps):

            # choose action (square to place marker)
            states = self.state
            x0, x1, x2, x3, x4, x5, x6, x7, x8 = states
            action = self.act(states)

            # place marker on square (if empty)
            if action == 0 and x0 == 0:
                x0 = 1
            elif action == 1 and x1 == 0:
                x1 = 1
            elif action == 2 and x2 == 0:
                x2 = 1
            elif action == 3 and x3 == 0:
                x3 = 1
            elif action == 4 and x4 == 0:
                x4 = 1
            elif action == 5 and x5 == 0:
                x5 = 1
            elif action == 6 and x6 == 0:
                x6 = 1
            elif action == 7 and x7 == 0:
                x7 = 1
            elif action == 8 and x8 == 0:
                x8 = 1

            # player 2 chooses random square
            player_2_action = random.randint(0, self.n_actions-1)

            # place player 2's marker (if square empty)
            if player_2_action == 0 and x0 == 0:
                x0 = 2
            elif player_2_action == 1 and x1 == 0:
                x1 = 2
            elif player_2_action == 2 and x2 == 0:
                x2 = 2
            elif player_2_action == 3 and x3 == 0:
                x3 = 2
            elif player_2_action == 4 and x4 == 0:
                x4 = 2
            elif player_2_action == 5 and x5 == 0:
                x5 = 2
            elif player_2_action == 6 and x6 == 0:
                x6 = 2
            elif player_2_action == 7 and x7 == 0:
                x7 = 2
            elif player_2_action == 8 and x8 == 0:
                x8 = 2

            # check for player 1 win
            player_1_wins = (x0 == 1 and x1 == 1 and x2 == 1) \
                   or (x3 == 1 and x4 == 1 and x5 == 1) \
                   or (x6 == 1 and x7 == 1 and x8 == 1) \
                   or (x0 == 1 and x3 == 1 and x6 == 1) \
                   or (x1 == 1 and x4 == 1 and x7 == 1) \
                   or (x2 == 1 and x5 == 1 and x8 == 1) \
                   or (x0 == 1 and x4 == 1 and x8 == 1) \
                   or (x2 == 1 and x4 == 1 and x6 == 1)
            player_1_wins = bool(player_1_wins)

            # check for player 2 win
            player_2_wins = (x0 == 2 and x1 == 2 and x2 == 2) \
                            or (x3 == 2 and x4 == 2 and x5 == 2) \
                            or (x6 == 2 and x7 == 2 and x8 == 2) \
                            or (x0 == 2 and x3 == 2 and x6 == 2) \
                            or (x1 == 2 and x4 == 2 and x7 == 2) \
                            or (x2 == 2 and x5 == 2 and x8 == 2) \
                            or (x0 == 2 and x4 == 2 and x8 == 2) \
                            or (x2 == 2 and x4 == 2 and x6 == 2)
            player_2_wins = bool(player_2_wins)

            # check for draw
            players_draw = (x0 != 0 and x1 != 0 and x2 != 0 and x3 != 0 and x4 != 0 and x5 != 0 and x6 != 0 and x7 != 0 and x8 != 0)

            # if game finished, allocate reward
            if player_1_wins:
                reward = 1
            elif player_2_wins:
                reward = -1
            else:
                reward = 0
            done = player_1_wins or player_2_wins or players_draw

            # update states and q table
            self.state = (x0, x1, x2, x3, x4, x5, x6, x7, x8)
            new_states = self.state
            states = list(states)
            self.update(states, action, new_states, reward)
            episode_reward += reward

            # end game if finished
            if done:
                print(self.state[0:3])
                print(self.state[3:6])
                print(self.state[6:9])
                if player_1_wins:
                    print('player 1 wins')
                elif player_2_wins:
                    print('player 2 wins')
                elif players_draw:
                    print('players draw')
                break
        return episode_reward
