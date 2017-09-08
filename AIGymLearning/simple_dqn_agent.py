import numpy as np
import random
import tensorflow as tf


# Simple Deep Q Network agent based largely on https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

# Function for creating tensorflow weight variables  - not currently being used!
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


# Function for creating tensorflow bias variables
def bias_variable(shape):
    initial = tf.constant(float(0), shape=shape)
    return tf.Variable(initial)


# Define the Agent class
class Agent:

    def __init__(self, n_states, n_actions, discount=0.99, min_explore=0.01, max_explore=1,
                 decay_explore=0.001, mem_capacity=100000, batch_size=64, n_hidden_neurons=64, fill_memory=False,
                 action_remap_gain=1):
        # Initialise number of steps as zero
        self.steps = 0
        self.steps_filled = 0
        # Store hyper-parameters
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount
        self.min_explore_rate = min_explore
        self.max_explore_rate = max_explore
        self.explore_decay_rate = decay_explore
        self.batch_size = batch_size
        self.action_remap_gain = action_remap_gain
        # Initialise brain and memory
        self.brain = Brain(n_states, n_actions, n_hidden_neurons)
        self.fill_memory = fill_memory
        self.memory = Memory(mem_capacity)

    def act(self, state):
        # Select the next action, depending on explore rate
        self.steps += 1
        explore_rate = self.get_explore_rate()
        if random.random() < explore_rate:
            return random.randint(0, self.n_actions-1)
        else:
            return np.argmax(self.brain.predict(np.reshape(state, (1, state.shape[0]))))

    def observe(self, sample):
        # Observe the result of an action and add this to agent's memory
        self.memory.add(sample)

    def replay(self):
        # Replay observations from memory and use these to train the agent
        batch = self.memory.sample(self.batch_size)
        batch_length = len(batch)

        no_state = np.zeros(self.n_states)

        states = np.array([observation[0] for observation in batch])
        new_states = np.array([(no_state if observation[3] is None else observation[3]) for observation in batch])

        predictions = self.brain.predict(states)
        new_predictions = self.brain.predict(new_states)

        inputs = np.zeros((batch_length, self.n_states))
        outputs = np.zeros((batch_length, self.n_actions))

        for i in range(batch_length):
            observation = batch[i]
            state = observation[0]
            action = observation[1]
            reward = observation[2]
            new_state = observation[3]

            target = predictions[i]
            if new_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor*np.amax(new_predictions[i])

            inputs[i] = state
            outputs[i] = target

        self.brain.train(inputs, outputs)

    def get_explore_rate(self):
        # Compute and return explore rate based on number of steps experienced by agent
        if self.fill_memory:
            if self.memory.get_memory_size()<self.memory.capacity:
                explore_rate = 1
                self.steps_filled = self.steps
            else:
                explore_rate = self.min_explore_rate + (self.max_explore_rate - self.min_explore_rate) * \
                                                       np.exp(-self.explore_decay_rate * (self.steps-self.steps_filled))
        else:
            explore_rate = self.min_explore_rate + (self.max_explore_rate - self.min_explore_rate) * \
                                               np.exp(-self.explore_decay_rate*self.steps)
        return explore_rate

    def execute_episode(self, max_time_steps, render, env):
        # Execute a single episode of the agent acting and learning in the environment
        episode_reward = 0
        states = env.reset()
        for time_step in range(max_time_steps):
            if render:
                env.render()
            action = self.act(states)
            new_states, reward, done, _ = env.step(action*self.action_remap_gain)

            if done:
                new_states = None

            self.observe((states, action, reward, new_states))  # Commit the state-action-reward set to memory
            if (not self.fill_memory) or self.memory.get_memory_size()>=self.memory.capacity:
                self.replay()  # Replay a set of samples from the agent's memory and use these to train the agent

            states = new_states
            episode_reward += reward

            if done:
                break
        return episode_reward

    def replay_episode(self, max_time_steps, render, env):
        # Execute a single episode of the trained agent acting in the environment, but not learning
        episode_reward = 0
        states = env.reset()
        for time_step in range(max_time_steps):
            if render:
                env.render()
            action = self.act(states)
            new_states, reward, done, _ = env.step(action)

            if done:
                new_states = None

            states = new_states
            episode_reward += reward

            if done:
                break
        return episode_reward


# Define the Brain class
class Brain:
    def __init__(self, n_states, n_actions, n_hidden_neurons):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden_neurons = n_hidden_neurons
        self.sess = self._create_sess()

    def _create_sess(self):
        # Create a tensorflow session, which defines the NN used in the agent's brain and the training approach
        tf.reset_default_graph()
        self.inputs1 = tf.placeholder(shape=[None, self.n_states], dtype=tf.float32)
        self.w1 = tf.get_variable("w1", shape=[self.n_states, self.n_hidden_neurons])
        self.b1 = bias_variable([self.n_hidden_neurons])
        self.w2 = tf.get_variable("w2", shape=[self.n_hidden_neurons, self.n_actions])
        self.b2 = bias_variable([self.n_actions])
        h1 = tf.nn.relu(tf.matmul(self.inputs1, self.w1) + self.b1)
        q_out = tf.matmul(h1, self.w2) + self.b2
        self.prediction = q_out
        self.next_q = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)
        loss = tf.reduce_mean(tf.squared_difference(self.next_q, q_out))
        trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025)
        self.update_model = trainer.minimize(loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        self.saver = tf.train.Saver([self.w1, self.w2, self.b1, self.b2])  # Define a saver to use later
        return sess

    def train(self, inputs, outputs):
        # Train the agent's brain
        self.sess.run([self.update_model], feed_dict={self.inputs1: inputs, self.next_q: outputs})

    def predict(self, states):
        # NN prediction of q values for a set of state inputs
        a = self.sess.run([self.prediction], feed_dict={self.inputs1: np.reshape(states, (states.shape[0], self.n_states))})
        return np.reshape(a, (states.shape[0], self.n_actions))

    def save_network(self, file_path):
        return self.saver.save(self.sess, file_path)

    def restore_network(self, file_path):
        self.saver.restore(self.sess, file_path)


# Define the Memory class
class Memory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)  # Remove the oldest memory entry if we have filled capacity

    def sample(self, n_samples):
        n_samples = min(n_samples, len(self.samples))
        return random.sample(self.samples, n_samples)

    def get_memory_size(self):
        return len(self.samples)
