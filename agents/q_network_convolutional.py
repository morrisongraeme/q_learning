import numpy as np
import random
import tensorflow as tf
import agents.tensorflow_utils as tf_utils


# Simple Deep Q Network agent based largely on https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

# Define the Agent class
class Agent:

    def __init__(self, n_states, n_actions, discount=0.99, min_explore=0.01, max_explore=1, decay_explore=0.001,
                 mem_capacity=100000, batch_size=64, update_target_interval=2000, loss_type='mse', double_dqn=True):
        # Initialise number of steps as zero
        self.steps = 0
        # Store hyper-parameters
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount
        self.min_explore_rate = min_explore
        self.max_explore_rate = max_explore
        self.explore_decay_rate = decay_explore
        self.batch_size = batch_size
        self.update_target_interval = update_target_interval
        self.double_dqn = double_dqn
        # Initialise brain and memory
        self.brain = Brain(n_states, n_actions, loss_type)
        self.memory = Memory(mem_capacity)

    def act(self, state, explore_override):
        # Select the next action, depending on explore rate
        if explore_override < 0:
            explore_rate = self.get_explore_rate()
        else:
            explore_rate = explore_override

        if random.random() < explore_rate:
            return random.randint(0, self.n_actions-1)
        else:
            return np.argmax(self.brain.predict(np.reshape(state, (1, state.shape[0]))))

    def observe(self, sample):
        # Observe the result of an action and add this to agent's memory
        self.memory.add(sample)

    def replay(self):
        # Replay observations from memory and use these to train the agent

        if self.steps % self.update_target_interval == 0:
            print('Copying q network parameters to target network')
            self.brain.copy_to_target()

        batch = self.memory.sample(self.batch_size)
        batch_length = len(batch)

        no_state = np.zeros(self.n_states)

        states = np.array([observation[0] for observation in batch])
        new_states = np.array([(no_state if observation[3] is None else observation[3]) for observation in batch])

        predictions = self.brain.predict(states)
        new_predictions = self.brain.predict(new_states, target=True)
        if self.double_dqn:
            new_predictions_main_network = self.brain.predict(new_states)

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
                if self.double_dqn:
                    target[action] = reward + self.discount_factor * \
                                              new_predictions[i][np.argmax(new_predictions_main_network[i])]
                else:
                    target[action] = reward + self.discount_factor*np.amax(new_predictions[i])

            inputs[i] = state
            outputs[i] = target

        self.brain.train(inputs, outputs)

    def get_explore_rate(self):
        # Compute and return explore rate based on number of steps experienced by agent
        explore_rate = self.min_explore_rate + (self.max_explore_rate - self.min_explore_rate) * \
                                               np.exp(-self.explore_decay_rate*self.steps)
        return explore_rate

    def execute_episode(self, render, env, explore_override=-1, learn=True, observe=True):
        # Execute a single episode of the agent acting and learning in the environment
        episode_reward = 0
        states = env.reset()
        while True:
            if render:
                env.render()
            action = self.act(states, explore_override)
            new_states, reward, done, _ = env.step(action)

            if done:
                new_states = None

            if observe:
                self.observe((states, action, reward, new_states))  # Commit the state-action-reward set to memory

            if learn:
                self.replay()  # Replay a set of samples from the agent's memory and use these to train the agent
                self.steps += 1  # Steps counter only increases if we are learning

            states = new_states
            episode_reward += reward

            if done:
                break
        return episode_reward


# Define the Brain class
class Brain:
    def __init__(self, n_states, n_actions, loss_type):
        self.n_states = n_states
        self.n_actions = n_actions
        self.sess = self._create_sess(loss_type)

    def _create_sess(self, loss_type):
        # Create a tensorFlow session, which defines the NN used in the agent's brain and the training approach
        tf.reset_default_graph()

        # Define input, variable and output placeholders for q network
        self.inputs1 = tf.placeholder(shape=[None, self.n_states], dtype=tf.float32)
        self.w1 = tf.get_variable("w1", shape=[self.n_states, 64])
        self.b1 = tf_utils.bias_variable([64])
        self.w2 = tf.get_variable("w2", shape=[64, self.n_actions])
        self.b2 = tf_utils.bias_variable([self.n_actions])
        h1 = tf.nn.relu(tf.matmul(self.inputs1, self.w1) + self.b1)
        self.prediction = tf.matmul(h1, self.w2) + self.b2
        self.next_q = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32)

        # Define target network variables and functions to copy them from q network
        self.w1_t = tf.get_variable("w1_t", shape=[self.n_states, 64])
        self.b1_t = tf_utils.bias_variable([64])
        self.w2_t = tf.get_variable("w2_t", shape=[64, self.n_actions])
        self.b2_t = tf_utils.bias_variable([self.n_actions])
        self.copy_w1 = self.w1_t.assign(self.w1)
        self.copy_w2 = self.w2_t.assign(self.w2)
        self.copy_b1 = self.b1_t.assign(self.b1)
        self.copy_b2 = self.b2_t.assign(self.b2)
        h1_t = tf.nn.relu(tf.matmul(self.inputs1, self.w1_t) + self.b1_t)
        self.prediction_t = tf.matmul(h1_t, self.w2_t) + self.b2_t

        # Define loss function and training
        if loss_type is 'mse':
            loss = tf.reduce_mean(tf.squared_difference(self.next_q, self.prediction))
        elif loss_type is 'huber':
            loss = tf.losses.huber_loss(self.next_q, self.prediction)
        trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025)
        self.update_model = trainer.minimize(loss)

        # Initialise session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Define a saver for saving off network parameters to a file
        # self.saver = tf.train.Saver([self.w1, self.w2, self.b1, self.b2])  # Define a saver to use later

        return sess

    def train(self, inputs, outputs):
        # Train the agent's brain
        self.sess.run([self.update_model], feed_dict={self.inputs1: inputs, self.next_q: outputs})

    def predict(self, states, target=False):
        # NN prediction of q values for a set of state inputs
        if target:
            a = self.sess.run([self.prediction_t],
                              feed_dict={self.inputs1: np.reshape(states, (states.shape[0], self.n_states))})
        else:
            a = self.sess.run([self.prediction],
                              feed_dict={self.inputs1: np.reshape(states, (states.shape[0], self.n_states))})
        return np.reshape(a, (states.shape[0], self.n_actions))

    # def save_network(self, file_path):
        # return self.saver.save(self.sess, file_path)

    # def restore_network(self, file_path):
        # self.saver.restore(self.sess, file_path)

    def copy_to_target(self):
        self.sess.run([self.copy_w1])
        self.sess.run([self.copy_w2])
        self.sess.run([self.copy_b1])
        self.sess.run([self.copy_b2])


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
