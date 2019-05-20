import numpy as np
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        # 'Dense' is the basic form of a neural network layer
        # Input layer of state size (4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # Hidden Layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):

        # Sample minibatch from the memory
        minibatch = random.sample(self.memory, batch_size)
        # Extract information from each memory
        for state, action, reward, next_state, done in minibatch:
            # If done, make out target reward
            target = reward

            if not done:
                # Predict the future disconuted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Make the agent to approximately map the current state to future discounted reward
            # We'll call the target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


if __name__ == '__main__':

    # Inizialize gym environment and the agent
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    # Iterate the game
    for e in range(EPISODES):

        # reset state in the beginnig of each game
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want ro render
            # env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make the previous state, action, reward and done
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # Print the score and break out of the loop
                print("Episode: {}/{}, score {}, e {:.2}".format(e, EPISODES,
                                                                 time_t, agent.epsilon))
                break

        # Train the agent with the experience of the episode
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
