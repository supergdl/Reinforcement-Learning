import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

EPISODES = 1000
resume = False

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.Q_model = self._build_model()
		self.Target_model = self._build_model()

	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() < self.epsilon:
			return random.randrange(self.action_size) 
		# act_value, numpy.ndarray, e.g. [[ 0.01849771 -0.00890147]], act_value.shape (1, 2)
		act_value = self.Q_model.predict(state)
		# returns action, act_values[0]=[ 0.01849771 -0.00890147]
		return np.argmax(act_value[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, _ in minibatch:
			target = reward # if done, it's the terminal state, target = current reward
			if not done:
				##### DDQN implementation 
				#: seperate action selection and action evaluation
				action_star = np.argmax(self.Q_model.predict(next_state))
				evaluation = self.Target_model.predict(next_state)
				target = reward + self.gamma*evaluation[0][action_star]

			target_f = self.Q_model.predict(state)
			# NOTE: since our y is a 2D array, the target reward is about the larger one, 
			# we need to replace the predicted action's reward with the target reward
			# so that the y remains 2D 
			target_f[0][action] = target
			self.Q_model.fit(state, target_f, epochs=1, verbose=0)


		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def load(self, name):
		self.Q_model.load_weights(name)


	def save(self, name):
		self.Q_model.save_weights(name)




if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	if resume:
		agent.load("./model/ddqn_offpolicy_carpole.weight.save")
	
	done = False
	batch_size = 32

	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])

		for time in range(500):
			env.render()

			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			rewar = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			
			agent.remember(state, action, reward, next_state, done)

			state = next_state

			if done:
				print("episode: {}/{}, score: {}, e:{:.2}"
					.format(e, EPISODES, time, agent.epsilon))
				break

			if len(agent.memory) > batch_size:
				agent.replay(batch_size)

		if e%10 == 0:
			agent.save("./model/ddqn_offpolicy_carpole.weight.save")


		# copy weights from Q_model to target_model
		if e%25 == 0:
			agent.Target_model.set_weights(agent.Q_model.get_weights()) 




