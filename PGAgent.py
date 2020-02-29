import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import Parameters 
import InputTrafficGeneration
import Envirement
from sklearn import preprocessing


class PGAgent:
    def __init__(self):
        self.pa = Parameters.Parameters()
        self.state_size = self.pa.state_size
        self.action_size = self.pa.action_size
        self.req_size = self.pa.req_size
        self.gamma = self.pa.gamma   # discount rate
        self.epsilon = self.pa.epsilon # exploration rate
        self.learning_rate = self.pa.learning_rate
        self.states = []#deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.gradients = []#deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.rewards = []#deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.probs = []#deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # for this model we set for hidden layer neurons number = 2*input_dim
        model.add(Dense(self.state_size*self.req_size*2, input_dim=self.state_size*self.req_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def memorize(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state,act_type):
        # this agent can perform 3 action types
        if act_type == 'RAND':
            action_rand =  random.randrange(self.action_size)
            action = action_rand
            prob = 0
        if act_type == 'BestCQI':
            x = np.reshape(state, [self.pa.state_size, 3])
            prob = (( (abs(x[:,0])+1) + abs(x[:,2]) ) *( abs(x[:,1])/x[:,1])).max()
            action_BestCQI = (( (abs(x[:,0])+1) + abs(x[:,2]) ) *( abs(x[:,1])/x[:,1])).argmax(axis=0)
            action = action_BestCQI
        # DQN action
        if act_type == 'DQN':
            aprob = self.model.predict(state, batch_size=1).flatten()
            self.probs.append(aprob)
            prob = aprob / np.sum(aprob)
            action_dqn = np.random.choice(self.action_size, 1, p=prob)[0]
            action = action_dqn
        return action, prob
    
    def predict(self, state):
        model = self.load('PGmodel/pgmodel.h5')
        aprob = model.predict(state, batch_size=1).flatten()
        action = np.argmax(aprob)
        return action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        if self.pa.epsilon > self.pa.epsilon_min:
            self.pa.epsilon *= self.pa.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

