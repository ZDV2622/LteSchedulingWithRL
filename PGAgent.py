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
from keras.layers import Conv1D,Conv2D, GlobalAveragePooling1D, MaxPooling1D


class PGAgent:
    def __init__(self):
        self.pa = Parameters.Parameters()
        self.state_size = self.pa.state_size
        self.action_size = self.pa.action_size
        self.req_size = self.pa.req_size
        self.gamma = self.pa.gamma   # discount rate
        self.epsilon = self.pa.epsilon # exploration rate
        self.learning_rate = self.pa.learning_rate
        self.states =deque(maxlen=30) #deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.gradients = deque(maxlen=30) #deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.rewards = deque(maxlen=30) #deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.probs = deque(maxlen=30) #deque(maxlen=self.pa.MAX_SF_QUE*self.pa.N_RBG*self.pa.DEQUE_SF_N)
        self.model = self._build_model()
        self.model.summary()
        
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model        
        # Neural Net for Deep-Q learning Model
        model = Sequential()  # Neural network is a set of sequential layers
        #model.add(Reshape((20,3,1),input_shape = (20,3))) test
        model.add(Conv1D(filters=5,#self.pa.state_size,
                         kernel_size = 1,
                         input_shape=(self.state_size,3),
                         strides=1,#self.pa.REQ_PARAM_N-3, 
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=1))
        #model.add(Conv1D(32, 3, activation='relu', kernel_initializer="he_normal"))
        #model.add(MaxPooling1D(pool_size=2, strides=1))
        #model.add(GlobalAveragePooling1D())
        model.add(Flatten())
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
        #self.rewards.append(prob)

    def act(self, state,act_type,sfi):
        # this agent can perform 3 action types
        if act_type == 'RAND':
            action_rand =  random.randrange(self.action_size)
            action = action_rand
            prob = 0
        if act_type == 'BestCQI':
            #x = np.reshape(state, [self.pa.state_size, self.pa.REQ_PARAM_N])
            x = state
            prob = (x[:,3]/6+x[:,0]/9+x[:,1]/300).max()
            action_BestCQI = (x[:,3]/6+x[:,0]/9+x[:,1]/300).argmax(axis=0)
            action = action_BestCQI
        # DQN action
        if act_type == 'DQN':
            aprob = self.model.predict(np.reshape(state, [1, self.state_size,3]), batch_size=1).flatten()
            #print(aprob.shape)
            self.probs.append(aprob)
            prob = aprob / np.sum(aprob)
            action_dqn = np.random.choice(self.action_size, 1, p=prob)[0]
            if np.random.rand() >= self.epsilon:
                action_dqn = aprob.argmax(axis=0)
            action = action_dqn
        return action, prob
    
    def predict(self, state):
        model = self.load('PGmodel/pgmodel.h5')
        aprob = model.predict(state, batch_size=1).flatten()
        action = np.argmax(aprob)
        return action

    # not used now
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
        #rewards = self.discount_rewards(rewards)
        #print('prob',self.probs,'grad',self.gradients)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        gradients *= rewards
        #X = np.squeeze(np.vstack([self.states]))
        X = self.states
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        for ii in range(0,len(rewards),1):
            self.model.fit(np.reshape(X[ii], [1,self.state_size,3]), np.reshape(Y[ii], [1,self.state_size]),epochs=1, verbose=0)
        #self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        if self.pa.epsilon > self.pa.epsilon_min:
            self.pa.epsilon *= self.pa.epsilon_decay
            
    # not used
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            #print(np.reshape(state, [1, 15,3]).shape)
            target = reward
            #if not done:
            #     target = reward + self.gamma * \
            #           np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(np.reshape(state, [1, 15,3]))
            #print('target f',action,target_f[0],target_f[0][0])
            target_f[0][action] = target
            self.model.fit(np.reshape(state, [1, 15,3]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

