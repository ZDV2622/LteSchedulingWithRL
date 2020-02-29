import numpy as np
import math
from collections import deque

class Parameters:
    def __init__(self):
        
        # LTE system parameters
        self.N_RB = 100 # Number of PRB in BW 
        self.RBG_SIZE = 4 # RBG Size for scheduling
        self.N_RBG = int(self.N_RB/self.RBG_SIZE)  # Number of RBG in all BW

        # Parameters for traffic generation
        self.MAX_UE_SF = 7 # max number of generated UE in SF
        self.UE_SF_FIXED = 1 # if 1 - then fixed number of UE=self.MAX_UE_SF will be in each SF
        self.QCI_N = 3 # number of qci types for ue
        self.UE_FIXED_QCI = 1 # 1 -then each QCI type will be presented for UE
        self.MAX_SF_QUE = self.MAX_UE_SF*self.QCI_N # max que size, wich can be scheduled in SF
        self.CQI_UE = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # for some ue select fixed CQI for all RBG and all SF
        self.MAX_REWARD_SF = 6*(14*12*self.N_RB) * self.QCI_N # max reward in SF(depends on the reward strategy)
        self.UE_MAX_BUFF = self.MAX_REWARD_SF+1
        self.UE_MIN_BUFF = self.MAX_REWARD_SF/self.N_RBG
        self.DEQUE_SF_N = 3


        ## model parameters
        self.batch_size = 1000 # for dqn model
        self.state_size =  self.MAX_SF_QUE # = to max que number in SF
        self.req_size = 2 # number of features related to request(exmpl: CQI, QCI, Buffer size)
        self.action_size = self.MAX_SF_QUE # = to max que number in SF
        self.memory = deque(maxlen=self.batch_size+20) # for dqn model
        self.memory_DQN = self.memory # for dqn model
        self.gamma = 1   # discount rate 
        self.epsilon = 0  # exploration rate (no impact to PGagent)
        self.epsilon_min = 0 # exploration rate (no impact to PGagent)
        self.epsilon_decay = 0.998 # exploration rate (no impact to PGagent)
        self.learning_rate = 0.0005  
        
