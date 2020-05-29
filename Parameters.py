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
        self.QCI_TYPE = [1,0,1,0,0,1,0,0,1] # QCI TYPE in NW
        self.QCI_N_TOTAL = 9
        self.QCI_N = sum(self.QCI_TYPE) # number of qci types
        self.QCI_DELAY = [300,300,300,300,300,300,300,300,300] # delay for QCI in MS
        #self.QCI_DELAY = [100,150,50,300,100,300,100,300,300] # delay for QCI in MS
        self.UE_FIXED_QCI = 0 # 1 -then each QCI type will be presented for UE
        self.MAX_SF_QUE = self.MAX_UE_SF*self.QCI_N # max que size, wich can be scheduled in SF
        self.CQI_UE = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # for some ue select fixed CQI for all RBG and all SF
        self.MAX_REWARD_SF = 1 # not used
        self.UE_MAX_BUFF = 6*(14*12*4)*10+1
        self.UE_MIN_BUFF = 6*(14*12*4)*10
        self.DEQUE_SF_N = 3
        self.REQ_PARAM_N = 4
        self.REQ_PARAM_N_FULL = 8


        ## model parameters
        self.batch_size = 1000 # for dqn model, not used now
        self.state_size =  self.MAX_SF_QUE # = to max que number in SF
        self.req_size = self.REQ_PARAM_N # number of features related to request(exmpl: CQI, QCI, delay)
        self.action_size = self.MAX_SF_QUE # = to max que number in SF
        self.memory = deque(maxlen=self.batch_size+20) # for dqn model
        self.gamma = 1   # discount rate 
        self.epsilon = 1  # exploration rate (no impact to PGagent)
        self.epsilon_min = 0.05 # exploration rate (no impact to PGagent)
        self.epsilon_decay = 0.9999 # exploration rate (no impact to PGagent)
        self.learning_rate = 0.005  
        
