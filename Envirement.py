import Parameters
import numpy as np
import DQNAgent


class Envirement:
    
    def __init__(self):
        self.pa = Parameters.Parameters()
        
    
    # input: alloc_req -allocated request, queue - all input requests in SF = sf_i and RBG - rbg_i
    # returns reward in rbg_i and sf_i - reward calcdepends on the strategy
    # updates queue buffer and returns next_state
    # queue - np matrix [state_size x 3]
    # next_state - np matrix [state_size x 3] with updated buffer

    def env_reward_next_state_sf_rbg(self,alloc_req,queue,sf_i,rbg_i):
        alloc = int(alloc_req)
        if queue[alloc,2]*(14*12*self.pa.RBG_SIZE)>=queue[alloc,1]:
            reward = (queue[alloc,1]) * abs(queue[alloc,0]+1)
        if queue[alloc,2]*(14*12*self.pa.RBG_SIZE)<=queue[alloc,1]:
            reward = (queue[alloc,2])*(14*12*self.pa.RBG_SIZE) * abs(queue[alloc,0]+1) # CQIxRBGx(QCI+1)
        if reward<0:
            reward = 0
        queue[alloc,1] = queue[alloc,1]-queue[alloc,2]*(14*12*self.pa.RBG_SIZE)
        if queue[alloc,1]<0:
            queue[alloc,0] = -1000
            queue[alloc,2] = -1000
        next_state = queue
        return reward,next_state
        
