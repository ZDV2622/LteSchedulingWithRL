import Parameters
import numpy as np
#import DQNAgent


class Envirement:
    
    def __init__(self):
        self.pa = Parameters.Parameters()
        
    
    # input: alloc_req -allocated request, queue - all input requests in SF = sf_i and RBG - rbg_i
    # returns reward in rbg_i and sf_i - reward calc depends on the strategy
    # updates queue buffer and returns next_state
    # queue - np matrix [state_size x 3]
    # next_state - np matrix [state_size x 3] with updated buffer

    def env_reward_next_state_sf_rbg(self,alloc_req,queue_n,queue_orig,sf_i,rbg_i):
        alloc = int(alloc_req)
        reward = queue_orig[alloc,3]/6 + queue_orig[alloc,0]/9 #+ queue_orig[alloc,1]/300 #(14*12*self.pa.RBG_SIZE)#+queue_orig[alloc,1] 
        #if queue[alloc,2]<=0 or queue[alloc,0]<0 or queue[alloc,1]<=0 or queue[alloc,3]<=0:
            #reward = 0
        queue_orig[alloc,2] = queue_orig[alloc,2]-abs(queue_orig[alloc,3])*(14*12*self.pa.RBG_SIZE)
        if queue_orig[alloc,2]<=0:
            #if reward > 0:
                #print('all data send', queue_orig[alloc,2])
            queue_orig[alloc,0] = 0
            queue_orig[alloc,1] = 0
            queue_orig[alloc,3] = 0
            queue_orig[alloc,2] = 0
        next_state = queue_orig
        #print('output req ',next_state,'\n',reward,'\n')
        return reward,next_state#,throughput
        
