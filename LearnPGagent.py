import Parameters
import PGAgent
import InputTrafficGeneration
import Envirement
from  Envirement import Envirement as env
import numpy as np
import random


# simulate learning
# LearnProcessPgagent starts simulation: 
# in each SF it generates requests, performs actions and estimates reward for each action

def LearnProcessPgagent(SF,sfi_train,sfi_save,sfi_show_reward):
    
    pa = Parameters.Parameters()
    state_size = pa.state_size
    action_size = pa.action_size
    agent = PGAgent.PGAgent()

    rew_sf_rand = np.zeros(SF)
    rew_sf_DQN = np.zeros(SF)
    rew_sf_BestCQI = np.zeros(SF)

    for sfi in range(0,SF,1):
        state_full_buffer = []
        state_full_buffer = InputTrafficGeneration.TrafficGeneration().full_que_sf_with_buffer(sfi)
        rbgi = 0
        for rbgi in range(pa.N_RBG-1):
            if rbgi == 0:
                state1_rand = state_full_buffer[:,[2,6,7]]
                state_rand = np.reshape(state1_rand, [1, state_size*3])
                state1_DQN = state_full_buffer[:,[2,6,7]]
                state3_DQN = state_full_buffer[:,[2,7]]
                state_DQN_fb = np.reshape(state1_DQN, [1, state_size*3])
                state_DQN_n = np.reshape(state3_DQN, [1, state_size*2])
                state1_BestCQI = state_full_buffer[:,[2,6,7]]
                state_BestCQI = np.reshape(state1_BestCQI, [1, state_size*3])
            
            action_rand = agent.act(state_rand,'RAND')[0]
            action_BestCQI = agent.act(state_BestCQI,'BestCQI')[0]
            action_DQN, prob_DQN = agent.act(state_DQN_n,'DQN')
            #print('act',action_DQN, 'prob',prob_DQN)
            
            reward_rand, next_state_rand = Envirement.Envirement().env_reward_next_state_sf_rbg(action_rand,np.reshape(state_rand, [state_size,3]),sfi,rbgi)
            reward_BestCQI, next_state_BestCQI = Envirement.Envirement().env_reward_next_state_sf_rbg(action_BestCQI,np.reshape(state_BestCQI,[state_size,3]),sfi,rbgi)
            reward_DQN, next_state_DQN_fb = Envirement.Envirement().env_reward_next_state_sf_rbg(action_DQN,np.reshape(state_DQN_fb, [state_size,3]),sfi,rbgi)
            
            next_state_rand[:,2] = state_full_buffer[:,7+rbgi+1]
            next_state_BestCQI[:,2] = state_full_buffer[:,7+rbgi+1]
            next_state_DQN_fb[:,2] = state_full_buffer[:,7+rbgi+1]
            next_state_DQN_n = next_state_DQN_fb[:,[0,2]]
            #next_state_DQN_n = np.zeros((pa.MAX_SF_QUE,1))
            #next_state_DQN_n[:,0] = (next_state_DQN_fb[:,0]+10*next_state_DQN_fb[:,2])*(next_state_DQN_fb[:,1]/abs(next_state_DQN_fb[:,1]))
            
            next_state_rand = np.reshape(next_state_rand, [1, state_size*3])
            next_state_BestCQI = np.reshape(next_state_BestCQI, [1, state_size*3])
            next_state_DQN_fb = np.reshape(next_state_DQN_fb, [1, state_size*3])
            next_state_DQN_n = np.reshape(next_state_DQN_n, [1, state_size*2])
            
            rew_sf_rand[sfi] = (int(rew_sf_rand[sfi]) + int(reward_rand))
            rew_sf_BestCQI[sfi] = (int(rew_sf_BestCQI[sfi]) + int(reward_BestCQI))
            rew_sf_DQN[sfi] = (int(rew_sf_DQN[sfi]) + int(reward_DQN))
            agent.memorize(state_DQN_n, action_DQN, prob_DQN, reward_DQN)
            
            state_rand = next_state_rand
            state_BestCQI = next_state_BestCQI
            state_DQN_fb = next_state_DQN_fb
            state_DQN_n = next_state_DQN_n
            
        rew_sf_rand[sfi] = rew_sf_rand[sfi]/pa.MAX_REWARD_SF
        rew_sf_BestCQI[sfi] = rew_sf_BestCQI[sfi]/pa.MAX_REWARD_SF
        rew_sf_DQN[sfi] = rew_sf_DQN[sfi]/pa.MAX_REWARD_SF

        if sfi%sfi_train==0:
            agent.train()
        if sfi%sfi_save==0:
            agent.save('PGmodel/pgmodel.h5')
        if (sfi)%sfi_show_reward==0:
            print('SFI: ', sfi, 'rewardRAND: ', rew_sf_rand[sfi])
            print('SFI: ', sfi, 'rewardDQN: ', rew_sf_DQN[sfi])
            print('SFI: ', sfi, 'rewardBestCQI: ', rew_sf_BestCQI[sfi])
            
    return rew_sf_rand, rew_sf_DQN, rew_sf_BestCQI