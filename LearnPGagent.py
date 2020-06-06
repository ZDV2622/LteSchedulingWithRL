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

        
    # takes requests with buffer in sf_id
    # [0] request index in sfi
    # [1] UE index
    # [2] QCI
    # [3] Buffer init for UE QCI
    # [4] SFI
    # [5] -1 - reserved value - save last SFI 
    # [6] delay target
    # [7] buffer current
    # [8-33] CQI in RBG

def LearnProcessPgagent(SF,sfi_train,sfi_save,sfi_show_reward):
    
    pa = Parameters.Parameters()
    state_size = pa.state_size
    action_size = pa.action_size
    agent = PGAgent.PGAgent()

    rew_sf_rand = np.zeros(SF)
    rew_sf_DQN = np.zeros(SF)
    rew_sf_BestCQI = np.zeros(SF)
    rew_sf_DQN_buff = np.zeros([SF,pa.QCI_N_TOTAL+1])
    

    for sfi in range(0,SF,1):
        qci_thr_dqn = np.zeros(pa.QCI_N_TOTAL+1)
        if sfi%1==0:
            # generate requests in sfi
            state_full_buffer = []
            state_full_buffer = InputTrafficGeneration.TrafficGeneration().full_que_sf_with_buffer(sfi)
            #state_full_buffer_normed = state_full_buffer / state_full_buffer.max(axis=0)
            
            state_rand_orig = state_full_buffer[:,[2,6,7,8]]  
            state_DQN_orig = state_full_buffer[:,[2,6,7,8]]
            # test
            state_DQN_n = state_DQN_orig / state_DQN_orig.max(axis=0)  
            state_DQN_act_n = state_DQN_n[:,[0,1,3]]
            state_DQN_act_n_3d = np.reshape(state_DQN_act_n, [1,pa.state_size,(pa.REQ_PARAM_N-1)])
            state_DQN_flat_n = np.reshape(state_DQN_n, [1, state_size*(pa.REQ_PARAM_N)])
            state_DQN_flat_act_n = np.reshape(state_DQN_act_n, [1, state_size*(pa.REQ_PARAM_N-1)])
            state_BestCQI_orig = state_full_buffer[:,[2,6,7,8]]
        #for ue in range(0,pa.MAX_UE_SF+1,1):
            #CQI = InputTrafficGeneration.TrafficGeneration().ue_traff_gen(ue)[0]
            #for req in range(0,pa.MAX_SF_QUE,1):
                #if state_full_buffer[req][1] == ue:
                    #state_full_buffer[req][pa.REQ_PARAM_N_FULL:pa.REQ_PARAM_N_FULL+pa.N_RBG] = CQI        
        rbgi = 0
        for rbgi in range(pa.N_RBG-1):
                
            action_rand = agent.act(state_rand_orig,'RAND',sfi)[0]
            action_BestCQI = agent.act(state_BestCQI_orig,'BestCQI',sfi)[0]
            action_DQN, prob_DQN = agent.act(state_DQN_act_n_3d,'DQN',sfi)

            reward_rand, next_state_rand_orig = Envirement.Envirement().env_reward_next_state_sf_rbg(action_rand,state_rand_orig,state_rand_orig,sfi,rbgi)
            reward_BestCQI, next_state_BestCQI_orig = Envirement.Envirement().env_reward_next_state_sf_rbg(action_BestCQI,state_BestCQI_orig,state_BestCQI_orig,sfi,rbgi)            
            reward_DQN, next_state_DQN_orig = Envirement.Envirement().env_reward_next_state_sf_rbg(action_DQN,state_DQN_orig,state_DQN_orig,sfi,rbgi)
            
            agent.memorize(state_DQN_act_n_3d, action_DQN, prob_DQN, reward_DQN)

            next_state_rand_orig[:,3] = state_full_buffer[:,pa.REQ_PARAM_N_FULL+rbgi+1]            
            next_state_BestCQI_orig[:,3] = state_full_buffer[:,pa.REQ_PARAM_N_FULL+rbgi+1]
            next_state_DQN_orig[:,3] = state_full_buffer[:,pa.REQ_PARAM_N_FULL+rbgi+1]
            #print('ns',next_state_DQN_orig[action_DQN,[0,1,2,3]])
            for ii in range(0,state_size,1):
                if next_state_rand_orig[ii,2] <= 0:
                    next_state_rand_orig[ii,0] = 0
                    next_state_rand_orig[ii,1] = 0
                    next_state_rand_orig[ii,3] = 0
                    next_state_rand_orig[ii,2] = 0
                if next_state_DQN_orig[ii,2] <= 0:
                    next_state_DQN_orig[ii,0] = 0
                    next_state_DQN_orig[ii,1] = 0
                    next_state_DQN_orig[ii,3] = 0
                    next_state_DQN_orig[ii,2] = 0
                if next_state_BestCQI_orig[ii,2] <= 0:
                    next_state_BestCQI_orig[ii,0] = 0
                    next_state_BestCQI_orig[ii,1] = 0
                    next_state_BestCQI_orig[ii,3] = 0
                    next_state_BestCQI_orig[ii,2] = 0
            
            rew_sf_rand[sfi] = (int(rew_sf_rand[sfi]) + int(reward_rand))
            rew_sf_BestCQI[sfi] = (int(rew_sf_BestCQI[sfi]) + int(reward_BestCQI))
            rew_sf_DQN[sfi] = (int(rew_sf_DQN[sfi]) + int(reward_DQN))
            qci_thr_dqn[int(state_DQN_orig[action_DQN,0])] = qci_thr_dqn[int(state_DQN_orig[action_DQN,0])] + 1
            rew_sf_DQN_buff[sfi,int(state_DQN_orig[action_DQN,0])] = rew_sf_DQN_buff[sfi,int(state_DQN_orig[action_DQN,0])] + 1

            
            state_rand_orig = next_state_rand_orig
            state_DQN_orig = next_state_DQN_orig
            state_DQN_n =  state_DQN_orig / state_DQN_orig.max(axis=0)
            state_DQN_act_n = state_DQN_n[:,[0,1,3]]
            state_DQN_act_n_3d = np.reshape(state_DQN_act_n, [1,(pa.REQ_PARAM_N-1),state_size])
            state_DQN_flat_n = np.reshape(state_DQN_n, [1, state_size*(pa.REQ_PARAM_N)])
            state_DQN_flat_act_n = np.reshape(state_DQN_act_n, [1, state_size*(pa.REQ_PARAM_N-1)])
            state_BestCQI_orig = next_state_BestCQI_orig
                        
        rew_sf_rand[sfi] = rew_sf_rand[sfi]#/pa.MAX_REWARD_SF
        rew_sf_BestCQI[sfi] = rew_sf_BestCQI[sfi]#/pa.MAX_REWARD_SF
        rew_sf_DQN[sfi] = rew_sf_DQN[sfi]#/pa.MAX_REWARD_SF

        if sfi%sfi_train==0 and sfi>1:
            agent.train()
        if sfi%sfi_save==0:
            agent.save('PGmodel/pgmodel.h5')
        if (sfi)%sfi_show_reward==0:
            print('SFI: ', sfi, 'rewardRAND: ', rew_sf_rand[sfi])
            print('SFI: ', sfi, 'rewardDQN: ', rew_sf_DQN[sfi])
            print('SFI: ', sfi, 'rewardBestCQI: ', rew_sf_BestCQI[sfi])
            print('QCI: ', sfi, 'thr: ', qci_thr_dqn )
            
    return rew_sf_rand, rew_sf_DQN, rew_sf_BestCQI,rew_sf_DQN_buff