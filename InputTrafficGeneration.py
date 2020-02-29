import Parameters
import numpy as np


class TrafficGeneration:
    
    # Class lets to generate traffic in subframe
    # it generates some number of UE, CQI - for each RBG, QCI and its buffer size
    # this model is very simple
    def __init__(self):
        # use parameters from Parameters file
        self.pa = Parameters.Parameters()
        
    # generate traffic for ue
    def ue_traff_gen(self,ue_id):
        pa = self.pa
        CQI_min = 2
        CQI_max = 6
        if pa.CQI_UE[ue_id] != 0:
            CQI_min = pa.CQI_UE[ue_id]
            CQI_max = pa.CQI_UE[ue_id]
        UE_CQI = [int(np.random.randint(CQI_min, CQI_max+1)/2)*2 for i in range(0,int(pa.N_RB/pa.RBG_SIZE),1)]
        UE_QCI = [np.random.randint(pa.UE_FIXED_QCI, 2) for i in range(0,pa.QCI_N,1)]
        UE_BUFF = [np.random.randint(pa.UE_MIN_BUFF, pa.UE_MAX_BUFF) for i in range(0,pa.QCI_N,1)]
        return UE_CQI,UE_QCI,UE_BUFF
        
        
    
    #generete traffic for all ue in sf_id
    def all_traff_gen_sf(self,sf_id):
        pa = self.pa
        # generates UE in SF
        UE_SF_ID = [np.random.randint(pa.UE_SF_FIXED , 2) for i in range(0,pa.MAX_UE_SF,1)]
        # number of UE, generated in SF
        UE_SF_N = np.array(UE_SF_ID).sum()
        ue_ind = UE_SF_ID
        
        # generate users requests and parameters for sf_id
        CQI = np.zeros((pa.MAX_UE_SF, int(pa.N_RB/pa.RBG_SIZE)))
        QCI = np.zeros((pa.MAX_UE_SF, pa.QCI_N))
        BUFF = np.zeros((pa.MAX_UE_SF, pa.QCI_N))
        for i in range(0,pa.MAX_UE_SF,1):
            if UE_SF_ID[i] == 1:
                UE_TRAF = self.ue_traff_gen(i)
                CQI[i][:] = UE_TRAF[0]
                QCI[i][:] = UE_TRAF[1]
                BUFF[i][:] = UE_TRAF[2]*QCI[i][:]
        SF_CQI = CQI
        QCI_BUFF = BUFF
        SF_ALL = np.concatenate((CQI, BUFF),axis=1)
        req = []
        r = 0
        for i in range(0,UE_SF_N,1):
            if UE_SF_ID[i] == 1:
                for j in range(0,pa.QCI_N,1):
                    if QCI_BUFF[i][j] != 0:
                        req.append([i,j,QCI_BUFF[i][j],sf_id,-1,QCI_BUFF[i][j]])
                        for k in range(0,int(pa.N_RB/pa.RBG_SIZE),):
                            req[r].append(int(SF_CQI[i][k]))
                        r = r+1
        return req
        
        
    # takes requests with buffer in sf_id
    def full_que_sf_with_buffer(self,sf_id):
        a = self.all_traff_gen_sf(sf_id) # new requests in SF_i
        pa = self.pa
        que_sf_with_buff = np.zeros((pa.state_size, 7+int(pa.N_RB/pa.RBG_SIZE)))-1000
        # create queue with active buffer
        i = 0
        for i in range(0,pa.MAX_SF_QUE,1):
            if len(a)>i:
                if a[i][5]>0:
                    que_sf_with_buff[i,:] = [i]+a[i][:]
        return que_sf_with_buff
        
        
        
        
        
