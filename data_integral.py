import roslibpy
import pickle
import numpy as np
import time
from copy import copy
import logging
import os
import pinocchio
import crocoddyl

import sys
import numpy.matlib
np.set_printoptions(threshold=sys.maxsize)
global client
global learn_type
from pinocchio.robot_wrapper import RobotWrapper
import ctypes

def talker():
    
    database = dict()
    database['Right'] = dict()
    data_processing = True
    
    for key in database.keys():
        database[key]['foot_poses'] = []
        database[key]['trajs'] = []
        database[key]['acc_trajs'] = []
        database[key]['x_inputs'] = []
        database[key]['vel_trajs'] = [] 
        database[key]['x_state'] = []        
        database[key]['u_trajs'] = []
        database[key]['data_phases_set'] = []
        database[key]['costs'] = [] 
        database[key]['iters'] = []

    database1 = dict()
    database1['Right'] = dict()
    for key in database.keys():
        if(key == 'Right'):
            database1[key]['foot_poses'] = []
            database1[key]['trajs'] = []
            database1[key]['acc_trajs'] = []
            database1[key]['x_inputs'] = []
            database1[key]['vel_trajs'] = [] 
            database1[key]['x_state'] = []        
            database1[key]['u_trajs'] = []
            database1[key]['data_phases_set'] = []
            database1[key]['costs'] = [] 
            database1[key]['iters'] = []

    k_ = ["filename5_9_02_-0.0007_0.txt",
"filename5_9_02_0.0007_0.txt",
"filename5_9_02_-0.0007_1.txt",
"filename5_9_02_0.0007_1.txt",
"filename5_9_02_-0.0021_0.txt",
"filename5_9_02_0.0021_0.txt",
"filename5_9_02_-0.0021_1.txt",
"filename5_9_02_-0.0033_0.txt",
"filename5_9_02_0.0033_0.txt",
"filename5_9_02_-0.0033_1.txt",
"filename5_9_02_0.0033_1.txt",
"filename5_9_02_-0.0033_2.txt",
"filename5_9_03_-0.0007_0.txt",
"filename5_9_03_0.0007_0.txt",
"filename5_9_03_-0.0007_1.txt",
"filename5_9_03_0.0007_1.txt",
"filename5_9_03_-0.0021_0.txt",
"filename5_9_03_0.0021_0.txt",
"filename5_9_03_-0.0021_1.txt",
"filename5_9_03_-0.0021_2.txt",
"filename5_9_03_-0.0021_3.txt",
"filename5_9_03_-0.0021_4.txt",
"filename5_9_03_-0.0021_5.txt",
"filename5_9_03_-0.0033_0.txt",
"filename5_9_03_0.0033_0.txt",
"filename5_9_03_-0.0033_1.txt",
"filename5_9_03_-0.0033_2.txt",
"filename5_9_12_-0.0007_0.txt",
"filename5_9_12_0.0007_0.txt",
"filename5_9_12_-0.0007_1.txt",
"filename5_9_12_0.0007_1.txt",
"filename5_9_12_-0.0007_2.txt",
"filename5_9_12_-0.0007_3.txt",
"filename5_9_12_-0.0007_4.txt",
"filename5_9_12_-0.0007_5.txt",
"filename5_9_12_-0.0007_6.txt",
"filename5_9_12_-0.0021_0.txt",
"filename5_9_12_0.0021_0.txt",
"filename5_9_12_-0.0021_1.txt",
"filename5_9_12_0.0021_1.txt",
"filename5_9_12_-0.0021_2.txt",
"filename5_9_12_-0.0033_0.txt",
"filename5_9_12_0.0033_0.txt",
"filename5_9_12_-0.0033_1.txt",
"filename5_9_12_0.0033_1.txt",
"filename5_9_12_0.0033_2.txt",
"filename5_9_13_-0.0007_0.txt",
"filename5_9_13_0.0007_0.txt",
"filename5_9_13_-0.0021_0.txt",
"filename5_9_13_0.0021_0.txt",
"filename5_9_13_-0.0021_1.txt",
"filename5_9_13_-0.0033_0.txt",
"filename5_9_13_0.0033_0.txt",
"filename5_9_13_0.0033_1.txt",
"filename5_9_22_-0.0007_0.txt",
"filename5_9_22_0.0007_0.txt",
"filename5_9_22_-0.0007_1.txt",
"filename5_9_22_0.0007_1.txt",
"filename5_9_22_-0.0007_2.txt",
"filename5_9_22_0.0007_2.txt",
"filename5_9_22_0.0007_3.txt",
"filename5_9_22_-0.0021_0.txt",
"filename5_9_22_0.0021_0.txt",
"filename5_9_22_-0.0021_1.txt",
"filename5_9_22_-0.0021_2.txt",
"filename5_9_22_-0.0033_0.txt",
"filename5_9_22_0.0033_0.txt",
"filename5_9_22_-0.0033_1.txt",
"filename5_9_22_0.0033_1.txt",
"filename5_9_22_0.0033_2.txt",
"filename5_9_23_-0.0007_0.txt",
"filename5_9_23_0.0007_0.txt",
"filename5_9_23_-0.0007_1.txt",
"filename5_9_23_0.0007_1.txt",
"filename5_9_23_0.0007_2.txt",
"filename5_9_23_0.0007_3.txt",
"filename5_9_23_-0.0021_0.txt",
"filename5_9_23_0.0021_0.txt",
"filename5_9_23_-0.0033_0.txt",
"filename5_9_23_0.0033_0.txt",
"filename5_9_32_-0.0007_0.txt",
"filename5_9_32_0.0007_0.txt",
"filename5_9_32_-0.0007_1.txt",
"filename5_9_32_0.0007_1.txt",
"filename5_9_32_-0.0021_0.txt",
"filename5_9_32_0.0021_0.txt",
"filename5_9_32_-0.0021_1.txt",
"filename5_9_32_-0.0021_2.txt",
"filename5_9_32_-0.0021_3.txt",
"filename5_9_32_-0.0033_0.txt",
"filename5_9_32_-0.0033_1.txt",
"filename5_9_32_0.0033_1.txt",
"filename5_9_32_0.0033_2.txt",
"filename5_9_32_0.0033_3.txt",
"filename5_9_32_0.00f33_0.txt",
"filename5_9_33_-0.0007_0.txt",
"filename5_9_33_0.0007_0.txt",
"filename5_9_33_-0.0007_1.txt",
"filename5_9_33_-0.0021_0.txt",
"filename5_9_33_0.0021_0.txt",
"filename5_9_33_-0.0021_1.txt",
"filename5_9_33_-0.0021_2.txt",
"filename5_9_33_-0.0021_3.txt",
"filename5_9_33_-0.0021_4.txt",
"filename5_9_33_-0.0033_0.txt",
"filename5_9_33_0.0033_0.txt",
"filename5_9_33_-0.0033_1.txt",
"filename5_9_33_-0.0033_2.txt",
"filename5_9_33_-0.0033_3.txt",
"filename5_9_42_-0.0007_0.txt",
"filename5_9_42_0.0007_0.txt",
"filename5_9_42_-0.0021_0.txt",
"filename5_9_42_0.0021_0.txt",
"filename5_9_42_-0.0021_1.txt",
"filename5_9_42_0.0021_1.txt",
"filename5_9_42_0.0021_2.txt",
"filename5_9_42_-0.0033_0.txt",
"filename5_9_42_0.0033_0.txt",
"filename5_9_43_-0.0007_0.txt",
"filename5_9_43_0.0007_0.txt",
"filename5_9_43_-0.0007_1.txt",
"filename5_9_43_-0.0021_0.txt",
"filename5_9_43_0.0021_0.txt",
"filename5_9_43_-0.0033_0.txt",
"filename5_9_43_0.0033_0.txt",
"filename5_9_43_-0.0033_1.txt",
"filename5_9_43_-0.0033_2.txt",
"filename5_9_52_-0.0007_0.txt",
"filename5_9_52_0.0007_0.txt",
"filename5_9_52_-0.0007_1.txt",
"filename5_9_52_-0.0021_0.txt",
"filename5_9_52_0.0021_0.txt",
"filename5_9_52_-0.0033_0.txt",
"filename5_9_52_0.0033_0.txt",
"filename5_9_52_-0.0033_1.txt",
"filename5_9_53_-0.0007_0.txt",
"filename5_9_53_0.0007_0.txt",
"filename5_9_53_-0.0007_1.txt",
"filename5_9_53_-0.0021_0.txt",
"filename5_9_53_0.0021_0.txt",
"filename5_9_53_-0.0021_1.txt",
"filename5_9_53_-0.0021_2.txt",
"filename5_9_53_-0.0033_0.txt",
"filename5_9_53_0.0033_0.txt",
"filename5_9_53_-0.0033_1.txt",
"filename5_9_53_-0.0033_2.txt",
"filename5_9_62_-0.0007_0.txt",
"filename5_9_62_0.0007_0.txt",
"filename5_9_62_-0.0007_1.txt",
"filename5_9_62_0.0007_1.txt",
"filename5_9_62_-0.0007_2.txt",
"filename5_9_62_0.0007_2.txt",
"filename5_9_62_-0.0007_3.txt",
"filename5_9_62_-0.0021_0.txt",
"filename5_9_62_0.0021_0.txt",
"filename5_9_62_-0.0021_1.txt",
"filename5_9_62_0.0021_1.txt",
"filename5_9_62_-0.0033_0.txt",
"filename5_9_62_0.0033_0.txt",
"filename5_9_62_-0.0033_1.txt",
"filename5_9_63_-0.0007_0.txt",
"filename5_9_63_0.0007_0.txt",
"filename5_9_63_-0.0021_0.txt",
"filename5_9_63_0.0021_0.txt",
"filename5_9_63_-0.0033_0.txt",
"filename5_9_63_0.0033_0.txt",
"filename5_9_72_-0.0007_0.txt",
"filename5_9_72_0.0007_0.txt",
"filename5_9_72_-0.0007_1.txt",
"filename5_9_72_0.0007_1.txt",
"filename5_9_72_-0.0007_2.txt",
"filename5_9_72_0.0007_2.txt",
"filename5_9_72_0.0007_3.txt",
"filename5_9_72_0.0007_4.txt",
"filename5_9_72_0.0007_5.txt",
"filename5_9_72_0.0007_6.txt",
"filename5_9_72_-0.0021_0.txt",
"filename5_9_72_0.0021_0.txt",
"filename5_9_72_-0.0021_1.txt",
"filename5_9_72_0.0021_1.txt",
"filename5_9_72_-0.0033_0.txt",
"filename5_9_72_0.0033_0.txt",
"filename5_9_72_-0.0033_1.txt",
"filename5_9_72_0.0033_1.txt",
"filename5_9_72_-0.0033_2.txt",
"filename5_9_72_-0.0033_3.txt",
"filename5_9_72_-0.0033_4.txt",
"filename5_9_72_-0.0033_5.txt",
"filename5_9_72_-0.0033_6.txt",
"filename5_9_73_-0.0007_0.txt",
"filename5_9_73_0.0007_0.txt",
"filename5_9_73_-0.0007_1.txt",
"filename5_9_73_-0.0021_0.txt",
"filename5_9_73_0.0021_0.txt",
"filename5_9_73_-0.0021_1.txt",
"filename5_9_73_-0.0033_0.txt",
"filename5_9_73_0.0033_0.txt",
"filename5_9_73_0.0033_1.txt",]
    
    for kkk in range(0,len(k_)):
        filename = '/home/jhk/Onedrive/fdyn/finish/timestep=1/'
        filename2 = k_[kkk]
        filename3 = filename + filename2
        print(filename3)
        with open(filename3, 'rb') as f:
            database1 = pickle.load(f,  encoding='iso-8859-1')
    
        for i in range(0, len(database1['Right']['trajs'])):
            aaa = True
            if aaa == True:
                database['Right']['trajs'].append(database1['Right']['trajs'][i])
                database['Right']['acc_trajs'].append(database1['Right']['acc_trajs'][i])
                database['Right']['vel_trajs'].append(database1['Right']['vel_trajs'][i])
                database['Right']['x_state'].append(database1['Right']['x_state'][i])
                database['Right']['u_trajs'].append(database1['Right']['u_trajs'][i])
                database['Right']['x_inputs'].append(database1['Right']['x_inputs'][i])
        
    with open('/home/jhk/Onedrive/fdyn/finish/timestep=1/filename_integral','wb') as f:
        pickle.dump(database,f)
    f.close()  
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()