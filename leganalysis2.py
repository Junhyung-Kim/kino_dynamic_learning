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
    #_1_386
    #with open('/home/jhk/ssd_mount/beforedata/ssp2/integral/Fdyn_data7_4_FULL.txt', 'rb') as f:
    with open('/home/jhk/ssd_mount/beforedata/ssp1/i=3,4,5/Fdyn_data8.txt', 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()

    for i in range(0, len(database['Right']['trajs'])):
        if len(database['Right']['trajs'][i]) < 80:
            print(i)
    '''
    database1 = dict()
    database1['Right'] = dict()
    
    for key in database1.keys():
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

    for i in range(0, len(database['Right']['trajs'])):
        if i != 169:
            database1['Right']['trajs'].append(database['Right']['trajs'][i])
            database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][i])
            database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][i])
            database1['Right']['x_state'].append(database['Right']['x_state'][i])
            database1['Right']['u_trajs'].append(database['Right']['u_trajs'][i])

    with open('/home/jhk/ssd_mount/beforedata/ssp1/i=9/Fdyn_data8.txt','wb') as f:
        pickle.dump(database1,f)
    f.close()
    '''
    

    print(len(database['Right']['trajs']))
    k= asdfasdfadf
    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()