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
    
    first = ['_72_', '_73_']
    second = ['-0.0033', '-0.0021', '-0.0007', '0.0007', '0.0021', '0.0033' ] 
    third = '_7'
    filename = '/home/jhk/ssd_mount2/beforedata/fdyn_int/timestep=13/integrate/Fdyn_data5_7'
    
    for i in range(0,2):
        for j in range(0,6):
            filename3 = filename + first[i] + second[j] + '.txt' 
            filename4 = filename + third + '.txt'
            if i == 0 and j == 0:
                with open(filename3, 'rb') as f:
                    database = pickle.load(f,  encoding='iso-8859-1')

                with open(filename4,'wb') as f:
                    pickle.dump(database,f)
    
            else:
                with open(filename4, 'rb') as f:
                    database = pickle.load(f,  encoding='iso-8859-1')

                with open(filename3, 'rb') as f:
                    database1 = pickle.load(f,  encoding='iso-8859-1')
    
                print([i, j, first[i], second[j]])
                '''
                print("a")
                print(database['Right']['x_inputs'][0])
                print(database['Right']['x_inputs'][len(database['Right']['trajs'])-1])
                print("b")
                print(database1['Right']['x_inputs'][0])
                print(database1['Right']['x_inputs'][len(database1['Right']['trajs'])-1])
                '''
                for k in range(0, len(database1['Right']['trajs'])):
                    aaa = True
                    if aaa == True:
                        database['Right']['trajs'].append(database1['Right']['trajs'][k])
                        database['Right']['acc_trajs'].append(database1['Right']['acc_trajs'][k])
                        database['Right']['vel_trajs'].append(database1['Right']['vel_trajs'][k])
                        database['Right']['x_state'].append(database1['Right']['x_state'][k])
                        database['Right']['u_trajs'].append(database1['Right']['u_trajs'][k])
                        database['Right']['x_inputs'].append(database1['Right']['x_inputs'][k])

                with open(filename4,'wb') as f:
                    pickle.dump(database,f)
    
    f.close()  
    print("e")
    print(database['Right']['x_inputs'][0])
    print(database['Right']['x_inputs'][len(database['Right']['trajs'])-1])
    a = 0
    
    aa = dafsdfasdfasd
    
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()