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
    for time_step in range(4, 5):
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
        
        file_name ='/home/jhk/ssd_mount/beforedata/integrate_fdyn/timestep='
        file_name2 = '/filename3_4_1'
        file_name3 = '.pkl'
        file_name4 = file_name + str(time_step) + file_name2 + file_name3#+ str(time_step) + file_name3
        with open(file_name4, 'rb') as f:
        #with open('/home/jhk/ssd_mount/beforedata/integrate_fdyn/Fdyn_data7.txt', 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        f.close()
        print(len(database['Right']['trajs']))

        #print(database['Right']['trajs'][1][0])

        #print(database['Right']['x_state'][1][0])
        #print(database['Right']['trajs'][len(database['Right']['trajs'])-1][4])
        #print(database['Right']['x_state'][len(database['Right']['trajs'])-1][4])
        print(database['Right']['trajs'][len(database['Right']['trajs'])-1][0])
        print(database['Right']['x_state'][len(database['Right']['trajs'])-1][0])
        
        #print(database['Right']['trajs'][6774][4])
        #print(database['Right']['x_state'][6774][4])
    
        sdfasdf = qqqqss
        
        
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

        with open('/home/jhk/ssd_mount/beforedata/integrate_fdyn/Fdyn_data7.txt', 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        f.close()
        print(len(database['Right']['trajs']))
        print(database['Right']['trajs'][0][0])
        #
        '''
        file_name ='/home/jhk/ssd_mount/beforedata/integrate_fdyn/timestep='
        file_name2 = '/filename3'
        file_name3 = '.pkl'
        file_name4 = file_name + str(time_step) + file_name2 + file_name3
        with open(file_name4, 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        f.close()
        print(len(database))
        print("Sss")
        '''
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
        file_name ='/home/jhk/ssd_mount/beforedata/integrate_fdyn/timestep='
        file_name2 = '/filename3_'
        file_name3 = '_-6775_1.pkl' #'_-6775.pkl'
        file_name4 = file_name + str(time_step) + file_name2 + str(time_step) + file_name3
        with open(file_name4, 'rb') as f:
            database1 = pickle.load(f,  encoding='iso-8859-1')
        f.close()
        k = len(database['Right']['trajs'])
        k1 = len(database1['Right']['trajs'])

        for i in range(0, k):
            database1['Right']['trajs'].append(database['Right']['trajs'][i])
            database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][i])
            database1['Right']['x_inputs'].append(database['Right']['x_inputs'][i])
            database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][i])
            database1['Right']['x_state'].append(database['Right']['x_state'][i])
            database1['Right']['u_trajs'].append(database['Right']['u_trajs'][i])
            database1['Right']['data_phases_set'].append(database['Right']['data_phases_set'][i])
            database1['Right']['costs'].append(database['Right']['costs'][i])
            database1['Right']['iters'].append(database['Right']['iters'][i])
        file_name ='/home/jhk/ssd_mount/beforedata/integrate_fdyn/timestep='
        file_name2 = '/filename3_2.pkl'
        file_name4 = file_name + str(time_step) + file_name2

        with open(file_name4,'wb') as f:
            pickle.dump(database1,f)
        f.close()
    
    print(len(database['Right']['trajs']))
    print(len(database1['Right']['trajs']))
    aa = True
    c = len(database['Right']['trajs'])
    d = len(database1['Right']['trajs'])
    '''
    for i in range(0, len(database['Right']['trajs'])):
        if(database['Right']['costs'][i] > 0.002):
            print(i)
            print(database['Right']['costs'][i])
    '''
    
    for i in range(0, c):
        aa = True
        for j in range(0, d):
            if(database['Right']['trajs'][i][0][0]) == (database1['Right']['trajs'][j][0][0]) and (database['Right']['trajs'][i][0][1]) == (database1['Right']['trajs'][j][0][1]) and (database['Right']['x_state'][i][0][2]) == (database1['Right']['x_state'][j][0][2]) and (database['Right']['trajs'][i][0][6]) == (database1['Right']['trajs'][j][0][6]):
                print(i)
                print(j)
                aa = False
        
        if aa == True:
            #database1['Right']['foot_poses'].append(database['Right']['foot_poses'][i])
            database1['Right']['trajs'].append(database['Right']['trajs'][i])
            database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][i])
            #database1['Right']['x_inputs'].append(database['Right']['x_inputs'][i])
            database1['Right']['x_state'].append(database['Right']['x_state'][i])
            database1['Right']['u_trajs'].append(database['Right']['u_trajs'][i])
            #database1['Right']['costs'].append(database['Right']['costs'][i])
            #database1['Right']['iters'].append(database['Right']['iters'][i])
            database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][i])
            #database1['Right']['data_phases_set'].append(database['Right']['data_phases_set'][i])

    print(len(database['Right']['trajs']))
    print(len(database1['Right']['trajs']))  
    
    '''
    with open('/home/jhk/ssd_mount/beforedata/integrate_fdyn/Fdyn_data7.txt','wb') as f:
        pickle.dump(database1,f)
    f.close()
    '''
    '''
    for i in range(0, len(database1['Right']['x_state'])):
        for j in range(0, len(database1['Right']['x_state'])):
            if(database1['Right']['trajs'][i][0][0]) == (database1['Right']['trajs'][j][0][0]) and (database1['Right']['trajs'][i][0][1]) == (database1['Right']['trajs'][j][0][1]) and i != j:
                print(i)
                print(j)
                aa = False     
                  
    print(len(database['Right']['trajs']))
    print(len(database1['Right']['trajs']))   
    for i in range(0, len(database1['Right']['x_state'])):
        if(database1['Right']['trajs'][i][0][0]) == 0 and (database1['Right']['trajs'][i][0][1]) == 5e-3:
            for j in range(len(database1['Right']['x_state'][i])):
                print(j)
                print(database1['Right']['x_state'][i][j])
    '''
    '''
    f3 = open("/home/jhk/5_tocabi_py.txt", 'w')
    f4 = open("/home/jhk/6_tocabi_py.txt", 'w')
    for i in range(0, len(database1['Right']['x_state'])):
        if(database1['Right']['trajs'][i][0][0]) == 0 and (database1['Right']['trajs'][i][0][1]) == 5e-3:
            for j in range(len(database1['Right']['trajs'][i])):
                print(j)
                print(database1['Right']['vel_trajs'][i][j])
                for k in range(len(database1['Right']['trajs'][i][j])):
                    f3.write(str(database1['Right']['trajs'][i][j][k]))
                    f3.write(", ")
                f3.write("\n")
            f3.close()
    '''
    
    
    
    
    
    
    '''       
            #if abs(database['Right']['trajs'][i][j][19]) > 0.08 or abs(database['Right']['trajs'][i][j][20]) > 0.07:
            if abs(database['Right']['trajs'][i][j][6]) < 0.9985:
                print(i)
                print(j)
                print(database['Right']['trajs'][i][j][19:])
    '''
    '''
    for i in range(0, len(database['Right']['trajs'])):
        for j in range(0, len(database['Right']['trajs'][i])):
            #if abs(database['Right']['trajs'][i][j][19]) > 0.08 or abs(database['Right']['trajs'][i][j][20]) > 0.07:
            if abs(database['Right']['trajs'][i][j][6]) < 0.9985:
                print(i)
                print(j)
                print(database['Right']['trajs'][i][j][19:])
    '''
    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()