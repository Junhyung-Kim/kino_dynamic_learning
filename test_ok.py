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

    
    #with open('/home/jhk/ssd_mount/beforeprocessing/SSP2/real/i=8/Fdyn_data5.txt', 'rb') as f:
    #with open('/home/jhkf/ssd_mount/beforeprocessing/SSP2/i=8/Fdyn_data5.txt', 'rb') as f:
    
    #with open('/home/jhk/ssd_mount/beforeprocessing/SSP2/none/i=9/Fdyn_data5.txt', 'rb') as f:
    #with open('/home/jhk/ssd_mount/beforeprocessing/SSP2/i=8/Fdyn_data7.txt', 'rb') as f:
    #with open('/home/jhk/ssd_mount/beforeprocessing/SSP2/real/i=5/Fdyn_data7.txt', 'rb') as f:
    #with open('/home/jhk/ssd_mount/beforedata/test.txt', 'rb') as f:
    # 42R -0.0007 timestep 27

    #with open('/home/jhk/ssd_mount2/beforedata/fdyn_int/timestep=41/Fdyn_data5_2_02_0.0007_0.txt', 'rb') as f:
    with open('/home/jhk/walkingdata/beforedata/fdyn/timestep=1/filename5_7_42_0.0021_0.pkl', 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()# 1,4
    print(database['Right']['costs'])
    for i in range(0, len(database['Right']['costs'])):
        if database['Right']['costs'][i] < 0.008:
            print(i)
            print(database['Right']['costs'][i])
    a = adsfasf

    print(len(database['Right']['trajs']))
    print(database['Right']['x_inputs'][0])
    k = asdfadsfsda
    print(database['Right']['x_inputs'][len(database['Right']['trajs'])-1])
    #k = sadfasdfsad
    print("a")
    print(database['Right']['x_state'][0][0])
    print(database['Right']['x_state'][0][59])
    print("B")
    print(database['Right']['x_state'][len(database['Right']['trajs'])-1][0])
    print(database['Right']['x_state'][len(database['Right']['trajs'])-1][59])
    
    F  = True
    a = 0
    #print(database['Right']['x_state'][170][50])
    
    for i in range(0,len(database['Right']['trajs'])):
        if (abs(database['Right']['x_state'][i][1][0] - (database['Right']['x_state'][i][0][0] + database['Right']['x_state'][i][0][1]*0.02)) < 0.005 ) and (abs(database['Right']['x_state'][i][1][4] - (database['Right']['x_state'][i][0][4] + database['Right']['x_state'][i][0][5]*0.02)) < 0.005 ):
            print(i)
            '''
            for j in range(0, len(database['Right']['trajs'][i])):
                print(j)
                print(database['Right']['x_state'][i][j])
                #print(database['Right']['x_state'][i][1]) 
            '''
            a = a + 1
    print(len(database['Right']['x_state']))
    print(a)

    print(database['Right']['x_state'][100][0])
    print(database['Right']['x_state'][101][1]) 
     
    #for i in range(0, 60):#len(database['Right']['trajs'][0])):
    #    print(i)
    #    print(database['Right']['x_state'][0][i])
        #if len(database['Right']['trajs'][i]) < 80:
   #         print(i)
   #         print("ss")
    
    a = asdfsdfsd
    
    #print(database['Right']['trajs'][3000][0])

    '''
    for j in range(0, 10):
        for k in range(0, 10):
            for l in range(0, 10):
                F = True
                for i in range(0,len(database['Right']['trajs'])):
                    if(int(round(database['Right']['trajs'][i][0][1]/0.01+5.5,2)) == j) and (int(round((database['Right']['x_state'][i][0][2] - database['Right']['x_state'][i][0][0])/0.01 + 5.5,2))== k) and (int(round((database['Right']['x_state'][i][0][6] - database['Right']['x_state'][i][0][4])/0.01 + 5.5,2))== l):
                        F = False
                        break  
                if F == True:
                    print("ss")
                    print([j, k, l])
                    a = a + 1
    print("a")
    print(a)
    ssss= sadfasdfasdfs
    '''
    #for i in range(0,len(database['Right']['trajs'])):
    #    print(round((database['Right']['x_state'][i][0][6] - database['Right']['x_state'][i][0][4])/0.01+5.5,2))
    #print(a)          

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

    with open('/home/jhk/ssd_mount/beforeprocessing/SSP/i=3,4,5/Fdyn_data7.txt', 'rb') as f:
        database1 = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(len(database1['Right']['trajs']))

    print(database['Right']['x_state'][0][50])
    print(database1['Right']['x_state'][0][0])
    print(database1['Right']['acc_trajs'][0][0])
    print(database1['Right']['x_state'][0][1])
    print(database1['Right']['acc_trajs'][0][1])

    aa = 0
    k = len(database['Right']['trajs'])
    k1 = len(database1['Right']['trajs'])
    aaa = True
    for i in range(0, k):
        aaa = True
        for j in range(0,k1):
            if(round(database['Right']['trajs'][i][0][0],4) == round(database1['Right']['trajs'][j][0][0],4)) and (round(database['Right']['trajs'][i][0][1],4)== round(database1['Right']['trajs'][j][0][1],4)) and (round(database['Right']['x_state'][i][0][2],4) == round(database1['Right']['x_state'][j][0][2],4)) and (round(database['Right']['x_state'][i][0][6],4) == round(database1['Right']['x_state'][j][0][6],4)):
                print("i")
                print(i)
                print(j)
                aa = aa + 1
                aaa = False
        if aaa == True:
            database1['Right']['trajs'].append(database['Right']['trajs'][i])
            database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][i])
            database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][i])
            database1['Right']['x_state'].append(database['Right']['x_state'][i])
            database1['Right']['u_trajs'].append(database['Right']['u_trajs'][i])
    
    '''
    print(database1['Right']['trajs'][len(database1['Right']['trajs'])-1][0])
    print(database1['Right']['trajs'][len(database1['Right']['trajs'])-2][0])
    print(database1['Right']['trajs'][len(database1['Right']['trajs'])-3][0])nnjn
    print(database1['Right']['x_state'][len(database1['Right']['trajs'])-1][0])
    print(database1['Right']['x_state'][len(database1['Right']['trajs'])-2][0])
    print(database1['Right']['x_state'][len(database1['Right']['trajs'])-3][0])
    '''
    print(len(database['Right']['trajs']))
    print(len(database1['Right']['trajs']))
      
   # dd = adsfsdfasdfsd
    with open('/home/jhk/ssd_mount/beforeprocessing/SSP/i=3,4,5/Fdyn_data7.txt','wb') as f:
        pickle.dump(database1,f)
    f.close()  
    
    
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
    
    with open('/home/jhk/ssd_mount/beforeprocessing/SSP/i=3,4,5/Fdyn_data7.txt', 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(len(database['Right']['trajs']))
    
    


if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()