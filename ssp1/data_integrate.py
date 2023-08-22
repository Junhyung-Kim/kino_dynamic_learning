#!/usr/bin/env python 
from __future__ import print_function
from tempfile import tempdir
import pinocchio
from pinocchio.utils import npToTuple
from pinocchio.rpy import matrixToRpy, rpyToMatrix
import sys
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import scipy.linalg
from sys import argv
from os.path import dirname, join, abspath
from control.matlab import *
from pinocchio.robot_wrapper import RobotWrapper
from copy import copy

##IK, only lower, data preprocessing
np.set_printoptions(threshold=sys.maxsize)

def loadmodel():
    global database, database1
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

    k_ = ["timestep=17_finish_",
        "timestep=17_finish_0_final",   
"timestep=17_finish_10_final", 
"timestep=17_finish_11_final", 
"timestep=17_finish_12_final", 
"timestep=17_finish_13_final",  
"timestep=17_finish_14_final",  
"timestep=17_finish_15_final",  
"timestep=17_finish_16_final", 
"timestep=17_finish_17_final",
"timestep=17_finish_18_final", 
"timestep=17_finish_19_final", 
"timestep=17_finish_1_final",  
"timestep=17_finish_20_final", 
"timestep=17_finish_21_final", 
"timestep=17_finish_22_final", 
"timestep=17_finish_23_final",  
"timestep=17_finish_24_final",  
"timestep=17_finish_25_final", 
"timestep=17_finish_26_final", 
"timestep=17_finish_27_final", 
"timestep=17_finish_28_final",  
"timestep=17_finish_29_final", 
"timestep=17_finish_2_final", 
"timestep=17_finish_3_final", 
"timestep=17_finish_4_final",  
"timestep=17_finish_5_final", 
"timestep=17_finish_6_final", 
"timestep=17_finish_7_final", 
"timestep=17_finish_8_final", 
"timestep=17_finish_9_final", 

  ]
    #e, 21
    j = []
    filename = '/home/jhk/Downloads/' #walkingdata/beforedata/ssp1/timestep=17_finish_/'
    filename2 = k_[0]

    filename3 = filename + filename2
    with open(filename3, 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(len(k_))
    for kkkk in range(1, len(k_)):
        filename = '/home/jhk/Downloads/'
        filename2 = k_[kkkk]
        filename3 = filename + filename2
        print(filename3)
        
        with open(filename3, 'rb') as f:
            database1 = pickle.load(f,  encoding='iso-8859-1')
        f.close()

        num_de = 0
        database1['Right']['ZMPerr'] = []
        print(len(database['Right']['trajs']))
        print(len(database1['Right']['trajs']))
        
        #if (database['Right']['trajs'][len(database['Right']['trajs'])-1][0][0] == database1['Right']['trajs'][len(database1['Right']['trajs'])-1][0][0]) and (database['Right']['trajs'][len(database['Right']['trajs'])-1][0][1] == database1['Right']['trajs'][len(database1['Right']['trajs'])-1][0][1]) and (database['Right']['vel_trajs'][len(database['Right']['trajs'])-1][0][0] == database1['Right']['vel_trajs'][len(database1['Right']['trajs'])-1][0][0]) and (database['Right']['vel_trajs'][len(database['Right']['trajs'])-1][0][1] == database1['Right']['vel_trajs'][len(database1['Right']['trajs'])-1][0][1]) and (database['Right']['x_state'][len(database['Right']['trajs'])-1][0][2] == database1['Right']['x_state'][len(database1['Right']['trajs'])-1][0][2]) and (database['Right']['x_state'][len(database['Right']['trajs'])-1][0][6] == database1['Right']['x_state'][len(database1['Right']['trajs'])-1][0][6]):
        #    print("ok")
        for kkk in range(0, len(database1['Right']['trajs'])):
            database['Right']['trajs'].append(database1['Right']['trajs'][kkk])
            database['Right']['acc_trajs'].append(database1['Right']['acc_trajs'][kkk])
            database['Right']['vel_trajs'].append(database1['Right']['vel_trajs'][kkk])
            database['Right']['x_state'].append(database1['Right']['x_state'][kkk])
            database['Right']['u_trajs'].append(database1['Right']['u_trajs'][kkk])
            database['Right']['x_inputs'].append(database1['Right']['x_inputs'][kkk])
            database['Right']['costs'].append(database1['Right']['costs'][kkk])

        #else:
        #    j.append(kkkk)
    
    filename = '/home/jhk/Downloads/'
    filename2 = "timestep=17_finish_"
    filename3 = filename + filename2
    with open(filename3, 'wb') as f:
        pickle.dump(database, f)
    f.close()
    print(len(database['Right']['costs']))

def talker():
    global LIPM_bool
    LIPM_bool = 1  #LIPM 0, LIPFM 1
    loadmodel()
    
if __name__=='__main__':
    talker()
