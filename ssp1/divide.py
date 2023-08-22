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

    

    k_ = ["timestep=25_finish"]
    #e, 21
    j = []
    filename = '/home/jhk/walkingdata/beforedata/ssp1/'
    filename2 = k_[0]

    filename3 = filename + filename2 +'_re' #
    with open(filename3, 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()


    for kkkk in range(0, 30):
        num_de = 0
        #if (database['Right']['trajs'][len(database['Right']['trajs'])-1][0][0] == database1['Right']['trajs'][len(database1['Right']['trajs'])-1][0][0]) and (database['Right']['trajs'][len(database['Right']['trajs'])-1][0][1] == database1['Right']['trajs'][len(database1['Right']['trajs'])-1][0][1]) and (database['Right']['vel_trajs'][len(database['Right']['trajs'])-1][0][0] == database1['Right']['vel_trajs'][len(database1['Right']['trajs'])-1][0][0]) and (database['Right']['vel_trajs'][len(database['Right']['trajs'])-1][0][1] == database1['Right']['vel_trajs'][len(database1['Right']['trajs'])-1][0][1]) and (database['Right']['x_state'][len(database['Right']['trajs'])-1][0][2] == database1['Right']['x_state'][len(database1['Right']['trajs'])-1][0][2]) and (database['Right']['x_state'][len(database['Right']['trajs'])-1][0][6] == database1['Right']['x_state'][len(database1['Right']['trajs'])-1][0][6]):
        #    print("ok")
        g = len(database['Right']['trajs'])

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

        if kkkk < 30:
            for kkk in range(int(g * kkkk/30),int( g * (kkkk+1)/30)):
                database1['Right']['trajs'].append(database['Right']['trajs'][kkk])
                database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][kkk])
                database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][kkk])
                database1['Right']['x_state'].append(database['Right']['x_state'][kkk])
                database1['Right']['u_trajs'].append(database['Right']['u_trajs'][kkk])
                database1['Right']['x_inputs'].append(database['Right']['x_inputs'][kkk])
                database1['Right']['costs'].append(database['Right']['costs'][kkk])
        else:
            for kkk in range(int(g * kkkk/30), len(database['Right']['trajs'])):
                database1['Right']['trajs'].append(database['Right']['trajs'][kkk])
                database1['Right']['acc_trajs'].append(database['Right']['acc_trajs'][kkk])
                database1['Right']['vel_trajs'].append(database['Right']['vel_trajs'][kkk])
                database1['Right']['x_state'].append(database['Right']['x_state'][kkk])
                database1['Right']['u_trajs'].append(database['Right']['u_trajs'][kkk])
                database1['Right']['x_inputs'].append(database['Right']['x_inputs'][kkk])
                database1['Right']['costs'].append(database['Right']['costs'][kkk])

        filename = '/home/jhk/walkingdata/beforedata/ssp1/'
        filename3 = filename + k_[0] + str(kkkk) 
        with open(filename3, 'wb') as f:
            pickle.dump(database1, f)
        f.close()


        print(len(database['Right']['trajs']))
        print(len(database1['Right']['trajs']))

        #else:
        #    j.append(kkkk)
    
    
    print(len(database['Right']['costs']))

def talker():
    global LIPM_bool
    LIPM_bool = 1  #LIPM 0, LIPFM 1
    loadmodel()
    
if __name__=='__main__':
    talker()
