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
    f = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')

    lines = f.readlines()
    lines2 = f2.readlines()
    lines1 = f1.readlines()

    N = 60

    array_boundx = [[] for i in range(int(len(lines2)))]
    array_boundy = [[] for i in range(int(len(lines2)))]

    array_boundx_ = [[] for i in range(N)]
    array_boundy_ = [[] for i in range(N)]

    array_boundRF = [[] for i in range(int(len(lines1)))]
    array_boundLF = [[] for i in range(int(len(lines1)))]

    array_boundRF_ = [[] for i in range(N)]
    array_boundLF_ = [[] for i in range(N)]

    
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    k = 1
    k1 = 1
    k3 = 1
    #PCAlearning()
    
    lines_array = []
    for i in range(0, len(lines)):
        lines_array.append(lines[i].split())

    lines1_array = []
    for i in range(0, len(lines1)):
        lines1_array.append(lines1[i].split())

    lines2_array = []
    for i in range(0, len(lines2)):
        lines2_array.append(lines2[i].split())

    for i in range(0, len(lines2_array)):
        for j in range(0, len(lines2_array[i])):
            if j == 0:
                array_boundx[i].append(float(lines2_array[i][j]))
            if j == 1:
                array_boundx[i].append(float(lines2_array[i][j]))
            if j == 2:
                array_boundy[i].append(float(lines2_array[i][j]))
            if j == 3:
                array_boundy[i].append(float(lines2_array[i][j]))

    print(array_boundy_)

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

    k_ = ["timestep=48_finish_" ,    ]

    j = []
    j1 = []
    
    for kkkk in range(0, len(k_), 1):
        filename = '/home/jhk/Downloads/'
        filename3 = filename  +'/' + k_[kkkk]

        if kkkk == 0:
            timestep=48
        '''
        elif kkkk == 1:
            timestep=483
        elif kkkk == 2:
            timestep=483
        elif kkkk == 3: 
            timestep=483
        elif kkkk == 4:
            timestep=48
        elif kkkk == 5:
            timestep=48
        elif kkkk == 6:
            timestep=48
        else:
            timestep = kkkk + 16
        '''
        print(filename3)
        print(timestep)
        #a = asdfasdfasd
        print(kkkk)
        with open(filename3, 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        prev = len(database['Right']['x_state'])
        f.close()
        j = 0

        
        for i in range(0, N):
            if i == 0:
                array_boundx_[i] = array_boundx[k3*i + timestep]
                array_boundy_[i] = array_boundy[k3*i + timestep]
            else:
                array_boundx_[i] = array_boundx[k3*(i) + timestep]
                array_boundy_[i] = array_boundy[k3*(i) + timestep]

        print(len(database['Right']['x_state']))
        for i in range(len(database['Right']['x_state'])-1,-1,-1):
            for k in range(1, N):
                
                if k == 1:
                    if(database['Right']['costs'][i] > 0.045):
                        print(database['Right']['costs'][i])
                        del(database[key]['trajs'][i])
                        del(database[key]['vel_trajs'][i])
                        del(database[key]['x_state'][i])
                        del(database[key]['acc_trajs'][i])
                        del(database[key]['u_trajs'][i])
                        del(database[key]['costs'][i])
                        del(database[key]['x_inputs'][i])
                        print("cost")
                        print([i,k])
                        j = j +1
                        break
                
                    
                
                if (database['Right']['x_state'][i][k][6] > array_boundy_[k][1]) or (database['Right']['x_state'][i][k][6] < array_boundy_[k][0]):
                    print("y")
                    print([i,j])
                    del(database[key]['trajs'][i])
                    del(database[key]['vel_trajs'][i])
                    del(database[key]['x_state'][i])
                    del(database[key]['acc_trajs'][i])
                    del(database[key]['u_trajs'][i])
                    del(database[key]['costs'][i])
                    del(database[key]['x_inputs'][i])
                    
                    j = j +1                    
                    break

                if (database['Right']['x_state'][i][k][2] > array_boundx_[k][1]) or (database['Right']['x_state'][i][k][2] < array_boundx_[k][0]):
                    print("X")
                    del(database[key]['trajs'][i])
                    del(database[key]['vel_trajs'][i])
                    del(database[key]['x_state'][i])
                    del(database[key]['acc_trajs'][i])
                    del(database[key]['u_trajs'][i])
                    del(database[key]['costs'][i])
                    del(database[key]['x_inputs'][i])
                    j = j +1
                    break
                    
                 
                
                
                    
              
        #print(database[key]['costs'][len(database['Right']['x_state'])-3])             
        #a = adsffsafd 
        
        #print(filename3)
        #print(j1)
        filename3 = filename3# + '_'
        print("total")
        print(len(database['Right']['x_state']))
        print(j)
        print("prev")
        print(prev)
        #with open(filename3, 'wb') as f:
        #    pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("final")

        print(len(database['Right']['x_state']))
        j1.append([prev, len(database['Right']['x_state'])])
        

def talker():
    global LIPM_bool
    LIPM_bool = 1  #LIPM 0, LIPFM 1
    loadmodel()
    
if __name__=='__main__':
    talker()
