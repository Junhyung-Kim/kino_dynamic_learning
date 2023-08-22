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

    timestep=23
    f = open("/home/jhk/walkingdata/beforedata/fdyn/lfoot2_final.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/fdyn/rfoot2_final.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')
    f5 = open("/home/jhk/ssd_mount/zmp5.txt", 'r')

    lines = f.readlines()
    lines2 = f2.readlines()
    lines3 = f5.readlines()  
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

    zmp_refx = [[] for i in range(len(lines3))]
    zmp_refy = [[] for i in range(len(lines3))]

    zmp_refx_ = [[] for i in range(N)]
    zmp_refy_ = [[] for i in range(N)]
    
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

    lines3_array = []
    for i in range(0, len(lines3)):
        lines3_array.append(lines3[i].split())

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

    k_ = [
        "timestep=1_finish",
        "timestep=2_finish",
        "timestep=3_finish",
        "timestep=4_finish",
        "timestep=5_finish",
        "timestep=6_finish",
        "timestep=7_finish",
        "timestep=8_finish",
        "timestep=9_finish",
        "timestep=10_finish",
        "timestep=11_finish",
        "timestep=12_finish",
        "timestep=13_finish",
        "timestep=14_finish",
        "timestep=15_finish",
        "timestep=16_finish",
        "timestep=17_finish",
        "timestep=18_finish",
        "timestep=19_finish",
        "timestep=20_finish",
        "timestep=21_finish",
        "timestep=22_finish",
        "timestep=23_finish",
        "timestep=24_finish",
        "timestep=25_finish",
        "timestep=26_finish",
        "timestep=27_finish",
        "timestep=28_finish",
        "timestep=29_finish",
        "timestep=30_finish",
        "timestep=31_finish",
        "timestep=32_finish",
        "timestep=33_finish",
        "timestep=34_finish",
        "timestep=35_finish",
        "timestep=36_finish",
        "timestep=37_finish",
        "timestep=38_finish",
        "timestep=39_finish",
        "timestep=40_finish",
        "timestep=41_finish",
        "timestep=42_finish",
        "timestep=43_finish",
        "timestep=44_finish",
        "timestep=45_finish",
        "timestep=46_finish",
        "timestep=47_finish",
        "timestep=48_finish",
        "timestep=49_finish",
    ]

    j = []
    j1 = []
    
    for kkkk in range(1, len(k_)):
        filename = '/home/jhk/walkingdata/beforedata/fdyn/'
        #filename2 = 'filename5_8_72_0.0007_1.txt'#k_[kkkk] 

        filename3 = filename + k_[kkkk] + '/' + k_[kkkk] 
        with open(filename3, 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        prev = len(database['Right']['x_state'])
        f.close()
        j = 0
        timestep = kkkk + 1
        for i in range(0, N):
            if i == 0:
                array_boundx_[i] = array_boundx[k3*i + timestep]
                array_boundy_[i] = array_boundy[k3*i + timestep]
            else:
                array_boundx_[i] = array_boundx[k3*(i) + timestep]
                array_boundy_[i] = array_boundy[k3*(i) + timestep]
            print(len(database['Right']['x_state']))
        for i in range(len(database['Right']['x_state'])-1,0,-1):
            for k in range(1, 60):
                if k == 1:
                    if(database['Right']['costs'][i] > 0.015):
                        del(database[key]['trajs'][i])
                        del(database[key]['vel_trajs'][i])
                        del(database[key]['x_state'][i])
                        del(database[key]['acc_trajs'][i])
                        del(database[key]['u_trajs'][i])
                        del(database[key]['costs'][i])
                        print("cost")
                        print([i,k])
                        j = j +1
                        break

                if (database['Right']['x_state'][i][k][6] > array_boundy_[k][1]) or (database['Right']['x_state'][i][k][6] < array_boundy_[k][0]):
                    print([i,k])
                    del(database[key]['trajs'][i])
                    del(database[key]['vel_trajs'][i])
                    del(database[key]['x_state'][i])
                    del(database[key]['acc_trajs'][i])
                    del(database[key]['u_trajs'][i])
                    del(database[key]['costs'][i])
                    j = j +1
                    break
           
        print(filename3)
        print(j1)
        with open(filename3, 'wb') as f:
            pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("final")
        print(len(database['Right']['x_state']))
        print("prev")
        print(prev)
        j1.append([prev, len(database['Right']['x_state'])])
        

def talker():
    global LIPM_bool
    LIPM_bool = 1  #LIPM 0, LIPFM 1
    loadmodel()
    
if __name__=='__main__':
    talker()
