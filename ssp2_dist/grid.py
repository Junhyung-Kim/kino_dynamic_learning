import pickle
import numpy as np
import time
from copy import copy
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from IPython.display import clear_output 
import logging
import os
import torch
import pinocchio
import crocoddyl

import sys
import numpy.matlib
np.set_printoptions(threshold=sys.maxsize)
global client
global learn_type
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torch.multiprocessing as multiprocessing
import ctypes
import pytorch_model_summary
import math


global q_traj, v_traj, acc_traj, x_traj, u_traj,q_traj_ae, v_traj_ae, acc_traj_ae, x_traj_ae, u_traj_ae
q_traj = multiprocessing.Array(ctypes.c_float, range(60*21))
v_traj = multiprocessing.Array(ctypes.c_float, range(60*20))
acc_traj= multiprocessing.Array(ctypes.c_float, range(60*20))
x_traj= multiprocessing.Array(ctypes.c_float, range(60*8))
u_traj = multiprocessing.Array(ctypes.c_float, range(60*4))


def define_RBF(dof=39, nbStates=60, offset=200, width=60, T=4000, coeff = 250):
    tList = np.arange(T)

    Mu = np.linspace(tList[0]-offset, tList[-1]+offset, nbStates)
    Sigma  = np.reshape(np.matlib.repmat(width, 1, nbStates),[1, 1, nbStates])
    Sigma.shape
    Phi = np.zeros((T, nbStates))
    for i in range(nbStates):
        Phi[:,i] = coeff*scipy.stats.norm(Mu[i], Sigma[0,0,i]).pdf(tList)
    return Phi

def apply_RBF(trajs, Phi, rcond=0.0001):
    w_trajs = []
    w,_,_,_ = np.linalg.lstsq(Phi, trajs, rcond=0.0001)
    w_trajs.append(w.flatten())
    return np.array(w_trajs)
    

if __name__=='__main__':
    global test_loader, test_vel_loader, test_acc_loader, test_u_loader, test_x_loader, x_inputs_test, pca, trajs, u_trajs, vel_trajs, x_trajs, acc_trajs, Phi
    global y_test_ae, y_vel_test_ae, y_acc_test_ae, y_x_test_ae, y_u_test_ae

    learn_type = 0
    array_ = [
"timestep=25_finish"
]
    
    for kk in range(0, len(array_),1):
        database = dict()
        database['left'] = dict()
        database['Right'] = dict()

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
        
        filename = '/home/jhk/walkingdata/beforedata/ssp1/'
        filename2 = filename + array_[kk] + '/' + array_[kk]
        with open(filename2, 'rb') as f:
            database = pickle.load(f,  encoding='iso-8859-1')
        f.close()

        init_trajs = dict()
        trajs = dict()
        x_inputs_init = dict()
        vel_trajs = dict()
        x_inputs = dict()
        acc_trajs = dict()
        foot_poses = dict()
        u_trajs = dict()
        x_trajs = dict()

        new_trajs = dict()
        new_vel_trajs = dict()
        new_u_trajs = dict()
        
        w_trajs = dict()
        w_vel_trajs = dict()
        w_x_trajs = dict()
        w_acc_trajs = dict()
        w_u_trajs = dict()

        w_trajs_pca = dict()
        pca = dict()

        w_x_trajs_pca = dict()
        pca_x = dict()

        w_vel_trajs_pca = dict()
        pca_vel = dict()

        w_acc_trajs_pca = dict()
        pca_acc = dict()

        w_u_trajs_pca = dict()
        pca_u = dict()
        
        #define dataset
        keys = ['Right']

        for key in keys:
            x_inputs[key] = []
            x_inputs[key] = np.array(database[key]['x_inputs'])
            trajs[key] = np.array(database[key]['trajs'])
            vel_trajs[key] = np.array(database[key]['vel_trajs'])
            x_trajs[key] = np.array(database[key]['x_state'])
        
        for key in keys:
            d = np.array([])
            for i in range(len(database[key]['trajs'])):
                c = np.append(np.append(trajs[key][i][0], vel_trajs[key][i][0], axis=0), np.array([x_trajs[key][i][0][2], x_trajs[key][i][0][6]]), axis=0)
                d = np.append(d, np.array([c]))
            d = d.reshape(len(database[key]['trajs']), 43)
            x_inputs[key] = d
        

        JJ=[]
        co = 0
        while len(JJ) < 20:
            co = co + 1
            ran = np.random.randint(len(database[key]['trajs']))
            if database['Right']['costs'][ran] < 0.015:
                JJ.append(ran)    
                print(JJ)
        
        print("start grid search")
        timestep = 60
        best = []
        best_cost = math.inf

        cost_temp=math.inf
        k2 = []
        for i in range(1, 20):
            for j in range(1,20):
                for k in range(1,20):
                    for f in range(50, 56):
                        rbf_num = f
                            
                        Phi = define_RBF(dof=43, nbStates = f, offset = i, width = j, T = timestep, coeff=k)
                        k3 = 0
                        cost = 0
                        traj_cost = 0
                        vel_cost = 0
                        x_cost = 0
                        for jj in JJ:
                            aa_ = apply_RBF(trajs[key][jj], Phi).reshape(rbf_num,-1)
                            bb_ = apply_RBF(vel_trajs[key][jj], Phi).reshape(rbf_num,-1)
                            ee_ = apply_RBF(x_trajs[key][jj], Phi).reshape(rbf_num,-1)

                            

                            for num in range(0, 60):
                                if num  == 0:
                                    traj_cost = np.linalg.norm(trajs[key][jj][num] - np.dot(Phi[num], aa_)) *np.linalg.norm(trajs[key][jj][num] - np.dot(Phi[num], aa_)) 
                                    vel_cost = np.linalg.norm(vel_trajs[key][jj][num] - np.dot(Phi[num], bb_)) * np.linalg.norm(vel_trajs[key][jj][num] - np.dot(Phi[num], bb_))
                                    x_cost =  np.linalg.norm(x_trajs[key][jj][num] - np.dot(Phi[num], ee_)) *np.linalg.norm(x_trajs[key][jj][num] - np.dot(Phi[num], ee_)) 
                                else:
                                    traj_cost = traj_cost + np.linalg.norm(trajs[key][jj][num] - np.dot(Phi[num], aa_)) *np.linalg.norm(trajs[key][jj][num] - np.dot(Phi[num], aa_)) 
                                    vel_cost = vel_cost + np.linalg.norm(vel_trajs[key][jj][num] - np.dot(Phi[num], bb_)) * np.linalg.norm(vel_trajs[key][jj][num] - np.dot(Phi[num], bb_))
                                    x_cost = x_cost + np.linalg.norm(x_trajs[key][jj][num] - np.dot(Phi[num], ee_)) *np.linalg.norm(x_trajs[key][jj][num] - np.dot(Phi[num], ee_)) 
                            if k3 == 0:
                                cost = traj_cost + vel_cost + x_cost
                            else:
                                cost = cost + traj_cost + vel_cost + x_cost
                            k3 = k3 + 1
                            if k3 == 20:
                                if(cost < cost_temp):
                                    k2 = [i, j, k, f, cost/14.0, traj_cost/14.0, vel_cost/14.0, x_cost/14.0]
                                    cost_temp = cost
                print(k2)
                print(array_[kk])
            print("jj= " + str(jj))
            print("offset= " + str(k2[0]) + " width= " + str(k2[1]) + " coeff= " + str(k2[2]) + " nbstates= " + str(k2[3]))
            
            if(cost_temp < best_cost):
                best.append(k2)
        best = np.array(best)
        print(best)
        filename = '/home/jhk/walkingdata/beforedata/ssp1/'
        filename2 = filename + array_[kk] + '/' + 'output11.txt'
        np.savetxt(filename2, best)
        '''
        with open("/home/jhk/walkingdata/beforedata/ssp1/timestep=12_finish/output11.txt", "w") as txt_file:
            for line in best:
                txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line

        '''