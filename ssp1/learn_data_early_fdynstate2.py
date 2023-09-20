import roslibpy
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
import intel_extension_for_pytorch as ipex
import logging
import os
import torch
import pinocchio
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
import torchvision

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

#manager = multiprocessing.Manager()
#thread_manager = manager.list()
#X_manager = manager.dict()

global q_traj, v_traj, acc_traj, x_traj, u_traj

def InversePCA(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[0] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            q_traj[:] = traj1.flatten()
            t2 = time.time()
            print("thread1")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[0] = 1

def InversePCA1(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[1] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            v_traj[:] = traj1.flatten()
            t2 = time.time()
            print("thread2")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[1] = 1

def InversePCA2(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[2] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            x_traj[:] = traj1.flatten()
            t2 = time.time()
            print("thread3")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[2] = 1

def InversePCA3(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[3] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            acc_traj[:] = traj1.flatten()
            t2 = time.time()
            print("thread4")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[3] = 1

def InversePCA4(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[4] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            u_traj[:] = traj1.flatten()
            t2 = time.time()
            print("thread5")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[4] = 1
   
class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = x.shape[0]
       
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
 
    def __len__(self):
        return self.len

class CNN(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 10, kernel_size=10, stride=1),
            torch.nn.ReLU(),
            )
        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=10, stride = 1),
            torch.nn.Flatten()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(in_features = input_size, out_features= int(input_size*2/3) + output_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features = int(input_size*2/3) + output_size, out_features = output_size)
            )

    def forward(self, x):
        out = self.layer3(x)
        return out

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
    for traj in trajs:
        w,_,_,_ = np.linalg.lstsq(Phi, traj, rcond=0.0001)
        w_trajs.append(w.flatten())
    return np.array(w_trajs)
   
def inverse_transform(w_pca, pca, Phi, rbf_num):
    w = pca.inverse_transform(w_pca)
    w = w.reshape(rbf_num,-1)
    traj = np.dot(Phi,w)
    return traj

def PCAlearning(time_step):
    global xs_pca_test
    global xs_pca
    global us_pca
    
    learn_type = 0
    learn_type1 = 0
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

    naming = [
        #"timestep=0_finish",  
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




        #"timestep=26_finish", 

#"timestep=12_finish",      "timestep=17_finish",      "timestep=21_finish",         "timestep=30_finish",      "timestep=39_finish",
#"timestep=13_finish",       "timestep=22_finish",      "timestep=27_finish",      "timestep=31_finish",      "timestep=40_finish",
#"timestep=15_finish",      "timestep=19_finish",       "timestep=28_finish",           
#"timestep=16_finish",      "timestep=20_finish",      

#"timestep=24_finish",      "timestep=29_finish",      "timestep=38_finish",       
]

    naming1 = [
#"timestep=0_finish_",  
"timestep=1_finish_",  
"timestep=2_finish_", 
"timestep=3_finish_",       
"timestep=4_finish_",    
"timestep=5_finish_",
"timestep=6_finish_",
"timestep=7_finish_",
"timestep=8_finish_",
"timestep=9_finish_",
"timestep=10_finish_",  
"timestep=11_finish_",  
"timestep=12_finish_", 
"timestep=13_finish_",       
"timestep=14_finish_",    
"timestep=15_finish_",
"timestep=16_finish_",
"timestep=17_finish_",
"timestep=18_finish_",
"timestep=19_finish_",
"timestep=20_finish_",  
"timestep=21_finish_",  
"timestep=22_finish_", 
"timestep=23_finish_",       
"timestep=24_finish_",    
"timestep=25_finish_",
"timestep=26_finish_",
"timestep=27_finish_",
"timestep=28_finish_",
"timestep=29_finish_",
"timestep=30_finish_",  
"timestep=31_finish_",  
"timestep=32_finish_", 
"timestep=33_finish_",       
"timestep=34_finish_",    
"timestep=35_finish_",
"timestep=36_finish_",
"timestep=37_finish_",
"timestep=38_finish_",
"timestep=39_finish_",
"timestep=40_finish_",  
"timestep=41_finish_",  
"timestep=42_finish_", 
"timestep=43_finish_",       
"timestep=44_finish_",    
"timestep=45_finish_",
"timestep=46_finish_",
"timestep=47_finish_",
"timestep=48_finish_",

 ]

    param = [
        [2,1,13,55], #1
        [2,1,5,55], #2
        [2,1,47,52], #3
        [1,1,3,54], #4
        [1,1,39,54], #5
        [3,1,37,55], #6
        [1,1,11,55], #7
        [3,1,11,53], #8
        [1,1,23,55], #9
        [1,1,3,55], #10
        [2,1,33,53], #11
        [1,1,11,55], #12
        [2,1,9,51], #13
        [1,1,49,55], #14

        [1,1,31,55], #15
        [1,1,7,55], #16
        [1,1,9,55], #17
        [2,1,31,53], #18
        [2,1,9,52], #19

        [1,1,23,54], #20
        [2,1,49,54], #21
        [2,1,21,55], #22
        [2,1,19,55], #23
        [2,1,39,55], #24
        [1,1,9,55], #25
        [1,1,43,55], #26
        [1,1,45,54], #27
        [1,1,11,54], #28
        [1,1,43,54], #29
        [1,1,41,53], #30
        [1,1,7,55], #31
        [1,1,7,55], #32
      [1,1,17,55], #33
      [1,1,17,55], #34
      [1,1,17,55], #35
      [1,1,49,55], #36
      [2,1,11,55], #37
      [2,1,13,55], #38
      [2,1,3,54], #39
      [2,1,3,54], #40
       [1,1,23,54], #41
       [1,1,49,55], #42
          [1,1,17,55], #43
           [1,1,41,55], #44
         [1,1,37,55], #45
             [1,1,31,55], #46
              [2,1,7,54], #47
               [1,1,35,54], #48
    ] 

    param=[
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
        [2,1,19,55],
    ]

    print(len(naming))
    print(len(param))
    #a = adsfsdfs
    
    file_name = "/home/jhk/walkingdata/beforedata/fdyn/"
    file_name2 = "/timestep="
    file_name3 = file_name +naming[time_step]+'/'+naming1[time_step]#+'_re'
    print(file_name3)
    

    with open(file_name3, 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(len(database['Right']['trajs']))


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
    num_desired = len(database[key]['vel_trajs'])
    keys = ['Right']
    num_data = dict()

    
    x_inputs_train = dict()
    x_inputs_test = dict()
    x_inputs_train_temp = dict()
    x_inputs_test_temp = dict()
    y_train = dict()
    y_test = dict()
    y_test_temp = dict()
    y_train_temp = dict()

    y_vel_train = dict()
    y_vel_test = dict()
    y_vel_test_temp = dict()
    y_vel_train_temp = dict()

    y_acc_train = dict()
    y_acc_test = dict()
    y_acc_train_temp = dict()
    y_acc_test_temp = dict()

    y_u_train = dict()
    y_u_test = dict()
    y_u_train_temp = dict()
    y_u_test_temp = dict()

    y_x_train = dict()
    y_x_test = dict()
    y_x_train_temp = dict()
    y_x_test_temp = dict()
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

    timestep = 60
    rbf_num = param[time_step][3]
    Phi = define_RBF(dof=19, nbStates =rbf_num, offset = param[time_step][0], width = param[time_step][1], T = timestep, coeff =param[time_step][2])

    if learn_type1 == 0:
        for key in keys:
            x_inputs[key] = []
            x_inputs[key] = np.array(database[key]['x_inputs'])[:num_desired]
            trajs[key] = np.array(database[key]['trajs'])[:num_desired]
            vel_trajs[key] = np.array(database[key]['vel_trajs'])[:num_desired]
            x_trajs[key] = np.array(database[key]['x_state'])[:num_desired]
            foot_poses[key] = database[key]['foot_poses'][:num_desired]
            num_data[key] = len(foot_poses[key])

        for key in keys:
            d = np.array([])
            for i in range(0, num_desired):
                c = np.append(np.append(trajs[key][i][0], vel_trajs[key][i][0], axis=0), x_trajs[key][i][0], axis=0)
                d = np.append(d, np.array([c]))
            d = d.reshape(num_desired, 49)
            x_inputs[key] = d

        #revise
        for key in keys:
            raw_u_trajs = database[key]['acc_trajs']
            raw_acc_trajs = database[key]['u_trajs']
            for i in range(len(raw_acc_trajs)):
                newrow1 = np.zeros(20)
                raw_acc_trajs[i] = numpy.vstack([raw_acc_trajs[i], newrow1])
            for i in range(len(raw_u_trajs)):
                newrow = np.zeros(4)
                raw_u_trajs[i] = numpy.vstack([raw_u_trajs[i],newrow])
            u_trajs[key] = np.array(raw_u_trajs)
            acc_trajs[key] = np.array(raw_acc_trajs)
        del(database)
        for key in keys:
            w_trajs[key] = apply_RBF(trajs[key], Phi)
            w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
            w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        #w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        #w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'Phi_normal'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(Phi, open(file_name4,"wb"))
        for key in keys:
            pca[key] = PCA(n_components = int(rbf_num))
            w_trajs_pca[key] = pca[key].fit_transform(w_trajs[key])
               
            pca_vel[key] = PCA(n_components=int(rbf_num))
            w_vel_trajs_pca[key] = pca_vel[key].fit_transform(w_vel_trajs[key])

            pca_x[key] = PCA(n_components= int(rbf_num))
            w_x_trajs_pca[key] = pca_x[key].fit_transform(w_x_trajs[key])

            #pca_acc[key] = PCA(n_components=int(rbf_num))
            #w_acc_trajs_pca[key] = pca_acc[key].fit_transform(w_acc_trajs[key])

            #pca_u[key] = PCA(n_components=int(rbf_num))
            #w_u_trajs_pca[key] = pca_u[key].fit_transform(w_u_trajs[key])

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'w_trajs_pca_Early_normal'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(pca, open(file_name4,"wb"))
        file_name2 = 'w_vel_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(pca_vel, open(file_name4,"wb"))
        file_name2 = 'w_x_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(pca_x, open(file_name4,"wb"))
        '''
        file_name2 = 'w_u_trajs_pca_'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(pca_u, open(file_name4,"wb"))
        file_name2 = 'w_acc_trajs_pca_'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pickle.dump(pca_acc, open(file_name4,"wb"))
        '''

        for key in keys:
            x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key], w_trajs_pca[key], test_size = 0.1, random_state=1)
            _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.1, random_state=1)
            #_,_, y_u_train[key], y_u_test[key] = train_test_split(x_inputs[key],w_u_trajs_pca[key], test_size = 0.1, random_state=1)
            #_,_, y_acc_train[key], y_acc_test[key] = train_test_split(x_inputs[key],w_acc_trajs_pca[key], test_size = 0.1, random_state=1)
            _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.1, random_state=1)

            x_inputs_train[key] = torch.FloatTensor(x_inputs_train[key])
            x_inputs_test[key] = torch.FloatTensor(x_inputs_test[key])
            y_test[key] = torch.FloatTensor( (y_test[key]))
            y_vel_test[key] = torch.FloatTensor( (y_vel_test[key]))
            #y_u_test[key] = torch.FloatTensor( (y_u_test[key]))
            #y_acc_test[key] = torch.FloatTensor( (y_acc_test[key]))
            y_x_test[key] = torch.FloatTensor( (y_x_test[key]))
            y_train[key] = torch.FloatTensor( (y_train[key]))
            y_vel_train[key] = torch.FloatTensor( (y_vel_train[key]))
            #y_u_train[key] = torch.FloatTensor( (y_u_train[key]))
            #y_acc_train[key] = torch.FloatTensor( (y_acc_train[key]))
            y_x_train[key] = torch.FloatTensor( (y_x_train[key]))

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'x_inputs_train_Early_normal'
        file_name3 = '.pt'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(x_inputs_train, file_name4)
        file_name2 = 'x_inputs_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(x_inputs_test, file_name4)
        file_name2 = 'y_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_test, file_name4)
        file_name2 = 'y_vel_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_vel_test, file_name4)
        file_name2 = 'y_u_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_u_test, file_name4)
        file_name2 = 'y_acc_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_acc_test, file_name4)
        file_name2 = 'y_x_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_x_test, file_name4)
        file_name2 = 'y_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_train, file_name4)
        file_name2 = 'y_vel_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_vel_train, file_name4)
        file_name2 = 'y_u_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_u_train, file_name4)
        file_name2 = 'y_acc_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_acc_train, file_name4)
        file_name2 = 'y_x_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        torch.save(y_x_train, file_name4)
        print("transform SAVE")
   
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")

    N = 60
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    learning_data_num = 27
    
    
    for i in range(0, 49, 1):#learning_data_num):
        PCAlearning(i)
    k = adsfasdff
    print("start")
   
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

