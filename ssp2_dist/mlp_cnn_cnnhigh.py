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
from pytorchtools import EarlyStopping

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
            torch.nn.Linear(in_features = input_size, out_features= 50),
            #torch.nn.LeakyReLU(10),
            #torch.nn.Linear(in_features = 50, out_features= 60),
            #torch.nn.LeakyReLU(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 50, out_features = output_size),
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
    learn_type1 = 1
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
    "timestep=0_finish",
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

    naming1 = [
       "timestep=0_finish",
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
    '''
    param = [
        [2,1,19,55], #0
        [2,1,13,55], #1
        [1,1,33,52], #2
        [1,1,45,55], #3
        [1,1,3,54], #4
        [1,1,13,54], #5
        [1,1,27,54], #6
        [1,1,35,53], #7
        [2,1,17,55], #8
        [2,1,19,54], #9
        [2,1,3,53], #10
        [2,1,23,53], #11
        [1,1,27,54], #12
        [2,1,23,53], #13
        [1,1,33,55], #14
        [1,1,23,55], #15
        [1,1,33,55], #16
        [1,1,35,54], #17
        [2,1,49,53], #18
        [2,1,3,53], #19
        [1,1,15,54], #20
        [2,1,25,52], #21
        [1,1,19,54], #22
        [1,1,13,55], #23
        [1,1,39,55], #24
        [3,1,35,55], #25
        [1,1,19,54], #26
        [1,1,19,54], #27
        [2,1,49,53], #28
        [1,1,27,54], #29
        [2,1,13,55], #30
        [1,1,17,53], #31
        [2,1,15,52], #32
        [1,1,7,55], #33
        [1,1,45,55], #34
        [1,1,7,55], #35
        [1,1,23,55], #36
        [1,1,49,53], #37
        [2,1,7,55], #38
        [3,1,17,53], #39
        [1,1,13,55], #40
        [2,1,35,54], #41
        [2,1,7,53], #42
        [1,1,19,55], #43
        [2,1,43,53], #44
        [1,1,41,55], #45
        [2,1,11,53], #46
        [2,1,21,52], #47
        [2,1,11,53], #48
        [2,1,47,53], #49
    ] 
    '''
    param=[
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
        [2,1,19,47],
    ]

    print(len(naming))
    print(len(param))
    #a = adsfsdfs
    #if time_step < 5:
    #    file_name = "/home/jhk/walkingdata/beforedata/ssp2/"
    #else:
    '''
    file_name = "/home/jhk/Downloads/"
    file_name2 = "/timestep="
    file_name3 = file_name + naming1[time_step]#+'_re'
    print(file_name3)
    

    with open(file_name3, 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(len(database['Right']['trajs']))
    '''

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
                c = np.append(np.append(trajs[key][i][0], vel_trajs[key][i][0], axis=0), np.array([x_trajs[key][i][0][2], x_trajs[key][i][0][6]]), axis=0)
                d = np.append(d, np.array([c]))
            d = d.reshape(num_desired, 43)
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
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/ssp2/'
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

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/ssp2/'
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
    else:
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'Phi_normal'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        #print(torch.load(file_name4))
        print(file_name4)
        Phi = pickle.load(open(file_name4,"rb"))

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'x_inputs_train_Early_normal'
        file_name3 = '.pt'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        x_inputs_train = torch.load(file_name4)
        file_name2 = 'x_inputs_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        x_inputs_test = torch.load(file_name4)
        file_name2 = 'y_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_test = torch.load(file_name4)
        file_name2 = 'y_vel_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_vel_test = torch.load(file_name4)
        file_name2 = 'y_x_test_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_x_test = torch.load(file_name4)
        file_name2 = 'y_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_train = torch.load(file_name4)
        file_name2 = 'y_vel_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_vel_train = torch.load(file_name4)
        file_name2 = 'y_x_train_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        y_x_train = torch.load(file_name4)

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'w_trajs_pca_Early_normal'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca = pickle.load(open(file_name4,'rb'))
        
        file_name2 = 'w_vel_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca_vel = pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_x_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca_x= pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_u_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        #pca_u = pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_acc_trajs_pca_Early_normal'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        #pca_acc = pickle.load(open(file_name4,'rb'))
        #explain =pca_x[key].explained_variance_ratio_
        #print(explain)
        #k = asdfasdf
    '''
    for epoch in range(1):
        losses = []
        print(np.shape(trajs[key][0]))
        for i in range(0,len(w_vel_trajs_pca[key])): 
            data = w_vel_trajs_pca[key][i]
            v_pred = pca_vel[key].inverse_transform(data)
            v_pred = v_pred.reshape(rbf_num,-1)
            v_pred = np.dot(Phi,v_pred)
            #batch_loss = loss(vel_trajs[key][i].flatten(), torch.FloatTensor(v_pred.flatten())) # difference between actual and reconstructed                   #losses.append(batch_loss.item())
             #running_loss = np.mean(losses)
            running_loss = numpy.linalg.norm(np.subtract(vel_trajs[key][i].flatten(),v_pred.flatten()))
            print(f"Epoch_vel {epoch}: {running_loss}")
    '''
    
    device = 'cpu'
    train_y = timeseries(x_inputs_train[key], y_train[key])
    test_y = timeseries(x_inputs_test[key], y_test[key])
    train_yvel = timeseries(x_inputs_train[key], y_vel_train[key])
    test_yvel = timeseries(x_inputs_test[key], y_vel_test[key])
    #train_yacc = timeseries(x_inputs_train[key], y_acc_train[key])
    #test_yacc = timeseries(x_inputs_test[key], y_acc_test[key])
    #train_yu = timeseries(x_inputs_train[key], y_u_train[key])
    #test_yu = timeseries(x_inputs_test[key], y_u_test[key])
    train_yx = timeseries(x_inputs_train[key], y_x_train[key])
    test_yx = timeseries(x_inputs_test[key], y_x_test[key])
   
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_y, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_y, batch_size=batch_size, shuffle=True)
    train_vel_loader = torch.utils.data.DataLoader(dataset=train_yvel, batch_size=batch_size, shuffle=True)
    test_vel_loader = torch.utils.data.DataLoader(dataset=test_yvel, batch_size=batch_size, shuffle=True)
    #train_acc_loader = torch.utils.data.DataLoader(dataset=train_yacc, batch_size=batch_size, shuffle=True)
    #test_acc_loader = torch.utils.data.DataLoader(dataset=test_yacc, batch_size=batch_size, shuffle=True)
    #train_u_loader = torch.utils.data.DataLoader(dataset=train_yu, batch_size=batch_size, shuffle=True)
    #test_u_loader = torch.utils.data.DataLoader(dataset=test_yu, batch_size=batch_size, shuffle=True)
    train_x_loader = torch.utils.data.DataLoader(dataset=train_yx, batch_size=batch_size, shuffle=True)
    test_x_loader = torch.utils.data.DataLoader(dataset=test_yx, batch_size=batch_size, shuffle=True)

    #q
    input_size = 43
    model = CNN(input_size=input_size,
                output_size = rbf_num,
                device=device).to(device)
       
    #qdot
    model1 = CNN(input_size=input_size,
                output_size = rbf_num,
                device=device).to(device)

    #x
    model2 = CNN(input_size=input_size,
                output_size = rbf_num,
                device=device).to(device)

    '''
    #acc
    input_size = 21
    sequence_length = 1
    num_layers = 10
    hidden_size = rbf_num

    model3 = CNN(input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)
   
    #u
    input_size = 19
    sequence_length = 1
    num_layers = 5
    hidden_size = rbf_num

    model4 = CNN(input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)
    '''

   
    #model3.train()
    #model4.train()

    if learn_type == 0:
        model.train()
        model1.train()
        model2.train()

        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_graph = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        input_size = 43
        sequence_length = 1
        best_loss = 10 ** 15
        patience_limit = 15
        train_loss = 0
        #model, optimizer = ipex.optimize(model, optimizer=optimizer)
        early_stopping = EarlyStopping(patience=patience_limit, verbose=True)
        
        for epoch in range(num_epochs):
            train_loss = 0
            train_num = 0
            model.train()
            for data in train_loader:
                train_num = train_num + 1
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model(X)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if epoch % 1 == 0:
                print ('Epoch [{}/{}],  Loss: {:.6f}'
                    .format(epoch+1, num_epochs, train_loss/train_num))
            #train_loss = loss.item()
            model.eval()
            val_loss = 0
            val_num = 0
            for data in test_loader:
                seq, y = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                val_num = val_num + 1
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()   
            early_stopping(val_loss, model)
            if early_stopping.early_stop: # 조건 만족 시 조기 종료
                break
        
        loss_graph[0] = [val_loss/val_num, train_loss/train_num]
        print("loss_graph")
        print(loss_graph[0])
        print([train_num, val_num])
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        sequence_length = 1
        best_loss = 10 ** 9
        #model1, optimizer1 = ipex.optimize(model1, optimizer=optimizer1)
        patience_limit = 15
        patience_check = 0 
        early_stopping = EarlyStopping(patience=patience_limit, verbose=True)
        for epoch in range(num_epochs):
            train_loss = 0
            model1.train()
            for data in train_vel_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model1(X)
                loss = criterion(out, target)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                train_loss += loss.item()

            if epoch % 1 == 0:
                print ('1Epoch [{}/{}],  Loss: {:.6f}'
                    .format(epoch+1, num_epochs, loss.item()))
            #train_loss = loss.item()

            model1.eval()
            val_loss = 0
            for data in test_vel_loader:
                seq, y = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                y_pred = model1(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()   
            ### early stopping 여부를 체크하는 부분 ###
            early_stopping(val_loss, model1)
            if early_stopping.early_stop: # 조건 만족 시 조기 종료
                break
            '''
            if val_loss > best_loss: # loss가 개선되지 않은 경우
                patience_check += 1
                if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                    break
            else: # loss가 개선된 경우
                best_loss = val_loss
                patience_check = 0
            '''
        loss_graph[1] = [val_loss/val_num, train_loss/train_num]
        print("loss_graph")
        print(loss_graph[1])
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 100
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        loss_graph[1] = [val_loss/val_num, train_loss/train_num]
        print("loss_graph")
        print(loss_graph[1])
        sequence_length = 1
        best_loss = 10 ** 9
        #model2, optimizer2 = ipex.optimize(model2, optimizer=optimizer2)
        patience_limit = 15
        patience_check = 0 
        early_stopping = EarlyStopping(patience=patience_limit, verbose=True)

        for epoch in range(num_epochs):
            train_loss = 0
            model2.train()
            for data in train_x_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model2(X)
                loss = criterion(out, target)
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                train_loss += loss.item()
                
            if epoch % 1 == 0:
                print ('2Epoch [{}/{}],  Loss: {:.6f}'
                    .format(epoch+1, num_epochs, loss.item()))
            #train_loss = loss.item()

            model2.eval()
            val_loss = 0
            for data in test_x_loader:
                seq, y = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                y_pred = model2(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
            early_stopping(val_loss, model2)
            if early_stopping.early_stop: # 조건 만족 시 조기 종료
                break
            '''
            ### early stopping 여부를 체크하는 부분 ###
            if val_loss > best_loss: # loss가 개선되지 않은 경우
                patience_check += 1
                if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                    break
            else: # loss가 개선된 경우
                best_loss = val_loss
                patience_check = 0
            '''
        loss_graph[2] = [val_loss/val_num, train_loss/train_num]
        print("loss_graph")
        print(loss_graph[2])
        file_name = '/home/jhk/ssd_mount/beforedata/cnnEarly_normal_leakyhigh'
        file_name2 = '0_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3       
        torch.save(model.state_dict(), file_name4)
        file_name2 = '1_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3   
        torch.save(model1.state_dict(), file_name4)
        file_name2 = '2_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3   
        torch.save(model2.state_dict(), file_name4)
        loss_graph[2] = [val_loss/val_num, train_loss/train_num]
        print("loss_graph")
        print(loss_graph[2])

        best = np.array(loss_graph)
        file_name = "/home/jhk/walkingdata/beforedata/fdyn/"
        file_name3 = file_name +'loss_normalsLeakyhigh' +naming[time_step] +  '.txt'
        np.savetxt(file_name3, best)
        
       
    else:
        file_name = '/home/jhk/ssd_mount/beforedata/cnn'
        file_name2 = '0_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3   
        model.load_state_dict(torch.load(file_name4))
        file_name2 = '1_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3   
        model1.load_state_dict(torch.load(file_name4))
        file_name2 = '2_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3   
        model2.load_state_dict(torch.load(file_name4))
    
    #for i  in range(0, len(trajs[key])):
    #    if(x_inputs_test[key][JJ][None,:][0][0] == trajs[key][i][0][0]) and (x_inputs_test[key][JJ][None,:][0][1] == trajs[key][i][0][1]) and (x_inputs_test[key][JJ][None,:][0][21] == x_trajs[key][i][0][2]) and (x_inputs_test[key][JJ][None,:][0][22] == x_trajs[key][i][0][6]):
    #        KKK = i
    #for i in range(0, 5):
    #    print(i)
    #    print(v_pca[i])
    #    print(vel_trajs[key][KKK][i])
    
   
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")

    N = 60
    T = 1

    MAXITER = 300
    dt_ = 1.2 / float(N)
    learning_data_num = 27
    
    
    for i in range(12, 49, 50):#learning_data_num):
        PCAlearning(i)
    k = adsfasdff
    print("start")
   
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

