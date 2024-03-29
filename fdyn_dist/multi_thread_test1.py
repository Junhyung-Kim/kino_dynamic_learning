import roslibpy
import pickle
import roslaunch
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
from multiprocessing import shared_memory
import sysv_ipc

class CShmReader : 
    def __init__(self) :
        pass
 
    def doReadShm(self , key) :
        memory = sysv_ipc.SharedMemory(key)
        memory_value = memory.read()
        c = np.ndarray((2,), dtype=np.double, buffer=memory_value)

    def doWriteShm(self, Input) :
        self.memory.write(Input)

def InversePCA(model, rbf_num, pca, Phi, tick, X, thread_manager):
    k = 0
    while True:
        if thread_manager[0] == 1:# and k == 0:
            c = torch.tensor(np.array(X[:]).reshape(1,1,43),dtype=torch.float32)
            w_traj = model[k].forward(c)[0].detach().numpy()
            w_traj = pca[k].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num[k+1][3],-1)
            q_traj[:] = np.dot(Phi[k],w_traj)
            thread_manager[0] = 0
            k = k + 1

def InversePCA1(model, rbf_num, pca, Phi, tick, X, thread_manager):
    k = 0
    while True:
        if thread_manager[1] == 1:# and k == 0:
            c = torch.tensor(np.array(X[:]).reshape(1,1,43),dtype=torch.float32)
            w_traj = model[k].forward(c).detach().numpy()
            #w_traj = w_traj[0].detach().numpy()
            w_traj = pca[k].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num[k+1][3],-1)
            v_traj[:] = np.dot(Phi[k],w_traj)
            a_traj[:] = np.subtract(v_traj[1:60,:], v_traj[0:59,:])/0.02
            thread_manager[1] = 0
            k = k + 1


def InversePCA2(model, rbf_num, pca, Phi, tick, X, thread_manager):
    k = 0
    while True:
        if thread_manager[2] == 1:# and k == 0:
            c = torch.tensor(np.array(X[:]).reshape(1,1,43),dtype=torch.float32)
            w_traj = model[k].forward(c).detach().numpy()
            #w_traj = w_traj[0].detach().numpy()
            w_traj = pca[k].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num[k+1][3],-1)
            x_traj[:] = np.dot(Phi[k],w_traj)
            
            u_traj[:,0:2] = np.subtract(x_traj[1:60,2:4], x_traj[0:59,2:4])/0.02
            u_traj[:,2:4] = np.subtract(x_traj[1:60,6:8], x_traj[0:59,6:8])/0.02
            
            #x_traj = np.dot(Phi[k],pca[k].inverse_transform([model[k].forward(c)[0].detach().numpy()[None,:]])[0].reshape(rbf_num,-1))
            
            thread_manager[2] = 0
            k = k + 1
            
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
            torch.nn.ReLU(),
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
    global xs_pca_test, rbf_num
    global xs_pca
    global us_pca

    learn_type = 1
    learn_type1 = 1
    
    file_name ='/home/jhk/kino_dynamic_learning/dataset/dataset1/'

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
    num_data = dict()
    key = 'Right'

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
]

    naming1 = [
"timestep=0_finish_",  
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
    timestep = 60
    rbf_num = param[time_step][3]
    Phi = define_RBF(dof=19, nbStates =rbf_num, offset = param[time_step][0], width = param[time_step][1], T = timestep, coeff =param[time_step][2])

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

    if learn_type1 == 0:
        for key in keys:
            w_trajs[key] = apply_RBF(trajs[key], Phi)
            w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
            w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
            w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
            w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)

        file_name2 = '/Phi.pkl'
        file_name3 = file_name + str(time_step) + file_name2
        pickle.dump(Phi, open(file_name3,"wb"))
       
        for key in keys:
            pca[key] = PCA(n_components = int(rbf_num))
            w_trajs_pca[key] = pca[key].fit_transform(w_trajs[key])
               
            pca_vel[key] = PCA(n_components=int(rbf_num))
            w_vel_trajs_pca[key] = pca_vel[key].fit_transform(w_vel_trajs[key])

            pca_x[key] = PCA(n_components= int(rbf_num))
            w_x_trajs_pca[key] = pca_x[key].fit_transform(w_x_trajs[key])
       
        file_name2 = 'w_trajs_pca_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pickle.dump(pca, open(file_name4,"wb"))
        file_name2 = 'w_vel_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pickle.dump(pca_vel, open(file_name4,"wb"))
        file_name2 = 'w_x_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pickle.dump(pca_x, open(file_name4,"wb"))
       
        for key in keys:
            x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key], w_trajs_pca[key], test_size = 0.1, random_state=1)
            _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.1, random_state=1)
            _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.1, random_state=1)

            x_inputs_train[key] = torch.FloatTensor(x_inputs_train[key])
            x_inputs_test[key] = torch.FloatTensor(x_inputs_test[key])
            y_test[key] = torch.FloatTensor( (y_test[key]))
            y_vel_test[key] = torch.FloatTensor( (y_vel_test[key]))
            y_x_test[key] = torch.FloatTensor( (y_x_test[key]))
            y_train[key] = torch.FloatTensor( (y_train[key]))
            y_vel_train[key] = torch.FloatTensor( (y_vel_train[key]))
            y_x_train[key] = torch.FloatTensor( (y_x_train[key]))

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset1/'
        file_name2 = 'x_inputs_train_'
        file_name3 = '.pt'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(x_inputs_train, file_name4)
        file_name2 = 'x_inputs_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(x_inputs_test, file_name4)
        file_name2 = 'y_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_test, file_name4)
        file_name2 = 'y_vel_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_vel_test, file_name4)
        file_name2 = 'y_u_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_u_test, file_name4)
        file_name2 = 'y_acc_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_acc_test, file_name4)
        file_name2 = 'y_x_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_x_test, file_name4)
        file_name2 = 'y_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_train, file_name4)
        file_name2 = 'y_vel_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_vel_train, file_name4)
        file_name2 = 'y_u_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_u_train, file_name4)
        file_name2 = 'y_acc_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_acc_train, file_name4)
        file_name2 = 'y_x_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        torch.save(y_x_train, file_name4)
        print("transform SAVE")
    else:
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'Phi'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        #print(torch.load(file_name4))
        print(file_name4)
        Phi = pickle.load(open(file_name4,"rb"))

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/fdyn/'
        file_name2 = 'w_trajs_pca_Early_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca = pickle.load(open(file_name4,'rb'))
        
        file_name2 = 'w_vel_trajs_pca_Early_'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca_vel = pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_x_trajs_pca_Early_'
        file_name4 = file_name  +file_name2+ naming[time_step]+ file_name3
        pca_x= pickle.load(open(file_name4,'rb'))
    
    device = 'cpu'
    
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

    if learn_type == 0:
        model.train()
        model1.train()
        model2.train()

        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 5
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_graph = []
        input_size = 23
        sequence_length = 1
        i = 0
        for epoch in range(num_epochs):
            for data in train_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model(X)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i = i + 1
                if i % 1000 == 0:
                    print ('Epoch [{}/{}],  Loss: {:.6f}'
                       .format(epoch+1, num_epochs, loss.item()))
       
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 20
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        loss_graph = []
        sequence_length = 1
        i = 0
        for epoch in range(num_epochs):
            for data in train_vel_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model1(X)
                loss = criterion(out, target)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                i = i + 1
                if i % 1000 == 0:
                    print ('1Epoch [{}/{}],  Loss: {:.6f}'
                        .format(epoch+1, num_epochs, loss.item()))
   
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        loss_graph = []
        sequence_length = 1
        i = 0
        for epoch in range(num_epochs):
            for data in train_x_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model2(X)
                loss = criterion(out, target)

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                i = i + 1
                if i % 1000 == 0:
                    print ('2Epoch [{}/{}],  Loss: {:.6f}'
                       .format(epoch+1, num_epochs, loss.item()))
                   
        file_name2 = '/cnn.pkl'
        file_name3 = file_name + str(time_step) + file_name2
        torch.save(model.state_dict(), file_name3)
        file_name2 = '/cnn1.pkl'
        file_name3 = file_name + str(time_step) + file_name2
        torch.save(model1.state_dict(), file_name3)
        file_name2 = '/cnn2.pkl'
        file_name3 = file_name + str(time_step) + file_name2
        torch.save(model2.state_dict(), file_name3)
       
    else:
        file_name = '/home/jhk/ssd_mount/beforedata/fdyn/cnnEarly'
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

    PCA_.append(pca[key])
    PCA_VEL.append(pca_vel[key])
    PCA_X.append(pca_x[key])

    NN_.append(model)
    NN_VEL.append(model1)
    NN_X.append(model2)
   
    if time_step == 0:
        global X_INIT
        X_INIT = x_inputs_test[key]

    PHI_.append(Phi)

def talker():
    global xs_pca_test, xs_pca, us_pca, rbf_num, talk
    global q_traj, v_traj, a_traj, x_traj, u_traj, param
    global PCA_, NN_, PCA_VEL, NN_VEL, PCA_X, NN_X, X_INIT, PHI_

    param = [
        [0,0,0,0], #0
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
[2,1,31,53], #12
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

    PCA_ = []
    NN_ = []
    PCA_VEL = []
    NN_VEL = []
    PCA_X = []
    NN_X = []
    PHI_= []
    X_INIT = np.array([])
    N = 60
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    total_time = 22

    crocs_data = dict()
    crocs_data['left'] = dict()
    crocs_data['Right'] = dict()

    for key in crocs_data.keys():
        crocs_data[key]['foot_poses'] = []
        crocs_data[key]['trajs'] = []
        crocs_data[key]['acc_trajs'] = []
        crocs_data[key]['x_inputs'] = []
        crocs_data[key]['vel_trajs'] = []
        crocs_data[key]['x_state'] = []        
        crocs_data[key]['u_trajs'] = []
        crocs_data[key]['data_phases_set'] = []
        crocs_data[key]['costs'] = []
        crocs_data[key]['iters'] = []
    
    for i in range(1, total_time):
        print("learning")
        print(i)
        PCAlearning(i)
    
    print("start")
    f = open("/home/jhk/walkingdata/beforedata/fdyn/lfoot2_final.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/fdyn/rfoot2_final.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')
    f5 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')

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

    for i in range(0, len(lines_array)):
        for j in range(0, len(lines_array[i])):
            if j == 0:
                array_boundRF[i].append(float(lines_array[i][j]))
            if j == 1:
                array_boundRF[i].append(float(lines_array[i][j]))
            if j == 2:
                array_boundRF[i].append(float(lines_array[i][j]))
    
    for i in range(0, len(lines1_array)):
        for j in range(0, len(lines1_array[i])):
            if j == 0:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 1:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 2:
                array_boundLF[i].append(float(lines1_array[i][j]))

    for i in range(0, len(lines_array)):
        array_boundRF[i] = np.sum([array_boundRF[i], [-0.03, 0.0, 0.15842]], axis = 0)
    for i in range(0, len(lines1_array)):
        array_boundLF[i] = np.sum([array_boundLF[i], [-0.03, 0.0, 0.15842]], axis = 0)
    
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

    for i in range(0, len(lines3_array)):
        for j in range(0, len(lines3_array[i])):
            if j == 0:
                zmp_refx[i].append(float(lines3_array[i][j]))
            if j == 1:
                zmp_refy[i].append(float(lines3_array[i][j]))

    f.close()
    f1.close()
    f2.close()

    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = RobotWrapper.BuildFromURDF("/usr/local/lib/python3.8/dist-packages/robot_properties_tocabi/resources/urdf/tocabi.urdf","/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/meshes",pinocchio.JointModelFreeFlyer())  
   
    pi = 3.14159265359
   
    jointsToLock = ['Waist1_Joint', 'Neck_Joint', 'Head_Joint',
    'L_Shoulder1_Joint', 'L_Shoulder2_Joint', 'L_Shoulder3_Joint', 'L_Armlink_Joint', 'L_Elbow_Joint', 'L_Forearm_Joint', 'L_Wrist1_Joint', 'L_Wrist2_Joint',
    'R_Shoulder1_Joint', 'R_Shoulder2_Joint', 'R_Shoulder3_Joint', 'R_Armlink_Joint', 'R_Elbow_Joint', 'R_Forearm_Joint', 'R_Wrist1_Joint', 'R_Wrist2_Joint']
    # Get the joint IDs
    jointsToLockIDs = []
   
    for jn in range(len(jointsToLock)):
        jointsToLockIDs.append(model.model.getJointId(jointsToLock[jn]))
    # Set initial configuration
   
    fixedJointConfig = np.matrix([0, 0, 0.82473, 0, 0, 0, 1,
    0.0, 0.0, -0.55, 1.26, -0.71, 0.0,
    0.0, 0.0, -0.55, 1.26, -0.71, 0.0,
    0, 0.0, 0.0,
    0.2, 0.6, 1.5, -1.47, -1, 0 ,-1, 0,
    0, 0,
    -0.2, -0.6 ,-1.5, 1.47, 1, 0, 1, 0]).T

    model = RobotWrapper.buildReducedRobot(model, jointsToLockIDs, fixedJointConfig)
    pi = 3.14159265359
   
    q = pinocchio.utils.zero(model.nq)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    q_init = [0, 0, 0.82473, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0]

    state = crocoddyl.StateKinodynamic(model.model)
    actuation = crocoddyl.ActuationModelKinoBase(state)
    x0 = np.array([0.] * (state.nx + 8))
    u0 = np.array([0.] * (22))
    for i in range(0,len(q_init)):
        x0[i] = q_init[i]
        q[i] = q_init[i]
    
    RFjoint_id = model.model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.model.getJointId("L_AnkleRoll_Joint")
    LFframe_id = model.model.getFrameId("L_Foot_Link")
    RFframe_id = model.model.getFrameId("R_Foot_Link")    
    data = model.model.createData()

    pinocchio.forwardKinematics(model.model, data, q, qdot)
    pinocchio.updateFramePlacements(model.model,data)
    pinocchio.centerOfMass(model.model, data, q, False)
    pinocchio.computeCentroidalMomentum(model.model,data,q,qdot)
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]

    LF_tran = data.oMi[LFjoint_id]
    RF_tran = data.oMi[RFjoint_id]

    x0[41] = data.com[0][0]
    x0[43] = data.com[0][0]
    x0[45] = data.com[0][1]
    x0[47] = data.com[0][1]
    
    weight_quad_camx = 2.9
    weight_quad_camy = 2.9
    weight_quad_zmp = np.array([1.0, 1.0])#([weight_quad_zmpx] + [weight_quad_zmpy])
    weight_quad_zmp1 = np.array([2.0, 2.0]) ##11
    weight_quad_cam = np.array([0.008, 0.008])#([weight_quad_camy] + [weight_quad_camx])
    weight_quad_upper = np.array([0.0005, 0.0005])
    weight_quad_com = np.array([30.0, 30.0, 3.0])#([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
    weight_quad_rf = np.array([10.0, 3.0, 5.0, 0.5, 0.5, 0.5])#np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
    weight_quad_lf = np.array([10.0, 3.0, 5.0, 0.5, 0.5, 0.5])#np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
    lb_ = np.ones([2, N])
    ub_ = np.ones([2, N])

    actuation_vector = [None] * (N)
    state_vector = [None] * (N)
    state_bounds = [None] * (N)
    state_bounds2 = [None] * (N)
    state_bounds3 = [None] * (N)
    state_activations = [None] * (N)
    state_activations1 = [None] * (N)
    state_activations2 = [None] * (N)
    state_activations3 = [None] * (N)
    xRegCost_vector = [None] * (N)
    uRegCost_vector = [None] * (N)
    stateBoundCost_vector = [None] * (N)
    stateBoundCost_vector1 = [None] * (N)
    stateBoundCost_vector2 = [None] * (N)
    camBoundCost_vector = [None] *  (N)
    comBoundCost_vector = [None] *  (N)
    rf_foot_pos_vector = [None] *  (N)
    lf_foot_pos_vector = [None] *  (N)
    pelvis_rot_vector = [None] *  (N)
    residual_FrameRF = [None] *  (N)
    residual_FramePelvis = [None] *  (N)
    residual_FrameLF = [None] *  (N)
    PelvisR = [None] *  (N)
    foot_trackR = [None] *  (N)
    foot_trackL = [None] *  (N)
    runningCostModel_vector = [None] * (N-1)
    runningDAM_vector = [None] * (N-1)
    runningModelWithRK4_vector = [None] * (N-1)
    xs = [None] * (N)
    us = [None] * (N-1)
    traj_= np.array([0.] * (state.nx + 8))
    
    rbf_num = 47
    tick = shared_memory.SharedMemory(create=True, size=sys.getsizeof(1))
    tick.buf[0] = 0
    
    time.sleep(1)

    print("Python start")
    mpc_signal = sysv_ipc.SharedMemory(1)
    mpc_signalv  = mpc_signal.read()
    mpc_signaldata =  np.ndarray(shape=(3,), dtype=np.int32, buffer=mpc_signalv)
    x_init = sysv_ipc.SharedMemory(2)
    x_initv  = x_init.read()
    statemachine = sysv_ipc.SharedMemory(3)
    statemachinedata = np.array([0, 0, 0], dtype=np.int8)
    statemachine.write(statemachinedata)
    desired_value = sysv_ipc.SharedMemory(4)
     
    thread_manager1 = []
    for i in range(0,3):
        thread_manager1.append(0)
   
    thread_manager = multiprocessing.Array(ctypes.c_int, thread_manager1)
   
    N = 60
    T = 1
    mpc_cycle = 0
   
    MAXITER = 300
    dt_ = 1.2 / float(N)
    time_step = 1
    k = 1
    k3 = 1

    for i in range(0, N):
        if i == 0:
            array_boundRF_[i] = array_boundRF[k*i + time_step]
        else:
            array_boundRF_[i] = array_boundRF[k*i + time_step]

    for i in range(0, N):
        if i == 0:
            array_boundLF_[i] = array_boundLF[k*i + time_step]
        else:
            array_boundLF_[i] = array_boundLF[k*i + time_step]
            
    for i in range(0, N):
        if i == 0:
            array_boundx_[i] = array_boundx[k3*i + time_step]
            array_boundy_[i] = array_boundy[k3*i + time_step]
        else:
            array_boundx_[i] = array_boundx[k3*(i) + time_step]
            array_boundy_[i] = array_boundy[k3*(i) + time_step]
        
    for i in range(0, N):
        if i == 0:
            zmp_refx_[i] = zmp_refx[k*i + time_step]
            zmp_refy_[i] = zmp_refy[k*i + time_step]
        else:
            zmp_refx_[i] = zmp_refx[k*(i)+ time_step]
            zmp_refy_[i] = zmp_refy[k*(i)+ time_step]

    tick = shared_memory.SharedMemory(create=True, size=sys.getsizeof(1))
    tick.buf[0] = 0
    X_temp = np.zeros(43)
    X_temp = X_temp.reshape(1,1,43)
    queue = multiprocessing.Array(ctypes.c_float, np.shape(X_temp)[2])
    
    a = np.ones([60,21])
    shm_q_traj = shared_memory.SharedMemory(create=True, size=a.nbytes)
    a = np.ones([60,20])
    shm_v_traj = shared_memory.SharedMemory(create=True, size=a.nbytes)
    a = np.ones([59,20])
    shm_acc_traj = shared_memory.SharedMemory(create=True, size=a.nbytes)
    a = np.ones([60,8])
    shm_x_traj = shared_memory.SharedMemory(create=True, size=a.nbytes)
    a = np.ones([59,4])
    shm_u_traj = shared_memory.SharedMemory(create=True, size=a.nbytes)
    
    KK = 0
    signal = False

    q_traj = np.ndarray(shape=(60,21), dtype=np.float32, buffer=shm_q_traj.buf)
    v_traj = np.ndarray(shape=(60,20), dtype=np.float32, buffer=shm_v_traj.buf)
    a_traj = np.ndarray(shape=(59,20), dtype=np.float32, buffer=shm_acc_traj.buf)
    x_traj = np.ndarray(shape=(60,8), dtype=np.float32, buffer=shm_x_traj.buf)
    u_traj = np.ndarray(shape=(59,4), dtype=np.float32, buffer=shm_u_traj.buf)
    
    p1 = multiprocessing.Process(target=InversePCA, args=(NN_, param, PCA_, PHI_, tick, queue, thread_manager))
    p2 = multiprocessing.Process(target=InversePCA1, args=(NN_VEL, param, PCA_VEL, PHI_, tick, queue, thread_manager))
    p3 = multiprocessing.Process(target=InversePCA2, args=(NN_X, param, PCA_X, PHI_, tick, queue, thread_manager))
    
    p1.start()
    p2.start()
    p3.start()

    for i in range(0,N-1):
        traj_[43] = (array_boundx_[i][0] + array_boundx_[i][1])/2 #zmp_refx_[i][0]
        traj_[47] = (array_boundy_[i][0] + array_boundy_[i][1])/2#zmp_refy_[i][0]
        state_vector[i] = crocoddyl.StateKinodynamic(model.model)
        actuation_vector[i] = crocoddyl.ActuationModelKinoBase(state_vector[i])
        state_bounds[i] = crocoddyl.ActivationBounds(lb_[:,i],ub_[:,i])
        state_activations[i] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[i], weight_quad_zmp)
        #state_activations1[i] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[i], weight_quad_upper)
        stateBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], state_activations[i], crocoddyl.ResidualFlyState(state_vector[i], actuation_vector[i].nu + 4))
        stateBoundCost_vector1[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_zmp1), crocoddyl.ResidualFlyState(state_vector[i], traj_, actuation_vector[i].nu + 4))
        stateBoundCost_vector2[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_upper), crocoddyl.ResidualFlyState1(state_vector[i], actuation_vector[i].nu + 4))
        camBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[i], actuation_vector[i].nu + 4))
        comBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[i], np.array([0.0, 0.0, data.com[0][2]]), actuation_vector[i].nu + 4))
        rf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        rf_foot_pos_vector[i].translation = copy(RF_tran.translation)
        lf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        pelvis_rot_vector[i] = pinocchio.SE3.Identity()
        lf_foot_pos_vector[i].translation = copy(LF_tran.translation)
        #residual_FramePelvis[i] = crocoddyl.ResidualFrameRotation(state_vector[i], Pelvis_id, pelvis_rot_vector[i], actuation_vector[i].nu + 4)
        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        #PelvisR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[i])
        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu + 4)
        
        if i >= 1:
            runningCostModel_vector[i].addCost("stateReg1", stateBoundCost_vector1[i], 1.0)
            
            #runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], 1.0)
            #runningCostModel_vector[i].addCost("stateReg2", stateBoundCost_vector2[i], 1.0)
            runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
            runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
        
        runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
        runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)

        #traj_[43] = zmp_refx_[N-1][0]
        #traj_[47] = zmp_refy_[N-1][0]

    state_vector[N-1] = crocoddyl.StateKinodynamic(model.model)
    actuation_vector[N-1] = crocoddyl.ActuationModelKinoBase(state_vector[N-1])
    state_bounds[N-1] = crocoddyl.ActivationBounds(lb_[:,N-1],ub_[:,N-1])
    state_activations[N-1] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[N-1], weight_quad_zmp)
    #state_activations1[N-1] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[N-1], weight_quad_upper)
    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
    stateBoundCost_vector1[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_zmp1), crocoddyl.ResidualFlyState(state_vector[N-1], traj_, actuation_vector[N-1].nu + 4))
    stateBoundCost_vector2[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_upper), crocoddyl.ResidualFlyState1(state_vector[N-1], actuation_vector[N-1].nu + 4))
    camBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[N-1], actuation_vector[N-1].nu + 4))
    comBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[N-1], np.array([0.0, 0.0, data.com[0][2]]), actuation_vector[N-1].nu + 4))
    rf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    rf_foot_pos_vector[N-1].translation = copy(RF_tran.translation)
    lf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    lf_foot_pos_vector[N-1].translation = copy(LF_tran.translation)
    pelvis_rot_vector[N-1] = pinocchio.SE3.Identity()
    #residual_FramePelvis[N-1] = crocoddyl.ResidualFrameRotation(state_vector[N-1], Pelvis_id, pelvis_rot_vector[N-1], actuation_vector[N-1].nu + 4)
    #PelvisR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[N-1])
    residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])
    
    terminalCostModel = crocoddyl.CostModelSum(state_vector[N-1], actuation_vector[N-1].nu + 4)
    #terminalCostModel.addCost("stateReg", stateBoundCost_vector[N-1], 1.0)
    #terminalCostModel.addCost("stateReg1", stateBoundCost_vector1[N-1], 1.0)
    terminalCostModel.addCost("stateReg2", stateBoundCost_vector2[N-1], 1.0)
    terminalCostModel.addCost("comReg", comBoundCost_vector[N-1], 1.0)
    #terminalCostModel.addCost("camReg", camBoundCost_vector[N-1], 1.0)
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
    
    #terminalCostModel.addCost("pelvisReg1", PelvisR[N-1], 1.0)
    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N-1], actuation_vector[N-1], terminalCostModel)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 20
    ddp = crocoddyl.SolverFDDP(problemWithRK4)

    for time_step in range(1, total_time):
        ok_ = False
        for i in range(1, N-1):  
            traj_[43] = (array_boundx[i + time_step][0] + array_boundx[i + time_step][1])/2 #zmp_refx_[i][0]
            traj_[47] = (array_boundy[i + time_step][0] + array_boundy[i + time_step][1])/2#zmp_refy_[i][0]
            state_bounds[i].lb[0] = copy(array_boundx[i + time_step][0])
            state_bounds[i].ub[0] = copy(array_boundx[i + time_step][1])
            state_bounds[i].lb[1] = copy(array_boundy[i + time_step][0])
            state_bounds[i].ub[1] = copy(array_boundy[i + time_step][1])
            state_activations[i].bounds = state_bounds[i]
            stateBoundCost_vector1[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_zmp1), crocoddyl.ResidualFlyState(state_vector[i], traj_, actuation_vector[i].nu + 4))
            stateBoundCost_vector[i].activation_ = state_activations[i]

            runningCostModel_vector[i].removeCost("stateReg1")
            runningCostModel_vector[i].addCost("stateReg1", stateBoundCost_vector1[i], 1.0)

            runningCostModel_vector[i].removeCost("stateReg")
            runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], 1.0)
           
            rf_foot_pos_vector[i].translation[0] = copy(array_boundRF[i + time_step][0])
            rf_foot_pos_vector[i].translation[1] = copy(array_boundRF[i + time_step][1])
            rf_foot_pos_vector[i].translation[2] = copy(array_boundRF[i + time_step][2])
            lf_foot_pos_vector[i].translation[0] = copy(array_boundLF[i + time_step][0])
            lf_foot_pos_vector[i].translation[1] = copy(array_boundLF[i + time_step][1])
            lf_foot_pos_vector[i].translation[2] = copy(array_boundLF[i + time_step][2])
            residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
            foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
           
            runningCostModel_vector[i].removeCost("footReg1")
            runningCostModel_vector[i].removeCost("footReg2")
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)  
           
        state_bounds[N-1].lb[0] = copy(array_boundx[N-1 + time_step][0])
        state_bounds[N-1].ub[0] = copy(array_boundx[N-1 + time_step][1])
        state_bounds[N-1].lb[1] = copy(array_boundy[N-1 + time_step][0])
        state_bounds[N-1].ub[1] = copy(array_boundy[N-1 + time_step][1])
        state_activations[N-1].bounds = state_bounds[N-1]
        stateBoundCost_vector[N-1].activation_ = state_activations[N-1]
        rf_foot_pos_vector[N-1].translation[0] = copy(array_boundRF[N-1 + time_step][0])
        rf_foot_pos_vector[N-1].translation[1] = copy(array_boundRF[N-1 + time_step][1])
        rf_foot_pos_vector[N-1].translation[2] = copy(array_boundRF[N-1 + time_step][2])
        lf_foot_pos_vector[N-1].translation[0] = copy(array_boundLF[N-1 + time_step][0])
        lf_foot_pos_vector[N-1].translation[1] = copy(array_boundLF[N-1 + time_step][1])
        lf_foot_pos_vector[N-1].translation[2] = copy(array_boundLF[N-1 + time_step][2])
        residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
        residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
        foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
        foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
       
       
        terminalCostModel.removeCost("footReg1")
        terminalCostModel.removeCost("footReg2")
        terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
        terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
        a2 = time.time()
        
        while signal == False:
            a = time.time()
            mpc_signalv  = mpc_signal.read()
            mpc_signaldata =  np.ndarray(shape=(3,), dtype=np.int32, buffer=mpc_signalv)
            if(mpc_signaldata[0] == 4):
                b = time.time()
                signal = True
        while ok_ == False:
            mpc_signalv  = mpc_signal.read()
            mpc_signaldata =  np.ndarray(shape=(3,), dtype=np.int32, buffer=mpc_signalv)
            if mpc_signaldata[0] == 1:
                x_initv  = x_init.read()
                X = np.ndarray(shape=(49,), dtype=np.float64, buffer=x_initv)
                queue[:41] = X[:41]
                queue[41] = X[43]
                queue[42] = X[47]
                
                thread_manager[:] = [1, 1, 1]    
                
                xs_pca = []
                us_pca = []
                xs_pca_test = []
                
                while True:
                    if (thread_manager[0] == 0 and thread_manager[1] == 0 and  thread_manager[2] == 0):
                        for i in range(0, 60): #q, v, x in zip(q_pca, v_pca, x_pca):
                            xs_pca.append(np.concatenate([q_traj[i,:], v_traj[i,:], x_traj[i,:]]))
                            if i != 59:
                                us_pca.append(np.concatenate([a_traj[i,:], u_traj[i,:]]))
                        break
                
                for i in range(0, len(x0)):
                    x0[i] = copy(X[i])

                for i in range(0, len(q)):
                    q[i] = x0[i]    
                for i in range(0, len(qdot)):
                    qdot[i] = x0[i+len(q)]                    

                pinocchio.forwardKinematics(model.model, data, q, qdot)
                pinocchio.updateFramePlacements(model.model,data)
                pinocchio.centerOfMass(model.model, data, q, qdot, False)
                pinocchio.computeCentroidalMomentum(model.model,data,q,qdot)
                
                x0[41] = data.com[0][0] 
                x0[45] = data.com[0][1]
            
                x0[42] = data.vcom[0][0] 
                x0[46] = data.vcom[0][1]

                x0[44] = data.hg.angular[1] 
                x0[48] = data.hg.angular[0] 
                
                #xs_pca_test = queue[:] 
            
                problemWithRK4.x0 = x0
                ddp.th_stop = 0.000001
                c_start = time.time()
                css = ddp.solve(xs_pca, us_pca, 100, False, 0.0001)
                c_end = time.time()
                duration = (1e3 * (c_end - c_start))

                X = ddp.xs[1]
                desired_value.write(X)
                statemachine.write(np.array([1, 0, 0], dtype=np.int8))

                print("end")
                avrg_duration = duration
                min_duration = duration #min(duration)
                max_duration = duration #max(duration)
                print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
                print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, ddp.cost))
                print("xs")
                print(x0)
                print(ddp.xs[1])
                print(["mpc_cycle", mpc_cycle, time_step])
                #print(mpc_cycle)

                ok_ = True

                if time_step == 12:
                    traj = np.array(ddp.xs)[:,0:21]
                    vel_traj = np.array(ddp.xs)[:,21:41]
                    x_traj = np.array(ddp.xs)[:, 41:49]
                    u_traj = np.array(ddp.us)[:,20:24]
                    acc_traj = np.array(ddp.us)[:, 0:20]

                    crocs_data['Right']['x_inputs'].append(copy(ddp.xs[0][0:21]))
                    crocs_data['Right']['vel_trajs'].append(copy(vel_traj))
                    crocs_data['Right']['x_state'].append(copy(x_traj))
                    crocs_data['Right']['costs'].append(copy(ddp.cost))
                    crocs_data['Right']['iters'].append(copy(ddp.iter))
                    crocs_data['Right']['trajs'].append(copy(traj))
                    crocs_data['Right']['u_trajs'].append(copy(acc_traj))
                    crocs_data['Right']['acc_trajs'].append(copy(u_traj))
                    print(len(crocs_data['Right']['trajs']))
                    
                    with open('/home/jhk/ssd_mount/filename3.pkl', 'wb') as f:
                        pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("success")

                if time_step == total_time - 1:
                    time.sleep(0.002)
                    statemachine.write(np.array([2, 0, 0], dtype=np.int8))
                    time.sleep(1000)
                mpc_cycle = mpc_cycle + 1  
                

            elif mpc_signaldata[0] == 2:
                statemachine.write(np.array([2, 0, 0], dtype=np.int8))
                
            '''
            if time_step == 0:
                a
                print(x_initv)
            else:
                X = np.array(ddp.xs[1][0:21])
                X = np.append(X, ddp.xs[1][43]) 
                X = np.append(X, ddp.xs[1][47])
                queue[:] = X
            k = 0        
            mpc_signalv  = mpc_signal.read()
            mpc_signaldata =  np.ndarray(shape=(3,), dtype=np.int32, buffer=mpc_signalv)
            thread_manager[:] = copy(mpc_signaldata[:])
            if(thread_manager[2] == 1):
                statemachinedata[0] = 2
                statemachine.write(statemachinedata)
                signal = True
            '''
            '''
            if(thread_manager[0] == 0 and thread_manager[1] == 0 and  thread_manager[2] == 0):
                break
            '''

        
       

def print_heard_talkling(message):    
    if  message['data'] == 'stateestimation':
        talk.publish(roslibpy.Message({'data': 'stateestimation'}))

   
if __name__=='__main__':
    global talk
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_controller/launch/simulation.launch'])
    launch.start()
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    listener = roslibpy.Topic(client, '/tocabi/command', 'std_msgs/String')
    listener.subscribe(print_heard_talkling)
    talk = roslibpy.Topic(client, '/chatter', 'std_msgs/String')
    #talk.publish(roslibpy.Message({'data': 'Hello World!'}))
    #PCAlearning()
    talker()

