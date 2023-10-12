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

#manager = multiprocessing.Manager()
#thread_manager = manager.list()
#X_manager = manager.dict()

global q_traj, v_traj, acc_traj, x_traj, u_traj
q_traj = multiprocessing.Array(ctypes.c_float, range(30*19))
v_traj = multiprocessing.Array(ctypes.c_float, range(30*18))
acc_traj= multiprocessing.Array(ctypes.c_float, range(30*18))
x_traj= multiprocessing.Array(ctypes.c_float, range(30*8))
u_traj = multiprocessing.Array(ctypes.c_float, range(30*4))

def InversePCA(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[0] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
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
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
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
            torch.nn.Linear(in_features = 19, out_features= 30),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 30, out_features = 40)
            )

    def forward(self, x):
        #out = self.layer1(x)
        #out = self.layer2(x)
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

def PCAlearning():
    global xs_pca_test
    global xs_pca
    global us_pca

    learn_type = 1
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

    with open('/home/jhk/ssd_mount/afterdata/integral/filename4.pkl', 'rb') as f:
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
    num_desired = 11130
    keys = ['Right']
    num_data = dict()

    for key in keys:
        x_inputs[key] = []
        x_inputs[key] = np.array(database[key]['x_inputs'])[:num_desired]
        trajs[key] = np.array(database[key]['trajs'])[:num_desired]
        vel_trajs[key] = np.array(database[key]['vel_trajs'])[:num_desired]
        x_trajs[key] = np.array(database[key]['x_state'])[:num_desired]
        foot_poses[key] = database[key]['foot_poses'][:num_desired]
        num_data[key] = len(foot_poses[key])

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

    timestep = 60
    rbf_num = 40
    
    Phi = define_RBF(dof=19, nbStates = rbf_num, offset = 1, width = 1, T = timestep, coeff =1)
    
    for key in keys:
        w_trajs[key] = apply_RBF(trajs[key], Phi)
        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)

    cost_temp = 0

    aa_ = w_trajs[key][0].reshape(rbf_num,-1)
    bb_ = w_vel_trajs[key][0].reshape(rbf_num,-1)
    dd_ = w_acc_trajs[key][0].reshape(rbf_num,-1)
    cc_ = w_u_trajs[key][0].reshape(rbf_num,-1)
    ee_ = w_x_trajs[key][0].reshape(rbf_num,-1)

    for i in range(1,100):
        for j in range(1,100):
            for k in range(1,100):
                for f in range(1,30):
                    rbf_num = f
                    Phi = define_RBF(dof=19, nbStates = f, offset = i, width = j, T = timestep, coeff=k)

                    for key in keys:
                        w_trajs[key] = apply_RBF(trajs[key], Phi)
                        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
                        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
                        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
                        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)
                    
                    aa_ = w_trajs[key][0].reshape(rbf_num,-1)
                    '''
                    print(np.shape(Phi))
                    print(np.shape(aa_))
                    print(trajs[key][0][0])
                    print(aa_)
                    print(Phi)
                    print(np.dot(Phi[0], aa_))
                    #print(trajs[key][0][0] )
                    print(np.linalg.norm(trajs[key][0][0] - np.dot(Phi[0], aa_)))
                    '''
                    cost = 0
                    for num in range(0, 60):
                        aa_ = w_trajs[key][10000].reshape(rbf_num,-1)
                        bb_ = w_vel_trajs[key][10000].reshape(rbf_num,-1)
                        dd_ = w_acc_trajs[key][10000].reshape(rbf_num,-1)
                        cc_ = w_u_trajs[key][10000].reshape(rbf_num,-1)
                        ee_ = w_x_trajs[key][10000].reshape(rbf_num,-1)
                     
                        if num  == 0:
                            cost = np.linalg.norm(trajs[key][10000][num] - np.dot(Phi[num], aa_)) *np.linalg.norm(trajs[key][10000][num] - np.dot(Phi[num], aa_))  + np.linalg.norm(vel_trajs[key][10000][num] - np.dot(Phi[num], bb_)) * np.linalg.norm(vel_trajs[key][10000][num] - np.dot(Phi[num], bb_)) + np.linalg.norm(acc_trajs[key][10000][num] - np.dot(Phi[num], dd_)) * np.linalg.norm(acc_trajs[key][10000][num] - np.dot(Phi[num], dd_)) + np.linalg.norm(u_trajs[key][10000][num] - np.dot(Phi[num], cc_)) * np.linalg.norm(u_trajs[key][10000][num] - np.dot(Phi[num], cc_)) + np.linalg.norm(x_trajs[key][10000][num] - np.dot(Phi[num], ee_)) * np.linalg.norm(x_trajs[key][10000][num] - np.dot(Phi[num], ee_))
                        else:
                            cost = cost+np.linalg.norm(trajs[key][10000][num] - np.dot(Phi[num], aa_)) *np.linalg.norm(trajs[key][10000][num] - np.dot(Phi[num], aa_))  + np.linalg.norm(vel_trajs[key][10000][num] - np.dot(Phi[num], bb_)) * np.linalg.norm(vel_trajs[key][10000][num] - np.dot(Phi[num], bb_)) + np.linalg.norm(acc_trajs[key][10000][num] - np.dot(Phi[num], dd_)) * np.linalg.norm(acc_trajs[key][10000][num] - np.dot(Phi[num], dd_)) + np.linalg.norm(u_trajs[key][10000][num] - np.dot(Phi[num], cc_)) * np.linalg.norm(u_trajs[key][10000][num] - np.dot(Phi[num], cc_)) + np.linalg.norm(x_trajs[key][10000][num] - np.dot(Phi[num], ee_)) * np.linalg.norm(x_trajs[key][10000][num] - np.dot(Phi[num], ee_))
                     
                    if i  == 1 and j == 1 and k == 1 and f ==1:
                        cost_temp = cost
                        print("start")
                    if(cost < cost_temp):
                        print("i")
                        print([i, j, k, f, cost])
                        cost_temp = cost
                        '''
                        print("j")
                        print(j)
                        print("k")
                        print(k)
                        print("f")
                        print(f)
                        print(cost)
                        '''
                        '''
                        print(trajs[key][0])
                        print(len(w_vel_trajs[key][0]))
                        print(vel_trajs[key][0][0])
                        print(np.dot(Phi[0], bb_))
                        
                        print("aa")
                        print(len(w_trajs[key][0]))
                        print(trajs[key][0][0])
                        print(np.dot(Phi[0], aa_))

                        print("cc")
                        print(len(w_u_trajs[key][0]))
                        print(u_trajs[key][0][0])
                        print(np.dot(Phi[0], cc_))

                        print("dd")
                        print(len(w_acc_trajs[key][0]))
                        print(acc_trajs[key][0][0])
                        print(np.dot(Phi[0], dd_))
                        '''
    k = b

    for key in keys:
        pca[key] = PCA(n_components = int(rbf_num))
        w_trajs_pca[key] = pca[key].fit_transform(w_trajs[key])

        pca_vel[key] = PCA(n_components=int(rbf_num))
        w_vel_trajs_pca[key] = pca_vel[key].fit_transform(w_vel_trajs[key])

        pca_x[key] = PCA(n_components= int(rbf_num))
        w_x_trajs_pca[key] = pca_x[key].fit_transform(w_x_trajs[key])

        pca_acc[key] = PCA(n_components=int(rbf_num))
        w_acc_trajs_pca[key] = pca_acc[key].fit_transform(w_acc_trajs[key])

        pca_u[key] = PCA(n_components=int(rbf_num))
        w_u_trajs_pca[key] = pca_u[key].fit_transform(w_u_trajs[key])

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

    for key in keys:
        x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key], w_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_u_train[key], y_u_test[key] = train_test_split(x_inputs[key],w_u_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_acc_train[key], y_acc_test[key] = train_test_split(x_inputs[key],w_acc_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.1666, random_state=1)

        x_inputs_train[key] = torch.FloatTensor(x_inputs_train[key])
        x_inputs_test[key] = torch.FloatTensor(x_inputs_test[key])
        y_test[key] = torch.FloatTensor( (y_test[key]))
        y_vel_test[key] = torch.FloatTensor( (y_vel_test[key]))
        y_u_test[key] = torch.FloatTensor( (y_u_test[key]))
        y_acc_test[key] = torch.FloatTensor( (y_acc_test[key]))
        y_x_test[key] = torch.FloatTensor( (y_x_test[key]))
        y_train[key] = torch.FloatTensor( (y_train[key]))
        y_vel_train[key] = torch.FloatTensor( (y_vel_train[key]))
        y_u_train[key] = torch.FloatTensor( (y_u_train[key]))
        y_acc_train[key] = torch.FloatTensor( (y_acc_train[key]))
        y_x_train[key] = torch.FloatTensor( (y_x_train[key]))
    
    device = 'cpu'
    train_y = timeseries(x_inputs_train[key], y_train[key])
    test_y = timeseries(x_inputs_test[key], y_test[key])
    train_yvel = timeseries(x_inputs_train[key], y_vel_train[key])
    test_yvel = timeseries(x_inputs_test[key], y_vel_test[key])
    train_yacc = timeseries(x_inputs_train[key], y_acc_train[key])
    test_yacc = timeseries(x_inputs_test[key], y_acc_test[key])
    train_yu = timeseries(x_inputs_train[key], y_u_train[key])
    test_yu = timeseries(x_inputs_test[key], y_u_test[key])
    train_yx = timeseries(x_inputs_train[key], y_x_train[key])
    test_yx = timeseries(x_inputs_test[key], y_x_test[key])

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_y, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_y, batch_size=batch_size, shuffle=True)
    train_vel_loader = torch.utils.data.DataLoader(dataset=train_yvel, batch_size=batch_size, shuffle=True)
    test_vel_loader = torch.utils.data.DataLoader(dataset=test_yvel, batch_size=batch_size, shuffle=True)
    train_acc_loader = torch.utils.data.DataLoader(dataset=train_yacc, batch_size=batch_size, shuffle=True)
    test_acc_loader = torch.utils.data.DataLoader(dataset=test_yacc, batch_size=batch_size, shuffle=True)
    train_u_loader = torch.utils.data.DataLoader(dataset=train_yu, batch_size=batch_size, shuffle=True)
    test_u_loader = torch.utils.data.DataLoader(dataset=test_yu, batch_size=batch_size, shuffle=True)
    train_x_loader = torch.utils.data.DataLoader(dataset=train_yx, batch_size=batch_size, shuffle=True)
    test_x_loader = torch.utils.data.DataLoader(dataset=test_yx, batch_size=batch_size, shuffle=True)

    #q
    input_size = 19
    sequence_length = 1
    num_layers = 5
    hidden_size = rbf_num

    model = CNN(input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)
        
    #qdot
    input_size = 19
    sequence_length = 1
    num_layers = 10
    hidden_size = rbf_num

    model1 = CNN(input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)

    #x
    input_size = 19
    sequence_length = 1
    num_layers = 5
    hidden_size = rbf_num

    model2 = CNN(input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)

    #acc
    input_size = 19
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

    model.train()
    model1.train()
    model2.train()
    model3.train()
    model4.train()

    if learn_type == 0:
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 5
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_graph = []

        for epoch in range(num_epochs):
            for data in train_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model(X)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 1 == 0:
                    print ('Epoch [{}/{}],  Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, loss.item()))
        
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 20
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        loss_graph = []

        for epoch in range(num_epochs):
            for data in train_vel_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model1(X)
                loss = criterion(out, target)

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

            if epoch % 1 == 0:
                print ('1Epoch [{}/{}],  Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, loss.item()))
    
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        loss_graph = []

        for epoch in range(num_epochs):
            for data in train_x_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model2(X)
                loss = criterion(out, target)

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                if epoch % 1 == 0:
                    print ('2Epoch [{}/{}],  Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, loss.item()))
        
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer3 = optim.Adam(model3.parameters(), lr=lr)
        loss_graph = []

        for epoch in range(num_epochs):
            for data in train_acc_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model3(X)
                loss = criterion(out, target)

                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()

            if epoch % 1 == 0:
                print ('3Epoch [{}/{}],  Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))

        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 50
        optimizer4 = optim.Adam(model4.parameters(), lr=lr)
        loss_graph = []

        for epoch in range(num_epochs):
            for data in train_u_loader:
                seq, target = data
                X = seq.reshape(batch_size, sequence_length, input_size).to(device)
                out = model(X)
                loss = criterion(out, target)

                optimizer4.zero_grad()
                loss.backward()
                optimizer4.step()

            if epoch % 1 == 0:
                print ('4Epoch [{}/{}],  Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))

        torch.save(model.state_dict(), '/home/jhk/ssd_mount/cnn.pkl')
        torch.save(model1.state_dict(), '/home/jhk/ssd_mount/cnn1.pkl')
        torch.save(model2.state_dict(), '/home/jhk/ssd_mount/cnn2.pkl')
        torch.save(model3.state_dict(), '/home/jhk/ssd_mount/cnn3.pkl')
        torch.save(model4.state_dict(), '/home/jhk/ssd_mount/cnn4.pkl')
        
    else:
        model.load_state_dict(torch.load('/home/jhk/ssd_mount/cnn.pkl'))
        model1.load_state_dict(torch.load('/home/jhk/ssd_mount/cnn1.pkl'))
        model2.load_state_dict(torch.load('/home/jhk/ssd_mount/cnn2.pkl'))
        model3.load_state_dict(torch.load('/home/jhk/ssd_mount/cnn3.pkl'))
        model4.load_state_dict(torch.load('/home/jhk/ssd_mount/cnn4.pkl'))

    '''
    JJ = np.random.randint(x_inputs_test[key].shape[0])
    X = x_inputs_test['Right'][JJ][None,:]
    X = X.reshape(1, sequence_length, input_size).to(device)        
    
    thread_manager1 = []
    for i in range(0,5):
        thread_manager1.append(0)

    thread_manager = multiprocessing.Array(ctypes.c_int, thread_manager1)
    queue = multiprocessing.Array(ctypes.c_float, X.numpy()[0][0].tolist())
    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
    p1 = multiprocessing.Process(target=InversePCA, args=(model, rbf_num, pca, Phi, queue, thread_manager))
    p2 = multiprocessing.Process(target=InversePCA1, args=(model1, rbf_num, pca_vel, Phi, queue, thread_manager))
    p3 = multiprocessing.Process(target=InversePCA2, args=(model2, rbf_num, pca_x, Phi, queue, thread_manager))
    p4 = multiprocessing.Process(target=InversePCA3, args=(model3, rbf_num, pca_acc, Phi, queue, thread_manager))
    p5 = multiprocessing.Process(target=InversePCA4, args=(model4, rbf_num, pca_u, Phi, queue, thread_manager))
   
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()  
    
    JJ = np.random.randint(x_inputs_test[key].shape[0])
    X = x_inputs_test['Right'][JJ][None,:]
    X = X.reshape(1, sequence_length, input_size).to(device)
    time.sleep(3)
    queue[:] = X.numpy()[0][0]

    if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
        for i in range(0,5):
            thread_manager[i] = 0

    while(thread_manager[0] == 0 or thread_manager[1] == 0 or thread_manager[2] == 0 or thread_manager[3] == 0 or thread_manager[4] ==  0):
        if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
            break
    '''
    '''
    for i in range(0, 1):
        tic = time.time()
        w_traj = model(X)
        w_traj = w_traj[0].detach().numpy()
        w_traj = pca[key].inverse_transform([w_traj[None,:]])[0]
        w_traj = w_traj.reshape(rbf_num,-1)
        tic1 = time.time()
        w_traj_dot = model1(X)
        w_traj_dot = w_traj_dot[0].detach().numpy()
        w_traj_dot = pca_vel[key].inverse_transform([w_traj_dot[None,:]])[0]
        w_traj_dot = w_traj_dot.reshape(rbf_num,-1)
        tic2 = time.time()
        w_traj_x = model2(X)
        w_traj_x = w_traj_x[0].detach().numpy()
        w_traj_x = pca_x[key].inverse_transform([w_traj_x[None,:]])[0]
        w_traj_x = w_traj_x.reshape(rbf_num,-1)
        tic3 = time.time()
        w_traj_acc = model3(X)
        w_traj_acc = w_traj_acc[0].detach().numpy()
        w_traj_acc = pca_acc[key].inverse_transform([w_traj_acc[None,:]])[0]
        w_traj_acc = w_traj_acc.reshape(rbf_num,-1)
        tic4 = time.time()
        w_traj_u = model4(X)
        w_traj_u = w_traj_u[0].detach().numpy()
        w_traj_u = pca_u[key].inverse_transform([w_traj_u[None,:]])[0]
        w_traj_u = w_traj_u.reshape(rbf_num,-1)
        tic5 = time.time()
        traj = np.dot(Phi,w_traj)
        traj_vel = np.dot(Phi,w_traj_dot)
        traj_acc = np.dot(Phi,w_traj_acc)
        traj_u = np.dot(Phi,w_traj_u)
        traj_x = np.dot(Phi,w_traj_x)
        toc = time.time()
        tt = tic - toc
        time.sleep(1)
    
        print(tic5 - tic4)
        print(tic4 - tic3)
        print(tic3 - tic2)
        print(tic2 - tic1)
        print(tic1 - tic)
    '''
    #traj = np.dot(Phi,q_traj)
    #traj_vel = np.dot(Phi,v_traj)
    #traj_x = np.dot(Phi,x_traj)
    #traj_acc = np.dot(Phi,acc_traj)
    #traj_u = np.dot(Phi,u_traj)
    
    q_pca = np.array(q_traj).reshape(30,19)
    v_pca = np.array(v_traj).reshape(30,18)
    x_pca = np.array(x_traj).reshape(30,8)
    acc_pca = np.array(acc_traj).reshape(30,18)
    u_pca = np.array(u_traj).reshape(30,4)
    '''
    print(acc_pca)
    q_pca = traj
    v_pca = traj_vel
    x_pca = traj_x
    acc_pca = traj_acc
    u_pca = traj_u
    print(acc_pca)
    '''
    xs_pca = []
    us_pca = []

    for q, v, x in zip(q_pca, v_pca, x_pca):
        xs_pca.append(np.concatenate([q, v, x]))
        
    for a, u in zip(acc_pca, u_pca):
        us_pca.append(np.concatenate([a, u]))
    del us_pca[-1]

    xs_pca_test = x_inputs_test[key][JJ][None,:][0]#.detach().numpy()[0]
    
    
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")
    N = 30
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    PCAlearning()
    '''
    for i in range(0,len(lines)):
        if lines[i].strip('\n') == 'walking_tick':
            loop = loop + 1
        if bool_u == 1:
            count_u_temp = count_u_temp + 1
            if count_u2 == 0:
                array_u[count_u].append(float(lines[i].strip('\n').strip(str(count_u)).strip('\t').strip(",").strip("ustate")))
            else:
                array_u[count_u].append(float(lines[i].strip('\n').strip(str(count_u)).strip('\t').strip(",")))
            count_u2 = count_u2 + 1
            if count_u_temp == 4:
                    count_u = count_u + 1
                    count_u2 = 0
                    count_u_temp = 0
                    bool_u = 0

        if lines[i].strip('\n').strip('\t') == "u" or bool_qdot == 1:
            if count_qddot2 == 29:
                count_qddot2 = 0

            bool_qdot = 1
            count_qddot_temp = count_qddot_temp + 1
            if count_qddot_temp > 1:
                array_qddot[count_qddot].append(float(lines[i].strip('\n').strip(str(count_qddot2)).strip('\t').strip(",")))
            
            if count_qddot_temp == 19:
                bool_qdot = 0
                count_qddot = count_qddot + 1
                count_qddot2 = count_qddot2 + 1
                count_qddot_temp = 0
                bool_u = 1

        if(i >= 6):
            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] >= 0 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 19:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 0:
                    array_q[count_q].append(float(lines[i].strip('\n').strip(str(count_q_temp)).strip('\t').strip(",")))
                else:
                    array_q[count_q].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 18:
                    count_q = count_q + 1
                    count_q_temp = count_q_temp + 1
                    if count_q_temp == 30:
                        count_q_temp = 0

            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] > 19 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 38:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 20:
                    array_qdot[count_qdot].append(float(lines[i].strip('\n').strip(str(count_qdot_temp)).strip('\t').strip(",")))
                else:
                    array_qdot[count_qdot].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 37:
                    count_qdot = count_qdot + 1
                    count_qdot_temp = count_qdot_temp + 1
                    if count_qdot_temp == 30:
                        count_qdot_temp = 0

            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] > 38 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 47:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 39:
                    array_xstate[count_xstate].append(float(lines[i].strip('\n').strip(str(count_xstate_temp)).strip('\t').strip(",")))
                else:
                    array_xstate[count_xstate].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 46:
                    count_xstate = count_xstate + 1
                    count_xstate_temp = count_xstate_temp + 1
                    if count_xstate_temp == 30:
                        count_xstate_temp = 0
    '''
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

