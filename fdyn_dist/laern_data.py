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
            torch.nn.Linear(in_features = input_size, out_features= 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features = 50, out_features = output_size)
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
    '''
    file_name = '/home/jhk/walkingdata/beforedata/fdyn/timestep='
    file_name2 = '_finish/timestep='
    file_name3 = file_name +str(time_step)+file_name2+str(time_step)+'_finish'
    print(file_name3)
    

    with open(file_name3, 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
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
    rbf_num = 47
    Phi = define_RBF(dof=19, nbStates =rbf_num, offset = 2, width = 1, T = timestep, coeff =47)

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

        for key in keys:
            w_trajs[key] = apply_RBF(trajs[key], Phi)
            w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
            w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        #w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        #w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/'
        file_name2 = 'Phi'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
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

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset1/'
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
        '''
        file_name2 = 'w_u_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pickle.dump(pca_u, open(file_name4,"wb"))
        file_name2 = 'w_acc_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pickle.dump(pca_acc, open(file_name4,"wb"))
        '''

        for key in keys:
            x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key], w_trajs_pca[key], test_size = 0.0001, random_state=1)
            _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.0001, random_state=1)
            #_,_, y_u_train[key], y_u_test[key] = train_test_split(x_inputs[key],w_u_trajs_pca[key], test_size = 0.1, random_state=1)
            #_,_, y_acc_train[key], y_acc_test[key] = train_test_split(x_inputs[key],w_acc_trajs_pca[key], test_size = 0.1, random_state=1)
            _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.0001, random_state=1)

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

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/'
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
        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset1/'
        file_name2 = 'Phi'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        #print(torch.load(file_name4))
        print(file_name4)
        Phi = pickle.load(open(file_name4,"rb"))

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset1/'
        file_name2 = 'x_inputs_train_'
        file_name3 = '.pt'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        x_inputs_train = torch.load(file_name4)
        file_name2 = 'x_inputs_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        x_inputs_test = torch.load(file_name4)
        file_name2 = 'y_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_test = torch.load(file_name4)
        file_name2 = 'y_vel_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_vel_test = torch.load(file_name4)
        file_name2 = 'y_x_test_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_x_test = torch.load(file_name4)
        file_name2 = 'y_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_train = torch.load(file_name4)
        file_name2 = 'y_vel_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_vel_train = torch.load(file_name4)
        file_name2 = 'y_x_train_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        y_x_train = torch.load(file_name4)

        file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/'
        file_name2 = 'w_trajs_pca_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pca = pickle.load(open(file_name4,'rb'))
        
        file_name2 = 'w_vel_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pca_vel = pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_x_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        pca_x= pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_u_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        #pca_u = pickle.load(open(file_name4,'rb'))
        file_name2 = 'w_acc_trajs_pca_'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
        #pca_acc = pickle.load(open(file_name4,'rb'))
        #explain =pca_x[key].explained_variance_ratio_
        #print(explain)
        #k = asdfasdf
    #print(y_train['Right'][0])
    #k = adsfsdfasdf
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
        num_epochs = 5
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_graph = []
        input_size = 43
        sequence_length = 1

        #model, optimizer = ipex.optimize(model, optimizer=optimizer)
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
                    print ('Epoch [{}/{}],  Loss: {:.6f}'
                       .format(epoch+1, num_epochs, loss.item()))
       
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 5
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        loss_graph = []
        sequence_length = 1
        #model1, optimizer1 = ipex.optimize(model1, optimizer=optimizer1)

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
                    print ('1Epoch [{}/{}],  Loss: {:.6f}'
                        .format(epoch+1, num_epochs, loss.item()))
   
        criterion = nn.MSELoss()
        lr = 0.001
        num_epochs = 5
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        loss_graph = []
        sequence_length = 1
        #model2, optimizer2 = ipex.optimize(model2, optimizer=optimizer2)

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
                    print ('2Epoch [{}/{}],  Loss: {:.6f}'
                       .format(epoch+1, num_epochs, loss.item()))
        
        file_name = '/home/jhk/ssd_mount/beforedata/cnn'
        file_name2 = '0_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3       
        torch.save(model.state_dict(), file_name4)
        file_name2 = '1_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3   
        torch.save(model1.state_dict(), file_name4)
        file_name2 = '2_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3   
        torch.save(model2.state_dict(), file_name4)
       
    else:
        file_name = '/home/jhk/ssd_mount/beforedata/cnn'
        file_name2 = '0_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3   
        model.load_state_dict(torch.load(file_name4))
        file_name2 = '1_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3   
        model1.load_state_dict(torch.load(file_name4))
        file_name2 = '2_'
        file_name3 = '.pkl'
        file_name4 = file_name  +file_name2+ str(time_step)+ file_name3   
        model2.load_state_dict(torch.load(file_name4))
    
    
    JJ = np.random.randint(x_inputs_test[key].shape[0])
    X = x_inputs_test['Right'][JJ][None,:]
    
    K = [ 0.01204123, -0.04866425,  0.82670249 ,-0.02236911,-0.01828107 , 0.01521162,
  0.99946687, -0.03219292,  0.1335083 , -0.48486977 , 1.23925489 ,-0.71781833,
 -0.08244633, -0.03223038,  0.13362632 ,-0.44887695 , 1.18784575 ,-0.70144142,
 -0.08262066,  0.018598,   -0.00480053 , 0.11080713 ,-0.16968124 ,-0.00951077,
 -0.10267765, -0.32088004,  0.11734803 ,-0.15215473 , 0.37951583 , 0.6371367,
 -0.21422519, -0.09421088, -0.26590537 ,-0.15064504 , 0.37541592 , 0.74677075,
 -0.34694619, -0.06874684, -0.26307327 , 0.04668285 , 0.01114374 ,-0.00276598,
 -0.10115297]
    print(X)
    for i in range(0, 43):
        X[0][i] = K[i]

    print(X)
    print("X")
    
    #
    X = X.reshape(1, 1, input_size).to(device)        
   
    ti = time.time()
    c = torch.tensor(X,dtype=torch.float32)
    w_traj = model.forward(c)
    w_traj = w_traj[0].detach().numpy()
    w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
    w_traj = w_traj.reshape(rbf_num,-1)
    traj1 = np.dot(Phi,w_traj)
    q_traj = traj1.flatten()
    t2 = time.time()  
    print(t2 - ti)

    ti = time.time()
    a = np.array(X[:])
    c = torch.tensor(a.reshape(1,1,input_size),dtype=torch.float32)
    w_traj_dot = model1.forward(c)
    w_traj_dot = w_traj_dot[0].detach().numpy()
    w_traj_dot = pca_vel['Right'].inverse_transform([w_traj_dot[None,:]])[0]
    w_traj_dot = w_traj_dot.reshape(rbf_num,-1)
    traj1 = np.dot(Phi,w_traj_dot)
    v_traj = traj1.flatten()
    t2 = time.time()
    acc_traj = np.subtract(traj1[1:60,:], traj1[0:59,:])/0.02
    t3 = time.time()
    print(t2 - ti)
    print(t3 - ti)

    ti = time.time()
    a = np.array(X[:])
    c = torch.tensor(a.reshape(1,1,input_size),dtype=torch.float32)
    w_traj_x = model2.forward(c)
    w_traj_x = w_traj_x[0].detach().numpy()
    w_traj_x = pca_x['Right'].inverse_transform([w_traj_x[None,:]])[0]
    w_traj_x = w_traj_x.reshape(rbf_num,-1)
    traj1 = np.dot(Phi,w_traj_x)
    x_traj = traj1.flatten()
    t2 = time.time()
    print(t2 - ti)
    u_traj = np.zeros([59, 4])
    for i in range(0, 59):
        u_traj[i][0] = (traj1[i + 1][2] - traj1[i][2])/0.02
        u_traj[i][1] = (traj1[i + 1][3] - traj1[i][3])/0.02
        u_traj[i][2] = (traj1[i + 1][6] - traj1[i][6])/0.02
        u_traj[i][3] = (traj1[i + 1][7] - traj1[i][7])/0.02

    q_pca = np.array(q_traj).reshape(60,21)
    v_pca = np.array(v_traj).reshape(60,20)
    x_pca = np.array(x_traj).reshape(60,8)
    acc_pca = np.array(acc_traj).reshape(59,20)
    u_pca = np.array(u_traj).reshape(59,4)

    xs_pca = []
    us_pca = []

    for q, v, x in zip(q_pca, v_pca, x_pca):
        xs_pca.append(np.concatenate([q, v, x]))
    for a, u in zip(acc_pca, u_pca):
        i = i + 1
        us_pca.append(np.concatenate([a, u]))

    xs_pca_test = x_inputs_test[key][JJ][None,:][0]

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
    
    
    for i in range(1, 50, 6):#learning_data_num):
        PCAlearning(i)
    k = adsfasdff
    print("start")
    #k = asdfasdfasd
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
    time_step = 23

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
   
    for i in range(0, N):
        if i == 0:
            array_boundRF_[i] = np.sum([array_boundRF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
        else:
            array_boundRF_[i] = np.sum([array_boundRF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
    
    for i in range(0, len(lines1_array)):
        for j in range(0, len(lines1_array[i])):
            if j == 0:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 1:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 2:
                array_boundLF[i].append(float(lines1_array[i][j]))

    for i in range(0, N):
        if i == 0:
            array_boundLF_[i] = np.sum([array_boundLF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
        else:
            array_boundLF_[i] = np.sum([array_boundLF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)

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

    for i in range(0, N):
        if i == 0:
            array_boundx_[i] = array_boundx[k3*i + time_step]
            array_boundy_[i] = array_boundy[k3*i + time_step]
        else:
            array_boundx_[i] = array_boundx[k3*(i) + time_step]
            array_boundy_[i] = array_boundy[k3*(i) + time_step]
    
    for i in range(0, len(lines3_array)):
        for j in range(0, len(lines3_array[i])):
            if j == 0:
                zmp_refx[i].append(float(lines3_array[i][j]))
            if j == 1:
                zmp_refy[i].append(float(lines3_array[i][j]))
           
    for i in range(0, N):
        if i == 0:
            zmp_refx_[i] = zmp_refx[k*i + time_step]
            zmp_refy_[i] = zmp_refy[k*i + time_step]
        else:
            zmp_refx_[i] = zmp_refx[k*(i)+ time_step]
            zmp_refy_[i] = zmp_refy[k*(i)+ time_step]

    f.close()
    f1.close()
    f2.close()

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
    q_init = [0, 0, 0.82473, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]

    e = 0
    for time1 in range(0, 1):
        for i in range(0, len(q)):    
            q[i] = xs_pca_test[i]
       
        state = crocoddyl.StateKinodynamic(model.model)
        actuation = crocoddyl.ActuationModelKinoBase(state)
        x0 = np.array([0.] * (state.nx + 8))
        u0 = np.array([0.] * (22))
        for i in range(0,len(q_init)):
            x0[i] = q[i]
       
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
        '''
        x0[41] = data.com[0][0]
        x0[43] = xs_pca_test[21]
        x0[45] = data.com[0][1]
        x0[47] = xs_pca_test[22]
        '''
        '''
        -0.45712259, -0.2939745,   0.01562487, -0.27835026,  0.60542595,  0.02710516,
        -0.00418088,  0.85618331, -1.83971559,  0.69214757,  0.54011833, -0.5761195,
        -0.00421453,  0.85235082, -1.70394928,  0.40980849,  0.6860741,  -0.57231448,
        0.00269425, -0.00275071,  
        '''
        xxx = [ 0.01204123, -0.04866425,  0.82670249, -0.02236911, -0.01828107 , 0.01521162,
  0.99946687, -0.03219292 , 0.1335083 , -0.48486977 , 1.23925489 ,-0.71781833,
 -0.08244633, -0.03223038 , 0.13362632 ,-0.44887695 , 1.18784575 ,-0.70144142,
 -0.08262066,  0.018598  , -0.00480053 , 0.11080713 ,-0.16968124 ,-0.00951077,
 -0.10267765, -0.32088004,  0.11734803 ,-0.15215473 , 0.37951583 , 0.6371367,
 -0.21422519, -0.09421088, -0.26590537 ,-0.15064504 , 0.37541592 , 0.74677075,
 -0.34694619, -0.06874684, -0.26307327 , 0.04668285 , 0.01114374 , 0.08497735,
  0.00250578, -0.00276598, -0.74720914 ,-0.02828383 ,-0.06704322, -0.10115297,
  0.98146212]
        
        for i in range(0, len(x0)):
            x0[i] = xxx[i]

        weight_quad_zmpx  = client.get_param("/dyros_practice/weight_quad_zmpx")
        weight_quad_zmpy  = client.get_param("/dyros_practice/weight_quad_zmpy")
        weight_quad_camx  = client.get_param("/dyros_practice/weight_quad_camx")
        weight_quad_camy  = client.get_param("/dyros_practice/weight_quad_camy")
        weight_quad_comx  = client.get_param("/dyros_practice/weight_quad_comx")
        weight_quad_comy  = client.get_param("/dyros_practice/weight_quad_comy")
        weight_quad_comz  = client.get_param("/dyros_practice/weight_quad_comz")
        weight_quad_rfx  = client.get_param("/dyros_practice/weight_quad_rfx")
        weight_quad_rfy  = client.get_param("/dyros_practice/weight_quad_rfy")
        weight_quad_rfz  = client.get_param("/dyros_practice/weight_quad_rfz")
        weight_quad_lfx  = client.get_param("/dyros_practice/weight_quad_lfx")
        weight_quad_lfy  = client.get_param("/dyros_practice/weight_quad_lfy")
        weight_quad_lfz  = client.get_param("/dyros_practice/weight_quad_lfz")
        weight_quad_rfroll  = client.get_param("/dyros_practice/weight_quad_rfroll")
        weight_quad_rfpitch  = client.get_param("/dyros_practice/weight_quad_rfpitch")
        weight_quad_rfyaw  = client.get_param("/dyros_practice/weight_quad_rfyaw")
        weight_quad_lfroll  = client.get_param("/dyros_practice/weight_quad_lfroll")
        weight_quad_lfpitch  = client.get_param("/dyros_practice/weight_quad_lfpitch")
        weight_quad_lfyaw  = client.get_param("/dyros_practice/weight_quad_lfyaw")
        weight_quad_camx = 2.9
        weight_quad_camy = 2.9
        weight_quad_zmp = np.array([10.0, 10.0])#([weight_quad_zmpx] + [weight_quad_zmpy])
        weight_quad_zmp1 = np.array([0.02, 0.02]) ##11
        weight_quad_cam = np.array([0.001, 0.001])#([weight_quad_camy] + [weight_quad_camx])
        weight_quad_upper = np.array([0.1, 0.1])
        weight_quad_com = np.array([5.0, 5.0, 1.0])#([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
        weight_quad_rf = np.array([5.0, 1.0, 30.0, 0.5, 0.5, 0.5])#np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
        weight_quad_lf = np.array([5.0, 1.0, 30.0, 0.5, 0.5, 0.5])#np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
        weight_quad_pelvis = np.array([weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
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
           
            #if i >= 2:
            runningCostModel_vector[i].addCost("stateReg1", stateBoundCost_vector1[i], 1.0)
           
            runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], 1.0)
            #runningCostModel_vector[i].addCost("stateReg2", stateBoundCost_vector2[i], 1.0)
            runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
            runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
           
            runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
            runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)

        traj_[43] = zmp_refx_[N-1][0]
        traj_[47] = zmp_refy_[N-1][0]
           
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
        terminalCostModel.addCost("stateReg1", stateBoundCost_vector1[N-1], 1.0)
        #terminalCostModel.addCost("stateReg2", stateBoundCost_vector2[N-1], 1.0)
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
       
        for i in range(0,N-1):  
            state_bounds[i].lb[0] = copy(array_boundx_[i][0])
            state_bounds[i].ub[0] = copy(array_boundx_[i][1])
            state_bounds[i].lb[1] = copy(array_boundy_[i][0])
            state_bounds[i].ub[1] = copy(array_boundy_[i][1])
            state_activations[i].bounds = state_bounds[i]
            stateBoundCost_vector[i].activation_ = state_activations[i]
           
            #runningCostModel_vector[i].removeCost("stateReg")
            #runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], 1.0)
           
            rf_foot_pos_vector[i].translation[0] = copy(array_boundRF_[i][0])
            rf_foot_pos_vector[i].translation[1] = copy(array_boundRF_[i][1])
            rf_foot_pos_vector[i].translation[2] = copy(array_boundRF_[i][2])
            lf_foot_pos_vector[i].translation[0] = copy(array_boundLF_[i][0])
            lf_foot_pos_vector[i].translation[1] = copy(array_boundLF_[i][1])
            lf_foot_pos_vector[i].translation[2] = copy(array_boundLF_[i][2])
            residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
            foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
           
            runningCostModel_vector[i].removeCost("footReg1")
            runningCostModel_vector[i].removeCost("footReg2")
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)  
           
           
        state_bounds[N-1].lb[0] = copy(array_boundx_[N-1][0])
        state_bounds[N-1].ub[0] = copy(array_boundx_[N-1][1])
        state_bounds[N-1].lb[1] = copy(array_boundy_[N-1][0])
        state_bounds[N-1].ub[1] = copy(array_boundy_[N-1][1])
        state_activations[N-1].bounds = state_bounds[N-1]
        stateBoundCost_vector[N-1].activation_ = state_activations[N-1]
        rf_foot_pos_vector[N-1].translation[0] = copy(array_boundRF_[N-1][0])
        rf_foot_pos_vector[N-1].translation[1] = copy(array_boundRF_[N-1][1])
        rf_foot_pos_vector[N-1].translation[2] = copy(array_boundRF_[N-1][2])
        lf_foot_pos_vector[N-1].translation[0] = copy(array_boundLF_[N-1][0])
        lf_foot_pos_vector[N-1].translation[1] = copy(array_boundLF_[N-1][1])
        lf_foot_pos_vector[N-1].translation[2] = copy(array_boundLF_[N-1][2])
        residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
        residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
        foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
        foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
       
       
        terminalCostModel.removeCost("footReg1")
        terminalCostModel.removeCost("footReg2")
        terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
        terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
       
        #terminalCostModel.removeCost("stateReg")
        #terminalCostModel.addCost("stateReg", stateBoundCost_vector[N-1], 1.0)
        print(xs_pca_test)
        print(x0)
        problemWithRK4.x0 = x0#xs_pca_test.detach().numpy() #xs[0]
        ddp.th_stop = 0.0001
        c_start = time.time()
        css = ddp.solve(xs_pca, us_pca, 100, False, 0.0001)
        c_end = time.time()
        duration = (1e3 * (c_end - c_start))

        #costs.cost.ref
        print("end")
        avrg_duration = duration
        min_duration = duration #min(duration)
        max_duration = duration #max(duration)
        print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
        print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, ddp.cost))


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
           
        with open('/home/jhk/ssd_mount/filename4.pkl', 'wb') as f:
	        pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("success")
   
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

