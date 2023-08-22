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
import sysv_ipc

#manager = multiprocessing.Manager()
#thread_manager = manager.list()
#X_manager = manager.dict()

global q_traj, v_traj, acc_traj, x_traj, u_traj
global p1, p2, p3, p4, p5
global PCA_1, PCA_2, PCA_3, PCA_4, PCA_5
global thread_manager
q_traj = multiprocessing.Array(ctypes.c_float, range(30*19))
v_traj = multiprocessing.Array(ctypes.c_float, range(30*18))
acc_traj= multiprocessing.Array(ctypes.c_float, range(30*18))
x_traj= multiprocessing.Array(ctypes.c_float, range(30*8))
u_traj = multiprocessing.Array(ctypes.c_float, range(30*4))
PCA_1 = True
PCA_2 = True
PCA_3 = True
PCA_4 = True
PCA_5 = True

class CShmReader :
    def __init__(self, key) :
        self.memory = sysv_ipc.SharedMemory(key)
 
    def doReadShm(self, sizex, sizey) :
        memory_value = self.memory.read()
        c = np.ndarray((sizex,sizey), dtype=np.int32, buffer=memory_value)
        return c

    def doReadShm1(self, sizex, sizey) :
        memory_value = self.memory.read()
        c = np.ndarray((sizex,sizey), dtype=np.float, buffer=memory_value)
        return c

    def doWriteShm(self, Input) :
        self.memory.write(Input)
        #print("c")
        #print(c)

def InversePCA(model, rbf_num, pca, Phi, X, thread_manager):
    global PCA_1
    while PCA_1:
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
            '''
            print("thread1")
            print(ti)
            print(t2)
            print(t2 - ti)
            '''
            thread_manager[0] = 1
        elif thread_manager[0] == 2:
            PCA_1 = False


def InversePCA1(model, rbf_num, pca, Phi, X, thread_manager):
    global PCA_2
    while PCA_2:
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
            '''
            print("thread2")
            print(ti)
            print(t2)
            print(t2 - ti)
            '''
            thread_manager[1] = 1
        elif thread_manager[1] == 2:
            PCA_2 = False

def InversePCA2(model, rbf_num, pca, Phi, X, thread_manager):
    global PCA_3
    while PCA_3:
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
            '''
            print("thread3")
            print(ti)
            print(t2)
            print(t2 - ti)
            '''
            thread_manager[2] = 1
        elif thread_manager[2] == 2:
            PCA_3 = False

def InversePCA3(model, rbf_num, pca, Phi, X, thread_manager):
    global PCA_4
    while PCA_4:
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
            #print("thread4")
           #print(ti)
           # print(t2)
           # print(t2 - ti)
            thread_manager[3] = 1
        elif thread_manager[3] == 2:
            PCA_4 = False

def InversePCA4(model, rbf_num, pca, Phi, X, thread_manager):
    global PCA_5
    while PCA_5:
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
            #print("thread5")
            #print(ti)
            #print(t2)
            #print(t2 - ti)
            thread_manager[4] = 1
        elif thread_manager[4] == 2:
            PCA_5 = False
    
class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = x.clone().detach()#torch.tensor(x,dtype=torch.float32)
        self.y = y.clone().detach()#torch.tensor(y,dtype=torch.float32)
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
    global thread_manager
    global p1, p2, p3, p4, p5

    shared_x = CShmReader(100)
    shared_u = CShmReader(101)
    ddp_start = CShmReader(102)
    ddp_finish = CShmReader(102)
    ddp_start1 = CShmReader(103)
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

    with open('/home/jhk/data/mpc/filename23.pkl', 'rb') as f:
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
    num_desired = 400
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
            newrow1 = np.zeros(18)
            raw_acc_trajs[i] = numpy.vstack([raw_acc_trajs[i], newrow1])
        for i in range(len(raw_u_trajs)):
            newrow = np.zeros(4)
            raw_u_trajs[i] = numpy.vstack([raw_u_trajs[i],newrow])
        u_trajs[key] = np.array(raw_u_trajs)
        acc_trajs[key] = np.array(raw_acc_trajs)

    timestep = 30
    rbf_num = 40
    
    Phi = define_RBF(dof=19, nbStates = rbf_num, offset = 1, width = 1, T = timestep, coeff =1)

    for key in keys:
        w_trajs[key] = apply_RBF(trajs[key], Phi)
        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)

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
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(f'{device} is available')
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

        torch.save(model.state_dict(), '/home/jhk/data/mpc/cnn.pkl')
        torch.save(model1.state_dict(), '/home/jhk/data/mpc/cnn1.pkl')
        torch.save(model2.state_dict(), '/home/jhk/data/mpc/cnn2.pkl')
        torch.save(model3.state_dict(), '/home/jhk/data/mpc/cnn3.pkl')
        torch.save(model4.state_dict(), '/home/jhk/data/mpc/cnn4.pkl')
        
    else:
        model.load_state_dict(torch.load('/home/jhk/data/mpc/cnn.pkl'))
        model1.load_state_dict(torch.load('/home/jhk/data/mpc/cnn1.pkl'))
        model2.load_state_dict(torch.load('/home/jhk/data/mpc/cnn2.pkl'))
        model3.load_state_dict(torch.load('/home/jhk/data/mpc/cnn3.pkl'))
        model4.load_state_dict(torch.load('/home/jhk/data/mpc/cnn4.pkl'))

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
    
    loop = 0.0

    ddpdata = ddp_start.doReadShm(1,4)
    ddpdata = []
    ddpdata = np.ndarray((1,4), dtype=np.int32, buffer=ddp_start.memory.read())
    finish = np.zeros((1,4), dtype=np.int32)
    newstart = np.zeros((1,4), dtype=np.int32)

    while True:
        a = np.ndarray((1,4), dtype=np.int32, buffer=ddp_start.memory.read())[0][0]
        if a == 1:
            JJ = np.random.randint(x_inputs_test[key].shape[0])
            X = x_inputs_test['Right'][JJ][None,:]
            X = X.reshape(1, sequence_length, input_size).to(device)

            queue[:] = X.numpy()[0][0]

            if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
                for i in range(0,5):
                    thread_manager[i] = 0

            while(thread_manager[0] == 0 or thread_manager[1] == 0 or thread_manager[2] == 0 or thread_manager[3] == 0 or thread_manager[4] ==  0):
                if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
                    break
                    
            
            xs_pca_test = x_inputs_test[key][JJ][None,:][0]
        
            q_pca = np.array(q_traj).reshape(30,19)
            v_pca = np.array(v_traj).reshape(30,18)
            x_pca = np.array(x_traj).reshape(30,8)
            acc_pca = np.array(acc_traj).reshape(30,18)
            u_pca = np.array(u_traj).reshape(30,4)
        
            xs_pca = np.empty((0,45))
            us_pca = np.empty((0,22))

            for q, v, x in zip(q_pca, v_pca, x_pca):
                xs_pca = np.append(xs_pca, np.array([np.concatenate([q, v, x])]), axis=0)
            for a, u in zip(acc_pca, u_pca):
                us_pca = np.append(us_pca, np.array([np.concatenate([a, u])]), axis=0)

            for i  in range(0, 19):
                us_pca[29][i] = xs_pca_test[i]

            if loop == 0:
                shared_x.doWriteShm(xs_pca)
                shared_u.doWriteShm(us_pca)
            else:
                shared_x.memory.write(xs_pca)
                shared_u.memory.write(us_pca)

            newstart[0] = 1
            ddp_start1.memory.write(newstart)
            ddp_finish.memory.write(finish)

            loop = loop + 1
            thread_manager[0] = 0
            thread_manager[1] = 0
            thread_manager[2] = 0
            thread_manager[3] = 0
            thread_manager[4] = 0
        elif a == 0:
            finish[0] = 0
            #ddp_finish.memory.write(finish)
        elif a == 2:
            break
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    PCAlearning()

