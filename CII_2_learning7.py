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

def InversePCA(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[0] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model.forward(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
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
            w_traj = pca['right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
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
            w_traj = pca['right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
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
            w_traj = pca['right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
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
            w_traj = pca['right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
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
            torch.nn.Linear(in_features = 19, out_features= 8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 8, out_features = 40)
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(x)
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
    learn_type = 0
    database = dict()
    database['left'] = dict()
    database['right'] = dict()

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
    keys = ['right']
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
    X = x_inputs_test['right'][JJ][None,:]
    X = X.reshape(1, sequence_length, input_size).to(device)        
    
    traj = np.zeros(19)
    traj_vel = np.zeros(18)
    traj_x = np.zeros(8)
    traj_acc = np.zeros(18)
    traj_u = np.zeros(4)

    thread_manager1 = []
    for i in range(0,5):
        thread_manager1.append(0)

    thread_manager = multiprocessing.Array(ctypes.c_int, thread_manager1)
    queue = multiprocessing.Array(ctypes.c_float, X.numpy()[0][0])
    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    '''
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
    X = x_inputs_test['right'][JJ][None,:]
    X = X.reshape(1, sequence_length, input_size).to(device)
    
    time.sleep(5)
    queue = multiprocessing.Array(ctypes.c_float, X.numpy()[0][0])
    print(thread_manager)
    if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
        for i in range(0,5):
            thread_manager[i] = 0

    while(thread_manager[0] == 0 or thread_manager[1] == 0 or thread_manager[2] == 0 or thread_manager[3] == 0 or thread_manager[4] ==  0):
        if(thread_manager[0] == 1 and thread_manager[1] == 1 and thread_manager[2] == 1 and thread_manager[3] == 1 and thread_manager[4] == 1):
            break
    

    timer.sleep(5)
    '''
    for i in range(0, 30):
        print(i)
        JJ = np.random.randint(x_inputs_test[key].shape[0])
        print(JJ)
        X = x_inputs_test['right'][JJ][None,:]
        X = X.reshape(1, sequence_length, input_size).to(device)   

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

    q_pca = traj
    v_pca = traj_vel
    x_pca = traj_x
    acc_pca = traj_acc
    u_pca = traj_u
    xs_pca = []
    us_pca = []
    for q, v, x in zip(q_pca, v_pca, x_pca):
        xs_pca.append(np.concatenate([q, v, x]))
        
    for a, u in zip(acc_pca, u_pca):
        us_pca.append(np.concatenate([a, u]))
    del us_pca[-1]
    xs_pca_test = x_inputs_test[key][JJ][None,:][0]#.detach().numpy()[0]
    '''
    
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")
    f = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f2 = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f1 = open("/home/jhk/data/mpc/3_tocabi_.txt", 'r')

    f6 = open("/home/jhk/data/mpc/6_tocabi_py1.txt", 'r')
    f5 = open("/home/jhk/data/mpc/5_tocabi_py1.txt", 'r')

    f3 = open("/home/jhk/data/mpc/5_tocabi_py.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_py.txt", 'w')

    lines = f.read().split(' ')
    lines1 = f2.readlines()
    lines2 = f1.readlines()
    lines3 = f6.readlines()
    loop = 0
    count_q = 0
    count_qdot = 0
    count_qddot = 0
    count_qddot2 = 0
    count_bound = 0
    count_u = 0
    count_u2 = 0
    count_xstate = 0
    count_q_temp = 0
    count_qdot_temp = 0
    count_xstate_temp = 0
    bool_qdot = 0
    bool_u = 0
    count_qddot_temp = 0
    count_u_temp = 0
    count_bound2 = 0
    array_qdot = [[] for i in range(int(len(lines1)/208) * 30)]
    array_q = [[] for i in range(int(len(lines1)/208) * 30)]
    array_xstate = [[] for i in range(int(len(lines1)/208) * 30)]
    array_u = [[] for i in range(int(len(lines1)/208)*29)]
    array_qddot = [[] for i in range(int(len(lines1)/208)*29)]

    array_boundx = [[] for i in range(int(len(lines1)/208) * 30)]
    array_boundy = [[] for i in range(int(len(lines1)/208) * 30)]

    array_boundRF = [[] for i in range(int(len(lines1)/208) * 30)]
    array_boundLF = [[] for i in range(int(len(lines1)/208) * 30)]

    array_qdot1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_q1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_xstate1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_u1 = [[] for i in range(int(len(lines3)/208)*29)]
    array_qddot1 = [[] for i in range(int(len(lines3)/208)*29)]

    bool_q = 0

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
    lines1_array = []
    for i in range(0, len(lines3)):
        lines1_array.append(lines3[i].split())

    lines2_array = []
    for i in range(0, len(lines2)):
        lines2_array.append(lines2[i].split()) 

    loop = 0
    count_q = 0
    count_qdot = 0
    count_qddot = 0
    count_qddot2 = 0
    count_bound = 0
    count_u = 0
    count_u2 = 0
    count_xstate = 0
    count_q_temp = 0
    count_qdot_temp = 0
    count_xstate_temp = 0
    bool_qdot = 0
    bool_u = 0
    count_qddot_temp = 0
    count_u_temp = 0
    count_bound2 = 0

    for i in range(0, len(lines1_array)):
        if len(lines1_array[i]) == 21:
            for j in range(0,19):
                array_q1[count_q].append(float(lines1_array[i][j].strip(',')))
                if j == 18:
                    count_q = count_q + 1 
        if len(lines1_array[i]) == 19:         
            for j in range(0,18):
                array_qddot1[count_qddot].append(float(lines1_array[i][j].strip(',')))
                if j == 17:
                    count_qddot = count_qddot + 1 
        if len(lines1_array[i]) == 20:         
            for j in range(0,18):
                array_qdot1[count_qdot].append(float(lines1_array[i][j].strip(',')))
                if j == 17:
                    count_qdot = count_qdot + 1  
        if len(lines1_array[i]) == 8:         
            for j in range(0,8):
                array_xstate1[count_xstate].append(float(lines1_array[i][j].strip(',')))
                if j == 7:
                    count_xstate = count_xstate + 1 
        if len(lines1_array[i]) == 4:         
            for j in range(0,4):
                array_u1[count_u].append(float(lines1_array[i][j].strip(',')))
                if j == 3:
                    count_u = count_u + 1 

    for i in range(0, len(lines2_array)):
        for j in range(0, len(lines2_array[i])):
            if divmod(int(j), int(len(lines2_array[i])))[1] == 3:
                array_boundx[i].append(float(lines2_array[i][j].strip('ub')))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 4:
                array_boundx[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 6:
                array_boundy[i].append(float(lines2_array[i][j].strip('ub')))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 7:
                array_boundy[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 8:
                array_boundRF[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 9:
                array_boundRF[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 10:
                array_boundRF[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 11:
                array_boundLF[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 12:
                array_boundLF[i].append(float(lines2_array[i][j]))
            if divmod(int(j), int(len(lines2_array[i])))[1] == 13:
                array_boundLF[i].append(float(lines2_array[i][j]))
    f.close()
    f1.close()
    f2.close()

    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = pinocchio.buildModelFromUrdf("/home/jhk/catkin_ws/src/tocabi_cc/robots/dyros_tocabi_with_redhands.urdf",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    q = pinocchio.randomConfiguration(model)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    #q_init = [1.00000000e-01,  0.00000000e+00,  8.07830000e-01,  0.00000000e+00, 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00, -4.58270337e-17,5.49813461e-19, -3.36530703e-01,  1.17717832e+00, -8.40647617e-01,1.23071442e-16 , 4.25905878e-16,  8.33360457e-17, -3.36530703e-01, 1.17717832e+00 ,-8.40647617e-01,  2.46142884e-16]

    for i in range(0, len(q)):
        q[i] = xs_pca_test[i]

    state = crocoddyl.StateKinodynamic(model)
    actuation = crocoddyl.ActuationModelKinoBase(state)
    x0 = np.array([0.] * (state.nx + 8))
    u0 = np.array([0.] * (22))
    for i in range(0,len(q_init)):
        x0[i] = xs_pca_test[i]
    
    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")
    LFframe_id = model.getFrameId("L_Foot_Link")
    RFframe_id = model.getFrameId("R_Foot_Link")    
    data = model.createData()

    pinocchio.forwardKinematics(model, data, q, qdot)
    pinocchio.updateFramePlacements(model,data)
    pinocchio.centerOfMass(model, data, q, False)
    pinocchio.computeCentroidalMomentum(model,data,q,qdot)
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]

    x0[37] = data.com[0][0]
    x0[39] = data.com[0][0]
    x0[41] = data.com[0][1]
    x0[43] = data.com[0][1]

    traj_= [0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.08, 0.0, 0.0, 0.0]
    u_traj_ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
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
    weight_quad_zmp = np.array([weight_quad_zmpx] + [weight_quad_zmpy])
    weight_quad_cam = np.array([weight_quad_camy] + [weight_quad_camx])
    weight_quad_com = np.array([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
    weight_quad_rf = np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
    weight_quad_lf = np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])

    lb_ = np.ones([2, N])
    ub_ = np.ones([2, N])

    for i in range(0,N-5):
        lb_[0,i] = 0.0
        ub_[0,i] = 0.2

    for i in range(N-5,N):
        lb_[0,i] = 0.15
        ub_[0,i] = 0.4

    for i in range(0,N-4):
        lb_[1,i] = -0.2
        ub_[1,i] = 0.2

    for i in range(N-4,N):
        lb_[1,i] = 0.05
        ub_[1,i] = 0.2
    
    actuation_vector = [None] * (N)
    state_vector = [None] * (N)
    state_bounds = [None] * (N)
    state_bounds2 = [None] * (N)
    state_bounds3 = [None] * (N)
    state_activations = [None] * (N)
    state_activations2 = [None] * (N)
    state_activations3 = [None] * (N)
    xRegCost_vector = [None] * (N)
    uRegCost_vector = [None] * (N)
    stateBoundCost_vector = [None] * (N)
    camBoundCost_vector = [None] *  (N)
    comBoundCost_vector = [None] *  (N)
    rf_foot_pos_vector = [None] *  (N)
    lf_foot_pos_vector = [None] *  (N)
    residual_FrameRF = [None] *  (N)
    residual_FrameLF = [None] *  (N)
    foot_trackR = [None] *  (N)
    foot_trackL = [None] *  (N)
    runningCostModel_vector = [None] * (N-1)
    runningDAM_vector = [None] * (N-1)
    runningModelWithRK4_vector = [None] * (N-1)
    xs = [None] * (N)
    us = [None] * (N-1)

    lb = []
    ub = []

    for i in range(0,N-1):
        state_vector[i] = crocoddyl.StateKinodynamic(model)
        actuation_vector[i] = crocoddyl.ActuationModelKinoBase(state_vector[i])
        state_bounds[i] = crocoddyl.ActivationBounds(lb_[:,i],ub_[:,i])
        state_activations[i] = crocoddyl.ActivationModelQuadraticBarrier(state_bounds[i])
        stateBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], state_activations[i], crocoddyl.ResidualFlyState(state_vector[i], actuation_vector[i].nu + 4))
        camBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[i], actuation_vector[i].nu + 4))
        comBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[i], actuation_vector[i].nu + 4))
        rf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        rf_foot_pos_vector[i].translation = copy(RF_tran.translation)
        lf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        lf_foot_pos_vector[i].translation = copy(LF_tran.translation)
        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu+4)
        runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], weight_quad_zmp[0])
        runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
        runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
        runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)
    
    state_vector[N-1] = crocoddyl.StateKinodynamic(model)
    actuation_vector[N-1] = crocoddyl.ActuationModelKinoBase(state_vector[N-1])
    state_bounds[N-1] = crocoddyl.ActivationBounds(lb_[:,N-1],ub_[:,N-1])
    state_activations[N-1] = crocoddyl.ActivationModelQuadraticBarrier(state_bounds[N-1])
    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
    camBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[N-1], actuation_vector[N-1].nu + 4))
    comBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[N-1], actuation_vector[N-1].nu + 4))
    rf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    rf_foot_pos_vector[N-1].translation = copy(RF_tran.translation)
    lf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    lf_foot_pos_vector[N-1].translation = copy(LF_tran.translation)
    residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])
    
    terminalCostModel = crocoddyl.CostModelSum(state_vector[N-1], actuation_vector[N-1].nu + 4)
    terminalCostModel.addCost("stateReg", stateBoundCost_vector[N-1], weight_quad_zmp[0])
    terminalCostModel.addCost("camReg", camBoundCost_vector[N-1], 1.0)
    terminalCostModel.addCost("comReg", comBoundCost_vector[N-1], 1.0)
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N-1], actuation_vector[N-1], terminalCostModel)

    for i in range(0,N):
        xs[i] = copy(x0)
    for i in range(0,N-1):
        us[i] = copy(u0)
    
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 6
    ddp = crocoddyl.SolverBoxFDDP(problemWithRK4)

    walking_tick = 23
    
    for i in range(0,N-1):
        state_bounds[i].lb[0] = copy(array_boundx[30*(walking_tick)+i][0])
        state_bounds[i].ub[0] = copy(array_boundx[30*(walking_tick)+i][1])
        state_bounds[i].lb[1] = copy(array_boundy[30*(walking_tick)+i][0])
        state_bounds[i].ub[1] = copy(array_boundy[30*(walking_tick)+i][1])
        state_activations[i].bounds = state_bounds[i]
        stateBoundCost_vector[i].activation_ = state_activations[i]
        rf_foot_pos_vector[i].translation[0] = copy(array_boundRF[30*(walking_tick)+i][0])
        rf_foot_pos_vector[i].translation[1] = copy(array_boundRF[30*(walking_tick)+i][1])
        rf_foot_pos_vector[i].translation[2] = copy(array_boundRF[30*(walking_tick)+i][2])
        lf_foot_pos_vector[i].translation[0] = copy(array_boundLF[30*(walking_tick)+i][0])
        lf_foot_pos_vector[i].translation[1] = copy(array_boundLF[30*(walking_tick)+i][1])
        lf_foot_pos_vector[i].translation[2] = copy(array_boundLF[30*(walking_tick)+i][2])
        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
        runningCostModel_vector[i].removeCost("footReg1")
        runningCostModel_vector[i].removeCost("footReg2")
        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)   

    state_bounds[N-1].lb[0] = copy(array_boundx[30*(walking_tick)+N-1][0])
    state_bounds[N-1].ub[0] = copy(array_boundx[30*(walking_tick)+N-1][1])
    state_bounds[N-1].lb[1] = copy(array_boundy[30*(walking_tick)+N-1][0])
    state_bounds[N-1].ub[1] = copy(array_boundy[30*(walking_tick)+N-1][1])
    state_activations[N-1].bounds = state_bounds[N-1]
    stateBoundCost_vector[N-1].activation_ = state_activations[N-1]
    rf_foot_pos_vector[N-1].translation[0] = copy(array_boundRF[30*(walking_tick)+N-1][0])
    rf_foot_pos_vector[N-1].translation[1] = copy(array_boundRF[30*(walking_tick)+N-1][1])
    rf_foot_pos_vector[N-1].translation[2] = copy(array_boundRF[30*(walking_tick)+N-1][2])
    lf_foot_pos_vector[N-1].translation[0] = copy(array_boundLF[30*(walking_tick)+N-1][0])
    lf_foot_pos_vector[N-1].translation[1] = copy(array_boundLF[30*(walking_tick)+N-1][1])
    lf_foot_pos_vector[N-1].translation[2] = copy(array_boundLF[30*(walking_tick)+N-1][2])
    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
    terminalCostModel.removeCost("footReg1")
    terminalCostModel.removeCost("footReg2")
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)

    problemWithRK4.x0 = xs[0]
    #ddp.alphas = [2**(-n) for n in range(10)]
    ddp.th_stop = 5.0
    c_start = time.time()
    css = ddp.solve(xs_pca, us_pca, 300, False, 0.5)
    c_end = time.time()
    duration = (1e3 * (c_end - c_start))

    '''
    print(ddp.xs[15])
   
    '''
    print("end")
    avrg_duration = duration
    min_duration = duration #min(duration)
    max_duration = duration #max(duration)
    print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
    print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))


    '''
    f4.write("walking_tick ")
    f4.write(str(walking_tick))
    f4.write(" css ")
    f4.write(str(ddp.iter))
    f4.write(" ")
    f4.write(str(css))
    f4.write(" ")
    f4.write(str(0))
    f4.write("\n")
    
    for i in range(0, N-1):
        f4.write("q ")
        f4.write(str(i))
        f4.write("\n")
        for j in range(0,19):
            f4.write(str(ddp.xs[i][j]))
            f4.write(", ")
        f4.write("qdot ")
        f4.write(str(i))
        f4.write("\n")            
        for j in range(19,37):
            f4.write(str(ddp.xs[i][j]))
            f4.write(", ")
        f4.write("x_state ")
        f4.write(str(i))
        f4.write("\n")  
        for j in range(37,45):
            f4.write(str(ddp.xs[i][j]))
            f4.write(", ")
        f4.write("\n")
        f4.write("u ")
        f4.write(str(i))
        f4.write("\n")  
        for j in range(0,18):
            f4.write(str(ddp.us[i][j]))
            f4.write(", ")
        f4.write("ustate ")
#            f4.write(str(i))
        f4.write("\n")  
        for j in range(18,22):
            f4.write(str(ddp.us[i][j]))
            f4.write(", ")
        f4.write("\n")
    f4.write("q ")
    f4.write(str(N))
    f4.write("\n")
    for j in range(0,19):
        f4.write(str(ddp.xs[N-1][j]))
        f4.write(", ")
    f4.write("qdot ")
    f4.write(str(N-1))
    f4.write("\n")            
    for j in range(19,37):
        f4.write(str(ddp.xs[N-1][j]))
        f4.write(", ")
    f4.write("x_state ")
    f4.write(str(N-1))
    f4.write("\n")  
    for j in range(37,45):
        f4.write(str(ddp.xs[N-1][j]))
        f4.write(", ")
    f4.write("\n")

    for i in range(0, N):
        f3.write(str(walking_tick))
        f3.write(" ")
        f3.write(str(i))
        f3.write(" ")
        f3.write("lb")
        f3.write(str(array_boundx[30*(walking_tick)+i][0]))
        f3.write("ub ")
        f3.write(str(array_boundx[30*(walking_tick)+i][1]))
        f3.write(" ")
        f3.write("lb ")
        f3.write(str(array_boundy[30*(walking_tick)+i][0]))
        f3.write("ub ")
        f3.write(str(array_boundy[30*(walking_tick)+i][1]))
        f3.write(" ")
        f3.write(str(array_boundRF[30*(walking_tick)+i][0]))
        f3.write(" ")
        f3.write(str(array_boundRF[30*(walking_tick)+i][1]))
        f3.write(" ")
        f3.write(str(array_boundRF[30*(walking_tick)+i][2]))
        f3.write(" ")
        f3.write(str(array_boundLF[30*(walking_tick)+i][0]))
        f3.write(" ")
        f3.write(str(array_boundLF[30*(walking_tick)+i][1]))
        f3.write(" ")
        f3.write(str(array_boundLF[30*(walking_tick)+i][2]))
        f3.write("\n")    
     
    c_start = time.time()
    css = ddp.solve(xs, us, 300, True, 0.1)
    c_end = time.time()
    duration = (1e3 * (c_end - c_start))
            
    avrg_duration = duration
    min_duration = duration #min(duration)
    max_duration = duration #max(duration)
    print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
    print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
     '''
    
    '''
    while client.is_connected:
        for i in range(0,N-1):
            state_bounds[i].lb[0] = copy(array_boundx[30*(walking_tick)+i][0])
            state_bounds[i].ub[0] = copy(array_boundx[30*(walking_tick)+i][1])
            state_bounds[i].lb[1] = copy(array_boundy[30*(walking_tick)+i][0])
            state_bounds[i].ub[1] = copy(array_boundy[30*(walking_tick)+i][1])
            state_activations[i].bounds = state_bounds[i]
            stateBoundCost_vector[i].activation_ = state_activations[i]
            rf_foot_pos_vector[i].translation[0] = copy(array_boundRF[30*(walking_tick)+i][0])
            rf_foot_pos_vector[i].translation[1] = copy(array_boundRF[30*(walking_tick)+i][1])
            rf_foot_pos_vector[i].translation[2] = copy(array_boundRF[30*(walking_tick)+i][2])
            lf_foot_pos_vector[i].translation[0] = copy(array_boundLF[30*(walking_tick)+i][0])
            lf_foot_pos_vector[i].translation[1] = copy(array_boundLF[30*(walking_tick)+i][1])
            lf_foot_pos_vector[i].translation[2] = copy(array_boundLF[30*(walking_tick)+i][2])
            #residual_FrameRF[i].pref = rf_foot_pos_vector[i]
            #residual_FrameLF[i].pref = lf_foot_pos_vector[i]
            #foot_trackR[i].residual_ = residual_FrameRF[i]
            #foot_trackL[i].residual_ = residual_FrameLF[i]
            residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
            foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
            foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
            runningCostModel_vector[i].removeCost("footReg1")
            runningCostModel_vector[i].removeCost("footReg2")
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)   

        state_bounds[N-1].lb[0] = copy(array_boundx[30*(walking_tick)+N-1][0])
        state_bounds[N-1].ub[0] = copy(array_boundx[30*(walking_tick)+N-1][1])
        state_bounds[N-1].lb[1] = copy(array_boundy[30*(walking_tick)+N-1][0])
        state_bounds[N-1].ub[1] = copy(array_boundy[30*(walking_tick)+N-1][1])
        state_activations[N-1].bounds = state_bounds[N-1]
        stateBoundCost_vector[N-1].activation_ = state_activations[N-1]
        rf_foot_pos_vector[N-1].translation[0] = copy(array_boundRF[30*(walking_tick)+N-1][0])
        rf_foot_pos_vector[N-1].translation[1] = copy(array_boundRF[30*(walking_tick)+N-1][1])
        rf_foot_pos_vector[N-1].translation[2] = copy(array_boundRF[30*(walking_tick)+N-1][2])
        lf_foot_pos_vector[N-1].translation[0] = copy(array_boundLF[30*(walking_tick)+N-1][0])
        lf_foot_pos_vector[N-1].translation[1] = copy(array_boundLF[30*(walking_tick)+N-1][1])
        lf_foot_pos_vector[N-1].translation[2] = copy(array_boundLF[30*(walking_tick)+N-1][2])
        foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
        foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
        terminalCostModel.removeCost("footReg1")
        terminalCostModel.removeCost("footReg2")
        terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
        terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)

        
        for j in range(0, N):
            for k in range(0, 19):
                xs[j][k] = copy(array_q1[30*(walking_tick) + j][k])
            for k in range(19, 37):
                xs[j][k] = copy(array_qdot1[30*(walking_tick) + j][k-19])
            for k in range(37, 45):
                xs[j][k] = copy(array_xstate1[30*(walking_tick) + j][k-37])
        for j in range(0, N-1):
            for k in range(0, 18):
                us[j][k] = copy(array_qddot1[29*(walking_tick) + j][k])
            for k in range(18, 22):
                us[j][k] = copy(array_u1[29*(walking_tick) + j][k-18])
        
    #    duration = []
    
        iter_ = 0
        T = 1
        for i in range(0,T):
            if walking_tick > 0:
                problemWithRK4.x0 = ddp.xs[1]
            
            ddp.th_stop = 0.01
            c_start = time.time()
            css = ddp.solve(xs_pca, us_pca, 300, True, 0.1)
            c_end = time.time()
            duration = (1e3 * (c_end - c_start))
            
            avrg_duration = duration
            min_duration = duration #min(duration)
            max_duration = duration #max(duration)
            print(iter_)
            print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
            print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
             
         
            for i in range(0,N):
                if i < N-1:
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['footReg1'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['footReg2'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['comReg'].cost.residual.reference[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['camReg'].cost.residual.reference[j]) > 0.05:
                            booltemp1 = False
                            break
                else:
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['footReg1'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['footReg2'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['comReg'].cost.residual.reference[j]) > 0.008:
                            booltemp1 = False
                            break
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['camReg'].cost.residual.reference[j]) > 0.05:
                            booltemp1 = False
                            break
                if booltemp1 == False:
                    break

                if booltemp1 == True and avrg_duration <= 30:
                    booltemp = False
                    break
            print("success")
            print(walking_tick)
            
            iter_ = iter_ + 1
        booltemp = True  

        for i in range(0,N-1):
            print(runningCostModel_vector[i].costs['comReg'].cost.residual)
            print(runningCostModel_vector[i].costs['camReg'].cost.residual)
            #print(runningCostModel_vector[i].costs['stateReg'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg1'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg2'].cost.residual)
        print(terminalCostModel.costs['comReg'].cost.residual)
        print(terminalCostModel.costs['camReg'].cost.residual)
        #print(runningCostModel_vector[i].costs['stateReg'].cost.residual)
        print(terminalCostModel.costs['footReg1'].cost.residual)
        print(terminalCostModel.costs['footReg2'].cost.residual)
        print(ddp.xs[N-1])
      
        f4.write("walking_tick ")
        f4.write(str(walking_tick))
        f4.write(" css ")
        f4.write(str(ddp.iter))
        f4.write(" ")
        f4.write(str(css))
        f4.write(" ")
        f4.write(str(0))
        f4.write("\n")
        
        for i in range(0, N-1):
            f4.write("q ")
            f4.write(str(i))
            f4.write("\n")
            for j in range(0,19):
                f4.write(str(ddp.xs[i][j]))
                f4.write(", ")
            f4.write("qdot ")
            f4.write(str(i))
            f4.write("\n")            
            for j in range(19,37):
                f4.write(str(ddp.xs[i][j]))
                f4.write(", ")
            f4.write("x_state ")
            f4.write(str(i))
            f4.write("\n")  
            for j in range(37,45):
                f4.write(str(ddp.xs[i][j]))
                f4.write(", ")
            f4.write("\n")
            f4.write("u ")
            f4.write(str(i))
            f4.write("\n")  
            for j in range(0,18):
                f4.write(str(ddp.us[i][j]))
                f4.write(", ")
            f4.write("ustate ")
#            f4.write(str(i))
            f4.write("\n")  
            for j in range(18,22):
                f4.write(str(ddp.us[i][j]))
                f4.write(", ")
            f4.write("\n")
        f4.write("q ")
        f4.write(str(N))
        f4.write("\n")
        for j in range(0,19):
            f4.write(str(ddp.xs[N-1][j]))
            f4.write(", ")
        f4.write("qdot ")
        f4.write(str(N-1))
        f4.write("\n")            
        for j in range(19,37):
            f4.write(str(ddp.xs[N-1][j]))
            f4.write(", ")
        f4.write("x_state ")
        f4.write(str(N-1))
        f4.write("\n")  
        for j in range(37,45):
            f4.write(str(ddp.xs[N-1][j]))
            f4.write(", ")
        f4.write("\n")

        for i in range(0, N):
            f3.write(str(walking_tick))
            f3.write(" ")
            f3.write(str(i))
            f3.write(" ")
            f3.write("lb")
            f3.write(str(array_boundx[30*(walking_tick)+i][0]))
            f3.write("ub ")
            f3.write(str(array_boundx[30*(walking_tick)+i][1]))
            f3.write(" ")
            f3.write("lb ")
            f3.write(str(array_boundy[30*(walking_tick)+i][0]))
            f3.write("ub ")
            f3.write(str(array_boundy[30*(walking_tick)+i][1]))
            f3.write(" ")
            f3.write(str(array_boundRF[30*(walking_tick)+i][0]))
            f3.write(" ")
            f3.write(str(array_boundRF[30*(walking_tick)+i][1]))
            f3.write(" ")
            f3.write(str(array_boundRF[30*(walking_tick)+i][2]))
            f3.write(" ")
            f3.write(str(array_boundLF[30*(walking_tick)+i][0]))
            f3.write(" ")
            f3.write(str(array_boundLF[30*(walking_tick)+i][1]))
            f3.write(" ")
            f3.write(str(array_boundLF[30*(walking_tick)+i][2]))
            f3.write("\n")
        walking_tick = walking_tick + 1
        if walking_tick == 3:
            break
    f3.close()
    f4.close()
    client.terminate()
    '''
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

