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

#manager = multiprocessing.Manager()
#thread_manager = manager.list()
#X_manager = manager.dict()

def InversePCA(model, rbf_num, pca, Phi, X, thread_manager):
    while True:
        if thread_manager[0] == 0:
            ti = time.time()
            a = np.array(X[:])
            c = torch.tensor(a.reshape(1,1,19),dtype=torch.float32)
            w_traj = model(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
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
            w_traj = model(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
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
            w_traj = model(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
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
            w_traj = model(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
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
            w_traj = model(c)
            w_traj = w_traj[0].detach().numpy()
            w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
            w_traj = w_traj.reshape(rbf_num,-1)
            traj1 = np.dot(Phi,w_traj)
            t2 = time.time()
            print("thread5")
            print(ti)
            print(t2)
            print(t2 - ti)
            thread_manager[4] = 1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 40),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 19),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()
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
        y_test[key] = torch.FloatTensor((y_test[key]))
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
  
    train_y = [
    (x_inputs_train[key][i], y_train[key][i]) for i in range(len(x_inputs_train[key]))
    ]
    train_yvel = [
    (x_inputs_train[key][i], y_vel_train[key][i]) for i in range(len(x_inputs_train[key]))
    ]
    train_yacc = [
    (x_inputs_train[key][i], y_acc_train[key][i]) for i in range(len(x_inputs_train[key]))
    ]
    train_yu = [
    (x_inputs_train[key][i], y_u_train[key][i]) for i in range(len(x_inputs_train[key]))
    ]
    train_yx = [
    (x_inputs_train[key][i], y_x_train[key][i]) for i in range(len(x_inputs_train[key]))
    ]

    test_y = [
    (x_inputs_test[key][i], y_test[key][i]) for i in range(len(x_inputs_test[key]))
    ]
    test_yvel = [
    (x_inputs_test[key][i], y_vel_test[key][i]) for i in range(len(x_inputs_test[key]))
    ]
    test_yacc = [
    (x_inputs_test[key][i], y_acc_test[key][i]) for i in range(len(x_inputs_test[key]))
    ]
    test_yu = [
    (x_inputs_test[key][i], y_u_test[key][i]) for i in range(len(x_inputs_test[key]))
    ]
    test_yx = [
    (x_inputs_test[key][i], y_x_test[key][i]) for i in range(len(x_inputs_test[key]))
    ]

    batch_size = 3
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

    discriminator = Discriminator()
    generator = Generator()

    lr = 0.0002
    num_epochs = 500
    criterion = nn.BCELoss()

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    BATCH_SIZE = batch_size
    for epoch in range(num_epochs):
        for n, (images, _) in enumerate(train_loader):
            images = images.reshape(BATCH_SIZE, -1).to('cpu')
            real_labels = torch.ones(BATCH_SIZE, 40).to('cpu')# [1,1,1...]
            fake_labels = torch.zeros(BATCH_SIZE, 40).to('cpu')# [0.0,0...]
            
            # 판별자가 진짜 이미지를 진짜로 인식하는 오차를 예산
            outputs = discriminator(images) # 진짜 이미지를 discriminator의 입력으로 제공
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            # 무작위 텐서로 가짜 이미지 생성
            z = torch.randn(BATCH_SIZE, 19).to('cpu')

            fake_images = generator(z) #G의 입력으로 랜덤 텐서 제공, G가 fake image 생성
            
            # 판별자가 가짜 이미지를 가짜로 인식하는 오차를 계산
            outputs = discriminator(fake_images)# 가짜 이미지를 discriminator의 입력으로 제공
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 Discriminator의 오차 계산
            d_loss = d_loss_real + d_loss_fake

            #------ Discriminator 학습 ------#
            # 역전파 알고리즘으로 Discriminator의 학습을 진행
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            d_loss.backward()
            optimizer_discriminator.step()# Discriminator 학습
            
            # 생성자가 판별자를 속였는지에 대한 오차(Generator의 loss)를 계산
            fake_images = generator(z)
            outputs = discriminator(fake_images) #한번 학습한 D가 fake image를 
            g_loss = criterion(outputs, real_labels)

            #------ Generator 학습 ------#
            
            # 역전파 알고리즘으로 생성자 모델의 학습을 진행
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            g_loss.backward()
            optimizer_generator.step()
            # Show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, epoch, d_loss.item(), g_loss.item(), 
                    real_score.mean().item(), fake_score.mean().item()))  

    JJ = np.random.randint(x_inputs_test[key].shape[0])
    X = x_inputs_test['Right'][JJ][None,:]
    #X = X.reshape(1, sequence_length, input_size).to(device)

    ti = time.time()
    w_traj = discriminator(X)
    w_traj = w_traj[0].detach().numpy()
    w_traj = pca['Right'].inverse_transform([w_traj[None,:]])[0]
    w_traj = w_traj.reshape(rbf_num,-1)
    traj1 = np.dot(Phi,w_traj)
    ta = time.time()
    print(X)
    print(traj1)
    print(ta-ti)


    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    PCAlearning()
    #talker()

