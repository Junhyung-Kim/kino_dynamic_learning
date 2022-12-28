import roslibpy
import pickle
import crocoddyl
import pinocchio
import numpy as np
import time
import example_robot_data
from copy import copy
#from regression import *
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from IPython.display import clear_output 
import GPy
import logging
import os
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import sys
import numpy.matlib
np.set_printoptions(threshold=sys.maxsize)
global client
global learn_type

class Regressor():
    def __init__(self, transform=None):
        self.transform = transform
        self.pca = None

    def save_to_file(self,filename):
        f = open(filename + '.pkl', 'wb')
        pickle.dump(self.__dict__,f)
        f.close()

    def load_from_file(self,filename):
        f = open(filename + '.pkl', 'rb')
        self.__dict__ = pickle.load(f)
class GPy_Regressor(Regressor):
    def __init__(self, dim_input, transform = None):
        self.transform = transform #whether the output should be transformed or not. Possible option: PCA, RBF, etc.
        self.dim_input = dim_input

    def fit(self,x,y, num_restarts = 10):
        kernel = gp.kernels.RBF(input_dim=self.dim_input, variance=torch.tensor(0.1), lengthscale=torch.tensor(0.3)) # + gp.kernels.White(input_dim=self.dim_input)
        Xu = torch.zeros(x.size(0)/5, x.size(1))
        #self.gp = gp.models.SparseGPRegression(x, y, kernel, Xu)
        self.gp = gp.models.GPRegression(x, y, kernel, noise=torch.tensor(0.1))
        #gpmodule = gp.models.SparseGPRegression(x, y, kernel, Xu)
        
        '''
        gplvm = gp.models.GPLVM(gpmodule)
        print("trainng...")
        gp.util.train(gplvm)  
        self.gp = gplvm
        print("trained")
        '''
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        variances = []
        lengthscales = []
        noises = []
        num_steps = num_restarts
        for i in range(num_steps):
            variances.append(self.gp.kernel.variance.item())
            lengthscales.append(self.gp.kernel.lengthscale.item())
            optimizer.zero_grad()
            loss = loss_fn(self.gp.model, self.gp.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(i)
            print(loss.item())
        plt.plot(losses)
        plt.show()
    


    def predict(self,x, is_transform = True):
        y,cov = self.gp.forward(x)
        y = y.detach().numpy().flatten()
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, cov
        else:
            return y,cov

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
        database = pickle.load(f)
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
        #for i in range(0,num_desired):
        #    x_inputs[key].append(np.array(database[key]['x_state'])[i][0])

        x_inputs[key] = np.array(database[key]['x_inputs'])[:num_desired]
        #for i in range(0,num_desired):
        #    x_inputs[key][i] = np.append(x_inputs[key][i], np.array(database[key]['x_state'])[i][0])
        #    print(x_input[key][i])
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

    print(raw_u_trajs[0])
    timestep = 30
    rbf_num = 68
    
    Phi = define_RBF(dof=19, nbStates = rbf_num, offset = 1, width = 1, T = timestep, coeff =1)
    #plt.plot(Phi)
    #plt.savefig('/home/jhk/data/mpc/filename.png')

    for key in keys:
        w_trajs[key] = apply_RBF(trajs[key], Phi)
        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)
    print("u")
    print(len(u_trajs[key]))
    print(len(acc_trajs[key]))
    print(len(x_trajs[key]))
    '''
    aa_ = w_trajs[key][0].reshape(rbf_num,-1)
    bb_ = w_vel_trajs[key][0].reshape(rbf_num,-1)
    dd_ = w_acc_trajs[key][0].reshape(rbf_num,-1)
    cc_ = w_u_trajs[key][0].reshape(rbf_num,-1)
    ee_ = w_x_trajs[key][0].reshape(rbf_num,-1)
    
    cost_temp = 0
    for i in range(1,1000):
        for j in range(1,1000):
            for k in range(1,1000):
                for f in range(30,120):
                    rbf_num = f
                    Phi = define_RBF(dof=19, nbStates = f, offset = i, width = j, T = timestep, coeff=k)

                    for key in keys:
                        w_trajs[key] = apply_RBF(trajs[key], Phi)
                        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
                        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
                        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
                        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)

                    aa_ = w_trajs[key][0].reshape(rbf_num,-1)
                    bb_ = w_vel_trajs[key][0].reshape(rbf_num,-1)
                    dd_ = w_acc_trajs[key][0].reshape(rbf_num,-1)
                    cc_ = w_u_trajs[key][0].reshape(rbf_num,-1)
                    ee_ = w_x_trajs[key][0].reshape(rbf_num,-1)
                    
                    print(trajs[key][0][0])
                    print(aa_)
                    print(Phi[0])
                    print(np.dot(Phi[0], aa_))
                    print(np.linalg.norm(trajs[key][0][0] - np.dot(Phi[0], aa_)))
    '''
    '''
                    cost = np.linalg.norm(trajs[key][0][0] - np.dot(Phi[0], aa_)) *np.linalg.norm(trajs[key][0][0] - np.dot(Phi[0], aa_))  + np.linalg.norm(vel_trajs[key][0][0] - np.dot(Phi[0], bb_)) * np.linalg.norm(vel_trajs[key][0][0] - np.dot(Phi[0], bb_)) + np.linalg.norm(acc_trajs[key][0][0] - np.dot(Phi[0], dd_)) * np.linalg.norm(acc_trajs[key][0][0] - np.dot(Phi[0], dd_)) + np.linalg.norm(u_trajs[key][0][0] - np.dot(Phi[0], cc_)) * np.linalg.norm(u_trajs[key][0][0] - np.dot(Phi[0], cc_)) + np.linalg.norm(x_trajs[key][0][0] - np.dot(Phi[0], ee_)) * np.linalg.norm(x_trajs[key][0][0] - np.dot(Phi[0], ee_))

                    if i  == 1 and j == 1 and k == 1 and f ==30:
                        cost_temp = cost
                        print("start")
                    if(cost < cost_temp):
                        print("i")
                        print(i)
                        print("j")
                        print(j)
                        print("k")
                        print(k)
                        print("f")
                        print(f)

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
    y_train = dict()
    y_test = dict()

    y_vel_train = dict()
    y_vel_test = dict()

    y_acc_train = dict()
    y_acc_test = dict()

    y_u_train = dict()
    y_u_test = dict()

    y_x_train = dict()
    y_x_test = dict()

    gpr = dict()
    gpr_vel = dict()
    gpr_u = dict()
    gpr_acc = dict()
    gpr_x = dict()
    
    for key in keys:
        x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key], w_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_u_train[key], y_u_test[key] = train_test_split(x_inputs[key],w_u_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_acc_train[key], y_acc_test[key] = train_test_split(x_inputs[key],w_acc_trajs_pca[key], test_size = 0.1666, random_state=1)
        _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.1666, random_state=1)

        x_inputs_train[key] = torch.FloatTensor(x_inputs_train[key])
        x_inputs_test[key] = torch.FloatTensor(x_inputs_test[key])
        y_train[key] = torch.FloatTensor(np.transpose(y_train[key]))
        y_test[key] = torch.FloatTensor(np.transpose(y_test[key]))
        y_vel_train[key] = torch.FloatTensor(np.transpose(y_vel_train[key]))
        y_vel_test[key] = torch.FloatTensor(np.transpose(y_vel_test[key]))
        y_u_train[key] = torch.FloatTensor(np.transpose(y_u_train[key]))
        y_u_test[key] = torch.FloatTensor(np.transpose(y_u_test[key]))
        y_acc_train[key] = torch.FloatTensor(np.transpose(y_acc_train[key]))
        y_acc_test[key] = torch.FloatTensor(np.transpose(y_acc_test[key]))
        y_x_train[key] = torch.FloatTensor(np.transpose(y_x_train[key]))
        y_x_test[key] = torch.FloatTensor(np.transpose(y_x_test[key]))
        
    if learn_type == 0:
        print("SAVE REGRESSION PROBLEM")
        for key in keys:
            gpr_vel[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_vel[key])
            gpr_vel[key].fit(x_inputs_train[key], y_vel_train[key],num_restarts=100)
            gpr[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca[key])
            gpr[key].fit(x_inputs_train[key], y_train[key],num_restarts=100)
            gpr_acc[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_acc[key])
            #gpr_acc[key].pca = pca_vel[key]
            gpr_acc[key].fit(x_inputs_train[key], y_acc_train[key],num_restarts=2000)
            gpr_x[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_x[key])
            #gpr_x[key].pca = pca_vel[key]
            gpr_x[key].fit(x_inputs_train[key], y_x_train[key],num_restarts=100)
            gpr_u[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_u[key])
            #gpr_u[key].pca = pca_vel[key]
            gpr_u[key].fit(x_inputs_train[key], y_u_train[key],num_restarts=100)

        clear_output()
        
        functions = dict()
        functions['gpr'] = gpr
        functions['gpr_x'] = gpr_x
        functions['gpr_vel'] = gpr_vel
        functions['gpr_acc'] = gpr_acc
        functions['gpr_u'] = gpr_u
        f = open('/home/jhk/data/mpc/functions.pkl', 'wb')
        pickle.dump(functions, f)
        f.close()
    else:
        print("LOAD REGRESSION PROBLEM")
        f = open('/home/jhk/data/mpc/functions.pkl', 'rb')
        functions = pickle.load(f)
        f.close()
        gpr = functions['gpr']
        gpr_vel = functions['gpr_vel']
        gpr_u = functions['gpr_u']
        gpr_x = functions['gpr_x']
        gpr_acc = functions['gpr_acc']
        
    indexes = [3]
    for i in (indexes): 
        if i%2 == 0:
            key = 'left'
        else:
            key = 'right'
            JJ = np.random.randint(x_inputs_test[key].shape[0])
            x_test = x_inputs_test[key][JJ][None,:]
            x = x_inputs_test[key][JJ][None,:]
            print("X")
            print(x)
            tic = time.time()
            w_traj,cov_traj = gpr[key].predict(x)
            w_vel,cov_vel = gpr_vel[key].predict(x)
            w_acc,cov_acc = gpr_acc[key].predict(x)
            w_x,cov = gpr_x[key].predict(x)
            w_u,cov_u = gpr_u[key].predict(x)
            toc = time.time()
            w_acc = w_acc.reshape(rbf_num,-1)
            acc_traj = np.dot(Phi,w_acc)
            w_vel = w_vel.reshape(rbf_num,-1)
            vel_traj = np.dot(Phi,w_vel)
            w_x = w_x.reshape(rbf_num,-1)
            x_traj = np.dot(Phi,w_x)
            w_u = w_u.reshape(rbf_num,-1)
            u_traj = np.dot(Phi,w_u)
            w_traj = w_traj.reshape(rbf_num,-1)
            traj = np.dot(Phi,w_traj)
            tac = time.time()

            tt = tic - toc
            tt1 = tac - toc

    q_pca = traj
    v_pca = vel_traj
    x_pca = x_traj
    acc_pca = acc_traj
    u_pca = u_traj
    xs_pca = []
    us_pca = []
    xs_pca_test = x
    for q, v, x in zip(q_pca, v_pca, x_pca):
        xs_pca.append(np.concatenate([q, v, x]))
        
    for a, u in zip(acc_pca, u_pca):
        us_pca.append(np.concatenate([a, u]))
    del us_pca[-1]

    print(xs_pca_test)
    print(xs_pca[0])

    xs_pca_test = x_test.detach().numpy()[0] #x_inputs_init[key][break_a]
    
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    PCAlearning()

