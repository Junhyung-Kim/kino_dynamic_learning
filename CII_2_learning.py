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
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from IPython.display import clear_output 
import GPy
import numpy.matlib

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
        kernel = GPy.kern.RBF(input_dim=self.dim_input, variance=0.1,lengthscale=0.3, ARD=True) + GPy.kern.White(input_dim=self.dim_input)
        self.gp = GPy.models.GPRegression(x, y, kernel)
        self.gp.optimize()

    def predict(self,x, is_transform = True):
        y,cov = self.gp.predict(x)
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
    print Phi
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

def talker():
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
        database = pickle.load(f)
    f.close()

    trajs = dict()
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
    rbf_num = 60
    Phi = define_RBF(dof=19, nbStates = rbf_num, offset = 200, width = 60, T = timestep)
    #Phi_input = define_RBF(dof=19, nbStates = rbf_num, offset = 200, width = 60, T = timestep -1)
    #plt.plot(Phi)
    #plt.savefig('/home/jhk/data/mpc/filename.png')

    for key in keys:
        w_trajs[key] = apply_RBF(trajs[key], Phi)
        w_vel_trajs[key] = apply_RBF(vel_trajs[key], Phi)
        w_x_trajs[key] = apply_RBF(x_trajs[key], Phi)
        w_u_trajs[key] = apply_RBF(u_trajs[key], Phi)    
        w_acc_trajs[key] = apply_RBF(acc_trajs[key], Phi)

    for key in keys:
        pca[key] = PCA(n_components=60)
        w_trajs_pca[key] = pca[key].fit_transform(w_trajs[key])
        pca_vel[key] = PCA(n_components=60)
        w_vel_trajs_pca[key] = pca_vel[key].fit_transform(w_vel_trajs[key])

        pca_x[key] = PCA(n_components=60)
        w_x_trajs_pca[key] = pca_x[key].fit_transform(w_x_trajs[key])

        pca_acc[key] = PCA(n_components=60)
        w_acc_trajs_pca[key] = pca_acc[key].fit_transform(w_acc_trajs[key])

        pca_u[key] = PCA(n_components=60)
        w_u_trajs_pca[key] = pca_u[key].fit_transform(w_u_trajs[key])

    key = 'Right'
    
    for i in range(10):
        w_pca = w_trajs_pca[key][i]
        w = pca[key].inverse_transform(w_pca)
        w = w.reshape(60,-1)
        traj = np.dot(Phi,w)

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
        x_inputs_train[key], x_inputs_test[key], y_train[key], y_test[key] = train_test_split(x_inputs[key],w_trajs_pca[key], test_size = 0.15666, random_state=1)
        _,_, y_vel_train[key], y_vel_test[key] = train_test_split(x_inputs[key],w_vel_trajs_pca[key], test_size = 0.15666, random_state=1)
        _,_, y_u_train[key], y_u_test[key] = train_test_split(x_inputs[key],w_u_trajs_pca[key], test_size = 0.15666, random_state=1)
        _,_, y_acc_train[key], y_acc_test[key] = train_test_split(x_inputs[key],w_acc_trajs_pca[key], test_size = 0.15666, random_state=1)
        _,_, y_x_train[key], y_x_test[key] = train_test_split(x_inputs[key],w_x_trajs_pca[key], test_size = 0.15666, random_state=1)

    if learn_type == 0:
        print("SAVE REGRESSION PROBLEM")
        for key in keys:
            gpr[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca[key])
            #gpr[key].pca = pca[key]
            gpr[key].fit(x_inputs_train[key], y_train[key],num_restarts=3)
            gpr_vel[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_vel[key])
            #gpr_vel[key].pca = pca_vel[key]
            gpr_vel[key].fit(x_inputs_train[key], y_vel_train[key],num_restarts=3)
            gpr_acc[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_acc[key])
            #gpr_acc[key].pca = pca_vel[key]
            gpr_acc[key].fit(x_inputs_train[key], y_acc_train[key],num_restarts=3)
            gpr_x[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_x[key])
            #gpr_x[key].pca = pca_vel[key]
            gpr_x[key].fit(x_inputs_train[key], y_x_train[key],num_restarts=3)
            gpr_u[key] = GPy_Regressor(dim_input=x_inputs_train[key].shape[1], transform = pca_u[key])
            #gpr_u[key].pca = pca_vel[key]
            gpr_u[key].fit(x_inputs_train[key], y_u_train[key],num_restarts=3)

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
            key = 'Right'
            JJ = np.random.randint(x_inputs_test[key].shape[0])
            x = x_inputs_test[key][JJ][None,:]

            tic = time.time()
            w_traj,cov_traj = gpr[key].predict(x)
            w_vel,cov_vel = gpr_vel[key].predict(x)
            w_acc,cov = gpr_acc[key].predict(x)
            w_x,cov = gpr_x[key].predict(x)
            toc = time.time()
            
            w_acc = w_acc.reshape(rbf_num,-1)
            acc_traj = np.dot(Phi,w_acc)
            w_vel = w_vel.reshape(rbf_num,-1)
            vel_traj = np.dot(Phi,w_vel)
            w_x = w_x.reshape(rbf_num,-1)
            x_traj = np.dot(Phi,w_x)
            w_u,cov_u = gpr_u[key].predict(x)
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
    
    print("aaa")
    print(xs_pca_test)
    print(xs_pca[0])

    print(tt)
    print(tt1)
    
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()

