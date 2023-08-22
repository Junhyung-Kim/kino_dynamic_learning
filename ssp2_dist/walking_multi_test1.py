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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(43,105),nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(nn.Linear(105,420),nn.LeakyReLU(0.1))
        self.fc3 = nn.Sequential(nn.Linear(420,1260),nn.LeakyReLU(0.1))
    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        x = z.reshape(-1,60,21)
        return x
    
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(43,120),nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(nn.Linear(120,400),nn.LeakyReLU(0.1))
        self.fc3 = nn.Sequential(nn.Linear(400,1200),nn.LeakyReLU(0.1))
    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        x = z.reshape(-1,60,20)
        return x
    
class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(43,80),nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(nn.Linear(80,240),nn.LeakyReLU(0.1))
        self.fc3 = nn.Sequential(nn.Linear(240,480),nn.LeakyReLU(0.1))
    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        x = z.reshape(-1,60,8)
        return x        

def PCAlearning(time_step):
    global xs_pca_test, rbf_num
    global xs_pca
    global us_pca

    learn_type = 1
    learn_type1 = 1
    
    file_name ='/home/jhk/kino_dynamic_learning/dataset/dataset1/'

    naming = [
        "timestep=0_finish_ssp2",  
        "timestep=1_finish_ssp2",  
"timestep=2_finish_ssp2", 
"timestep=3_finish_ssp2",       
"timestep=4_finish_ssp2",    
"timestep=5_finish_ssp2",
"timestep=6_finish_ssp2",
"timestep=7_finish_ssp2",
"timestep=8_finish_ssp2",
"timestep=9_finish_ssp2",
"timestep=10_finish_ssp2",  
"timestep=11_finish_ssp2",  
"timestep=12_finish_ssp2", 
"timestep=13_finish_ssp2",       
"timestep=14_finish_ssp2",    
"timestep=15_finish_ssp2",
"timestep=16_finish_ssp2",
"timestep=17_finish_ssp2", ####
"timestep=18_finish_ssp2",
"timestep=19_finish_ssp2",
"timestep=20_finish_ssp2",  
"timestep=21_finish_ssp2",  
"timestep=22_finish_ssp2", 
"timestep=23_finish_ssp2",       
"timestep=24_finish_ssp2",    
"timestep=25_finish_ssp2",
"timestep=26_finish_ssp2",
"timestep=27_finish_ssp2",
"timestep=28_finish_ssp2",
"timestep=29_finish_ssp2",
"timestep=30_finish_ssp2",  
"timestep=31_finish_ssp2",  
"timestep=32_finish_ssp2", 
"timestep=33_finish_ssp2",       
"timestep=34_finish_ssp2",    
"timestep=35_finish_ssp2",
"timestep=36_finish_ssp2",
"timestep=37_finish_ssp2",
"timestep=38_finish_ssp2",
"timestep=39_finish_ssp2",
"timestep=40_finish_ssp2",  
"timestep=41_finish_ssp2",  
"timestep=42_finish_ssp2", 
"timestep=43_finish_ssp2",       
"timestep=44_finish_ssp2",    
"timestep=45_finish_ssp2",
"timestep=46_finish_ssp2",
"timestep=47_finish_ssp2",
"timestep=48_finish_ssp2",
]

    #if time_step 

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
    key = 'Right'
    
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
    device = 'cpu'
    model = Decoder().to(device)
    model1 = Decoder1().to(device)
    model2 = Decoder3().to(device)

    model.eval()
    model1.eval()
    model2.eval()

    file_name = '/home/jhk/ssd_mount/beforedata/ssp2/mlpEarly' 
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

    file_name = '/home/jhk/kino_dynamic_learning/dataset/dataset2/'
    file_name2 = 'x_inputs_train_'
    file_name3 = '.pt'
    file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
    x_inputs_train = torch.load(file_name4)
    file_name2 = 'x_inputs_test_'
    file_name4 = file_name  +file_name2+ str(time_step)+ file_name3
    x_inputs_test = torch.load(file_name4)

    NN_.append(model)
    NN_VEL.append(model1)
    NN_X.append(model2)
   
    if time_step == 0:
        global X_INIT
        X_INIT = x_inputs_test[key]
    
    
def talker():
    global xs_pca_test, xs_pca, us_pca, rbf_num
    global q_traj, v_traj, a_traj, x_traj, u_traj
    global PCA_, NN_, PCA_VEL, NN_VEL, PCA_X, NN_X, X_INIT, PHI_
    PCA_ = []
    NN_ = []
    PCA_VEL = []
    NN_VEL = []
    PCA_X = []
    NN_X = []
    PHI_= []
    X_INIT = np.array([])
    '''
    print("start")
    mpc_signal = sysv_ipc.SharedMemory(1)
    mpc_signalv  = mpc_signal.read()
    mpc_signaldata =  np.ndarray(shape=(3,), dtype=np.int32, buffer=mpc_signalv)
    x_init = sysv_ipc.SharedMemory(2)
    x_initv  = x_init.read()
    statemachine = sysv_ipc.SharedMemory(3)
    statemachinedata = np.array([0], dtype=np.int8)
    
    thread_manager1 = []
    for i in range(0,40):
        thread_manager1.append(0)
    '''
    #thread_manager = multiprocessing.Array(ctypes.c_int, thread_manager1)
   
    N = 60
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    total_time = 49

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
    f = open("/home/jhk/walkingdata/beforedata/ssp2/lfoot1.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/ssp2/rfoot2.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/ssp2/zmp3.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')
    f5 = open("/home/jhk/walkingdata/beforedata/ssp2/zmp3.txt", 'r')

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

    for time_step in range(1, total_time):
        for i in range(0, N):
            if i == 0:
                array_boundRF_[i] = np.sum([array_boundRF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
            else:
                array_boundRF_[i] = np.sum([array_boundRF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
   
        for i in range(0, N):
            if i == 0:
                array_boundLF_[i] = np.sum([array_boundLF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
            else:
                array_boundLF_[i] = np.sum([array_boundLF[k*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)

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
        if time_step == 1:
            for i in range(0, len(q)):    
                q[i] = q_init[i]
       
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

        x0[41] = data.com[0][0]
        x0[43] = data.com[0][0]
        x0[45] = data.com[0][1]
        x0[47] = data.com[0][1]
        
        if time_step == 1:    
            time_step_ = time_step
        
            X = np.zeros(43)
            for i in range(0, len(q)):
                X[i] = q[i]
            for i in range(len(q), len(q) + len(qdot)):
                X[i] = qdot[i - len(q)]
            X[41] = x0[43]
            X[42] = x0[47]

            #X = np.concatenate([q, qdot, x0[43], x0[47]])
            
            X = X.reshape(-1, 43)    
            
            ti = time.time()
            c = torch.tensor(X,dtype=torch.float32)
            w_traj = NN_[time_step_-1].forward(c)
            print(np.shape(w_traj))
            print(w_traj[0][0])
            w_traj = w_traj.reshape(60,21).detach().numpy()
            print(w_traj[0])
            print(qdot)
            print("qend")
            
            q_traj = w_traj.flatten()
            t2 = time.time()  
            print(t2 - ti)
            print(x0)
            print(w_traj[0])

            ti = time.time()
            w_traj_dot = NN_VEL[time_step_-1].forward(c)
            w_traj_dot = w_traj_dot.reshape(60,20).detach().numpy()
            v_traj = w_traj_dot.flatten()
            t2 = time.time()
            acc_traj = np.subtract(w_traj_dot[1:60,:], w_traj_dot[0:59,:])/0.02
            t3 = time.time()
            print(t2 - ti)
            print(t3 - ti)

            ti = time.time()
            w_traj_x = NN_X[time_step_-1].forward(c)
            w_traj_x = w_traj_x.reshape(60,8).detach().numpy()
            x_traj = w_traj_x.flatten()
            t2 = time.time()
            print(t2 - ti)
            u_traj = np.zeros([59, 4])
            for i in range(0, 59):
                u_traj[i][0] = (w_traj_x[i + 1][2] - w_traj_x[i][2])/0.02
                u_traj[i][1] = (w_traj_x[i + 1][3] - w_traj_x[i][3])/0.02
                u_traj[i][2] = (w_traj_x[i + 1][6] - w_traj_x[i][6])/0.02
                u_traj[i][3] = (w_traj_x[i + 1][7] - w_traj_x[i][7])/0.02

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
        else:
            x0 = ddp.xs[1]
            time_step_ = time_step

            for i in range(0, 21):
                q[i] = ddp.xs[1][i]
            for i in range(0, 20):
                qdot[i] = ddp.xs[1][i+21]
            
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
        
            
            print(time_step_)
            print("com")
            print(data.com[0])
            print([ddp.xs[1][41], ddp.xs[1][45]])
            print([data.hg.angular[0], data.hg.angular[1]])
            print([ddp.xs[1][48], ddp.xs[1][44]])
            print(data.oMf[RFframe_id].translation)
            print(array_boundRF_[1])
            print(data.oMf[LFframe_id].translation)
            print(array_boundLF_[1])
            
            X = np.array(ddp.xs[1][0:21])
            X = np.append(X, ddp.xs[1][21:41])
            X = np.append(X, ddp.xs[1][43]) 
            X = np.append(X, ddp.xs[1][47])
            X = X.reshape(-1, 43)    
            
            c = torch.tensor(X,dtype=torch.float32)
            ti = time.time()
            w_traj = NN_[time_step_-1].forward(c)
            t2 = time.time()  
            w_traj = w_traj.reshape(60,21).detach().numpy()
            q_traj = w_traj.flatten()
            
            print("q")
            print(t2-ti)

            ti = time.time()
            w_traj_dot = NN_VEL[time_step_-1].forward(c)
            w_traj_dot = w_traj_dot.reshape(60,20).detach().numpy()
            v_traj = w_traj_dot.flatten()
            t2 = time.time()
            acc_traj = np.subtract(w_traj_dot[1:60,:], w_traj_dot[0:59,:])/0.02
            t3 = time.time()
            print(t2 - ti)
            print(t3 - ti)

            ti = time.time()
            w_traj_x = NN_X[time_step_-1].forward(c)
            w_traj_x = w_traj_x.reshape(60,8).detach().numpy()
            x_traj = w_traj_x.flatten()
            t2 = time.time()
            print(t2 - ti)
            u_traj = np.zeros([59, 4])
            for i in range(0, 59):
                u_traj[i][0] = (w_traj_x[i + 1][2] - w_traj_x[i][2])/0.02
                u_traj[i][1] = (w_traj_x[i + 1][3] - w_traj_x[i][3])/0.02
                u_traj[i][2] = (w_traj_x[i + 1][6] - w_traj_x[i][6])/0.02
                u_traj[i][3] = (w_traj_x[i + 1][7] - w_traj_x[i][7])/0.02

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
            '''
            print("0000")
            print(xs_pca[0])
            print("1111")
            print(xs_pca[1])
            print("x0")
            print(x0)
            '''

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
        weight_quad_zmp = np.array([1.0, 1.0])#([weight_quad_zmpx] + [weight_quad_zmpy])
        weight_quad_zmp1 = np.array([5.0, 8.0]) ##11
        weight_quad_cam = np.array([0.003, 0.003])#([weight_quad_camy] + [weight_quad_camx])
        weight_quad_upper = np.array([0.0005, 0.0005])
        weight_quad_com = np.array([20.0,20.0, 3.0])#([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
        weight_quad_rf = np.array([20.0, 3.0, 5.0, 0.5, 0.5, 0.5])#np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
        weight_quad_lf = np.array([20.0, 3.0, 5.0, 0.5, 0.5, 0.5])#np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
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
           
            if i >= 0:
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
       

        for i in range(0, N-1):  
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
        #print(x0)
        problemWithRK4.x0 = x0
        ddp.th_stop = 0.0000005
        c_start = time.time()
        css = ddp.solve(xs_pca, us_pca, False, True, 0.00001)
        c_end = time.time()
        duration = (1e3 * (c_end - c_start))

        #costs.cost.ref
        print("timestep")
        print(time_step)
        avrg_duration = duration
        min_duration = duration #min(duration)
        max_duration = duration #max(duration)
        print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
        print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, ddp.cost))
        #if time_step == 15:
        #    s = asdfasds
        
        traj = np.array(ddp.xs)[:,0:21]
        vel_traj = np.array(ddp.xs)[:,21:41]
        x_traj = np.array(ddp.xs)[:, 41:49]
        u_traj = np.array(ddp.us)[:,20:24]
        acc_traj = np.array(ddp.us)[:, 0:20]
        
        crocs_data['Right']['x_inputs'].append(copy(ddp.xs[0][0:21]))
        crocs_data['Right']['vel_trajs'].append(copy(vel_traj[0][0:20]))
        crocs_data['Right']['x_state'].append(copy(x_traj[0][0:8]))
        crocs_data['Right']['costs'].append(copy(ddp.cost))
        crocs_data['Right']['iters'].append(copy(ddp.iter))
        crocs_data['Right']['trajs'].append(copy(traj[0][0:21]))
        crocs_data['Right']['u_trajs'].append(copy(acc_traj[0][0:20]))
        crocs_data['Right']['acc_trajs'].append(copy(u_traj[0][0:4]))
           
        with open('/home/jhk/ssd_mount/filename3.pkl', 'wb') as f:
	        pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("success")
        '''
        if time_step == 26:
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
            
            
            with open('/home/jhk/ssd_mount/filename3.pkl', 'wb') as f:
                pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print("success")
            print(x_traj[0])
            print(x_traj[1])
            print([array_boundy_[0], array_boundy_[1]])
            a =sdfasdf
        '''
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()