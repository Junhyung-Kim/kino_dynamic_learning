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
    
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")

    shared_x = CShmReader(100)
    shared_u = CShmReader(101)
    ddp_start = CShmReader(103)
    ddp_finish = CShmReader(103)
    ddp_restart = CShmReader(104)
    ddp_sol = CShmReader(105)

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
    q_ = pinocchio.utils.zero(model.nq)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    loop = 0
     
    xs_pca =[]
    us_pca =[]
    xs_pca_test = []
    
    xs_pca_ = np.ndarray((30,45), dtype=np.float, buffer=shared_x.memory.read())
    us_pca_ = np.ndarray((30,22), dtype=np.float, buffer=shared_u.memory.read())
    
    for q in zip(xs_pca_):
        xs_pca.append(np.concatenate([q[0]]))
    for q in zip(us_pca_[:N-1]):
        us_pca.append(np.concatenate([q[0]]))

    xs_pca_test = us_pca_[N-1]
    us_pca = us_pca[:N-1]

    for i in range(0, len(q_)):
        q_[i] = xs_pca_test[i]
    
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

    pinocchio.forwardKinematics(model, data, q_, qdot)
    pinocchio.updateFramePlacements(model,data)
    pinocchio.centerOfMass(model, data, q_, False)
    pinocchio.computeCentroidalMomentum(model,data,q_,qdot)
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
    problemWithRK4.nthreads = 2
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
    residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
    terminalCostModel.removeCost("footReg1")
    terminalCostModel.removeCost("footReg2")
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)

    finish = np.zeros((1,4), dtype=np.int32)
    newstart = np.zeros((1,4), dtype=np.int32)
    sol = np.zeros((1,45), dtype=np.double)

    while True:
        a = np.ndarray((1,4), dtype=np.int32, buffer=ddp_start.memory.read())[0][0]
        if a == 1:
            #ddp_start
            xs_pca =[]
            us_pca =[]
            xs_pca_test = []
        
            xs_pca_ = np.ndarray((30,45), dtype=np.float, buffer=shared_x.memory.read())
            us_pca_ = np.ndarray((30,22), dtype=np.float, buffer=shared_u.memory.read())
        
            for q in zip(xs_pca_):
                xs_pca.append(np.concatenate([q[0]]))
            for q in zip(us_pca_[:N-1]):
                us_pca.append(np.concatenate([q[0]]))

            xs_pca_test = us_pca_[N-1]
            us_pca = us_pca[:N-1]
            
            for i in range(0,len(q_init)):
                x0[i] = xs_pca_test[i]

            for i in range(0, len(q_)):
                q_[i] = xs_pca_test[i]
        
            pinocchio.forwardKinematics(model, data, q_, qdot)
            pinocchio.updateFramePlacements(model,data)
            pinocchio.centerOfMass(model, data, q_, False)
            pinocchio.computeCentroidalMomentum(model,data,q_,qdot)

            x0[37] = data.com[0][0]
            x0[39] = data.com[0][0]
            x0[41] = data.com[0][1]
            x0[43] = data.com[0][1]

            problemWithRK4.x0 = x0
            ddp.th_stop = 5.0
            c_start = time.time()
            css = ddp.solve(xs_pca, us_pca, 300, False, 20.0)
            c_end = time.time()
            duration = (1e3 * (c_end - c_start))
            sol = ddp.xs[1]
            ddp_sol.memory.write(sol)
            newstart[0] = 1
            ddp_restart.memory.write(newstart)
            
            avrg_duration = duration
            min_duration = duration #min(duration)
            max_duration = duration #max(duration)
            print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
            print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
            ddp_finish.memory.write(finish)
        elif a == 0:
            finish[0] = 0
            #ddp_finish.memory.write(finish)
        elif a == 2:
            break

    '''
    global thread_manager
    global p1, p2, p3, p4, p5

    for i in range(0,5):
        thread_manager[i] = 2
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
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

