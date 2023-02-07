import roslibpy
import crocoddyl
import pinocchio
import numpy as np
import time
import example_robot_data
from copy import copy
import random
import pickle
#from regression import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

global client

def talker():
    print("start")
    f = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f2 = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f1 = open("/home/jhk/data/mpc/10_tocabi_.txt", 'r')

    f6 = open("/home/jhk/data/mpc/6_tocabi_py1.txt", 'r')
    f7 = open("/home/jhk/data/mpc/6_tocabi_py.txt", 'r')
    f5 = open("/home/jhk/data/mpc/6_tocabi_py_save.txt", 'r')

    f3 = open("/home/jhk/data/mpc/5_tocabi_py.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_py.txt", 'w')

    lines = f.read().split(' ')
    lines1 = f2.readlines()
    lines2 = f1.readlines()
    lines3 = f6.readlines()
    lines4 = f5.readlines()
    lines5 = f7.readlines()
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

    array_qdot1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_q1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_xstate1 = [[] for i in range(int(len(lines3)/208) * 30)]
    array_u1 = [[] for i in range(int(len(lines3)/208)*29)]
    array_qddot1 = [[] for i in range(int(len(lines3)/208)*29)]

    bool_q = 0

    N = 40    
    T = 1
    MAXITER = 300
    dt_ = 0.03
    lines1_array = []
    for i in range(0, len(lines3)):
        lines1_array.append(lines3[i].split())

    lines2_array = []
    for i in range(0, len(lines2)):
        lines2_array.append(lines2[i].split())

    lines6_array = []
    for i in range(0, len(lines5)):
        lines6_array.append(lines5[i].split())
    
    lines5_array = []
    for i in range(0, len(lines4)):
        lines5_array.append(lines4[i].split())

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


    array_boundx = [[] for i in range(int(len(lines2_array)))]
    array_boundy = [[] for i in range(int(len(lines2_array)))]

    array_boundRF = [[] for i in range(int(len(lines2_array)))]
    array_boundLF = [[] for i in range(int(len(lines2_array)))]

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
        array_boundx[i].append(float(lines2_array[i][1]))
        array_boundx[i].append(float(lines2_array[i][2]))
        array_boundy[i].append(float(lines2_array[i][3]))
        array_boundy[i].append(float(lines2_array[i][4]))

        array_boundRF[i].append(float(lines2_array[i][5]))
        array_boundRF[i].append(float(lines2_array[i][6]))
        array_boundRF[i].append(float(lines2_array[i][7]))

        array_boundLF[i].append(float(lines2_array[i][8]))
        array_boundLF[i].append(float(lines2_array[i][9]))
        array_boundLF[i].append(float(lines2_array[i][10]))

    f.close()
    f1.close()
    f2.close()

    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = pinocchio.buildModelFromUrdf("/home/jhk/catkin_ws/src/tocabi_cc/robots/dyros_tocabi_with_redhands2.urdf",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    q = pinocchio.randomConfiguration(model)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    
    for i in range(0, len(q)):
        q[i] = q_init[i]
    
    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")
    LHjoint_id = model.getJointId("L_HipYaw_Joint")
    RHjoint_id = model.getJointId("R_HipYaw_Joint")
    LFframe_id = model.getFrameId("L_Foot_Link")
    RFframe_id = model.getFrameId("R_Foot_Link")
    PELVframe_id = model.getFrameId("Pelvis_Link")
    PELVjoint_id = model.getJointId("root_joint")
    Pelvis_id = model.getFrameId("Pelvis_Link")  

    contactPointLF = pinocchio.SE3.Identity()
    contactPointRF = pinocchio.SE3.Identity()
    
    contactPointLF.translation.T.flat = [0.03, 0, -0.1585]
    contactPointRF.translation.T.flat = [0.03, 0, -0.1585]

    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")

    model.addBodyFrame("LF_contact", LFjoint_id, contactPointLF, LFframe_id)
    model.addBodyFrame("RF_contact", RFjoint_id, contactPointRF, RFframe_id)

    LFcframe_id = model.getFrameId("LF_contact")
    RFcframe_id = model.getFrameId("RF_contact")

    data = model.createData()

    pinocchio.forwardKinematics(model, data, q, qdot)
    pinocchio.updateFramePlacements(model,data)
    pinocchio.updateGlobalPlacements(model,data)
    pinocchio.computeJointJacobians(model, data, q)
    pinocchio.centerOfMass(model, data, q, False)
    pinocchio.computeCentroidalMomentum(model,data,q,qdot)

    state = crocoddyl.StateKinodynamic(model)
    actuation = crocoddyl.ActuationModelKinoBase(state)
    traj_= [0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.08, 0.0, 0.0, 0.0]
    u_traj_ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    LF_tran = data.oMi[LFjoint_id].translation
    RF_tran = data.oMi[RFjoint_id].translation
    LF_rot = data.oMi[LFjoint_id].rotation
    RF_rot = data.oMi[RFjoint_id].rotation

    RFc_tran_init = data.oMf[RFcframe_id].translation
    LFc_tran_init = data.oMf[LFcframe_id].translation

    PELV_tran = data.oMf[PELVframe_id].translation
    LF_tran = data.oMi[LFjoint_id].translation
    RF_tran = data.oMi[RFjoint_id].translation
    HLR_tran_init = data.oMi[LHjoint_id].translation
    HRR_tran_init = data.oMi[RHjoint_id].translation
    HLR_rot_init = data.oMi[LHjoint_id].rotation
    HRR_rot_init = data.oMi[RHjoint_id].rotation
    LHjoint_id = model.getJointId("L_HipYaw_Joint")
    RHjoint_id = model.getJointId("R_HipYaw_Joint")
    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")
    PELV_rot = data.oMf[PELVframe_id].rotation
    RF_rot = data.oMf[RFframe_id].rotation
    LF_rot = data.oMf[LFframe_id].rotation

    PELV_tran_init = np.add(data.oMi[PELVjoint_id].translation, model.inertias[PELVjoint_id].lever)
    CPELV_tran_init = data.oMi[PELVjoint_id].translation 
    PELV_rot_init = data.oMi[PELVjoint_id].rotation
    
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
    #weight_quad_pelvis = np.array([0.0] +[0.0] +[0.0] +[weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
    weight_quad_pelvis = np.array([weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])

    print(weight_quad_zmp)
    print(weight_quad_cam)
    print(weight_quad_com)
    print(weight_quad_rf)

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
        rf_foot_pos_vector[i].translation = copy(RF_tran)
        lf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        pelvis_rot_vector[i] = pinocchio.SE3.Identity()
        lf_foot_pos_vector[i].translation = copy(LF_tran)
        residual_FramePelvis[i] = crocoddyl.ResidualKinoFrameRotation(state_vector[i], Pelvis_id, pelvis_rot_vector[i].rotation, actuation_vector[i].nu + 4)
        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        PelvisR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[i])
        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu+4)
        runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], weight_quad_zmp[0])
        runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
        runningCostModel_vector[i].addCost("pelvisReg1", PelvisR[i], 1.0)
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
    rf_foot_pos_vector[N-1].translation = copy(RF_tran)
    lf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    lf_foot_pos_vector[N-1].translation = copy(LF_tran)
    pelvis_rot_vector[N-1] = pinocchio.SE3.Identity()
    residual_FramePelvis[N-1] = crocoddyl.ResidualKinoFrameRotation(state_vector[N-1], Pelvis_id, pelvis_rot_vector[N-1].rotation, actuation_vector[N-1].nu + 4)
    PelvisR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[N-1])
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
    terminalCostModel.addCost("pelvisReg1", PelvisR[N-1], 1.0)
    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N-1], actuation_vector[N-1], terminalCostModel)
    
    walking_tick = 23

    #model IK
    x0 = np.array([0.] * (state.nx + 8))
    u0 = np.array([0.] * (22))
    for i in range(0,len(q_init)):
        x0[i] = q_init[i]

    x0[37] = 1.12959174e-01
    x0[39] = 1.12959174e-01

    for i in range(0,N):
        xs[i] = copy(x0)
    for i in range(0,N-1):
        us[i] = copy(u0)
        #elif i >= 6 * end_k 


    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 6
    ddp = crocoddyl.SolverBoxFDDP(problemWithRK4)

    crocs_data = dict()
    crocs_data['left'] = dict()
    crocs_data['right'] = dict()
    
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

    
    for k in range(0, len(q)):
        q[k] = q_init[k]
        pinocchio.forwardKinematics(model, data, q, qdot)
        pinocchio.updateFramePlacements(model,data)
        pinocchio.updateGlobalPlacements(model,data)
        pinocchio.computeJointJacobians(model, data, q)
        pinocchio.centerOfMass(model, data, q, False)

    for l in range(0,len(q_init)):
        x0[l] = q[l]

    x0[37] = data.com[0][0]
    x0[39] = data.com[0][0]

    x0[41] = data.com[0][1]
    x0[43] = data.com[0][1]

    #xa = [0.0032530184588006545, -0.1087520714290202, 0.8015646022673689, 2.1729979745988774e-05, -5.03686773466601e-05, -1.7157640401248844e-05, 0.9999999983482099, 1.5954918468562534e-05, 0.19180860766098048, -0.7091508734585984, 1.2368432236753062, -0.5275639659891859, -0.19187336640794853, -3.614881197853655e-05, 0.19338611153762006, -0.554126309481918, 1.261619741821, -0.7073669610158738, -0.19339488921204567,0.00020186360374745398, -0.11594855764412491, 0.02262471439770103, -0.005193437605771742, -0.003403872770827001, -0.002611626653977417, 0.003365079538628551, 0.09851986749457962, 0.016438340968377412, -0.07323179931116768, 0.06810302849025111, -0.09604412159700464, -0.005365253082123043, 0.11543235463447504, -0.0862148267007832, -0.0024546483277480755, 0.08663855894951594, -0.1066289703687552,0.13491752095618878, 0.014318792738289, 0.1252321829327175, -0.35152585016562715, -0.06745227034682875, -0.23653931775715392, 0.05084450905067229, 0.4674350607626813]

   # for l in range(0,len(x0)):
   #     x0[l] = xa[l]
    
    for l in range(0,N):
        xs[l] = copy(x0)

    #print(xs)


    walking_tick = int(27)

    while True:    
        for l in range(0,N-1):
            state_bounds[l].lb[0] = copy(array_boundx[walking_tick + 3*l][0])
            state_bounds[l].ub[0] = copy(array_boundx[walking_tick + 3*l][1])
            state_bounds[l].lb[1] = copy(array_boundy[walking_tick + 3*l][0])
            state_bounds[l].ub[1] = copy(array_boundy[walking_tick + 3*l][1])
            state_activations[l].bounds = state_bounds[l]
            stateBoundCost_vector[l].activation_ = state_activations[l]
            rf_foot_pos_vector[l].translation[0] = copy(array_boundRF[walking_tick + 3*l][0] + 0.0011286)
            rf_foot_pos_vector[l].translation[1] = copy(array_boundRF[walking_tick + 3*l][1])
            rf_foot_pos_vector[l].translation[2] = copy(array_boundRF[walking_tick + 3*l][2] - 0.016954)
            lf_foot_pos_vector[l].translation[0] = copy(array_boundLF[walking_tick + 3*l][0] + 0.0011286)
            lf_foot_pos_vector[l].translation[1] = copy(array_boundLF[walking_tick + 3*l][1])
            lf_foot_pos_vector[l].translation[2] = copy(array_boundLF[walking_tick + 3*l][2] - 0.016954)
            residual_FrameRF[l] = crocoddyl.ResidualKinoFramePlacement(state_vector[l], RFframe_id, rf_foot_pos_vector[l], actuation_vector[l].nu + 4)
            residual_FrameLF[l] = crocoddyl.ResidualKinoFramePlacement(state_vector[l], LFframe_id, lf_foot_pos_vector[l], actuation_vector[l].nu + 4)
            foot_trackR[l] = crocoddyl.CostModelResidual(state_vector[l], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[l])
            foot_trackL[l] = crocoddyl.CostModelResidual(state_vector[l], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[l])
            runningCostModel_vector[l].removeCost("footReg1")
            runningCostModel_vector[l].removeCost("footReg2")
            runningCostModel_vector[l].addCost("footReg1", foot_trackR[l], 1.0)
            runningCostModel_vector[l].addCost("footReg2", foot_trackL[l], 1.0)

        state_bounds[N-1].lb[0] = copy(array_boundx[walking_tick + 3*(N-1)][0])
        state_bounds[N-1].ub[0] = copy(array_boundx[walking_tick + 3*(N-1)][1])
        state_bounds[N-1].lb[1] = copy(array_boundy[walking_tick + 3*(N-1)][0])
        state_bounds[N-1].ub[1] = copy(array_boundy[walking_tick + 3*(N-1)][1])
        state_activations[N-1].bounds = state_bounds[N-1]
        stateBoundCost_vector[N-1].activation_ = state_activations[N-1]
        rf_foot_pos_vector[N-1].translation[0] = copy(array_boundRF[walking_tick + 3*(N-1)][0] + 0.0011286)
        rf_foot_pos_vector[N-1].translation[1] = copy(array_boundRF[walking_tick + 3*(N-1)][1])
        rf_foot_pos_vector[N-1].translation[2] = copy(array_boundRF[walking_tick + 3*(N-1)][2]- 0.016954)
        lf_foot_pos_vector[N-1].translation[0] = copy(array_boundLF[walking_tick + 3*(N-1)][0] + 0.0011286)
        lf_foot_pos_vector[N-1].translation[1] = copy(array_boundLF[walking_tick + 3*(N-1)][1])
        lf_foot_pos_vector[N-1].translation[2] = copy(array_boundLF[walking_tick + 3*(N-1)][2]- 0.016954)
        residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
        residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)    
        foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
        foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
        terminalCostModel.removeCost("footReg1")
        terminalCostModel.removeCost("footReg2")
        terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
        terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
        k_temp = 0
        iter_ = 0
        booltemp = True
        a_iter = 0

        if(walking_tick < 60):
            ddp.th_stop = 0.0001
        else:
            ddp.th_stop = 1e-5
        start_k = 1
        end_k = 40
        x0_count = 0

        if(walking_tick ==27):
            for i in range(0, len(lines5_array)):
                if i >= 8*start_k and i <=8*end_k -1:
                    if i == 8*start_k:
                        print(le)
                    if i % 8 == 2:
                        for j in range(0, 19):
                            xs[x0_count][j] = float(lines5_array[i][j].strip(','))
                    if i % 8 == 3:
                        for j in range(0, 18): 
                            xs[x0_count][j+19] = float(lines5_array[i][j].strip(','))
                    if i % 8 == 4:
                        for j in range(0, 8): 
                            xs[x0_count][j+37] = float(lines5_array[i][j].strip(','))
                    if i % 8 == 6:
                        for j in range(0, 18): 
                            us[x0_count][j] = float(lines5_array[i][j].strip(','))
                    if i % 8 == 7:
                        for j in range(0, 4): 
                            us[x0_count][j+18] = float(lines5_array[i][j].strip(','))
                        x0_count = x0_count +1

            #for j in range(x0_count-1, N):
            #    xs[j] = xs[j]
            xs[N-1] = xs[N-2]
            x0 = xs[0]
            print(x0_count)
            
            problemWithRK4.x0 = x0     
            print(xs)       
            
            #ddp.solve(xs,us,3000,False)
        else:
            print(walking_tick)
            problemWithRK4.x0 = ddp.xs[start_k]
            for i in range(0, N-start_k):
                ddp.xs[i] = ddp.xs[i+start_k]
            for i in range(0, N-1-start_k):
                ddp.us[i] = ddp.us[i+start_k]
            for i in range(N-start_k, N):
                ddp.xs[i] = ddp.xs[N-1-start_k]

        while booltemp == True:
            booltemp1 = True

            c_start = time.time()
            if(walking_tick ==27):
                css = ddp.solve(xs, us, 10000, False, 500)     
            else:
                css = ddp.solve(ddp.xs, ddp.us, 10000, False, 500)
            c_end = time.time()
            duration = (1e3 * (c_end - c_start))

            avrg_duration = duration
            min_duration = duration #min(duration)
            max_duration = duration #max(duration)
            if(walking_tick >= 496):
                KKK = N
            else:
                KKK = N - 5
            for l in range(1,KKK):
                if l < N-1:
                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['footReg1'].cost.residual.reference.translation[a]) > 0.008:
                            print("11")
                            print(l)
                            print(runningCostModel_vector[l].costs['footReg1'].cost.residual.reference.translation)
                            booltemp1 = False
                            
                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['footReg2'].cost.residual.reference.translation[a]) > 0.008:
                            print("22")
                            print(l)
                            print(runningCostModel_vector[l].costs['footReg2'].cost.residual.reference.translation)
                            booltemp1 = False
                            
                    for a in range(0,2):
                        if abs(runningCostModel_vector[l].costs['comReg'].cost.residual.reference[a]) > 0.006:
                            print("33")
                            print(l)
                            print(runningCostModel_vector[l].costs['comReg'].cost.residual.reference)
                            booltemp1 = False
                            
                    if abs(runningCostModel_vector[l].costs['comReg'].cost.residual.reference[2]) > 0.008:
                            print("33")
                            print(l)
                            print(runningCostModel_vector[l].costs['comReg'].cost.residual.reference)
                            booltemp1 = False
                            
                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['camReg'].cost.residual.reference[a]) > 0.008:
                            print("44")
                            print(l)
                            print(runningCostModel_vector[l].costs['camReg'].cost.residual.reference)
                            booltemp1 = False
                            
                    
                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['footReg1'].cost.residual.reference.rotation[a,a]) > 0.02:
                            print("ori1")
                            #print(l)
                            #print(runningCostModel_vector[l].costs['footReg1'].cost.residual.reference.rotation)
                            booltemp1 = False
                            
                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['footReg2'].cost.residual.reference.rotation[a,a]) > 0.02:
                            print("ori2")
                            #print(l)
                            #print(runningCostModel_vector[l].costs['footReg2'].cost.residual.reference.rotation)
                            booltemp1 = False

                    for a in range(0,3):
                        if abs(runningCostModel_vector[l].costs['pelvisReg1'].cost.residual.reference[a,a]) > 0.02:
                            print("ori3")
                            #print(l)
                            #print(runningCostModel_vector[l].costs['pelvisReg1'].cost.residual.reference)
                            booltemp1 = False
                    
                else:
                    for a in range(0,3):
                        if abs(terminalCostModel.costs['footReg1'].cost.residual.reference.translation[a]) > 0.008:
                            print("111")
                            print(terminalCostModel.costs['footReg1'].cost.residual.reference.translation)
                            booltemp1 = False

                    for a in range(0,3):
                        if abs(terminalCostModel.costs['footReg2'].cost.residual.reference.translation[a]) > 0.008:
                            booltemp1 = False
                            print("222")
                            print(terminalCostModel.costs['footReg2'].cost.residual.reference.translation)

                    for a in range(0,2):
                        if abs(terminalCostModel.costs['comReg'].cost.residual.reference[a]) > 0.006:
                            booltemp1 = False
                            print("333")
                            print(terminalCostModel.costs['comReg'].cost.residual.reference)
                            
                    if abs(terminalCostModel.costs['comReg'].cost.residual.reference[2]) > 0.0065:
                            booltemp1 = False
                            print("333")
                            print(terminalCostModel.costs['comReg'].cost.residual.reference)

                    for a in range(0,3):
                        if abs(terminalCostModel.costs['camReg'].cost.residual.reference[a]) > 0.008:
                            print("444")
                            print(terminalCostModel.costs['camReg'].cost.residual.reference)
                            booltemp1 = False
                    
                    for a in range(0,3):
                        if abs(terminalCostModel.costs['footReg1'].cost.residual.reference.rotation[a,a]) > 0.02:
                            print("ori11")
                            booltemp1 = False
                    for a in range(0,3):
                        if abs(terminalCostModel.costs['footReg2'].cost.residual.reference.rotation[a,a]) > 0.02:
                            print("ori22")
                            booltemp1 = False
                    for a in range(0,3):
                        if abs(terminalCostModel.costs['pelvisReg1'].cost.residual.reference[a,a]) > 0.02:
                            print("ori33")
                            booltemp1 = False

            if booltemp1 == False:
                print("booltemp1 error")
        
            if(iter_ == 50):
                booltemp1 = True

            if css== True:
                booltemp1 = True
            
            print("iter")
            print(iter_)
            print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, css, max_duration))
            print('ddp.iter {0},{1},{2}'.format(ddp.iter, ddp.cost, walking_tick))
           
            
            if booltemp1 == True and css == True:# and ddp.cost < 0.02: 
                a_iter = 0        
                print(ddp.xs[1][:7])      
                if(walking_tick >= 496):
                    for k in range(0, start_k):
                        f4.write("walking_tick ")
                        f4.write(str(walking_tick + 3*k))
                        f4.write(" css ")
                        f4.write(str(ddp.iter))
                        f4.write(" ")
                        f4.write(str(css))
                        f4.write(" ")
                        f4.write(str(1.0))
                        f4.write("\n")
                        f4.write("q ")
                        f4.write(str(k))
                        f4.write("\n")
                        for a in range(0,19):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("qdot ")
                        f4.write(str(k))
                        f4.write("\n")            
                        for a in range(19,37):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("x_state ")
                        f4.write(str(k))
                        f4.write("\n")  
                        for a in range(37,45):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("\n")
                        f4.write("u ")
                        f4.write(str(k))
                        f4.write("\n")  
                else:
                    for k in range(0, N):
                        f4.write("walking_tick ")
                        f4.write(str(walking_tick + 3*k))
                        f4.write(" css ")
                        f4.write(str(ddp.iter))
                        f4.write(" ")
                        f4.write(str(css))
                        f4.write(" ")
                        f4.write(str(1.0))
                        f4.write("\n")
                        f4.write("q ")
                        f4.write(str(k))
                        f4.write("\n")
                        for a in range(0,19):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("qdot ")
                        f4.write(str(k))
                        f4.write("\n")            
                        for a in range(19,37):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("x_state ")
                        f4.write(str(k))
                        f4.write("\n")  
                        for a in range(37,45):
                            f4.write(str(ddp.xs[k][a]))
                            f4.write(", ")
                        f4.write("\n")
                        f4.write("u ")
                        f4.write(str(k))
                        f4.write("\n") 
                    
                for k in range(0, N-1):
                    f3.write("walking_tick ")
                    f3.write(str(walking_tick + 3*k))
                    f3.write(" css ")
                    f3.write(str(ddp.iter))
                    f3.write(" ")
                    f3.write(str(css))
                    f3.write(" ")
                    f3.write(str(1.0))
                    f3.write("\n")
                    f3.write("q ")
                    f3.write(str(k))
                    f3.write("\n")
                    for a in range(0,19):
                        f3.write(str(ddp.xs[k][a]))
                        f3.write(", ")
                    f3.write("qdot ")
                    f3.write(str(k))
                    f3.write("\n")            
                    for a in range(19,37):
                        f3.write(str(ddp.xs[k][a]))
                        f3.write(", ")
                    f3.write("x_state ")
                    f3.write(str(k))
                    f3.write("\n")  
                    for a in range(37,45):
                        f3.write(str(ddp.xs[k][a]))
                        f3.write(", ")
                    f3.write("\n")
                    f3.write("u ")
                    f3.write(str(k))
                    f3.write("\n")
                    for a in range(0,18):
                        f3.write(str(ddp.us[k][a]))
                        f3.write(", ")
                    f3.write("ustate ")
                    f3.write("\n")  
                    for a in range(18,22):
                        f3.write(str(ddp.us[k][a]))
                        f3.write(", ")
                    f3.write("\n")
                f3.write("q ")
                f3.write(str(N))
                f3.write("\n")
                for a in range(0,19):
                    f3.write(str(ddp.xs[N-1][a]))
                    f3.write(", ")
                f3.write("qdot ")
                f3.write(str(N-1))
                f3.write("\n")            
                for a in range(19,37):
                    f3.write(str(ddp.xs[N-1][a]))
                    f3.write(", ")
                f3.write("x_state ")
                f3.write(str(N-1))
                f3.write("\n")  
                for a in range(37,45):
                    f3.write(str(ddp.xs[N-1][a]))
                    f3.write(", ")
                f3.write("\n")
                print("Data save")
                walking_tick = walking_tick + 3*start_k
                booltemp = False
                break
            else:
                if booltemp1 == True:
                    print("Time fail")
                else:
                    print("error")
            
            iter_ = iter_ + 1
            k_temp = k_temp + 1
    '''
    for k in range(0, N):
        f4.write(str(walking_tick))
        f4.write(" ")
        f4.write(str(i))
        f4.write(" ")
        f4.write("lb")
        f4.write(str(array_boundx[(walking_tick)+i*2][0]))
        f4.write("ub ")
        f4.write(str(array_boundx[(walking_tick)+i*2][1]))
        f4.write(" ")
        f4.write("lb ")
        f4.write(str(array_boundy[(walking_tick)+i*2][0]))
        f4.write("ub ")
        f4.write(str(array_boundy[(walking_tick)+i*2][1]))
        f4.write(" ")
        f4.write(str(array_boundRF[(walking_tick)+i*2][0]))
        f4.write(" ")
        f4.write(str(array_boundRF[(walking_tick)+i*2][1]))
        f4.write(" ")
        f4.write(str(array_boundRF[(walking_tick)+i*2][2]))
        f4.write(" ")
        f4.write(str(array_boundLF[(walking_tick)+i*2][0]))
        f4.write(" ")
        f4.write(str(array_boundLF[(walking_tick)+i*2][1]))
        f4.write(" ")
        f4.write(str(array_boundLF[(walking_tick)+i*2][2]))
        f4.write("\n")
    '''
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()
