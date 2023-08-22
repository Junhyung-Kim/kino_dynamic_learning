import roslibpy
import pickle
import numpy as np
import time
from copy import copy
import logging
import os
import pinocchio
import crocoddyl

import sys
import numpy.matlib
np.set_printoptions(threshold=sys.maxsize)
global client
global learn_type
from pinocchio.robot_wrapper import RobotWrapper
import ctypes

def PCAlearning():
    global xs_pca_test
    global xs_pca
    global us_pca

    learn_type = 1
    database = dict()
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

    with open('/home/jhk/ssd_mount/kdyn4_data.txt', 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()

    '''
    print(len(database['Right']['trajs'][0]))

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
    '''
    
def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")
    f = open("/home/jhk/ssd_mount/lfoot1.txt", 'r')
    f1 = open("/home/jhk/ssd_mount/rfoot1.txt", 'r')
    f2 = open("/home/jhk/ssd_mount/zmp.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')

    lines = f.readlines()
    lines2 = f2.readlines()
    lines1 = f1.readlines()

    array_boundx = [[] for i in range(int(len(lines2)))]
    array_boundy = [[] for i in range(int(len(lines2)))]

    array_boundx_ = [[] for i in range(30)]
    array_boundy_ = [[] for i in range(30)]

    array_boundRF = [[] for i in range(int(len(lines1)))]
    array_boundLF = [[] for i in range(int(len(lines1)))]

    array_boundRF_ = [[] for i in range(30)]
    array_boundLF_ = [[] for i in range(30)]


    N = 30
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    k = 8
    k1 = 8
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

    for i in range(0, len(lines_array)):
        for j in range(0, len(lines_array[i])):
            if j == 0:
                array_boundRF[i].append(float(lines_array[i][j]))
            if j == 1:
                array_boundRF[i].append(float(lines_array[i][j]))
            if j == 2:
                array_boundRF[i].append(float(lines_array[i][j]))
    
    for i in range(0, 30):
        array_boundRF_[i] = np.sum([array_boundRF[k*i], [-0.03, 0.0, 0.15842]], axis = 0)

    for i in range(0, len(lines1_array)):
        for j in range(0, len(lines1_array[i])):
            if j == 0:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 1:
                array_boundLF[i].append(float(lines1_array[i][j]))
            if j == 2:
                array_boundLF[i].append(float(lines1_array[i][j]))

    for i in range(0, 30):
        array_boundLF_[i] = np.sum([array_boundLF[k*i], [-0.03, 0.0, 0.15842]], axis = 0)

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

    for i in range(0, 30):
        array_boundx_[i] = array_boundx[k*i]
        array_boundy_[i] = array_boundy[k*i]


    f.close()
    f1.close()
    f2.close()
    
    database = dict()
    database['Right'] = dict()
    data_processing = True

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

    with open('/home/jhk/ssd_mount/kdyn4_data.txt', 'rb') as f:
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()

    if data_processing == True:
        database_ = dict()
        database_['Right'] = dict()
        database_1 = dict()
        database_1['Right'] = dict()

        for key in database_.keys():
            database_[key]['foot_poses'] = []
            database_[key]['trajs'] = []
            database_[key]['acc_trajs'] = []
            database_[key]['x_inputs'] = []
            database_[key]['vel_trajs'] = [] 
            database_[key]['x_state'] = []        
            database_[key]['u_trajs'] = []
            database_[key]['data_phases_set'] = []
            database_[key]['costs'] = [] 
            database_[key]['iters'] = []

        for i in range(0, len(database[key]['trajs'])):
            database_['Right']['x_inputs'].append(database['Right']['trajs'][i][0])
            trajs_temp = []
            acc_temp = []
            x_temp = []
            vel_temp = []
            u_temp = []
            for j in range(0, 30):
                trajs_temp.append(database['Right']['trajs'][i][k1*j])
                acc_temp.append(database['Right']['acc_trajs'][i][k1*j])
                x_temp.append(database['Right']['x_state'][i][k1*j])
                vel_temp.append(database['Right']['vel_trajs'][i][k1*j])
                u_temp.append(database['Right']['u_trajs'][i][k1*j])
                if j == 29:
                    database_['Right']['trajs'].append(trajs_temp)
                    database_['Right']['acc_trajs'].append(acc_temp)
                    database_['Right']['x_state'].append(x_temp)
                    database_['Right']['vel_trajs'].append(vel_temp)
                    database_['Right']['u_trajs'].append(u_temp)
    database_1 = database_                
    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = RobotWrapper.BuildFromURDF("/usr/local/lib/python3.8/dist-packages/robot_properties_tocabi/resources/urdf/tocabi.urdf","/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/meshes",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    
    jointsToLock = ['Waist1_Joint', 'Waist2_Joint', 'Upperbody_Joint', 'Neck_Joint', 'Head_Joint', 
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
    0, 0, 0,  
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
    
    with open('/home/jhk/ssd_mount/filename1.pkl', 'rb') as f:
        crocs_data = pickle.load(f,  encoding='iso-8859-1')
    f.close()

    for i in range(0, len(q)):
        q[i] = crocs_data['Right']['trajs'][0][15][i]
    for i in range(0, len(qdot)):
        qdot[i] = crocs_data['Right']['vel_trajs'][0][15][i]

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
    
    print("ff")
    print(LF_tran)
    print(RF_tran)
    print(array_boundLF_[5])
    print(array_boundRF_[5])
    print(data.com[0])
    print(crocs_data['Right']['trajs'][0][5])
    print(data.hg)


    for i in range(0, N):
        print(i)
        a = [crocs_data['Right']['x_state'][0][i][2], crocs_data['Right']['x_state'][0][i][6]]
        print("zmp")
        print(a)
        print(array_boundx_[i])
        print(array_boundy_[i])
        print("com")
        for j in range(0, len(q)):
            q[j] = crocs_data['Right']['trajs'][0][i][j]
        for j in range(0, len(qdot)):
            qdot[j] = crocs_data['Right']['vel_trajs'][0][i][j]
        pinocchio.forwardKinematics(model.model, data, q, qdot)
        pinocchio.updateFramePlacements(model.model,data)
        pinocchio.centerOfMass(model.model, data, q, False)
        pinocchio.computeCentroidalMomentum(model.model,data,q,qdot)
        a = [crocs_data['Right']['x_state'][0][i][0], crocs_data['Right']['x_state'][0][i][4], 0.81]
        print(data.com[0] -a)
        print("cam")
        a = [crocs_data['Right']['x_state'][0][i][7], crocs_data['Right']['x_state'][0][i][3], 0.0]
        print(data.hg.angular - a)
        #print(crocs_data['Right']['acc_trajs'][0][i])
    print("dd")
    for i in range(0, N):
        if (crocs_data['Right']['x_state'][0][i][2] > array_boundx_[i][0]) and (crocs_data['Right']['x_state'][0][i][2] < array_boundx_[i][1]):
            a1 = 0
        else:
            print("zmpx not ok")
            print(i)
        if (crocs_data['Right']['x_state'][0][i][6] > array_boundy_[i][0]) and (crocs_data['Right']['x_state'][0][i][6] < array_boundy_[i][1]):
            a1 = 0
        else:
            print("zmpy not ok")
            print(i)
        

    q_init = [-0.02456656147909892, -0.08181577602709429, 0.8301948278049228, -0.04201139979930479, -0.02797947630489103, 0.02482648311555383, 0.9984166649894822, -0.06190311912385309, 0.2399469984801109, -0.6249800589636709, 1.3266767175519611, -0.641756991434395, -0.15941886655851734, -0.0598247687956362, 0.22567650180896426, -0.46704070676668963, 1.1614411560468993, -0.6368622219771066, -0.1476579659640173]
    q_dot = [0.35372132554790586, 0.24686497570533733, -0.1358164800850723, 0.2695733393719637, -0.32390132053146997, 0.8168856169112034, -0.014752626408420334, -0.4147987848872471, -0.29226420454731167, 0.8387169115820758, -0.31599003992647867, -0.4282398672859218, 0.9116964920940118, 0.5210480108010829, 0.5875388401793415, 0.9391627601902741, -0.32419231150449784, -1.2897265739277992]
    for i in range(0, len(q)):
        q[i] = q_init[i]
    for i in range(0, len(qdot)):
        qdot[i] = q_dot[i]

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
    #pinocchio.updateFramePlacement(model.model,data,37)
    #pinocchio.updateFramePlacement(model.model,data,51)
    pinocchio.computeCentroidalMomentum(model.model,data,q,qdot)
    pinocchio.jacobianCenterOfMass(model.model, data, q)
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]
    '''vvvvvvvvvvvvvv
    print("Before")
    print(LF_tran)
    print(RF_tran)
    print(data.com[0])
    LF_tran = data.oMi[LFjoint_id]
    RF_tran = data.oMi[RFjoint_id]
    print("After")
    print(LF_tran)
    print(RF_tran)
    print(data.com[0])
    print(data.hg)
    '''
    x0[37] = data.com[0][0]
    x0[39] = data.com[0][0]
    x0[41] = data.com[0][1]
    x0[43] = data.com[0][1]
    
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
    weight_quad_pelvis = np.array([weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])

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
    '''
    for i in range(0,N-1):
        
        state_vector[i] = crocoddyl.StateKinodynamic(model.model)
        actuation_vector[i] = crocoddyl.ActuationModelKinoBase(state_vector[i])
        state_bounds[i] = crocoddyl.ActivationBounds(lb_[:,i],ub_[:,i])
        state_activations[i] = crocoddyl.ActivationModelQuadraticBarrier(state_bounds[i])
        stateBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], state_activations[i], crocoddyl.ResidualFlyState(state_vector[i], actuation_vector[i].nu + 4))
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
        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu+4)
        runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], weight_quad_zmp[0])
        runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
        runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
        runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)

    state_vector[N-1] = crocoddyl.StateKinodynamic(model.model)
    actuation_vector[N-1] = crocoddyl.ActuationModelKinoBase(state_vector[N-1])
    state_bounds[N-1] = crocoddyl.ActivationBounds(lb_[:,N-1],ub_[:,N-1])
    state_activations[N-1] = crocoddyl.ActivationModelQuadraticBarrier(state_bounds[N-1])
    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
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
    terminalCostModel.addCost("stateReg", stateBoundCost_vector[N-1], weight_quad_zmp[0])
    terminalCostModel.addCost("camReg", camBoundCost_vector[N-1], 1.0)
    terminalCostModel.addCost("comReg", comBoundCost_vector[N-1], 1.0)
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
    #terminalCostModel.addCost("pelvisReg1", PelvisR[N-1], 1.0)
    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N-1], actuation_vector[N-1], terminalCostModel)

    database_['Right']['trajs'].append(trajs_temp)
    database_['Right']['acc_trajs'].append(acc_temp)
    database_['Right']['x_state'].append(x_temp)
    database_['Right']['vel_trajs'].append(vel_temp)
    database_['Right']['u_trajs'].append(u_temp)

    for i in range(0,N):
        xs[i] = np.append(np.append(database_['Right']['trajs'][0][i], database_['Right']['vel_trajs'][0][i]),database_['Right']['x_state'][0][i])
    
    for i in range(0,N-1):
        us[i] = np.append(database_['Right']['u_trajs'][0][i], database_['Right']['acc_trajs'][0][i])
    
    print(weight_quad_zmpx)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 6
    ddp = crocoddyl.SolverBoxFDDP(problemWithRK4)
    #array_boundRF_
    for i in range(0,N-1):
        state_bounds[i].lb[0] = copy(array_boundx_[i][0])
        state_bounds[i].ub[0] = copy(array_boundx_[i][1])
        state_bounds[i].lb[1] = copy(array_boundy_[i][0])
        state_bounds[i].ub[1] = copy(array_boundy_[i][1])
        state_activations[i].bounds = state_bounds[i]
        stateBoundCost_vector[i].activation_ = state_activations[i]
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
    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
    terminalCostModel.removeCost("footReg1")
    terminalCostModel.removeCost("footReg2")
    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
    '''
    '''
    print("crocs")
    print(crocs_data['Right']['x_state'][0][0])
    print(crocs_data['Right']['x_state'][0][1])
    print(crocs_data['Right']['x_state'][0][2])
    print(crocs_data['Right']['acc_trajs'][0][0])
    print(crocs_data['Right']['acc_trajs'][0][1])
    print(crocs_data['Right']['acc_trajs'][0][2])
    print(weight_quad_com)
    print(len(crocs_data['Right']['acc_trajs']))
    '''
    '''
    problemWithRK4.x0 = xs[0]
    ddp.th_stop = 10.0
    c_start = time.time()
    css = ddp.solve(xs, us, 3000, False, 1.0)
    c_end = time.time()
    duration = (1e3 * (c_end - c_start))

    print("end")
    avrg_duration = duration
    min_duration = duration #min(duration)
    max_duration = duration #max(duration)
    print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
    print('ddp.iter {0},{1}'.format(ddp.iter, css))
    
    walking_tick = 0
    booltemp = True
    while client.is_connected:  
        iter_ = 0
        T = 1
        while booltemp == True:
            booltemp1 = True
            c_start = time.time()
            css = ddp.solve(ddp.xs, ddp.us, 3000, False, 1.0)
            c_end = time.time()
            duration = (1e3 * (c_end - c_start))
                
            avrg_duration = duration
            min_duration = duration #min(duration)
            max_duration = duration #max(duration)
            print(iter_)
            print(ddp.xs[0])
            print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
            print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
            print(ddp.cost)
            for i in range(0,N):
                print("loop")
                print(i)
                if i < N-1:
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['footReg1'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            print("foot1")
                            print(runningCostModel_vector[i].costs['footReg1'].cost.residual.reference.translation[j])
                    for j in range(0,3):
                        if abs(runningCostModel_vector[i].costs['footReg2'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            print("foot2")
                            print(runningCostModel_vector[i].costs['footReg2'].cost.residual.reference.translation[j])
                    for j in range(0,2):
                        if abs(runningCostModel_vector[i].costs['comReg'].cost.residual.reference[j]) > 0.008:
                            print(j)
                            booltemp1 = False
                            break
                    for j in range(0,2):
                        if abs(runningCostModel_vector[i].costs['camReg'].cost.residual.reference[j]) > 0.05:
                            booltemp1 = False
                            print("cam")
                            print(runningCostModel_vector[i].costs['camReg'].cost.residual.reference[j])
                            
                else:
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['footReg1'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            print("foot1")
                    for j in range(0,3):
                        if abs(terminalCostModel.costs['footReg2'].cost.residual.reference.translation[j]) > 0.008:
                            booltemp1 = False
                            print("foot2")
                    for j in range(0,2):
                        if abs(terminalCostModel.costs['comReg'].cost.residual.reference[j]) > 0.008:
                            booltemp1 = False
                            break
                    
                    for j in range(0,2):
                        if abs(terminalCostModel.costs['camReg'].cost.residual.reference[j]) > 0.05:
                            booltemp1 = False
                            print("cam")
                #if booltemp1 == False:
                #    break

            if booltemp1 == True:
                booltemp = False
                print("success")
                break

            if iter_ == 5:
                booltemp = False

            iter_ = iter_ + 1

        for i in range(0,N-1):
            #print(runningCostModel_vector[i].costs['comReg'].cost.residual)
            print(runningCostModel_vector[i].costs['camReg'].cost.residual)
            #print(runningCostModel_vector[i].costs['stateReg'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg1'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg2'].cost.residual)
        #print(terminalCostModel.costs['comReg'].cost.residual)
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
            f3.write(str(array_boundx_[i][0]))
            f3.write("ub ")
            f3.write(str(array_boundx_[i][1]))
            f3.write(" ")
            f3.write("lb ")
            f3.write(str(array_boundy_[i][0]))
            f3.write("ub ")
            f3.write(str(array_boundy_[i][1]))
            f3.write(" ")
            f3.write(str(array_boundRF_[i][0]))
            f3.write(" ")
            f3.write(str(array_boundRF_[i][1]))
            f3.write(" ")
            f3.write(str(array_boundRF_[i][2]))
            f3.write(" ")
            f3.write(str(array_boundLF_[i][0]))
            f3.write(" ")
            f3.write(str(array_boundLF_[i][1]))
            f3.write(" ")
            f3.write(str(array_boundLF_[i][2]))
            f3.write("\n")
        walking_tick = 3
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


