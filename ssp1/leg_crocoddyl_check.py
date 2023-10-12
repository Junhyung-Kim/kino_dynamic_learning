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

    with open('/home/jhk/ssd_mount/filename_result.pkl', 'rb') as f:
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
    f = open("/home/jhk/walkingdata/beforedata/ssp1/lfoot1.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/ssp1/rfoot2.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/ssp1/zmp3.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')
    f5 = open("/home/jhk/ssd_mount/zmp5.txt", 'r')


    lines = f.readlines()
    lines2 = f2.readlines()
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
    
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)
    k = 1
    k1 = 1
    k3 = 1
    #PCAlearning()
    time_step = 10
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
    
    for i in range(0, N):
        if i == 0:
            array_boundRF_[i] = np.sum([array_boundRF[k*i+ time_step], [-0.03, 0.0, 0.15842]], axis = 0)
        else:
            array_boundRF_[i] = np.sum([array_boundRF[k*i+ time_step], [-0.03, 0.0, 0.15842]], axis = 0)

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
            array_boundLF_[i] = np.sum([array_boundLF[k*i+ time_step], [-0.03, 0.0, 0.15842]], axis = 0)
        else:
            array_boundLF_[i] = np.sum([array_boundLF[k*(i)+ time_step], [-0.03, 0.0, 0.15842]], axis = 0)

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
            array_boundx_[i] = array_boundx[k3*i+ time_step]
            array_boundy_[i] = array_boundy[k3*i+ time_step]

        else:
            array_boundx_[i] = array_boundx[k3*(i)+ time_step]
            array_boundy_[i] = array_boundy[k3*(i)+ time_step]


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

    #with open('/home/jhk/ssd_mount/afterdata/integral/filename4.pkl', 'rb') as f:
    with open('/home/jhk/walkingdata/beforedata/ssp1/timestep=10_finish_re', 'rb') as f:
    #with open('/home/jhk/ssd_mount/ssp1_data6.txt', 'rb') as f:
    #file_name ='/home/jhk/ssd_mount/filenametest_'
    #file_name2 = '.pkl'
    #file_name3 = file_name + str(time_step) + file_name2
    #with open(file_name3, 'rb') as f:     
        database = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    print(database['Right']['trajs'][0][0])
    
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
       
        for i in range(30000,30001):#len(database[key]['trajs'])-1, len(database[key]['trajs'])):
            database_['Right']['x_inputs'].append(database['Right']['trajs'][i][0])
            trajs_temp = []
            acc_temp = []
            x_temp = []
            vel_temp = []
            u_temp = []
            for j in range(0, N):
                trajs_temp.append(database['Right']['trajs'][i][k1*j])
                x_temp.append(database['Right']['x_state'][i][k1*j])
                vel_temp.append(database['Right']['vel_trajs'][i][k1*j])
                if j != N-1:
                    acc_temp.append(database['Right']['acc_trajs'][i][k1*j])
                    u_temp.append(database['Right']['u_trajs'][i][k1*j])
                if j == N-1:
                    database_['Right']['trajs'].append(trajs_temp)
                    database_['Right']['acc_trajs'].append(acc_temp)
                    database_['Right']['x_state'].append(x_temp)
                    database_['Right']['vel_trajs'].append(vel_temp)
                    database_['Right']['u_trajs'].append(u_temp)
    database_1 = database_                
    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = RobotWrapper.BuildFromURDF("/usr/local/lib/python3.8/dist-packages/robot_properties_tocabi/resources/urdf/tocabi.urdf","/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/meshes",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    
    jointsToLock = ['Waist1_Joint',  'Neck_Joint', 'Head_Joint', 
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
    
    #with open('/home/jhk/ssd_mount/afterdata/integral/filename4.pkl', 'rb') as f:
    #//ho11me/jhk/ssd_mount/beforeprocessing/SSP/i=0-8,j=13/filename3.pkl
    #with open('/home/jhk/ssd_mount/beforedata/integrate_ssp1/filename3_40_-6775.pkl', 'rb') as f:
    
    with open('/home/jhk/walkingdata/beforedata/ssp1/timestep=10_finish_re', 'rb') as f:
    #file_name ='/home/jhk/ssd_mount/filenametest_'
    #file_name2 = '.pkl'
    #file_name3 = file_name + str(time_step) + file_name2
    #with open(file_name3, 'rb') as f:
        
        crocs_data = pickle.load(f,  encoding='iso-8859-1')
    f.close()
    ''''''
    for i in range(0, len(crocs_data['Right']['trajs'][0][15])):
        q[i] = crocs_data['Right']['trajs'][0][0][i]
    for i in range(0, len(crocs_data['Right']['vel_trajs'][0][15])):
        qdot[i] = crocs_data['Right']['vel_trajs'][0][0][i]

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
    dh_dq = np.zeros([6, model.nv])
    dhd_dq = np.zeros([6, model.nv])
    dhd_dv = np.zeros([6, model.nv])
    dhd_da = np.zeros([6, model.nv])
    
    #pinocchio.computeCentroidalDynamicsDerivatives(model.model, data, q, qdot, qdot, dh_dq, dhd_dq, dhd_dv, dhd_da)
    
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]

    LF_tran = data.oMi[LFjoint_id]
    RF_tran = data.oMi[RFjoint_id]

    #for i in range(0,  len(crocs_data['Right']['x_state'])):
    #    if (crocs_data['Right']['costs'][i] < 0.2):
    #        time1d = i
    time1d = len(crocs_data['Right']['x_state'])- 1
    for i in range(0, N-1):
        print(i)
        print(crocs_data['Right']['x_state'][time1d][i][2])
        a = [crocs_data['Right']['x_state'][time1d][i][2], crocs_data['Right']['x_state'][time1d][i][6]]
        b = [database['Right']['x_state'][time1d][i][2], database['Right']['x_state'][time1d][i][6]]
        #print("zmp")
        #print(a)
        #print(b)
        
        for j in range(0, len(crocs_data['Right']['trajs'][time1d][i])):
            q[j] = crocs_data['Right']['trajs'][time1d][i][j]
        for j in range(0, len(crocs_data['Right']['vel_trajs'][time1d][i])):
            qdot[j] = crocs_data['Right']['vel_trajs'][time1d][i][j]
        if i != 0:
            for j in range(0, len(crocs_data['Right']['u_trajs'][time1d][i])):
                qddot[j] = crocs_data['Right']['u_trajs'][time1d][i-1][j]
        else:
            for j in range(0, len(crocs_data['Right']['u_trajs'][time1d][i])):
                qddot[j] = 0.0

        pinocchio.forwardKinematics(model.model, data, q, qdot)
        pinocchio.updateFramePlacements(model.model,data)
        pinocchio.centerOfMass(model.model, data, q, False)
        pinocchio.computeCentroidalMomentumTimeVariation(model.model,data,q,qdot,qddot)
        a = [crocs_data['Right']['x_state'][time1d][i][0], crocs_data['Right']['x_state'][time1d][i][4], 0.81]
        b = [database['Right']['x_state'][time1d][i][0], database['Right']['x_state'][time1d][i][4], 0.81]
        
        print("com")
        print(a - data.com[0])
        
        
        print("cam")
        a = [crocs_data['Right']['x_state'][time1d][i][7], crocs_data['Right']['x_state'][time1d][i][3], 0.0]
        print(data.hg.angular - a)
        #print(data.hg.angular)
        #print(a)
        #print(data.hg.angular)
        #print(a)
        
        print("RFLF")
        print(RF_tran.translation - array_boundRF_[i])
        print(LF_tran.translation - array_boundLF_[i])
        print(array_boundLF_[i])

    
    for i in range(0, N):
        if (crocs_data['Right']['x_state'][time1d][i][2] > array_boundx_[i][0]) and (crocs_data['Right']['x_state'][time1d][i][2] < array_boundx_[i][1]):
            a1 = 0
        else:
            print("zmpx not ok")
            a = [crocs_data['Right']['x_state'][time1d][i][2], crocs_data['Right']['x_state'][time1d][i][3]]
            print(a[0])
            print(array_boundx_[i])
            print(i)

        if (crocs_data['Right']['x_state'][time1d][i][6] > array_boundy_[i][0]) and (crocs_data['Right']['x_state'][time1d][i][6] < array_boundy_[i][1]):
            a1 = 0
        else:
            print("zmpy not ok")
            a = [crocs_data['Right']['x_state'][time1d][i][6], crocs_data['Right']['x_state'][time1d][i][3]]
            print(a[0])
            print(array_boundy_[i])
            print(i)
    
    for i in range(0, N):
        print(i)
        print(crocs_data['Right']['trajs'][time1d][i][19:])
        #print(crocs_data['Right']['trajs'][time1d][i][6])
       # print(crocs_data['Right']['x_state'][time1d][i][4])
        print([array_boundx_[i],array_boundy_[i]])
        print([crocs_data['Right']['x_state'][time1d][i][2], crocs_data['Right']['x_state'][time1d][i][6]])
        
    print(time1d)
    '''
    for i in range(len(crocs_data['Right']['costs'])):
        if(crocs_data['Right']['costs'][i] > 0.001):
            print(i)
            print(crocs_data['Right']['costs'][i])
    
    '''
    print("ddd")
    print(crocs_data['Right']['trajs'][time1d][0])
    print(crocs_data['Right']['trajs'][time1d][59])
    print(crocs_data['Right']['x_state'][time1d][59])

    #print(crocs_data['Right']['vel_trajs'][time1d][0])
    #print(crocs_data['Right']['u_trajs'][time1d][0])
    print(crocs_data['Right']['costs'][time1d])

    '''
    for i in range(0, N):
        print(i)
        print(crocs_data['Right']['x_state'][time1d][i])
    '''
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()

