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
def rotateWithY(pitch_angle):
    rotate_with_y = np.zeros((3,3))

    rotate_with_y[0, 0] = np.cos(pitch_angle)
    rotate_with_y[1, 0] = 0.0
    rotate_with_y[2, 0] = -1 * np.sin(pitch_angle)

    rotate_with_y[0, 1] = 0.0
    rotate_with_y[1, 1] = 1.0
    rotate_with_y[2, 1] = 0.0

    rotate_with_y[0, 2] = np.sin(pitch_angle)
    rotate_with_y[1, 2] = 0.0
    rotate_with_y[2, 2] = np.cos(pitch_angle)

    return rotate_with_y

def rotateWithX(roll_angle):
    rotate_with_x = np.zeros((3,3))

    rotate_with_x[0, 0] = 1.0
    rotate_with_x[1, 0] = 0.0
    rotate_with_x[2, 0] = 0.0

    rotate_with_x[0, 1] = 0.0
    rotate_with_x[1, 1] = np.cos(roll_angle)
    rotate_with_x[2, 1] = np.sin(roll_angle)

    rotate_with_x[0, 2] = 0.0
    rotate_with_x[1, 2] = -1 * np.sin(roll_angle)
    rotate_with_x[2, 2] = np.cos(roll_angle)

    return rotate_with_x  

def rotateWithZ(yaw_angle):
    rotate_with_z = np.zeros((3,3))

    rotate_with_z[0, 0] = np.cos(yaw_angle)
    rotate_with_z[1, 0] = np.sin(yaw_angle)
    rotate_with_z[2, 0] = 0.0

    rotate_with_z[0, 1] = -1 * np.sin(yaw_angle)
    rotate_with_z[1, 1] = np.cos(yaw_angle)
    rotate_with_z[2, 1] = 0.0

    rotate_with_z[0, 2] = 0.0
    rotate_with_z[1, 2] = 0.0
    rotate_with_z[2, 2] = 1.0

    return rotate_with_z
def inverseKinematics(time, LF_rot_c, RF_rot_c, PELV_rot_c, LF_tran_c, RF_tran_c, PELV_tran_c, HRR_tran_init_c, HLR_tran_init_c, HRR_rot_init_c, HLR_rot_init_c, PELV_tran_init_c, PELV_rot_init_c, CPELV_tran_init_c):
    global leg_q, leg_qdot, leg_qddot, leg_qs, leg_qdots, leg_qddots
    M_PI = 3.14159265358979323846
    if time == 0:
        leg_q = np.zeros(12)
        leg_qdot = np.zeros(12)
        leg_qddot = np.zeros(12)
        total_tick =1
        leg_qs = np.zeros((int(total_tick), 12))
        leg_qdots = np.zeros((int(total_tick), 12))
        leg_qddots = np.zeros((int(total_tick), 12))

    l_upper = 0.35
    l_lower = 0.35

    offset_hip_pitch = 0.0
    offset_knee_pitch = 0.0
    offset_ankle_pitch = 0.0

    lpt = np.subtract(PELV_tran_c, LF_tran_c)
    rpt = np.subtract(PELV_tran_c, RF_tran_c)
    lp = np.matmul(np.transpose(LF_rot_c), lpt)
    rp = np.matmul(np.transpose(RF_rot_c), rpt)
   
    PELF_rot = np.matmul(np.transpose(PELV_rot_c), LF_rot_c)
    PERF_rot = np.matmul(np.transpose(PELV_rot_c), RF_rot_c)

    ld = np.zeros(3)  
    rd = np.zeros(3)

    ld[0] = HLR_tran_init_c[0] - PELV_tran_init_c[0]
    ld[1] = HLR_tran_init_c[1] - PELV_tran_init_c[1]
    ld[2] = -(CPELV_tran_init_c[2] - HLR_tran_init_c[2]) + (CPELV_tran_init_c[2] - PELV_tran_init_c[2])

    rd[0] = HRR_tran_init_c[0] - PELV_tran_init_c[0]
    rd[1] = HRR_tran_init_c[1] - PELV_tran_init_c[1]
    rd[2] = -(CPELV_tran_init_c[2] - HRR_tran_init_c[2]) + (CPELV_tran_init_c[2] - PELV_tran_init_c[2])

    ld = np.matmul(np.transpose(LF_rot_c), ld)
    rd = np.matmul(np.transpose(RF_rot_c), rd)

    lr = lp + ld
    rr = rp + rd

    lc = np.linalg.norm(lr)

    leg_q[3] = -1 * np.arccos((l_upper * l_upper + l_lower * l_lower - lc * lc) / (2 * l_upper * l_lower)) + M_PI
    l_ankle_pitch = np.arcsin((l_upper * np.sin(M_PI - leg_q[3])) / lc)
   
    leg_q[4] = -1 * np.arctan2(lr[0], np.sqrt(lr[1] * lr[1] + lr[2] * lr[2])) - l_ankle_pitch
    leg_q[5] = np.arctan2(lr[1], lr[2])

    r_tl2 = np.zeros((3,3))
    r_l2l3 = np.zeros((3,3))
    r_l3l4 = np.zeros((3,3))
    r_l4l5 = np.zeros((3,3))

    r_l2l3 = rotateWithY(leg_q[3])
    r_l3l4 = rotateWithY(leg_q[4])
    r_l4l5 = rotateWithX(leg_q[5])

    r_tl2 = np.matmul(np.matmul(np.matmul(PELF_rot, np.transpose(r_l4l5)),np.transpose(r_l3l4)),np.transpose(r_l2l3))
    leg_q[1] = np.arcsin(r_tl2[2, 1])

    c_lq5 = np.divide(-r_tl2[0, 1], np.cos(leg_q[1]))

    if c_lq5 > 1.0:
        c_lq5 = 1.0
    elif c_lq5 < -1.0:
        c_lq5 = -1.0
   
    leg_q[0] = -1 * np.arcsin(c_lq5)
    leg_q[2] = -1 * np.arcsin(r_tl2[2, 0] / np.cos(leg_q[1])) + offset_hip_pitch
    leg_q[3] = leg_q[3] - offset_knee_pitch
    leg_q[4] = leg_q[4] - offset_ankle_pitch

    rc = np.linalg.norm(rr)
    leg_q[9] = -1 * np.arccos((l_upper * l_upper + l_lower * l_lower - rc * rc) / (2 * l_upper * l_lower)) + M_PI

    r_ankle_pitch = np.arcsin((l_upper * np.sin(M_PI - leg_q[9])) / rc)
    leg_q[10] = -1 * np.arctan2(rr[0], np.sqrt(rr[1] * rr[1] + rr[2] * rr[2])) - r_ankle_pitch
    leg_q[11] = np.arctan2(rr[1], rr[2])
    r_tr2 = np.zeros((3,3))
    r_r2r3 = np.zeros((3,3))
    r_r3r4 = np.zeros((3,3))
    r_r4r5 = np.zeros((3,3))

    r_r2r3 = rotateWithY(leg_q[9])
    r_r3r4 = rotateWithY(leg_q[10])
    r_r4r5 = rotateWithX(leg_q[11])

    r_tr2 = np.matmul(np.matmul(np.matmul(PERF_rot, np.transpose(r_r4r5)),np.transpose(r_r3r4)),np.transpose(r_r2r3))
    leg_q[7] = np.arcsin(r_tr2[2,1])
    c_rq5 = -r_tr2[0, 1] / np.cos(leg_q[7])

    if c_rq5 > 1.0:
        c_rq5 = 1.0
    elif c_rq5 < -1.0:
        c_rq5 = -1.0
   
    leg_q[6] = -1* np.arcsin(c_rq5)
    leg_q[8] = np.arcsin(r_tr2[2, 0] / np.cos(leg_q[7])) - offset_hip_pitch
    leg_q[9] = -1 * leg_q[9] + offset_knee_pitch
    leg_q[10] = -1 * leg_q[10] + offset_ankle_pitch

    leg_q[0] = leg_q[0] * (-1)
    leg_q[6] = leg_q[6] * (-1)
    leg_q[8] = leg_q[8] * (-1)
    leg_q[9] = leg_q[9] * (-1)
    leg_q[10] = leg_q[10] * (-1)

    #leg_qs[time,:] = leg_q
    '''
    if(time == 0):
        leg_qs[time,:] = leg_q
        leg_qdots[time,:] = np.zeros(12)
        leg_qddots[time,:] = np.zeros(12)
    else:
        leg_qs[time,:] = leg_q
        leg_qdots[time,:] = np.subtract(leg_qs[time,:], leg_qs[time-1,:]) * hz
        leg_qddots[time,:] = np.subtract(leg_qdots[time,:], leg_qdots[time-1,:]) * hz
    '''
    return leg_q

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

    with open('/home/jhk/ssd_mount/Fdyn_data5.txt', 'rb') as f:
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
    f = open("/home/jhk/ssd_mount/lfoot2_3_2.txt", 'r')
    f1 = open("/home/jhk/ssd_mount/rfoot2_3_2.txt", 'r')
    f2 = open("/home/jhk/ssd_mount/zmp2_3_2.txt", 'r')
    f3 = open("/home/jhk/data/mpc/5_tocabi_data.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_data.txt", 'w')
    f5 = open("/home/jhk/ssd_mount/zmp5.txt", 'r')


    lines = f.readlines()
    lines2 = f2.readlines()
    lines3 = f5.readlines()  
    lines1 = f1.readlines()

    N = 60
    
    for time_step in range(0, 1):
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
        
        for i in range(0, N):
            if i == 0:
                if time_step == 0:
                    array_boundRF_[i] = np.sum([array_boundRF[k3*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
                else:
                    array_boundRF_[i] = np.sum([array_boundRF[k3*i + time_step - 1], [-0.03, 0.0, 0.15842]], axis = 0)
            else:
                array_boundRF_[i] = np.sum([array_boundRF[k3*i + time_step - 1], [-0.03, 0.0, 0.15842]], axis = 0)

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
                if time_step == 0:
                    array_boundLF_[i] = np.sum([array_boundLF[k3*i + time_step], [-0.03, 0.0, 0.15842]], axis = 0)
                else:
                    array_boundLF_[i] = np.sum([array_boundLF[k3*i + time_step - 1], [-0.03, 0.0, 0.15842]], axis = 0)
            else:
                array_boundLF_[i] = np.sum([array_boundLF[k3*i + time_step - 1], [-0.03, 0.0, 0.15842]], axis = 0)

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
                if time_step == 0:
                    array_boundx_[i] = array_boundx[k3*i + time_step]
                    array_boundy_[i] = array_boundy[k3*i + time_step]
                else:
                    array_boundx_[i] = array_boundx[k3*i + time_step - 1]
                    array_boundy_[i] = array_boundy[k3*i + time_step - 1]
            else:
                array_boundx_[i] = array_boundx[k3*(i) + time_step - 1]
                array_boundy_[i] = array_boundy[k3*(i) + time_step - 1]
        for i in range(0, len(lines3_array)):
            for j in range(0, len(lines3_array[i])):
                if j == 0:
                    zmp_refx[i].append(float(lines3_array[i][j]))
                if j == 1:
                    zmp_refy[i].append(float(lines3_array[i][j]))
                
        for i in range(0, N):
            if i == 0:
                zmp_refx_[i] = zmp_refx[k*i + time_step -1]
                zmp_refy_[i] = zmp_refy[k*i + time_step -1]
            else:
                zmp_refx_[i] = zmp_refx[k*(i)+ time_step -1]
                zmp_refy_[i] = zmp_refy[k*(i)+ time_step -1]
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
            crocs_data[key]['time'] = []

        
        #file_name ='/home/jhk/ssd_mount/filenametest_'
        #file_name2 = '.pkl'
        #file_name3 = file_name + str(time_step) + file_name2
        #with open(file_name3, 'rb') as f:
        
        with open('/home/jhk/ssd_mount/beforedata/ssp1/i=6/filename4_0.pkl', 'rb') as f:
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
        qdot_g = pinocchio.utils.zero(model.nv)
        qdot_init = pinocchio.utils.zero(model.nv)
        qddot = pinocchio.utils.zero(model.nv)
        q_init = [0, 0, 0.82473, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]

        RFjoint_id = model.model.getJointId("R_AnkleRoll_Joint")
        LFjoint_id = model.model.getJointId("L_AnkleRoll_Joint")
        LFframe_id = model.model.getFrameId("L_Foot_Link")
        RFframe_id = model.model.getFrameId("R_Foot_Link")    

        RFjoint_id1 = model.model.getJointId("R_Foot_Joint")
        LFjoint_id1 = model.model.getJointId("L_Foot_Joint")

        contactPointLF = pinocchio.SE3.Identity()
        contactPointRF = pinocchio.SE3.Identity()
    
        contactPointLF.translation.T.flat = [0.03, 0, -0.1585]
        contactPointRF.translation.T.flat = [0.03, 0, -0.1585]

        model.model.addBodyFrame("LF_contact", LFjoint_id, contactPointLF, LFframe_id)
        model.model.addBodyFrame("RF_contact", RFjoint_id, contactPointRF, RFframe_id)

        LFcframe_id = model.model.getFrameId("LF_contact")
        RFcframe_id = model.model.getFrameId("RF_contact")
        data = model.model.createData()

        for i in range(0, len(q)-2):
            q[i] = q_init[i]
       
        pinocchio.forwardKinematics(model.model, data, q, qdot, qddot)
        pinocchio.updateFramePlacements(model.model,data)
        pinocchio.updateGlobalPlacements(model.model,data)
        pinocchio.computeJointJacobians(model.model, data, q)
        pinocchio.computeMinverse(model.model, data, q)
        pinocchio.centerOfMass(model.model,data,False)

        LF_tran = data.oMi[LFjoint_id].translation
        RF_tran = data.oMi[RFjoint_id].translation
        LF_rot = data.oMi[LFjoint_id].rotation
        RF_rot = data.oMi[RFjoint_id].rotation
        PELVjoint_id = model.model.getJointId("root_joint")
        RFc_tran_init = data.oMf[RFcframe_id].translation
        LFc_tran_init = data.oMf[LFcframe_id].translation

        PELV_tran = np.add(data.oMi[PELVjoint_id].translation, model.model.inertias[PELVjoint_id].lever)
        PELV_rot = data.oMi[PELVjoint_id].rotation
        LHjoint_id = model.model.getJointId("L_HipYaw_Joint")
        RHjoint_id = model.model.getJointId("R_HipYaw_Joint")
        RFjoint_id = model.model.getJointId("R_AnkleRoll_Joint")
        LFjoint_id = model.model.getJointId("L_AnkleRoll_Joint")
    
        RFjoint_id1 = model.model.getJointId("R_Foot_Joint")
        LFjoint_id1 = model.model.getJointId("L_Foot_Joint")
        virtual_init = copy(data.oMi[PELVjoint_id].translation)

        LF_tran_init = copy(data.oMi[LFjoint_id].translation)
        RF_tran_init = copy(data.oMi[RFjoint_id].translation)
        HLR_tran_init = copy(data.oMi[LHjoint_id].translation)
        HRR_tran_init = copy(data.oMi[RHjoint_id].translation)
        LF_rot_init = copy(data.oMi[LFjoint_id].rotation)
        RF_rot_init = copy(data.oMi[RFjoint_id].rotation)
        HLR_rot_init = copy(data.oMi[LHjoint_id].rotation)
        HRR_rot_init = copy(data.oMi[RHjoint_id].rotation)
        COM_tran_init = data.com[0]

        COM_tran_init = data.com[0]
        PELV_tran_init = np.add(data.oMi[PELVjoint_id].translation, model.model.inertias[PELVjoint_id].lever)
        CPELV_tran_init = data.oMi[PELVjoint_id].translation
        PELV_rot_init = data.oMi[PELVjoint_id].rotation

        virtual_init = data.oMf[PELVjoint_id].translation

        foot_distance = LF_tran_init - RF_tran_init

        TranFVi = np.zeros((4,4))
        TranFVi[0:3,0:3] = np.identity(3)
        TranFVi[0:3,3] = PELV_tran
        TranFVi[3,3] = 1.0

        TranFRi = np.zeros((4,4))
        TranFRi[0:3,0:3] = RF_rot_init
        TranFRi[0:3,3] = RF_tran_init
        TranFRi[3,3] = 1.0

        TranFLi = np.zeros((4,4))
        TranFLi[0:3,0:3] = LF_rot_init
        TranFLi[0:3,3] = LF_tran_init
        TranFLi[3,3] = 1.0

        TranVRi = np.matmul(np.linalg.inv(TranFVi),TranFRi)
        TranVLi = np.matmul(np.linalg.inv(TranFVi),TranFLi)

        print("time1")
        print(len(database_['Right']['u_trajs']))
        e = 0
        for time1 in range(0, len(database_['Right']['u_trajs'])):
            for i in range(0, len(q)):    
                q[i] = database_['Right']['trajs'][time1][0][i]
            
            print("Q")
            print(q)
            state = crocoddyl.StateKinodynamic(model.model)
            actuation = crocoddyl.ActuationModelKinoBase(state)
            
            pinocchio.forwardKinematics(model.model, data, q, qdot_g)
            pinocchio.updateFramePlacements(model.model,data)
            pinocchio.centerOfMass(model.model, data, q, qdot_g, False)
            pinocchio.computeCentroidalMomentum(model.model,data,q,qdot_g)

            LF_tran = data.oMf[LFframe_id].translation
            RF_tran = data.oMf[RFframe_id].translation
            print("on")
            print(data.oMi[PELVjoint_id].translation)
            print(LF_tran)
            print(RF_tran)
            print(data.com[0])
            for i1 in range(1,2):
                for j1 in range(0,5): #4
                    for i in range(0, len(database_['Right']['trajs'][time1][0])):    
                        q[i] = database_['Right']['trajs'][time1][0][i]
                        
                    for j in range(0, len(database_['Right']['vel_trajs'][time1][0])):    
                        qdot_g[j] = database_['Right']['vel_trajs'][time1][0][j]

                    pinocchio.forwardKinematics(model.model, data, q, qdot_g)
                    pinocchio.updateFramePlacements(model.model,data)
                    pinocchio.centerOfMass(model.model, data, q, qdot_g, False)
                    pinocchio.computeCentroidalMomentum(model.model,data,q,qdot_g)

                    print("vel")
                    print([time1, i1, j1])
                    print(database_['Right']['vel_trajs'][time1][0][0])
                    PELV_move = [-database_['Right']['vel_trajs'][time1][0][0] *0.02 -0.002*(i1-2.5), -database_['Right']['vel_trajs'][time1][0][1] *0.02 -0.001*(j1-2.5), 0.0] 
                    PELV_de = data.oMi[PELVjoint_id].translation + PELV_move
                    print(PELV_de)
                    PELV_tran1 = np.add(PELV_de,model.model.inertias[PELVjoint_id].lever)
                    print(PELV_tran1)
                    print(LF_tran)
                    print(RF_tran)
    
                    COMz = copy(data.com[0][2])
                    q_prev = inverseKinematics(0.0, LF_rot, RF_rot, PELV_rot, LF_tran, RF_tran, PELV_tran1, HRR_tran_init, HLR_tran_init, HRR_rot_init, HLR_rot_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init)
                    print(q_prev)

                    for i in range(0, 12):
                        qdot_g[i+6] = (q[i+7] - q_prev[i]) / dt_
                    for i in range(0, 3):
                        qdot_g[i] = (-PELV_move[i]) / dt_

                    for i in range(0, 3):
                        q[i] = PELV_de[i]
                    for i in range(0, 12):
                        q[i+7] = q_prev[i]

                    pinocchio.forwardKinematics(model.model, data, q, qdot_g)
                    pinocchio.updateFramePlacements(model.model,data)
                    pinocchio.centerOfMass(model.model, data, q, qdot_g, False)
                    pinocchio.computeCentroidalMomentum(model.model,data,q,qdot_g)
                    LF_tran = data.oMf[LFframe_id].translation
                    RF_tran = data.oMf[RFframe_id].translation

                    print(data.oMi[PELVjoint_id].translation)
                    print(LF_tran)
                    print(RF_tran)
                    print("com")
                    print(data.com[0])
                    print(data.vcom[0])
                    comV = copy(data.vcom[0])
                    hgV = copy(data.hg.angular)
                    print(database_['Right']['x_state'][time1][0])

                    for i in range(0, len(q)-2):    
                        q[i] = database_['Right']['trajs'][time1][0][i]

                    x0 = np.array([0.] * (state.nx + 8))
                    u0 = np.array([0.] * (22))
                    for i in range(0,len(q_init)):
                        x0[i] = q[i]

                    x0[41] = data.com[0][0]
                    x0[43] = data.com[0][0]
                    x0[45] = data.com[0][1]
                    x0[47] = data.com[0][1]
                    
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
                    '''
                    weight_quad_zmp = np.array([2.0, 2.0])#([weight_quad_zmpx] + [weight_quad_zmpy])
                    weight_quad_zmp1 = np.array([0.3, 0.3]) ##11
                    weight_quad_cam = np.array([0.5, 0.5])#([weight_quad_camy] + [weight_quad_camx])
                    weight_quad_upper = np.array([0.8, 0.8])
                    weight_quad_com = np.array([3.0, 3.0, 1.0])#([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
                    weight_quad_rf = np.array([3.0, 2.0, 3.0, 0.5, 0.5, 0.5])#np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
                    weight_quad_lf = np.array([3.0, 2.0, 3.0, 0.5, 0.5, 0.5])#np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
                    '''
                    weight_quad_zmp = np.array([2.0, 2.0])#([weight_quad_zmpx] + [weight_quad_zmpy])
                    weight_quad_zmp1 = np.array([0.5, 0.8]) ##11
                    weight_quad_cam = np.array([0.4, 0.4])#([weight_quad_camy] + [weight_quad_camx])
                    weight_quad_upper = np.array([0.05, 0.05])
                    weight_quad_com = np.array([4.0, 4.0, 1.0])#([weight_quad_comx] + [weight_quad_comy] + [weight_quad_comz])
                    weight_quad_rf = np.array([2.0, 1.0, 2.0, 0.5, 0.5, 0.5])#np.array([weight_quad_rfx] + [weight_quad_rfy] + [weight_quad_rfz] + [weight_quad_rfroll] + [weight_quad_rfpitch] + [weight_quad_rfyaw])
                    weight_quad_lf = np.array([2.0, 1.0, 2.0, 0.5, 0.5, 0.5])#np.array([weight_quad_lfx] + [weight_quad_lfy] + [weight_quad_lfz] + [weight_quad_lfroll] + [weight_quad_lfpitch] + [weight_quad_lfyaw])
                    
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
                        comBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[i], np.array([0.0, 0.0, COMz]), actuation_vector[i].nu + 4))
                        rf_foot_pos_vector[i] = pinocchio.SE3.Identity()
                        rf_foot_pos_vector[i].translation = copy(RF_tran)
                        lf_foot_pos_vector[i] = pinocchio.SE3.Identity()
                        pelvis_rot_vector[i] = pinocchio.SE3.Identity()
                        lf_foot_pos_vector[i].translation = copy(LF_tran)
                        #residual_FramePelvis[i] = crocoddyl.ResidualFrameRotation(state_vector[i], Pelvis_id, pelvis_rot_vector[i], actuation_vector[i].nu + 4)
                        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
                        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
                        #PelvisR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[i])
                        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
                        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
                        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu + 4)
                        
                        #if i >= 2:
                        #    runningCostModel_vector[i].addCost("stateReg1", stateBoundCost_vector1[i], 1.0)
                        
                        runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], 1.0)
                        runningCostModel_vector[i].addCost("stateReg2", stateBoundCost_vector2[i], 1.0)
                        runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1.0)
                        runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1.0)
                        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
                        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
                        
                        runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
                        runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)

                    traj_[43] = zmp_refx_[N-1][0]
                    traj_[47] = zmp_refy_[N-1][0]
                        
                    state_vector[N-1] = crocoddyl.StateKinodynamic(model.model)
                    actuation_vector[N-1] = crocoddyl.ActuationModelKinoBase(state_vector[N-1])
                    state_bounds[N-1] = crocoddyl.ActivationBounds(lb_[:,N-1],ub_[:,N-1])
                    state_activations[N-1] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[N-1], weight_quad_zmp)
                    #state_activations1[N-1] = crocoddyl.ActivationModelWeightedQuadraticBarrier(state_bounds[N-1], weight_quad_upper)
                    stateBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], state_activations[N-1], crocoddyl.ResidualFlyState(state_vector[N-1], actuation_vector[N-1].nu + 4))
                    stateBoundCost_vector1[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_zmp1), crocoddyl.ResidualFlyState(state_vector[N-1], traj_, actuation_vector[N-1].nu + 4))
                    stateBoundCost_vector2[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_upper), crocoddyl.ResidualFlyState1(state_vector[N-1], actuation_vector[N-1].nu + 4))
                    camBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[N-1], actuation_vector[N-1].nu + 4))
                    comBoundCost_vector[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[N-1], np.array([0.0, 0.0, COMz]), actuation_vector[N-1].nu + 4))
                    rf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
                    rf_foot_pos_vector[N-1].translation = copy(RF_tran)
                    lf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
                    lf_foot_pos_vector[N-1].translation = copy(LF_tran)
                    pelvis_rot_vector[N-1] = pinocchio.SE3.Identity()
                    #residual_FramePelvis[N-1] = crocoddyl.ResidualFrameRotation(state_vector[N-1], Pelvis_id, pelvis_rot_vector[N-1], actuation_vector[N-1].nu + 4)
                    #PelvisR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_pelvis), residual_FramePelvis[N-1])
                    residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
                    residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
                    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
                    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])
                    
                    terminalCostModel = crocoddyl.CostModelSum(state_vector[N-1], actuation_vector[N-1].nu + 4)
                    terminalCostModel.addCost("stateReg", stateBoundCost_vector[N-1], 1.0)
                    #terminalCostModel.addCost("stateReg1", stateBoundCost_vector1[N-1], 1.0)
                    terminalCostModel.addCost("stateReg2", stateBoundCost_vector2[N-1], 1.0)
                    terminalCostModel.addCost("comReg", comBoundCost_vector[N-1], 1.0)
                    #terminalCostModel.addCost("camReg", camBoundCost_vector[N-1], 1.0)
                    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
                    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
                    
                    #terminalCostModel.addCost("pelvisReg1", PelvisR[N-1], 1.0)
                    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N-1], actuation_vector[N-1], terminalCostModel)

                    database_['Right']['trajs'].append(trajs_temp)
                    database_['Right']['acc_trajs'].append(acc_temp)
                    database_['Right']['x_state'].append(x_temp)
                    database_['Right']['vel_trajs'].append(vel_temp)
                    database_['Right']['u_trajs'].append(u_temp)

                    '''
                    for i in range(0,N):
                        xs[i] = np.append(np.append(database_['Right']['trajs'][time1][i], database_['Right']['vel_trajs'][time1][i]),database_['Right']['x_state'][time1][i])
                    
                    for i in range(0,N-1):
                        us[i] = np.append(database_['Right']['u_trajs'][time1][i],database_['Right']['acc_trajs'][time1][i])#np.append(np.append(database_['Right']['u_trajs'][time1][i+1], database_['Right']['acc_trajs'][time1][i][0]), database_['Right']['acc_trajs'][time1][i][2])
                    '''
                    u0 = np.zeros(24)
                    
                    for i in range(0,N):
                        if i == 0:
                            '''
                            database_['Right']['x_state'][time1][i][1] = comV[0]
                            database_['Right']['x_state'][time1][i][5] = comV[1]
                            database_['Right']['x_state'][time1][i][3] =  hgV[1]
                            database_['Right']['x_state'][time1][i][7] =  hgV[0]
                            '''
                            c11 = copy(database_['Right']['x_state'][time1][i])
                            c11[1] = comV[0]
                            c11[5] = comV[1]
                            c11[3] =  hgV[1]
                            c11[7] =  hgV[0]

                            d11 = copy(database_['Right']['vel_trajs'][time1][i])

                            for j in range(0, len(qdot_g)):
                                d11[j] = qdot_g[j]
                        else:
                            c11 = copy(database_['Right']['x_state'][time1][i])
                            d11 = copy(database_['Right']['vel_trajs'][time1][i])
                        xs[i] = np.append(np.append(database_['Right']['trajs'][time1][i], d11),c11)
                        
                    for i in range(0,N-1):
                        us[i] = np.append(database_['Right']['u_trajs'][time1][i],database_['Right']['acc_trajs'][time1][i])#np.append(np.append(database_['Right']['u_trajs'][time1][i+1], database_['Right']['acc_trajs'][time1][i][0]), database_['Right']['acc_trajs'][time1][i][2])
                    
                    print("xs")
                    print(xs[0])
                    print(xs[1])
                    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
                    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
                    problemWithRK4.nthreads = 12
                    ddp = crocoddyl.SolverFDDP(problemWithRK4)
                    
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
                    residual_FrameRF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], RFframe_id, rf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
                    residual_FrameLF[N-1] = crocoddyl.ResidualKinoFramePlacement(state_vector[N-1], LFframe_id, lf_foot_pos_vector[N-1], actuation_vector[N-1].nu + 4)
                    foot_trackR[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[N-1])
                    foot_trackL[N-1] = crocoddyl.CostModelResidual(state_vector[N-1], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[N-1])    
                    
                    
                    terminalCostModel.removeCost("footReg1")
                    terminalCostModel.removeCost("footReg2")
                    terminalCostModel.addCost("footReg1", foot_trackR[N-1], 1.0)
                    terminalCostModel.addCost("footReg2", foot_trackL[N-1], 1.0)
                    
                    problemWithRK4.x0 = xs[0]
                    
                    c_start = time.time()
                    c_end = time.time()
                    duration = (1e3 * (c_end - c_start))

                    print("end")
                    avrg_duration = duration
                    min_duration = duration #min(duration)
                    max_duration = duration #max(duration)
                    print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
                    #print('ddp.iter {0},{1}'.format(ddp.iter, css))
                    walking_tick = 0
                    booltemp = True
                    #while client.is_connected:  
                    iter_ = 0
                    T = 1
                
                    cost_prev = 0
                    while booltemp == True:
                        booltemp1 = True
                        c_start = time.time()
                        if iter_ == 0:
                            ddp.th_stop = 0.00000001
                            print("a")
                            css = ddp.solve(xs, us, 300, False, 0.01)
                            #css = ddp.solve(xs, us, 1, False, 0.1)

                        else:
                            print("b")
                            ddp.th_stop = 0.00000001
                            css = ddp.solve(ddp.xs, ddp.us, 300, False)
                        c_end = time.time()
                        
                        duration = (1e3 * (c_end - c_start))
                            
                        avrg_duration = duration
                        min_duration = duration #min(duration)
                        max_duration = duration #max(duration)
                        print("iter_")
                        print(iter_)
                        print(ddp.xs[-1])
                        print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
                        print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
                        print(ddp.cost)
                        print("time")
                        print(time1)
                        print("loop")
                        print(e)
                        print(ddp.us[0])
                        #if iter_ == 1:
                        
                        booltemp1 = False

                        if booltemp1 == False:
                            if iter_ >= 30 or abs(ddp.cost - cost_prev) < 0.000001:
                                booltemp = False
                            cost_prev = ddp.cost
                            if iter_ == 0:
                                for key in crocs_data.keys():
                                    if key == 'left':
                                        for l in range(0,3):
                                            crocs_data[key]['foot_poses'].append([lf_foot_pos_vector[l].translation[0], lf_foot_pos_vector[l].translation[1], lf_foot_pos_vector[l].translation[2]])
                                    else:
                                        for l in range(0,3):
                                            crocs_data[key]['foot_poses'].append([rf_foot_pos_vector[l].translation[0], rf_foot_pos_vector[l].translation[1], rf_foot_pos_vector[l].translation[2]])    
                                traj = np.array(ddp.xs)[:,0:21]
                                vel_traj = np.array(ddp.xs)[:,21:41]
                                x_traj = np.array(ddp.xs)[:, 41:49]
                                u_traj = np.array(ddp.us)[:,20:24]
                                acc_traj = np.array(ddp.us)[:, 0:20]
                                
                                crocs_data[key]['u_trajs'].append(copy(acc_traj))
                                crocs_data[key]['acc_trajs'].append(copy(u_traj))

                                crocs_data['Right']['x_inputs'].append(copy(ddp.xs[0][0:21]))
                                crocs_data['Right']['vel_trajs'].append(copy(vel_traj))
                                crocs_data['Right']['x_state'].append(copy(x_traj))
                                crocs_data['Right']['costs'].append(copy(ddp.cost))
                                crocs_data['Right']['iters'].append(copy(ddp.iter))
                                crocs_data['Right']['trajs'].append(copy(traj))
                                crocs_data['Right']['data_phases_set'].append([css, avrg_duration])
                                crocs_data['Right']['time'].append([i1, j1, -0.002*(i1-2.5) -0.001*(j1-2.5)])
                            else:
                                traj = np.array(ddp.xs)[:,0:21]
                                vel_traj = np.array(ddp.xs)[:,21:41]
                                x_traj = np.array(ddp.xs)[:, 41:49]
                                u_traj = np.array(ddp.us)[:,20:24]
                                acc_traj = np.array(ddp.us)[:, 0:20]
                                crocs_data['Right']['x_inputs'][e] =copy(ddp.xs[0][0:21])
                                crocs_data['Right']['vel_trajs'][e] =copy(vel_traj)
                                crocs_data['Right']['x_state'][e] =copy(x_traj)
                                crocs_data['Right']['costs'][e] =copy(ddp.cost)
                                crocs_data['Right']['iters'][e] =copy(ddp.iter)
                                crocs_data['Right']['trajs'][e] =copy(traj)
                                crocs_data['Right']['data_phases_set'][e] =[css, avrg_duration]
                                crocs_data['Right']['u_trajs'][e] = copy(acc_traj)
                                crocs_data['Right']['acc_trajs'][e] = copy(u_traj)
                                crocs_data['Right']['time'][e] = [i1, j1, -0.002*(i1-2.5) -0.001*(j1-2.5)]
                                #database_[key]['data_phases_set'].append(css)
                            print(time1)
                            print("QQ")
                            print(ddp.xs[N-1][0:21])
                            
                        print(database['Right']['x_state'][time1][0])
                        print(database['Right']['x_state'][time1][2])
                        print(ddp.xs[0])
                        iter_ = iter_ + 1
                        if booltemp == False:
                            e = e+1
                        
                        if time1 % 100 == 0:
                            file_name ='/home/jhk/ssd_mount/beforedata/ssp1/i=6/filename4_40-5_dist_'
                            file_name2 = '.pkl'
                            file_name3 = file_name + str(time_step) + file_name2
                            with open(file_name3, 'wb') as f:
                                pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                            print("success")

                        #a = asdfasdfasdfasdf
                        
            file_name ='/home/jhk/ssd_mount/beforedata/ssp1/i=6/filename4_105_dist_'
            file_name2 = '.pkl'
            file_name3 = file_name + str(time_step) + file_name2
            with open(file_name3, 'wb') as f:
                pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("success")
if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    #PCAlearning()
    talker()