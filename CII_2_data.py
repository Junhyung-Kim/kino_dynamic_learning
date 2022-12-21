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
        leg_qs = np.zeros((int(1), 12))
        leg_qdots = np.zeros((int(1), 12))
        leg_qddots = np.zeros((int(1), 12))

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

    lr = np.add(lp, ld)
    rr = np.add(rp, rd)

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

    print("legq")
    print(leg_q)
    '''
    else:
        leg_qdots[time,:] = np.subtract(leg_qs[time,:], leg_qs[time-1,:]) * hz
        leg_qddots[time,:] = np.subtract(leg_qdots[time,:], leg_qdots[time-1,:]) * hz
    '''
def talker():
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
    array_qdot = [[] for i in xrange(int(len(lines1)/208) * 30)]
    array_q = [[] for i in xrange(int(len(lines1)/208) * 30)]
    array_xstate = [[] for i in xrange(int(len(lines1)/208) * 30)]
    array_u = [[] for i in xrange(int(len(lines1)/208)*29)]
    array_qddot = [[] for i in xrange(int(len(lines1)/208)*29)]

    array_boundx = [[] for i in xrange(int(len(lines1)/208) * 30)]
    array_boundy = [[] for i in xrange(int(len(lines1)/208) * 30)]

    array_boundRF = [[] for i in xrange(int(len(lines1)/208) * 30)]
    array_boundLF = [[] for i in xrange(int(len(lines1)/208) * 30)]

    array_qdot1 = [[] for i in xrange(int(len(lines3)/208) * 30)]
    array_q1 = [[] for i in xrange(int(len(lines3)/208) * 30)]
    array_xstate1 = [[] for i in xrange(int(len(lines3)/208) * 30)]
    array_u1 = [[] for i in xrange(int(len(lines3)/208)*29)]
    array_qddot1 = [[] for i in xrange(int(len(lines3)/208)*29)]

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
        rf_foot_pos_vector[i].translation = copy(RF_tran)
        lf_foot_pos_vector[i] = pinocchio.SE3.Identity()
        lf_foot_pos_vector[i].translation = copy(LF_tran)
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
    rf_foot_pos_vector[N-1].translation = copy(RF_tran)
    lf_foot_pos_vector[N-1] = pinocchio.SE3.Identity()
    lf_foot_pos_vector[N-1].translation = copy(LF_tran)
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

    walking_tick = 0

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
    
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 6
    ddp = crocoddyl.SolverBoxFDDP(problemWithRK4)
    ddp.solve(xs,us,300)
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

    data_finish = True
    while data_finish == True:
        for i in range(0, 40):
            for j in range(0, 40):
                if i != 20 and j != 20:
                    for k in range(0, len(q)):
                        q[k] = q_init[k]
                    pinocchio.forwardKinematics(model, data, q, qdot)
                    pinocchio.updateFramePlacements(model,data)
                    pinocchio.updateGlobalPlacements(model,data)
                    pinocchio.computeJointJacobians(model, data, q)
                    pinocchio.centerOfMass(model, data, q, False)

                    PELV_tran = np.add(data.oMi[PELVjoint_id].translation, model.inertias[PELVjoint_id].lever)
                    Pelv_Move = [ 0.15/(i -20), 0.15/(j -20), 0.0]
                    PELV_tran[0] = Pelv_Move[0] + PELV_tran[0]
                    PELV_tran[1] = Pelv_Move[1] + PELV_tran[1]
                    PELV_tran[2] = Pelv_Move[2] + PELV_tran[2]
                    inverseKinematics(0.0, LF_rot, RF_rot, PELV_rot, LF_tran, RF_tran, PELV_tran, HRR_tran_init, HLR_tran_init, HRR_rot_init, HLR_rot_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init)

                    for a in range(0, 3):
                        q[a] = q[a] + Pelv_Move[a]
                    for b in range(7,19):
                        q[b] = leg_q[b-7]
                        
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

                    for l in range(0,N):
                        xs[l] = copy(x0)

                    print("kk")
                    print(i)
                    print(j)

                    for l in range(0,N-1):
                        state_bounds[l].lb[0] = copy(array_boundx[30*(walking_tick)+l][0])
                        state_bounds[l].ub[0] = copy(array_boundx[30*(walking_tick)+l][1])
                        state_bounds[l].lb[1] = copy(array_boundy[30*(walking_tick)+l][0])
                        state_bounds[l].ub[1] = copy(array_boundy[30*(walking_tick)+l][1])
                        state_activations[l].bounds = state_bounds[l]
                        stateBoundCost_vector[l].activation_ = state_activations[l]
                        rf_foot_pos_vector[l].translation[0] = copy(array_boundRF[30*(walking_tick)+l][0])
                        rf_foot_pos_vector[l].translation[1] = copy(array_boundRF[30*(walking_tick)+l][1])
                        rf_foot_pos_vector[l].translation[2] = copy(array_boundRF[30*(walking_tick)+l][2])
                        lf_foot_pos_vector[l].translation[0] = copy(array_boundLF[30*(walking_tick)+l][0])
                        lf_foot_pos_vector[l].translation[1] = copy(array_boundLF[30*(walking_tick)+l][1])
                        lf_foot_pos_vector[l].translation[2] = copy(array_boundLF[30*(walking_tick)+l][2])
                        #residual_FrameRF[i].pref = rf_foot_pos_vector[i]
                        #residual_FrameLF[i].pref = lf_foot_pos_vector[i]
                        #foot_trackR[i].residual_ = residual_FrameRF[i]
                        #foot_trackL[i].residual_ = residual_FrameLF[i]
                        residual_FrameRF[l] = crocoddyl.ResidualKinoFramePlacement(state_vector[l], RFframe_id, rf_foot_pos_vector[l], actuation_vector[l].nu + 4)
                        residual_FrameLF[l] = crocoddyl.ResidualKinoFramePlacement(state_vector[l], LFframe_id, lf_foot_pos_vector[l], actuation_vector[l].nu + 4)
                        foot_trackR[l] = crocoddyl.CostModelResidual(state_vector[l], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[l])
                        foot_trackL[l] = crocoddyl.CostModelResidual(state_vector[l], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[l])
                        runningCostModel_vector[l].removeCost("footReg1")
                        runningCostModel_vector[l].removeCost("footReg2")
                        runningCostModel_vector[l].addCost("footReg1", foot_trackR[l], 1.0)
                        runningCostModel_vector[l].addCost("footReg2", foot_trackL[l], 1.0)

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
                    k_temp = 0
                    iter_ = 0
                    booltemp = True
                    problemWithRK4.x0 = xs[0]
                    while booltemp == True:
                        booltemp1 = True
                        c_start = time.time()
                        if k_temp == 0:
                            css = ddp.solve(xs, us, 3000)
                        else:
                            css = ddp.solve(ddp.xs, ddp.us, 3000)
                        c_end = time.time()
                        duration = (1e3 * (c_end - c_start))

                        avrg_duration = duration
                        min_duration = duration #min(duration)
                        max_duration = duration #max(duration)
                        print("iter")
                        print(iter_)
                        print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
                        print('ddp.iter {0},{1},{2}'.format(ddp.iter, ddp.cost, walking_tick))
                        
                        for l in range(0,N):
                            if l < N-1:
                                for a in range(0,3):
                                    if abs(runningCostModel_vector[l].costs['footReg1'].cost.residual.reference.translation[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(runningCostModel_vector[l].costs['footReg2'].cost.residual.reference.translation[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(runningCostModel_vector[l].costs['comReg'].cost.residual.reference[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(runningCostModel_vector[l].costs['camReg'].cost.residual.reference[a]) > 0.01:
                                        booltemp1 = False
                                        break
                            else:
                                for a in range(0,3):
                                    if abs(terminalCostModel.costs['footReg1'].cost.residual.reference.translation[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(terminalCostModel.costs['footReg2'].cost.residual.reference.translation[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(terminalCostModel.costs['comReg'].cost.residual.reference[a]) > 0.006:
                                        booltemp1 = False
                                        break
                                for a in range(0,3):
                                    if abs(terminalCostModel.costs['camReg'].cost.residual.reference[a]) > 0.01:
                                        booltemp1 = False
                                        break
                            if booltemp1 == False:
                                break

                        if booltemp1 == True and avrg_duration <= 30:
                            for key in crocs_data.keys():
                                if key == 'left':
                                    for l in range(0,3):
                                        crocs_data[key]['foot_poses'].append([lf_foot_pos_vector[l].translation[0], lf_foot_pos_vector[l].translation[1], lf_foot_pos_vector[l].translation[2]])
                                else:
                                    for l in range(0,3):
                                        crocs_data[key]['foot_poses'].append([rf_foot_pos_vector[l].translation[0], rf_foot_pos_vector[l].translation[1], rf_foot_pos_vector[l].translation[2]])
                                #for l in range(0,N):
                                traj = np.array(ddp.xs)[:,0:19]
                                vel_traj = np.array(ddp.xs)[:,19:37]
                                x_traj = np.array(ddp.xs)[:, 37:45]
                                crocs_data[key]['x_inputs'].append(copy(ddp.xs[0][0:19]))
                                crocs_data[key]['vel_trajs'].append(copy(vel_traj))
                                crocs_data[key]['x_state'].append(copy(x_traj))
                                crocs_data[key]['costs'].append(copy(ddp.cost))
                                crocs_data[key]['iters'].append(copy(ddp.iter))
                                crocs_data[key]['trajs'].append(copy(traj))
                                #for l in range(0,N-1):
                                u_traj = np.array(ddp.us)[:,18:22]
                                acc_traj = np.array(ddp.us)[:, 0:18]
                                crocs_data[key]['u_trajs'].append(copy(acc_traj))
                                crocs_data[key]['acc_trajs'].append(copy(u_traj))
                                
                            booltemp = False
                            break
                        else:
                            if booltemp1 == True:
                                print("Time fail")
                        
                        iter_ = iter_ + 1
                        k_temp = k_temp + 1

        if i == 99 and j == 99:
            with open('/home/jhk/data/mpc/filename.pkl', 'wb') as f:
	            pickle.dump(crocs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data_finish = False
                    
    '''
    while client.is_connected:
        T = 10
        #problemWithRK4.x0 = copy(ddp.xs[1])
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

        booltemp = True
        booltemp1 = True
        iter_ = 0
        T = 1
        #for i in range(0,T):
        while booltemp == True:
            booltemp1 = True
            c_start = time.time()
            css = ddp.solve(ddp.xs,ddp.us, 3000)
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

        crocs_data = dict()
        crocs_data['left'] = dict()
        crocs_data['right'] = dict()
        for key in crocs_data.keys():
            crocs_data[key]['foot_poses'] = []
            crocs_data[key]['trajs'] = []
            crocs_data[key]['x_inputs'] = []
            crocs_data[key]['vel_trajs'] = []        
            crocs_data[key]['u_trajs'] = []
            crocs_data[key]['data_phases_set'] = []
            crocs_data[key]['costs'] = []
            crocs_data[key]['iters'] = []   
        
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
        if walking_tick == 1:
            break
    f3.close()
    f4.close()
    client.terminate()
    '''
    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()

