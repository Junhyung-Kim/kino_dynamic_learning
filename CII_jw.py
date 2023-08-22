#!/usr/bin/env python 
from __future__ import print_function
from tempfile import tempdir
import pinocchio
from pinocchio.utils import npToTuple
from pinocchio.rpy import matrixToRpy, rpyToMatrix
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import roslibpy
import time
from sys import argv
from os.path import dirname, join, abspath

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

    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = pinocchio.buildModelFromUrdf("/home/jhk/catkin_ws/src/tocabi_cc/robots/dyros_tocabi_with_redhands.urdf",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    data = model.createData()
    q = pinocchio.utils.zero(model.nq)
    print(model.nq)
    print(q)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    q_init = [0.11363161 ,-0.18341768,  1.1555157 ,  0.17879308,  0.26401635,  0.10616599,
  0.94183713 ,-0.15896429 ,-0.2115861 , -0.9237448 ,  1.05840019, -0.77066255,
 -0.08788558 ,-0.11192483, -0.19807116, -0.9252402 ,  1.14118755, -0.77312606,
 -0.08781195]
    qdot_init = [-0.23055242, -0.31445128,  0.0835999 ,  0.70254367,  1.30148494,
  0.39196236, -0.38865854, -0.62590964, -0.81337598, -0.47783407, -0.12150462,
 -0.21752914 ,-0.25896493, -0.58831107, -0.81620195 ,-0.27236269, -0.1329973,
 -0.21657379 ]
   # q_init = [-0.0471697, -0.0319821,   0.845637, -0.0401819, -0.0252661,  0.0393611,   0.998097, -0.0503212,  0.0575037 , -0.656835 ,   1.32996 , -0.821296, -0.0848094, -0.0469608  ,  0.06052 , -0.656833  ,  1.32996 , -0.821296, -0.0848094]
    
    for i in range(0, len(q_init)):
        q[i] = q_init[i]
    for i in range(0, len(qdot_init)):
        qdot[i] = qdot_init[i]
    
    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")
    LHjoint_id = model.getJointId("L_HipYaw_Joint")
    RHjoint_id = model.getJointId("R_HipYaw_Joint")
    LFframe_id = model.getFrameId("L_Foot_Link")
    RFframe_id = model.getFrameId("R_Foot_Link")
    PELVframe_id = model.getFrameId("Pelvis_Link")
    PELVjoint_id = model.getJointId("root_joint")
    
    print("aaaa")
    
    '''

    '''
    t1 = time.time()
    '''
    pinocchio.centerOfMass(model,data,q)
    pinocchio.jacobianCenterOfMass(model, data, q)
    pinocchio.computeCentroidalMomentum(model, data, q, qdot)
    pinocchio.computeCentroidalDynamicsDerivatives(model, data, q, qdot, qdot)
    pinocchio.updateFramePlacement(model,data,3)
    pinocchio.updateFramePlacement(model,data,4)
    pinocchio.updateFramePlacement(model,data,5)
    '''
    Jc = np.zeros((6,18))
    f = np.zeros(6)
    t2 = time.time()
    #pinocchio.getCentroidalDynamicsDerivatives(model,data)
    #pinocchio.aba(model, data, q, qdot, qdot)
   # pinocchio.updateGlobalPlacements(model,data)
    
  #  pinocchio.computeAllTerms(model,data,q, qdot)
    pinocchio.computeCentroidalMomentum(model, data, q, qdot)  
   # pinocchio.computeRNEADerivatives(model, data, q, qdot,qdot)
  #  pinocchio.computeJointJacobians(model, data, q)
   # pinocchio.rnea(model, data, q, qdot,qdot)
  #  pinocchio.updateFramePlacement(model,data,RFframe_id)
    #Jc = pinocchio.getFrameJacobian(model, data, RFframe_id, pinocchio.LOCAL)
    
    #pinocchio.forwardDynamics(model, data, qddot, Jc, f, 0.0)
    #pinocchio.getKKTContactDynamicMatrixInverse(model, data, Jc)
    
    #pinocchio.updateAcceleration(model, data, qddot)
    #pinocchio.updateForce(model,)
  #  pinocchio.computeABADerivatives(model, data, q, qdot, qdot)
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)
    
    #print(data.Jcom)
    print("ccc")
    #print(pinocchio.getFrameJacobian(model, data, LFframe_id, pinocchio.LOCAL))
    
    #pinocchio.computeJointJacobians(model, data, q)
    #pinocchio.centerOfMass(model, data, q, False)
    #pinocchio.computeCentroidalMomentum(model,data,q,qdot)
    #LF_tran = data.oMf[LFframe_id].translation
    #RF_tran = data.oMf[RFframe_id].translation
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

    LF_rot12 = data.oMf[LFframe_id].translation
    RF_rot12 = data.oMf[RFframe_id].translation

    #PELV_tran_init = np.add(data.oMi[PELVjoint_id].translation, model.inertias[PELVjoint_id].lever)
    #CPELV_tran_init = data.oMi[PELVjoint_id].translation 
    #PELV_rot_init = data.oMi[PELVjoint_id].rotation
    print("com")
    print(data.com[0])
    print(data.hg)
   # print(data.mass[0])
   # print(data.mass[0] * data.com[0][2])
    #print(9.81 / data.com[0][2])
    PELV_tran = np.add(data.oMi[PELVjoint_id].translation, model.inertias[PELVjoint_id].lever)

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()
