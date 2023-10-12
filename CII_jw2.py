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
from pinocchio.visualize import GepettoVisualizer
from pinocchio.robot_wrapper import RobotWrapper

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
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf("/usr/local/lib/python3.8/dist-packages/robot_properties_tocabi/resources/urdf/tocabi.urdf","/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/meshes",pinocchio.JointModelFreeFlyer())  
    #model = RobotWrapper.BuildFromURDF("/usr/local/lib/python3.8/dist-packages/robot_properties_tocabi/resources/urdf/tocabi.urdf","/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/meshes",pinocchio.JointModelFreeFlyer())  
    pi = 3.14159265359

    jointsToLock = ['Waist1_Joint', 'Waist2_Joint', 'Upperbody_Joint', 'Neck_Joint', 'Head_Joint', 
    'L_Shoulder1_Joint', 'L_Shoulder2_Joint', 'L_Shoulder3_Joint', 'L_Armlink_Joint', 'L_Elbow_Joint', 'L_Forearm_Joint', 'L_Wrist1_Joint', 'L_Wrist2_Joint',
    'R_Shoulder1_Joint', 'R_Shoulder2_Joint', 'R_Shoulder3_Joint', 'R_Armlink_Joint', 'R_Elbow_Joint', 'R_Forearm_Joint', 'R_Wrist1_Joint', 'R_Wrist2_Joint']
    # Get the joint IDs
    jointsToLockIDs = []
    
    for jn in range(len(jointsToLock)):
        jointsToLockIDs.append(model.getJointId(jointsToLock[jn]))
    # Set initial configuration
    
    
    fixedJointConfig = np.matrix([0, 0, 0.80783, 0, 0, 0, 1, 
    0, 0, -0.725348, 1.25572, -0.53037, 0, 0, 0, -0.265866, 1.134302, -0.86843, 0,
    0, 0, 0,  
    0.2, 0.6, 1.5, -1.47, -1, 0 ,-1, 0, 
    0, 0, 
    -0.2, -0.6 ,-1.5, 1.47, 1, 0, 1, 0]).T
    '''
    fixedJointConfig = np.matrix([
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    '''
    #reducedModel, geom_models = pinocchio.buildReducedModel(model, [visual_model, collision_model], jointsToLockIDs, fixedJointConfig)
    print("ddd")
    #print(reducedModel.nq)
    #visual_model1 = geom_models[0]
    #collision_model1 = geom_models[1]
    
    #model = RobotWrapper.buildReducedRobot(model, jointsToLockIDs, fixedJointConfig)
    viz = GepettoVisualizer(model, collision_model, visual_model)
    try:
        #model.initViewer(True)
        viz.initViewer()
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install gepetto-viewer")
        print(err)
        sys.exit(0)
    viz.loadViewerModel("pinocchio")

    q = pinocchio.utils.zero(model.nq)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.725348, 1.25572, -0.53037, 0, 0, 0, -0.265866, 1.134302, -0.86843, 0]
    
    for i in range(0, len(q_init)):
        q[i] = q_init[i]

    data = model.createData()
    print(q)
    viz.display(q)
    pinocchio.centerOfMass(model,data,q)
    print(data.com[0])

    '''
    viz = GepettoVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer()
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install gepetto-viewer")
        print(err)
        sys.exit(0)
    viz.loadViewerModel("pinocchio")
    q = pinocchio.utils.zero(model.nq)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    
    for i in range(0, len(q_init)):
        q[i] = q_init[i]

    print(q)
    viz.display(q)
    '''

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()
