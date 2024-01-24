import roslibpy
import pickle
import numpy as np
import time
from copy import copy
import logging
import os
import scipy
import pinocchio
import crocoddyl
import quadprog

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

def talker():
    global xs_pca_test, xs_pca, us_pca
    print("start")
    f = open("/home/jhk/walkingdata/beforedata/fdyn/lfoot2_final.txt", 'r')
    f1 = open("/home/jhk/walkingdata/beforedata/fdyn/rfoot2_final.txt", 'r')
    f2 = open("/home/jhk/walkingdata/beforedata/fdyn/zmp2_ssp1_1.txt", 'r')
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
    time_step = 11
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
        array_boundLF_[i] = np.sum([array_boundLF[k*i+ time_step], [-0.03, 0.0, 0.15842]], axis = 0)
        
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
        array_boundx_[i] = array_boundx[k3*i+ time_step]
        array_boundy_[i] = array_boundy[k3*i+ time_step]


    f.close()
    f1.close()
    f2.close()
    
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
    '''
    fixedJointConfig = np.matrix([0, 0, 0.82473, 0, 0, 0, 1, 
    0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 
    0.0, 0.0, -0.55, 1.26, -0.71, 0.0,
    0, 0, 0,  
    0.2, 0.6, 1.5, -1.47, -1, 0 ,-1, 0, 
    0, 0, 
    -0.2, -0.6 ,-1.5, 1.47, 1, 0, 1, 0]).T
    '''
    fixedJointConfig = np.matrix([ 0.00184906,
 -3.3417e-05,
    0.824229,
 1.01508e-05,
 0.000554211,
 0.000143354,
           1,
-0.000223601,
-2.87115e-05,
   -0.548433,
     1.26119,
   -0.714696,
-1.06773e-05,
 -0.00029453,
  -3.807e-05,
   -0.548386,
     1.26124,
   -0.714656,
  9.1817e-05,
 -0.00010485,
 -0.00164512,
-1.09408e-05,
    0.200089,
    0.597312,
     1.49998,
    -1.47006,
   -0.999883,
-1.46526e-05,
   -0.999985,
 8.16923e-07,
-7.55479e-06,
  0.00015246,
   -0.199954,
   -0.597039,
    -1.50003,
        1.47,
           1,
 1.17954e-05,
           1,
-9.60788e-08]).T

    model = RobotWrapper.buildReducedRobot(model, jointsToLockIDs, fixedJointConfig)
    pi = 3.14159265359
    
    q = pinocchio.utils.zero(model.nq)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)

    q_init = [-0.00391439,
       4.4739e-05,
    0.819632,
 0, 0,
 0,      1,
8.98911e-05, -1.07153e-05,   -0.546695,
     1.27746,    -0.715484,  2.44323e-05,  
     8.36442e-05, -2.60782e-05,
   -0.546681,      1.27736,    -0.715455,
  9.1817e-05, 
  0, 0]

    for i in range(0, len(q)):
        q[i] = q_init[i]

    RFjoint_id = model.model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.model.getJointId("L_AnkleRoll_Joint")
    LFframe_id = model.model.getFrameId("L_Foot_Link")
    RFframe_id = model.model.getFrameId("R_Foot_Link")  
    contactPointLF = pinocchio.SE3.Identity()
    contactPointRF = pinocchio.SE3.Identity()
    contactPointLF.translation.T.flat = [0.03, 0, -0.1585]
    contactPointRF.translation.T.flat = [0.03, 0, -0.1585]

    RFjoint_id = model.model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.model.getJointId("L_AnkleRoll_Joint")

    model.model.addBodyFrame("LF_contact", LFjoint_id, contactPointLF, LFframe_id)
    model.model.addBodyFrame("RF_contact", RFjoint_id, contactPointRF, RFframe_id)

    LFcframe_id = model.model.getFrameId("LF_contact")
    RFcframe_id = model.model.getFrameId("RF_contact")

    data = model.model.createData()

    pinocchio.forwardKinematics(model.model, data, q, qdot)
    pinocchio.updateFramePlacements(model.model,data)
    pinocchio.centerOfMass(model.model, data, q, False)
    pinocchio.computeCentroidalMomentum(model.model,data,q,qdot)
    pinocchio.computeAllTerms(model.model, data, q, qdot)
    RFj = pinocchio.computeFrameJacobian(model.model, data, q, RFcframe_id)
    LFj = pinocchio.computeFrameJacobian(model.model, data, q, LFcframe_id)
    print(q)
    print(RFj)
    #s = asdfasdfs
    dh_dq = np.zeros([6, model.nv])
    dhd_dq = np.zeros([6, model.nv])
    dhd_dv = np.zeros([6, model.nv])
    dhd_da = np.zeros([6, model.nv])
    
    qv = []
    qv_prev = []

    Jc1 = np.zeros([model.nv, model.nv])

    Fc = np.zeros(12)
    Jc = np.zeros([12, model.nv])

    Jc[0:6, 0:model.nv] = RFj
    Jc[6:12, 0:model.nv] = LFj
    
    Jc1[0:model.nv,0] = np.transpose(RFj[0, 0:model.nv])
    Jc1[0:model.nv,1] = np.transpose(RFj[1, 0:model.nv])
    Jc1[0:model.nv,2] = np.transpose(RFj[5, 0:model.nv])
    Jc1[0:model.nv,3] = np.transpose(LFj[0, 0:model.nv])
    Jc1[0:model.nv,4] = np.transpose(LFj[1, 0:model.nv])
    Jc1[0:model.nv,5] = np.transpose(LFj[5, 0:model.nv])
    Jc1[6:model.nv, 6:model.nv] = numpy.identity(model.nv-6)

    Fc[2] = 0.5 * data.nle[2]
    Fc[3] = 11.9678
    Fc[4] = 2.6225
    Fc[8] = 0.5 * data.nle[2]
    Fc[9] = 36.1556
    Fc[10] = 7.9229
    
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]

    LF_tran = data.oMi[LFjoint_id]
    RF_tran = data.oMi[RFjoint_id]

    k = data.nle - np.matmul(np.transpose(Jc), Fc)

    S = np.matmul(np.linalg.inv(Jc1),k)
    
    Fc[0] = S[0]
    Fc[1] = S[1]
    Fc[5] = S[2]
    Fc[6] = S[3]
    Fc[7] = S[4]
    Fc[11] = S[5]

    #print(Jc1[0:6, 0:12])
    print(((Fc[3]/Fc[2]-0.1025)*Fc[2]+(Fc[9]/Fc[8]+0.1025)*Fc[8])/(Fc[2]+Fc[8])) 
    print(np.matmul(np.transpose(Jc), Fc)[0:6])
    print(data.nle[0:6])
    print(np.matmul(np.transpose(Jc), Fc)[6:] + S[6:])
    print(data.nle[6:])
    print("KKK")
    print(np.linalg.matrix_rank(Jc1))
    print(np.shape(Jc1))
    #print(np.matmul(Jc1, (Jc1)))
    #print(RFj[0, 0:model.nv])
    #print(S[:])

    
    Jc2 = np.zeros([6, 12])
    Jc3 = np.zeros([6, 12])
    Fc3 = np.zeros(12)
    nle2 = np.zeros(6)

    print("np2")
    Jc2[:,0:6] = np.transpose(RFj[0:6, 0:6])
    Jc2[:,6:12] = np.transpose(LFj[0:6, 0:6])

    print(np.linalg.matrix_rank(Jc2))

    #kas= afsd

    Fc3[2] = 0.5 * data.nle[2]
    Fc3[3] = 20
    Fc3[4] = 0.0
    Fc3[8] = 0.5 * data.nle[2]
    Fc3[9] = 20
    Fc3[10] = 0.0

    nle2 = np.matmul(Jc2, Fc3)
    print("kk")
    print(nle2)
    nle2 = np.subtract(data.nle[0:6], nle2)

    Jc3[0:6,0] = np.transpose(RFj)[0:6,0]
    Jc3[0:6,1] = np.transpose(RFj)[0:6,1]
    Jc3[0:6,2] = np.transpose(RFj)[0:6,2]
    Jc3[0:6,3] = np.transpose(LFj)[0:6,0]
    Jc3[0:6,4] = np.transpose(LFj)[0:6,1]
    Jc3[0:6,5] = np.transpose(LFj)[0:6,2]
    Jc3[0:6,6:] = np.identity(6)
    print(np.linalg.matrix_rank(Jc3))
    a = np.matmul(scipy.linalg.pinv(Jc3), nle2[0:6])
    print(np.matmul(scipy.linalg.pinv(Jc3),Jc3))

    #print(np.linalg.matrix_rank(Jc3))
    #print(np.linalg.matrix_rank(Jc3[0:5,:]))
    print(a)
    #k = asdfasdfs

    nle2 = data.nle[:6]
    Fc2 = np.matmul(scipy.linalg.pinv(Jc2),nle2)
    print("ssss")
    print(((Fc3[3]/Fc3[2]-0.1025)*Fc3[2]+(Fc3[9]/Fc3[8]+0.1025)*Fc3[8])/(Fc3[2]+Fc3[8])) 
    print(np.linalg.matrix_rank(Jc2))
    print(Fc3)
    print(data.M[0:3,:])
    print("dddd")
    print(RFj)
    #sprint(data.nle - np.matmul(np.transpose(Jc), Fc))




    
if __name__=='__main__':
    talker()

