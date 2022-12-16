import roslibpy
import crocoddyl
import pinocchio
import numpy as np
import time
import example_robot_data
from copy import copy
#from regression import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

global client

def talker():
    print("start")
    f = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f2 = open("/home/jhk/data/mpc/4_tocabi_.txt", 'r')
    f1 = open("/home/jhk/data/mpc/3_tocabi_.txt", 'r')

    f3 = open("/home/jhk/data/mpc/5_tocabi_py.txt", 'w')
    f4 = open("/home/jhk/data/mpc/6_tocabi_py.txt", 'w')

    lines = f.read().split(' ')
    lines1 = f2.readlines()
    lines2 = f1.readlines()
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

    bool_q = 0

    N = 30
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)

    for i in range(0,len(lines)):
        if lines[i].strip('\n') == 'walking_tick':
            loop = loop + 1

        if bool_u == 1:
            count_u_temp = count_u_temp + 1
            if count_u2 == 0:
                array_u[count_u].append(float(lines[i].strip('\n').strip(str(count_u)).strip('\t').strip(",").strip("ustate")))
            else:
                array_u[count_u].append(float(lines[i].strip('\n').strip(str(count_u)).strip('\t').strip(",")))
            count_u2 = count_u2 + 1
            if count_u_temp == 4:
                    count_u = count_u + 1
                    count_u2 = 0
                    count_u_temp = 0
                    bool_u = 0

        if lines[i].strip('\n').strip('\t') == "u" or bool_qdot == 1:
            if count_qddot2 == 29:
                count_qddot2 = 0

            bool_qdot = 1
            count_qddot_temp = count_qddot_temp + 1
            if count_qddot_temp > 1:
                array_qddot[count_qddot].append(float(lines[i].strip('\n').strip(str(count_qddot2)).strip('\t').strip(",")))
            
            if count_qddot_temp == 19:
                bool_qdot = 0
                count_qddot = count_qddot + 1
                count_qddot2 = count_qddot2 + 1
                count_qddot_temp = 0
                bool_u = 1

        if(i >= 6):
            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] >= 0 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 19:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 0:
                    array_q[count_q].append(float(lines[i].strip('\n').strip(str(count_q_temp)).strip('\t').strip(",")))
                else:
                    array_q[count_q].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 18:
                    count_q = count_q + 1
                    count_q_temp = count_q_temp + 1
                    if count_q_temp == 30:
                        count_q_temp = 0

            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] > 19 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 38:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 20:
                    array_qdot[count_qdot].append(float(lines[i].strip('\n').strip(str(count_qdot_temp)).strip('\t').strip(",")))
                else:
                    array_qdot[count_qdot].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 37:
                    count_qdot = count_qdot + 1
                    count_qdot_temp = count_qdot_temp + 1
                    if count_qdot_temp == 30:
                        count_qdot_temp = 0

            if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] > 38 and divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] < 47:
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 39:
                    array_xstate[count_xstate].append(float(lines[i].strip('\n').strip(str(count_xstate_temp)).strip('\t').strip(",")))
                else:
                    array_xstate[count_xstate].append(float(lines[i].strip('\n').strip('\t').strip(",")))
                if divmod(int(i - 6 * loop - 2106*(loop - 1)), int(71))[1] == 46:
                    count_xstate = count_xstate + 1
                    count_xstate_temp = count_xstate_temp + 1
                    if count_xstate_temp == 30:
                        count_xstate_temp = 0
    
    lines2_array = []
    for i in range(0, len(lines2)):
        lines2_array.append(lines2[i].split()) 

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
    LFframe_id = model.getFrameId("L_Foot_Link")
    RFframe_id = model.getFrameId("R_Foot_Link")
    data = model.createData()

    pinocchio.forwardKinematics(model, data, q, qdot)
    pinocchio.updateFramePlacements(model,data)
    pinocchio.centerOfMass(model, data, q, False)
    pinocchio.computeCentroidalMomentum(model,data,q,qdot)
    LF_tran = data.oMf[LFframe_id]
    RF_tran = data.oMf[RFframe_id]

    state = crocoddyl.StateKinodynamic(model)
    actuation = crocoddyl.ActuationModelKinoBase(state)
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
    
    actuation_vector = [None] * N
    state_vector = [None] * N
    state_bounds = [None] * N
    state_bounds2 = [None] * N
    state_bounds3 = [None] * N
    state_activations = [None] * N
    state_activations2 = [None] * N
    state_activations3 = [None] * N
    xRegCost_vector = [None] * N
    uRegCost_vector = [None] * N
    stateBoundCost_vector = [None] * N
    camBoundCost_vector = [None] * N
    comBoundCost_vector = [None] * N
    rf_foot_pos_vector = [None] * N
    lf_foot_pos_vector = [None] * N
    residual_FrameRF = [None] * N
    residual_FrameLF = [None] * N
    foot_trackR = [None] * N
    foot_trackL = [None] * N
    runningCostModel_vector = [None] * N
    runningDAM_vector = [None] * N
    runningModelWithRK4_vector = [None] * N
    xs = [None] * (N+1)
    us = [None] * (N)

    lb = []
    ub = []

    for i in range(0,N):
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

    
    terminalCostModel = crocoddyl.CostModelSum(state_vector[N - 1], actuation_vector[N - 1].nu + 4)
    terminalCostModel.addCost("stateReg", stateBoundCost_vector[N - 1], weight_quad_zmp[0])
    terminalCostModel.addCost("camReg", camBoundCost_vector[N - 1], 1.0)
    terminalCostModel.addCost("comReg", comBoundCost_vector[N - 1], 1.0)
    terminalCostModel.addCost("footReg1", foot_trackR[N - 1], 1.0)
    terminalCostModel.addCost("footReg2", foot_trackL[N - 1], 1.0)
    terminalDAM = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[N - 1], actuation_vector[N - 1], terminalCostModel)

    x0 = np.array([0.] * (state.nx + 8))
    u0 = np.array([0.] * (22))
    for i in range(0,len(q_init)):
        x0[i] = q_init[i]

    x0[37] = 1.12959174e-01
    x0[39] = 1.12959174e-01

    for i in range(0,N+1):
        xs[i] = x0
    for i in range(0,N):
        us[i] = u0
    print(weight_quad_zmp[0])
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, dt_)
    problemWithRK4 = crocoddyl.ShootingProblem(x0, runningModelWithRK4_vector, terminalModel)
    problemWithRK4.nthreads = 1
    ddp = crocoddyl.SolverBoxFDDP(problemWithRK4)
    ddp.solve(xs,us,300)

    walking_tick = 0
    while client.is_connected:
        T = 30
        problemWithRK4.x0 = ddp.xs[1]
        for i in range(0,N):
            state_bounds[i].lb[0] = array_boundx[30*(walking_tick)+i][0]
            state_bounds[i].ub[0] = array_boundx[30*(walking_tick)+i][1]
            state_bounds[i].lb[1] = array_boundy[30*(walking_tick)+i][0]
            state_bounds[i].ub[1] = array_boundy[30*(walking_tick)+i][1]
            state_activations[i].bounds = state_bounds[i]
            stateBoundCost_vector[i].activation_ = state_activations[i]
            rf_foot_pos_vector[i].translation[0] = array_boundRF[30*(walking_tick)+i][0]
            rf_foot_pos_vector[i].translation[1] = array_boundRF[30*(walking_tick)+i][1]
            rf_foot_pos_vector[i].translation[2] = array_boundRF[30*(walking_tick)+i][2]
            lf_foot_pos_vector[i].translation[0] = array_boundLF[30*(walking_tick)+i][0]
            lf_foot_pos_vector[i].translation[1] = array_boundLF[30*(walking_tick)+i][1]
            lf_foot_pos_vector[i].translation[2] = array_boundLF[30*(walking_tick)+i][2]
            residual_FrameRF[i].pref = rf_foot_pos_vector[i]
            residual_FrameLF[i].pref = lf_foot_pos_vector[i]
            foot_trackR[i].residual_ = residual_FrameRF[i]
            foot_trackL[i].residual_ = residual_FrameLF[i]
            runningCostModel_vector[i].removeCost("footReg1")
            runningCostModel_vector[i].removeCost("footReg2")
            runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1.0)
            runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1.0)
            print("qq")
            print(lf_foot_pos_vector[i].translation[2])
            print(array_boundLF[30*(walking_tick)+i][2])
            if i > 1:
                print(lf_foot_pos_vector[i-1].translation[2])

        '''
        if walking_tick >= 2:
            for j in range(1, N):
                for k in range(0, 19):
                    ddp.xs[j][k] = array_q[30*(walking_tick) + j][k]
                
                for k in range(19, 37):
                    ddp.xs[j][k] = array_qdot[30*(walking_tick) + j][k-19]
                
                for k in range(37, 45):
                    ddp.xs[j][k] = array_xstate[30*(walking_tick) + j][k-37]

                for k in range(0, 18):
                    ddp.us[j][k] = array_qddot[29*(walking_tick) + j][k]

                for k in range(18, 22):
                    ddp.us[j][k] = array_u[29*(walking_tick) + j][k-18]
        '''
        duration = []
        for i in range(0,T):
            c_start = time.time()
            css = ddp.solve(ddp.xs,ddp.us,300)
            c_end = time.time()
            duration.append(1e3 * (c_end - c_start))
            avrg_duration = duration[i]
            min_duration = min(duration)
            max_duration = max(duration)
            print(i)
            print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
            print('ddp.iter {0},{1},{2}'.format(ddp.iter, css, walking_tick))
        for i in range(0,N):
            print(runningCostModel_vector[i].costs['comReg'].cost.residual)
            print(runningCostModel_vector[i].costs['camReg'].cost.residual)
            print(runningCostModel_vector[i].costs['stateReg'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg1'].cost.residual)
            print(runningCostModel_vector[i].costs['footReg2'].cost.residual)
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
        f4.write(str(N-1))
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
        print(walking_tick)
        if walking_tick == 52:
            break

    f3.close()
    f4.close()
    client.terminate()

    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()
