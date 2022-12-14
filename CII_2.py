import roslibpy
import crocoddyl
import pinocchio
import numpy as np

global client

def talker():
    print("start")
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

    #pinocchio.forwardKinematics(model, data, q, qdot)
    #pinocchio.updateFramePlacements(model,data)
    #pinocchio.centerOfMass(model, data, q, False)
    #pinocchio.computeCentroidalMomentum(model,data,q,qdot)
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
    
    N = 30
    T = 1
    MAXITER = 300
    dt_ = 1.2 / float(N)

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
       
    RF_tran.translation[0] = 0.06479871
    RF_tran.translation[1] = -0.1025
    RF_tran.translation[2] = 0.14151976
    LF_tran.translation[0] = 0.06479871
    LF_tran.translation[1] = 0.1025
    LF_tran.translation[2] = 0.14151976
    print(RF_tran)
    for i in range(0,N):
        state_vector[i] = crocoddyl.StateKinodynamic(model)
        actuation_vector[i] = crocoddyl.ActuationModelKinoBase(state_vector[i])
        state_bounds[i] = crocoddyl.ActivationBounds(lb_[:,i],ub_[:,i])
        state_activations[i] = crocoddyl.ActivationModelQuadraticBarrier(state_bounds[i])
        stateBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], state_activations[i], crocoddyl.ResidualFlyState(state_vector[i], actuation_vector[i].nu + 4))
        camBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_cam), crocoddyl.ResidualModelCentroidalAngularMomentum(state_vector[i], actuation_vector[i].nu + 4))
        comBoundCost_vector[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_com), crocoddyl.ResidualModelCoMKinoPosition(state_vector[i], actuation_vector[i].nu + 4))
        rf_foot_pos_vector[i] = RF_tran
        lf_foot_pos_vector[i] = LF_tran
        residual_FrameRF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        residual_FrameLF[i] = crocoddyl.ResidualKinoFramePlacement(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i].nu + 4)
        foot_trackR[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_rf), residual_FrameRF[i])
        foot_trackL[i] = crocoddyl.CostModelResidual(state_vector[i], crocoddyl.ActivationModelWeightedQuad(weight_quad_lf), residual_FrameLF[i])
        runningCostModel_vector[i] = crocoddyl.CostModelSum(state_vector[i], actuation_vector[i].nu+4)
        runningCostModel_vector[i].addCost("stateReg", stateBoundCost_vector[i], weight_quad_zmp[0])
        runningCostModel_vector[i].addCost("camReg", camBoundCost_vector[i], 1e0)
        runningCostModel_vector[i].addCost("comReg", comBoundCost_vector[i], 1e0)
        runningCostModel_vector[i].addCost("footReg1", foot_trackR[i], 1e0)
        runningCostModel_vector[i].addCost("footReg2", foot_trackL[i], 1e0)
        runningDAM_vector[i] = crocoddyl.DifferentialActionModelKinoDynamics(state_vector[i], actuation_vector[i], runningCostModel_vector[i])
        runningModelWithRK4_vector[i] = crocoddyl.IntegratedActionModelEuler(runningDAM_vector[i], dt_)

    
    terminalCostModel = crocoddyl.CostModelSum(state_vector[N - 1], actuation_vector[N - 1].nu + 4)
    terminalCostModel.addCost("stateReg", stateBoundCost_vector[N - 1], weight_quad_zmp[0])
    terminalCostModel.addCost("camReg", camBoundCost_vector[N - 1], 1e0)
    terminalCostModel.addCost("comReg", comBoundCost_vector[N - 1], 1e0)
    terminalCostModel.addCost("footReg1", foot_trackR[N - 1], 1e0)
    terminalCostModel.addCost("footReg2", foot_trackL[N - 1], 1e0)
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
    
    print("lb")
    print(lb_)
    print(ub_)
    print("Xs")
    print(ddp.xs[N-6])
    print("Xu")
    print(ddp.us[-1])
    print("finish")

    while client.is_connected:
        T = 1

        for i in range(0,N-5):
            state_bounds[i].lb[0] = 0.0
            state_bounds[i].ub[0] = 0.2

        for i in range(N-5,N):
            state_bounds[i].lb[0] = 0.15
            state_bounds[i].ub[0] = 0.4

        for i in range(0,N-4):
            state_bounds[i].lb[1] = -0.2
            state_bounds[i].ub[1] = 0.2

        for i in range(N-4,N):
            state_bounds[i].lb[1] = 0.05
            state_bounds[i].ub[1] = 0.2

        for i in range(0,N):
            state_activations[i].bounds = state_bounds[i]
            rf_foot_pos_vector[i].translation = RF_tran.translation
            lf_foot_pos_vector[i].translation = LF_tran.translation
            residual_FrameRF[i].pref = rf_foot_pos_vector[i]
            residual_FrameLF[i].pref = lf_foot_pos_vector[i]
            foot_trackR[i].residual_ = residual_FrameRF[i]
            foot_trackL[i].residual_ = residual_FrameLF[i]


        print("Aa")
        ddp.solve(xs,us,300)

    client.terminate()
    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()

