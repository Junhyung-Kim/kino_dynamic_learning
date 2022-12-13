import roslibpy
import pinocchio
import crocoddyl
import numpy as np

global client

def talker():
    print("start")
    global model, foot_distance, data, LFframe_id, RFframe_id, PELVjoint_id, LHjoint_id, RHjoint_id, LFjoint_id, q_init, RFjoint_id, LFcframe_id, RFcframe_id, q, qdot, qddot, LF_tran, RF_tran, PELV_tran, LF_rot, RF_rot, PELV_rot, qdot_z, qddot_z, HRR_rot_init, HLR_rot_init, HRR_tran_init, HLR_tran_init, LF_rot_init, RF_rot_init, LF_tran_init, RF_tran_init, PELV_tran_init, PELV_rot_init, CPELV_tran_init, q_command, qdot_command, qddot_command, robotIginit, q_c
    model = pinocchio.buildModelFromUrdf("/home/jhk/catkin_ws/src/tocabi_cc/robots/dyros_tocabi_with_redhands.urdf",pinocchio.JointModelFreeFlyer())  
    
    pi = 3.14159265359
    data = model.createData()
    q = pinocchio.randomConfiguration(model)
    qdot = pinocchio.utils.zero(model.nv)
    qdot_init = pinocchio.utils.zero(model.nv)
    qddot = pinocchio.utils.zero(model.nv)
    q_init = [0, 0, 0.80783, 0, 0, 0, 1, 0, 0, -0.55, 1.26, -0.71, 0, 0, 0, -0.55, 1.26, -0.71, 0]
    #q_init = [0.01294101076, 0.003120668277, 0.8135094529, 0.008556023321, -0.0006148309733, -0.01279588653, 0.9998813338, 0.001752930024, -0.02157127951, -0.6016562578, 1.339193349, -0.7282713371, -0.000160158865, -0.01343052232, -0.0189677757, -0.5246394697, 1.243313265, -0.7113368054, -0.0007128223061]
    #qdot_init = [0.03597615471, 0.0214605209, 0.03173298171, 0.05051464353, -0.02421889668, -0.02116851465, 0.01012380136, -0.04675341303, -0.1098650157, 0.1358372283, -0.1006766063, 0.0782513364, 0.009917746835, -0.05568175805, 0.0759440997, -0.02251223385, -0.07128019866, 0.08248878455]
   
    for i in range(0, len(q)):
        q[i] = q_init[i]
    for i in range(0, len(qdot_init)):
        qdot[i] = qdot_init[i]
    
    RFjoint_id = model.getJointId("R_AnkleRoll_Joint")
    LFjoint_id = model.getJointId("L_AnkleRoll_Joint")
    LFframe_id = model.getFrameId("L_Foot_Link")
    RFframe_id = model.getFrameId("R_Foot_Link")
    pinocchio.forwardKinematics(model, data, q, qdot)
    pinocchio.updateFramePlacements(model,data)
    pinocchio.centerOfMass(model, data, q, False)
    pinocchio.computeCentroidalMomentum(model,data,q,qdot)
    LF_tran = data.oMf[LFframe_id].translation
    RF_tran = data.oMf[RFframe_id].translation

    print(data.com[0])
    print(data.hg)
    print(RF_tran)
    print(LF_tran)

    state = crocoddyl.StateKinodynamic(model)
    actuation = crocoddyl.ActuationModelKinoBase(state)
    traj_= [0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.08, 0.0, 0.0, 0.0]
    u_traj_ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    weight_quad_zmpx  = client.get_param("/dyros_practice_1/weight_quad_zmpx")
    weight_quad_zmpy  = client.get_param("/dyros_practice_1/weight_quad_zmpy")
    weight_quad_camx  = client.get_param("/dyros_practice_1/weight_quad_camx")
    weight_quad_camy  = client.get_param("/dyros_practice_1/weight_quad_camy")
    weight_quad_comx  = client.get_param("/dyros_practice_1/weight_quad_comx")
    weight_quad_comy  = client.get_param("/dyros_practice_1/weight_quad_comy")
    weight_quad_comz  = client.get_param("/dyros_practice_1/weight_quad_comz")
    weight_quad_rfx  = client.get_param("/dyros_practice_1/weight_quad_rfx")
    weight_quad_rfy  = client.get_param("/dyros_practice_1/weight_quad_rfy")
    weight_quad_rfz  = client.get_param("/dyros_practice_1/weight_quad_rfz")
    weight_quad_lfx  = client.get_param("/dyros_practice_1/weight_quad_lfx")
    weight_quad_lfy  = client.get_param("/dyros_practice_1/weight_quad_lfy")
    weight_quad_lfz  = client.get_param("/dyros_practice_1/weight_quad_lfz")
    weight_quad_rfroll  = client.get_param("/dyros_practice_1/weight_quad_rfroll")
    weight_quad_rfpitch  = client.get_param("/dyros_practice_1/weight_quad_rfpitch")
    weight_quad_rfyaw  = client.get_param("/dyros_practice_1/weight_quad_rfyaw")
    weight_quad_lfroll  = client.get_param("/dyros_practice_1/weight_quad_lfroll")
    weight_quad_lfpitch  = client.get_param("/dyros_practice_1/weight_quad_lfpitch")
    weight_quad_lfyaw  = client.get_param("/dyros_practice_1/weight_quad_lfyaw")
    weight_quad_zmp =[weight_quad_zmpx, weight_quad_zmpy]
    weight_quad_cam =[ weight_quad_camy, weight_quad_camx]
    weight_quad_com =[ weight_quad_comx, weight_quad_comy, weight_quad_comz]
    weight_quad_rf =[ weight_quad_rfx, weight_quad_rfy, weight_quad_rfz, weight_quad_rfroll, weight_quad_rfpitch, weight_quad_rfyaw]
    weight_quad_lf =[ weight_quad_lfx, weight_quad_lfy, weight_quad_lfz, weight_quad_lfroll, weight_quad_lfpitch, weight_quad_lfyaw]
    
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
        lb_[0,i] = -0.2
        ub_[0,i] = 0.2

    for i in range(N-4,N):
        lb_[0,i] = 0.05
        ub_[0,i] = 0.2
    
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

    lb = []
    ub = []

    for i in range(0,N):
        state_vector[i] = crocoddyl.StateKinodynamic(model)
        print(i)
        lb=lb_[i,:]
        state_bounds[i] = crocoddyl.ActivationBounds(lb, lb)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb_[:,i], ub_[:,i]))
        

    print(lb_)


    print("finish")
    

if __name__=='__main__':
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    talker()
