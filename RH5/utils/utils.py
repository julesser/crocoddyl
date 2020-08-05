import crocoddyl
import pinocchio
import numpy as np
import csv

# from libcrocoddyl_pywrap import *
# from libcrocoddyl_pywrap import __version__

def plotSolution(ddp, fs, dirName, num_knots, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if bounds: 
        rmodel, xs, us, accs, X, U, F, A, X_LB, X_UB, U_LB, U_UB = mergeDataFromSolvers(ddp, fs, bounds)
    else: 
         rmodel, xs, us, accs, X, U, F, A = mergeDataFromSolvers(ddp, fs, bounds)
    nx, nq, nu, nf, na = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0], accs[0].shape[0]
    # print('nx: ', nx)
    # print('nq: ', nq)
    # print('na: ', na)

    # Plotting the joint state: positions, velocities and torques
    plt.figure(figIndex, figsize=(16,9)) # (16,9) for bigger headings
    torsoJointNames = ['BodyPitch','BodyRoll','BodyYaw']
    legJointNames = ['Hip1', 'Hip2', 'Hip3', 'Knee', 'AnkleRoll', 'AnklePitch']
    # torso
    plt.subplot(3, 3, 1)
    plt.title('Joint Position [rad]')
    [plt.plot(X[k], label=torsoJointNames[i]) for i, k in enumerate(range(7, 10))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 10))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 10))]
    plt.ylabel('Torso')
    plt.subplot(3, 3, 2)
    plt.title('Joint Velocity [rad/s]')
    [plt.plot(X[k], label=torsoJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('Torso')
    plt.subplot(3, 3, 3)
    plt.title('Joint Acceleration [rad/s²]')
    [plt.plot(A[k], label=torsoJointNames[i]) for i, k in enumerate(range(6, 9))]
    plt.ylabel('Torso')
    plt.legend()
    # left foot
    plt.subplot(3, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 16))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(10, 16))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(10, 16))]
    plt.ylabel('LF')
    plt.subplot(3, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 15))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 15))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 15))]
    plt.ylabel('LF')
    plt.subplot(3, 3, 6)
    [plt.plot(A[k], label=legJointNames[i]) for i, k in enumerate(range(9, 15))]
    plt.ylabel('LF')
    plt.legend()
    # right foot
    plt.subplot(3, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 22))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(16, 22))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(16, 22))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.subplot(3, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 21))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 21))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 21))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.subplot(3, 3, 9)
    [plt.plot(A[k], label=legJointNames[i]) for i, k in enumerate(range(15, 21))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.legend()
    plt.savefig(dirName + 'JointState.png', dpi = 300)

    # Plotting the joint torques
    plt.figure(figIndex+1, figsize=(16,9))
    # Torso
    plt.subplot(3, 1, 1)
    plt.title('Joint Torque [Nm]')
    [plt.plot(U[k], label=torsoJointNames[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 3))]
    plt.ylabel('Torso')
    plt.xlabel('Knots')
    plt.legend()
    plt.subplot(3, 1, 2)
    # Left foot
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 9))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(3, 9))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(3, 9))]
    plt.ylabel('LF')
    plt.xlabel('Knots')
    plt.legend()
    # Right foot
    plt.subplot(3, 1, 3)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 15))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(9, 15))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(9, 15))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.legend()
    plt.savefig(dirName + 'JointTorques.png', dpi = 300)
    
    # Get 3 dim CoM, get feet poses
    rdata = rmodel.createData()
    lfId = rmodel.getFrameId('FL_SupportCenter')
    rfId = rmodel.getFrameId('FR_SupportCenter')
    Cx = []
    Cy = []
    Cz = []
    lfPoses = []
    rfPoses = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
        Cz.append(np.asscalar(c[2]))
        pinocchio.forwardKinematics(rmodel, rdata, q)
        pinocchio.updateFramePlacements(rmodel, rdata)
        lfPoses.append(rdata.oMf[lfId].translation) 
        rfPoses.append(rdata.oMf[rfId].translation)
        # print(rdata.oMf[lfId]) # Pose specified via rotation matrix + translation vector
        # print(pinocchio.SE3ToXYZQUATtuple(rdata.oMf[lfId])) # Pose specified via quaternion + translation vector

    nfeet = lfPoses[0].shape[0]
    lfPose, rfPose = [0.] * nfeet, [0.] * nfeet       
    for i in range(nfeet):
        lfPose[i] = [np.asscalar(p[i]) for p in lfPoses]
        rfPose[i] = [np.asscalar(p[i]) for p in rfPoses]
    knots = list(range(0,len(Cz)))

    # Analyse CoP cost
    copCost, frictionConeCost = 0, 0
    com2DTrackCost, footTrackCost = 0, 0 
    jointLimCost, stateRecoveryCost, stateRegCost, ctrlRegCost = 0, 0, 0, 0
    if isinstance(ddp, list):
        for s in ddp:
            for j in range(s.problem.T):
                # print(s.problem.runningModels[j].differential.costs.costs)
                for costName in s.problem.runningModels[j].differential.costs.costs:
                    costTerm = s.problem.runningDatas[j].differential.costs.costs[costName]
                    if costName == "FR_SupportCenter_CoP" or costName == "FL_SupportCenter_CoP":
                        copCost += costTerm.cost
                        # print(costName)
                        # print("r: " + str(costTerm.r))
                        # print("cost: " + str(costTerm.cost))
                        # print("----------------------------")
                        # if costTerm.cost != 0: 
                        #     print("#####################################################")
                    elif costName == "FR_SupportCenter_frictionCone" or costName == "FL_SupportCenter_frictionCone":
                        frictionConeCost += costTerm.cost
                        # print(costName)
                        # print("r: " + str(costTerm.r))
                        # print("cost: " + str(costTerm.cost))
                        # print("----------------------------")
                        # if costTerm.cost != 0: 
                        #     print("#####################################################")
                    elif costName == "com2DTrack":
                        com2DTrackCost += costTerm.cost
                    elif costName == "FR_SupportCenter_footTrack" or costName == "FL_SupportCenter_footTrack":
                        footTrackCost += costTerm.cost
                    elif costName == "jointLim":
                        jointLimCost += costTerm.cost
                    elif costName == "stateReg":
                        stateRegCost += costTerm.cost
                    elif costName == "ctrlReg":
                        ctrlRegCost += costTerm.cost
                    elif costName == "stateRecovery":
                        stateRecoveryCost += costTerm.cost
        print("total copCost: " + str(copCost))
        print("total frictionConeCost: " + str(frictionConeCost))
        print("total com2DTrack: " + str(com2DTrackCost))
        print("total footTrackCost: " + str(footTrackCost))
        print("total jointLimCost: " + str(jointLimCost))
        print("total stateRegCost: " + str(stateRegCost))
        print("total ctrlRegCost: " + str(ctrlRegCost))
        print("total stateRecoveryCost: " + str(stateRecoveryCost))
        print("..total costs then are multiplied with the assigned weight")
    
    # Get forces from solver(s)
    forces = getCartesianForcesLocalCS(ddp)
    # Calc CoP traejctory
    CoPs = calcCoPs(forces)
    # Transform CoP to image plane (world CS)
    CoPLF = np.zeros((2, len(CoPs)))
    CoPRF = np.zeros((2, len(CoPs)))
    CoPLFx, CoPLFy, CoPRFx, CoPRFy = [], [], [], []
    print("timepoints: " + str(len(CoPs)))
    for k in range(len(CoPs)): 
        for CoP in CoPs[k]: # Iterate if DS
            if CoP["key"] == "10":  # LF
                # print(CoP["CoP"][0])
                CoPLF[0][k] = CoP["CoP"][0] + lfPose[0][k]
                CoPLF[1][k] = CoP["CoP"][1] + lfPose[1][k]
                CoPLFx.append(CoP["CoP"][0] + lfPose[0][k])
                CoPLFy.append(CoP["CoP"][1] + lfPose[1][k])
            elif CoP["key"] == "16":  # RF
                CoPRF[0][k] = CoP["CoP"][0] + rfPose[0][k]
                CoPRF[1][k] = CoP["CoP"][1] + rfPose[1][k]
                CoPRFx.append(CoP["CoP"][0] + rfPose[0][k])
                CoPRFy.append(CoP["CoP"][1] + rfPose[1][k])

    # Stability Analysis: XY-Plot of CoM Projection and Feet Positions
    footLength, footWidth = 0.2, 0.08
    total_knots = sum(num_knots)
    # relTimePoints = [0,(2*total_knots)+num_knots[1]-1] # TaskSpecific: Walking 2 steps (stabilization)
    relTimePoints = [0,(2*total_knots)-1] # TaskSpecific: Walking 2 steps
    # relTimePoints = [0,(2*total_knots)-1, (4*total_knots)-1,(6*total_knots)+num_knots[1]-1] # TaskSpecific: Walking Long Gait
    # relTimePoints = [0,40,100] # TaskSpecific: Squats
    # relTimePoints = [0] # TaskSpecific: Balancing
    numPlots = list(range(1,len(relTimePoints)+1))
    plt.figure(figIndex + 2, figsize=(16,9))
    # (1) Variant with subplots
    # for i, t in zip(numPlots, relTimePoints):
    #     plt.subplot(1, len(relTimePoints), i)
    #     plt.plot(Cx[t], Cy[t], marker='x', markersize = '10', label='CoM')
    #     [plt.plot(lfPose[0][t], lfPose[1][t], marker='x', markersize = '10', label='LF'), plt.plot(rfPose[0][t], rfPose[1][t], marker='x', markersize = '10', label='RF')]
    #     plt.xlabel('X [m]')
    #     plt.ylabel('Y [m]')
    #     plt.legend()
    #     plt.axis('scaled')
    #     plt.xlim(0, .4)
    #     plt.ylim(-.2, .2)
    #     currentAxis = plt.gca()
    #     currentAxis.add_patch(Rectangle((lfPose[0][t] - footLength/2, lfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor='k', fill=False))
    #     currentAxis.add_patch(Rectangle((rfPose[0][t] - footLength/2, rfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor='k', fill=False))
    # (2) Variant with just one plot
    # plt.subplot(1,2,1)
    plt.plot(Cx[1:-1], Cy[1:-1], label='CoM')
    plt.plot(Cx[0], Cy[0], marker='o', linestyle='', label='CoMStart')
    plt.plot(Cx[num_knots[1]-1], Cy[num_knots[1]-1], marker='o', linestyle='', label='CoMRFLiftOff') # TaskSpecific: Walking ff.
    plt.plot(Cx[total_knots-1], Cy[total_knots-1], marker='o', linestyle='', label='CoMRFTouchDown') 
    plt.plot(Cx[total_knots + num_knots[1]-1], Cy[total_knots + num_knots[1]-1], marker='o', linestyle='', label='CoMLFLiftOff')
    plt.plot(Cx[2*(total_knots)-1], Cy[2*(total_knots)-1], marker='o', linestyle='', label='CoMLFTouchDown')
    plt.plot(Cx[-1], Cy[-1], marker='o', linestyle='', label='CoMEnd') 
    [plt.plot(lfPose[0][0], lfPose[1][0], marker='>', markersize = '10', linestyle='', label='LFStart'), plt.plot(rfPose[0][0], rfPose[1][0], marker='>', markersize = '10', linestyle='', label='RFT1')]
    [plt.plot(lfPose[0][-1], lfPose[1][-1], marker='>', markersize = '10', linestyle='', label='LFEnd'), plt.plot(rfPose[0][-1], rfPose[1][-1], marker='>', markersize = '10', linestyle='', label='RFT2')]
    [plt.plot(CoPLFx, CoPLFy, marker='x', linestyle='', label='LFCoP')]
    [plt.plot(CoPRFx, CoPRFy, marker='x', linestyle='', label='RFCoP')]
    plt.legend()
    plt.axis('scaled')
    plt.xlim(0, 0.5)
    plt.ylim(-0.2, 0.2)
    # plt.xlim(0, 0.9) TaskSpecific: LongGait or large steps
    # plt.ylim(-0.3, 0.3)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    currentAxis = plt.gca()
    for t in relTimePoints:
        # if smaller region: draw dotted rectangle
        currentAxis.add_patch(Rectangle((lfPose[0][t] - footLength/4, lfPose[1][t] - footWidth/4), footLength/2, footWidth/2, edgecolor = 'silver', linestyle=':', fill=False))
        currentAxis.add_patch(Rectangle((rfPose[0][t] - footLength/4, rfPose[1][t] - footWidth/4), footLength/2, footWidth/2, edgecolor = 'silver', linestyle=':', fill=False))
        # currentAxis.add_patch(Rectangle((lfPose[0][t] - footLength/8, lfPose[1][t] - footWidth/8), footLength/4, footWidth/4, edgecolor = 'silver', linestyle=':', fill=False))
        # currentAxis.add_patch(Rectangle((rfPose[0][t] - footLength/8, rfPose[1][t] - footWidth/8), footLength/4, footWidth/4, edgecolor = 'silver', linestyle=':', fill=False))
        if t != relTimePoints[-1]:
            # black rectangles 
            currentAxis.add_patch(Rectangle((lfPose[0][t] - footLength/2, lfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor = 'k', fill=False))
            currentAxis.add_patch(Rectangle((rfPose[0][t] - footLength/2, rfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor = 'k', fill=False))
        else: 
            # red rectangles for last double support
            currentAxis.add_patch(Rectangle((lfPose[0][t] - footLength/2, lfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor = 'r', fill=False))
            currentAxis.add_patch(Rectangle((rfPose[0][t] - footLength/2, rfPose[1][t] - footWidth/2), footLength, footWidth, edgecolor = 'r', fill=False))
    # # Additionally plot CoM height TaskSpecific: Squats
    # plt.subplot(1, 2, 2)
    # plt.plot(knots, Cz)
    # plt.xlabel('Knots')
    # plt.ylabel('CoM Z [m]')
    plt.savefig(dirName + 'StabilityAnalysis.png', dpi = 300)

    # Plotting the Task Space: Center of Mass and Feet (x,y,z over knots)
    plt.figure(figIndex + 3, figsize=(16,9))
    plt.subplot(3, 3, 1)
    [plt.plot(knots, CoPLF[0], label='LF'), plt.plot(knots, CoPRF[0], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('CoP X [m]')
    plt.legend()
    plt.subplot(3, 3, 2)
    [plt.plot(knots, CoPLF[1], label='LF'), plt.plot(knots, CoPRF[1], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('CoP Y [m]')
    plt.legend()
    plt.subplot(3, 3, 4)
    plt.plot(knots, Cx)
    plt.xlabel('Knots')
    plt.ylabel('CoM X [m]')
    plt.subplot(3, 3, 5)
    plt.plot(knots, Cy)
    plt.xlabel('Knots')
    plt.ylabel('CoM Y [m]')
    plt.subplot(3, 3, 6)
    plt.plot(knots, Cz)
    plt.xlabel('Knots')
    plt.ylabel('CoM Z [m]')
    plt.subplot(3, 3, 7)
    [plt.plot(knots, lfPose[0], label='LF'), plt.plot(knots, rfPose[0], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot X [m]')
    plt.legend()
    plt.subplot(3, 3, 8)
    [plt.plot(knots, lfPose[1], label='LF'), plt.plot(knots, rfPose[1], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot Y [m]')
    plt.legend()
    plt.subplot(3, 3, 9)
    [plt.plot(knots, lfPose[2], label='LF'), plt.plot(knots, rfPose[2], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot Z [m]')
    plt.legend()
    plt.savefig(dirName + 'TaskSpace.png', dpi = 300)

    # Plotting the Contact Wrenches
    contactForceNames = ['Fx','Fy','Fz'] 
    contactMomentNames = ['Tx','Ty','Tz']
    plt.figure(figIndex + 4, figsize=(16,9))
    plt.subplot(2,2,1)
    plt.title('Contact Forces [N]')
    [plt.plot(F[k], label=contactForceNames[i]) for i, k in enumerate(range(0, 3))]
    plt.xlabel('Knots')
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2,2,2)
    plt.title('Contact Moment [Nm]')
    [plt.plot(F[k], label=contactMomentNames[i]) for i, k in enumerate(range(3, 6))]
    plt.plot()
    plt.xlabel('Knots')
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2,2,3)
    [plt.plot(F[k], label=contactForceNames[i]) for i, k in enumerate(range(6, 9))]
    plt.xlabel('Knots')
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(2,2,4)
    [plt.plot(F[k], label=contactMomentNames[i]) for i, k in enumerate(range(9, nf))]
    plt.plot()
    plt.xlabel('Knots')
    plt.ylabel('RF')
    plt.legend()
    plt.savefig(dirName + 'ContactWrenches.png', dpi = 300)

    # Plotting the Floating Base Position, Velocity and Acceleration
    PosFBNames = ['X', 'Y', 'Z']
    VelFBNames = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    AccFBNames = ['vxd', 'vyd', 'vzd', 'wxd', 'wyd', 'wzd']
    plt.figure(figIndex + 5, figsize=(16,9))
    ax1 = plt.subplot(3,1,1)
    [plt.plot(X[k], label=PosFBNames[i]) for i, k in enumerate(range(0, 3))]
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylabel('Position')
    plt.legend()
    ax2 = plt.subplot(3,1,2)
    [plt.plot(X[k], label=VelFBNames[i]) for i, k in enumerate(range(nq, nq+6))]
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('Velocity')
    plt.legend()
    plt.subplot(3,1,3)
    [plt.plot(A[k], label=AccFBNames[i]) for i, k in enumerate(range(0, 6))]
    plt.xlabel('Knots')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.savefig(dirName + 'Base.png', dpi = 300)


def logSolution(ddp, fs, timeStep, logPath):
    # Stack together all data contained in multiple solvers
    rmodel, xs, us, accs, X, U, F, A = mergeDataFromSolvers(ddp, fs, bounds=False)
    nx, nq, nu, nf, na = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0], accs[0].shape[0]
    # Collect time steps
    time = []
    for t in range(len(xs)):
        time.append(round(timeStep * t, 2))

    filename = logPath + 'logJointSpace.csv'
    rangeRelJoints = list(range(7, nq)) + list(range(nq + 6, nq + nx-nq))  # Ignore floating base (fixed joints)
    rangeRelAccs = list(range(6, na)) # Acceleration is 6-dim
    XRel = []
    ARel = []
    # Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
    for k in rangeRelJoints:
        XRel.append(X[k])
    # Get relevant accelerations (x_LF, x_RF, v_LF, v_RF)
    for l in rangeRelAccs: 
        ARel.append(A[l])
    sol = list(map(list, zip(*XRel))) # Transpose
    solU = list(map(list, zip(*U))) # Transpose
    solA = list(map(list, zip(*ARel)))  # Transpose
    # Include time and effort columns
    for m in range(len(sol)):
        sol[m] = [time[m]] + sol[m] + solA[m] + solU[m] 
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t[s]',
                         'q_BodyPitch', 'q_BodyRoll', 'q_BodyYaw',
                         'q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch',
                         'q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch',
                         'qd_BodyPitch', 'qd_BodyRoll', 'qd_BodyYaw',
                         'qd_LLHip1', 'qd_LLHip2', 'qd_LLHip3', 'qd_LLKnee', 'qd_LLAnkleRoll', 'qd_LLAnklePitch',
                         'qd_LRHip1', 'qd_LRHip2', 'qd_LRHip3', 'qd_LRKnee', 'qd_LRAnkleRoll', 'qd_LRAnklePitch',
                         'qdd_BodyPitch', 'qdd_BodyRoll', 'qdd_BodyYaw',
                         'qdd_LLHip1', 'qdd_LLHip2', 'qdd_LLHip3', 'qdd_LLKnee', 'qdd_LLAnkleRoll', 'qdd_LLAnklePitch',
                         'qdd_LRHip1', 'qdd_LRHip2', 'qdd_LRHip3', 'qdd_LRKnee', 'qdd_LRAnkleRoll', 'qdd_LRAnklePitch',
                         'Tau_BodyPitch', 'Tau_BodyRoll', 'Tau_BodyYaw',
                         'Tau_LLHip1', 'Tau_LLHip2', 'Tau_LLHip3', 'Tau_LLKnee', 'Tau_LLAnkleRoll', 'Tau_LLAnklePitch',
                         'Tau_LRHip1', 'Tau_LRHip2', 'Tau_LRHip3', 'Tau_LRKnee', 'Tau_LRAnkleRoll', 'Tau_LRAnklePitch'])
        writer.writerows(sol)

    filename = logPath + 'logBase.csv'
    rangeRelJoints = list(range(0, 7)) + list(range(nq, nq + 6)) # Ignore other joints
    rangeRelAccs = list(range(0, 6)) # Acceleration is 6-dim
    XRel = []
    ARel = []
    sol = []
    # Get relevant joints states (floating base)
    for k in rangeRelJoints:
        XRel.append(X[k])
    sol = list(map(list, zip(*XRel)))  # Transpose
    # Get relevant accelerations (floating base)
    for l in rangeRelAccs: 
        ARel.append(A[l])
    solA = list(map(list, zip(*ARel)))  # Transpose
    # Include time column
    for m in range(len(sol)):
        sol[m] = [time[m]] + sol[m] + solA[m] 
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t[s]',
                         'X', 'Y', 'Z', 'Qx', 'Qy', 'Qz', 'Qw',
                         'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                         'vxd', 'vyd', 'vzd', 'wxd', 'wyd', 'wzd'])
        writer.writerows(sol)

    filename = logPath + 'logContactWrenches.csv'
    sol = np.zeros([len(time), 13])
    # Include time column
    for l in range(len(time)):
        sol[l] = [*[time[l]],*fs[l]] 
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t[s]',
                         'Fx_FL_SupportCenter', 'Fy_FL_SupportCenter', 'Fz_FL_SupportCenter', 'Tx_FL_SupportCenter', 'Ty_FL_SupportCenter', 'Tz_FL_SupportCenter',
                         'Fx_FR_SupportCenter', 'Fy_FR_SupportCenter', 'Fz_FR_SupportCenter', 'Tx_FR_SupportCenter', 'Ty_FR_SupportCenter', 'Tz_FR_SupportCenter'])
        writer.writerows(sol)

    filename = logPath + 'logTaskSpace.csv'
    cs = []
    lfPoses, rfPoses = [], []
    lfPosition, rfPosition = [], []
    sol = np.zeros([len(time), 21])
    rdata = rmodel.createData()
    lfId = rmodel.getFrameId('FL_SupportCenter')
    rfId = rmodel.getFrameId('FR_SupportCenter')
    # Calculate CoM and foot poses for all states
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        cs.append(c)
        pinocchio.forwardKinematics(rmodel, rdata, q)
        pinocchio.updateFramePlacements(rmodel, rdata)
        # print(rdata.oMf[lfId])  # Pose specified via rotation matrix + translation vector
        # print(pinocchio.SE3ToXYZQUATtuple(rdata.oMf[lfId]))  # Pose specified via quaternion + translation vector
        lfPoses.append(pinocchio.SE3ToXYZQUATtuple(rdata.oMf[lfId]))
        rfPoses.append(pinocchio.SE3ToXYZQUATtuple(rdata.oMf[rfId]))
        lfPosition.append(rdata.oMf[lfId].translation) 
        rfPosition.append(rdata.oMf[rfId].translation)
    # Transform foot poses to image plane
    nfeet = lfPosition[0].shape[0]
    lfPosImg, rfPosImg = [0.] * nfeet, [0.] * nfeet       
    for i in range(nfeet):
        lfPosImg[i] = [np.asscalar(p[i]) for p in lfPosition]
        rfPosImg[i] = [np.asscalar(p[i]) for p in rfPosition]
    # Compute CoP
    forces = getCartesianForcesLocalCS(ddp)
    CoPs = calcCoPs(forces)
    # Transform CoP to image plane
    CoPLF = np.zeros((2, len(CoPs)))
    CoPRF = np.zeros((2, len(CoPs)))
    for k in range(len(CoPs)): 
        for CoP in CoPs[k]: # Iterate if DS
            if CoP["key"] == "10":  # LF
                CoPLF[0][k] = CoP["CoP"][0] + lfPosImg[0][k]
                CoPLF[1][k] = CoP["CoP"][1] + lfPosImg[1][k]
            elif CoP["key"] == "16":  # RF
                CoPRF[0][k] = CoP["CoP"][0] + rfPosImg[0][k]
                CoPRF[1][k] = CoP["CoP"][1] + rfPosImg[1][k]
    for l in range(len(time)):
        sol[l] = [*cs[l], CoPLF[0][l], CoPLF[1][l], CoPRF[0][l], CoPRF[1][l], *lfPoses[l], *rfPoses[l]]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cx', 'Cy', 'Cz',
                         'CoPx_FL', 'CoPy_FL', 'CoPx_FR', 'CoPy_FR',  
                         'X_FL_SupportCenter', 'Y_FL_SupportCenter', 'Z_FL_SupportCenter', 'Qx_FL_SupportCenter', 'Qy_FL_SupportCenter', 'Qz_FL_SupportCenter', 'Qw_FL_SupportCenter',
                         'X_FR_SupportCenter', 'Y_FR_SupportCenter', 'Z_FR_SupportCenter', 'Qx_FR_SupportCenter', 'Qy_FR_SupportCenter', 'Qz_FR_SupportCenter', 'Qw_FR_SupportCenter',])
        writer.writerows(sol)


def setLimits(rmodel):
    # Add the free-flyer joint limits (floating base)
    ub = rmodel.upperPositionLimit
    ub[:7] = 1
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1
    rmodel.lowerPositionLimit = lb

    # Artificially reduce the torque limits
    lims = rmodel.effortLimit
    # lims *= 0.5 
    # lims[11] = 70
    rmodel.effortLimit = lims


def mergeDataFromSolvers(ddp, fs, bounds):
    xs, us, accs = [], [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(ddp, list):
        rmodel = ddp[0].problem.runningModels[0].state.pinocchio
        for s in ddp:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            for j in range(s.problem.T):
                accs.append(s.problem.runningDatas[j].differential.xout)
            if bounds:
                models = s.problem.runningModels + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = ddp.problem.runningModels[0].state.pinocchio
        xs, us = ddp.xs, ddp.us
        for j in range(ddp.problem.T):
                accs.extend(s.problem.runningDatas[j].differential.xout)
        if bounds:
            models = s.problem.runningModels + [s.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state, control and wrench trajectories
    nx, nq, nu, nf, na = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0], accs[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    F = [0.] * 12
    A = [0.] * na
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(na):
        A[i] = [np.asscalar(a[i]) for a in accs]
    for i in range(nf):
        F[i] = [np.asscalar(f[i]) for f in fs]
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    if bounds: 
        return rmodel, xs, us, accs, X, U, F, A, X_LB, X_UB, U_LB, U_UB
    else: 
        return rmodel, xs, us, accs, X, U, F, A


def getCartesianForcesLocalCS(ddp):
    # Idea: Calc individual CoP for each foot and merge geometrically via Fz weight
    forces = []
    if isinstance(ddp, list):
        for s in ddp:
            models = s.problem.runningModels
            datas = s.problem.runningDatas
            # m_externalForces = m_robotData->liMi[1].act(m_robotData->f[1]);
            for i, data in enumerate(datas):
                model = models[i]
                force_k = []
                # 1) Compute cartesian forces expressed in world frame
                for key, contact in data.differential.multibody.contacts.contacts.items():  # Iterate all available contacts (2 for DS)
                    if model.differential.contacts.contacts[key].active:
                        # Old Approach from getForceTrajectoryFromSolver()
                        force = contact.jMf.actInv(contact.f)
                        # New Approach: Transform contact forces (in local contact frame) to cartesian forces (in world frame)
                        # fiMo = pinocchio.SE3(contact.pinocchio.oMi[contact.joint].rotation.T, contact.jMf.translation)
                        # force = fiMo.actInv(contact.f)
                        force_k.append({"key": str(contact.joint), "f": force})
                forces.append(force_k)
    else: 
        models = ddp.problem.runningModels
        datas = ddp.problem.runningDatas
        for i, data in enumerate(datas):
            model = models[i]
            force_k = []
            for key, contact in data.differential.multibody.contacts.contacts.items():
                if model.differential.contacts.contacts[key].active:
                    force = contact.jMf.actInv(contact.f)
                    force_k.append({"key": str(contact.joint), "f": force})
            forces.append(force_k)

    return forces

def calcCoPs(forces):
    CoPs = []
    for force in forces:
        CoP_k = []
        for i in range(len(force)):
            f = force[i]["f"]
            key = force[i]["key"]
            CoP_k_i = [np.asscalar(-f.angular[1] / f.linear[2]),  # CoPX = tauY / fZ
                       np.asscalar(f.angular[0] / f.linear[2]),  # CoPY = tauX / fZ
                       0.0]
            CoP_k.append({"key": key, "f": f, "CoP": CoP_k_i})
        CoPs.append(CoP_k)
    return CoPs 

# TODO: ZMP calculation for cross-checking (should equal CoP results for flat floor)
def calcZMPs(ma_G, H_G):
    pass
    # eq.(20): OD = (n x M_o^gi) / (R^gi x n)       (zero moment point)
    # eq.(8):  RGi = mg - ma_G                      (gravity plus inertia forces, ma_g acceleration of G)
    # eq.(9):  M_Q^gi = QG x mg - QG x ma_G - H_G   (gravity plus inertia moments, H_G rate of angular momentum at G)

    # Input from Justin
    # CoM acceleration: computeCenterOfMass(model,data,q,v,a) access via data.acom[0]
    # Gravity field:    model.gravity.linear
    # Rate of angular momentum: computeCentroidalMomentumTimeVariation() access via data.dhg.angular
