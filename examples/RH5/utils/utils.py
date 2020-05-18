import crocoddyl
import pinocchio
import numpy as np
import csv


def plotSolution(ddp, fs, dirName, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    if bounds: 
        rmodel, xs, us, accs, X, U, F, A, X_LB, X_UB, U_LB, U_UB = mergeDataFromSolvers(ddp, fs, bounds)
    else: 
         rmodel, xs, us, accs, X, U, F, A = mergeDataFromSolvers(ddp, fs, bounds)
    nx, nq, nu, nf, na = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0], accs[0].shape[0]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex, figsize=(16,9)) # (16,9) for bigger headings
    legJointNames = ['Hip1', 'Hip2', 'Hip3', 'Knee', 'AnkleRoll', 'AnklePitch'] # Hip 1-3: Yaw, Roll, Pitch 
    # left foot
    plt.subplot(2, 3, 1)
    plt.title('Joint Position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 13))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title('Joint Velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title('Joint Torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 6))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 6))]
    plt.ylabel('LF')
    plt.legend()
    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(13, 19))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(13, 19))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.legend()
    plt.subplot(2, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 12))]
    plt.ylabel('RF')
    plt.xlabel('Knots')
    plt.legend()
    plt.savefig(dirName + 'Solution.png', bbox_inches = 'tight', dpi = 300)


    # Plotting floating base coordinates
    plt.figure(figIndex + 1, figsize=(16,9))
    baseTranslationNames = ['X', 'Y', 'Z']
    [plt.plot(X[k], label=baseTranslationNames[i]) for i, k in enumerate(range(0, 3))]
    plt.xlabel('Knots')
    plt.ylabel('Translation [m]')
    plt.legend()
    plt.savefig(dirName + 'FloatingBase.png', bbox_inches = 'tight', dpi = 300)
        

    # Get 3 dim CoM, get feet poses
    rdata = rmodel.createData()
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
        lfId = rmodel.getFrameId('FL_SupportCenter')
        rfId = rmodel.getFrameId('FR_SupportCenter')
        lfPoses.append(rdata.oMf[lfId].translation) # TODO: Add rotation as seperate vector
        rfPoses.append(rdata.oMf[rfId].translation)
    
    nfeet = lfPoses[0].shape[0]
    lfPose, rfPose = [0.] * nfeet, [0.] * nfeet       
    for i in range(nfeet):
        lfPose[i] = [np.asscalar(p[i]) for p in lfPoses]
        rfPose[i] = [np.asscalar(p[i]) for p in rfPoses]

    knots = list(range(0,len(Cz)))

    # Plotting the Center of Mass and Feet (x,y,z over knots)
    plt.figure(figIndex + 2, figsize=(16,9))
    plt.subplot(2, 3, 1)
    plt.plot(knots, Cx)
    plt.xlabel('Knots')
    plt.ylabel('CoM X [m]')
    plt.subplot(2, 3, 2)
    plt.plot(knots, Cy)
    plt.xlabel('Knots')
    plt.ylabel('CoM Y [m]')
    plt.subplot(2, 3, 3)
    plt.plot(knots, Cz)
    plt.xlabel('Knots')
    plt.ylabel('CoM Z [m]')
    plt.subplot(2, 3, 4)
    [plt.plot(knots, lfPose[0], label='LF'), plt.plot(knots, rfPose[0], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot X [m]')
    plt.legend()
    plt.subplot(2, 3, 5)
    [plt.plot(knots, lfPose[1], label='LF'), plt.plot(knots, rfPose[1], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot Y [m]')
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(knots, lfPose[2], label='LF'), plt.plot(knots, rfPose[2], label='RF')]
    plt.xlabel('Knots')
    plt.ylabel('Foot Z [m]')
    plt.legend()
    plt.savefig(dirName + 'CoMAndFeet.png', bbox_inches = 'tight', dpi = 300)

    # # Plotting the Center of Mass (y,z over x)
    # plt.figure(figIndex + 3, figsize=(16,9))
    # plt.subplot(1, 2, 1)
    # plt.plot(Cx, Cy)
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # plt.subplot(1, 2, 2)
    # plt.plot(Cx, Cz)
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.savefig(dirName + 'CoM2.png', bbox_inches = 'tight', dpi = 300)


    # Plotting the contact wrenches
    contactForceNames = ['Fx','Fy','Fz'] 
    contactMomentNames = ['Tx','Ty','Tz']
    plt.figure(figIndex + 3, figsize=(16,9))

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
    plt.savefig(dirName + 'ContactWrenches.png', bbox_inches = 'tight', dpi = 300)


    # Plotting the Acceleration
    AccFBNames = ['vxd', 'vyd', 'vzd', 'wxd', 'wyd', 'wzd']
    plt.figure(figIndex + 4, figsize=(16,9))

    plt.subplot(3,1,1)
    [plt.plot(A[k], label=AccFBNames[i]) for i, k in enumerate(range(0, 6))]
    plt.xlabel('Knots')
    plt.ylabel('FB')
    plt.legend()
    plt.subplot(3,1,2)
    [plt.plot(A[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    plt.xlabel('Knots')
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(3,1,3)
    [plt.plot(A[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    plt.xlabel('Knots')
    plt.ylabel('RF')
    plt.legend()
    plt.savefig(dirName + 'Acceleration.png', bbox_inches = 'tight', dpi = 300)


def logSolution(ddp, fs, timeStep, logPath):
    # Stack together all data contained in multiple solvers
    rmodel, xs, us, accs, X, U, F, A = mergeDataFromSolvers(ddp, fs, bounds=False)
    nx, nq, nu, nf, na = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0], accs[0].shape[0]
    # Collect time steps
    time = []
    for t in range(len(xs)):
        time.append(round(timeStep * t, 2))

    filename = logPath + 'logJointStatesAndEffort.csv'
    rangeRelJoints = list(range(7, nq)) + list(range(nq + 6, nq + 18))  # Ignore floating base (fixed joints)
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
                         'q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch',
                         'q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch',
                         'qd_LLHip1', 'qd_LLHip2', 'qd_LLHip3', 'qd_LLKnee', 'qd_LLAnkleRoll', 'qd_LLAnklePitch',
                         'qd_LRHip1', 'qd_LRHip2', 'qd_LRHip3', 'qd_LRKnee', 'qd_LRAnkleRoll', 'qd_LRAnklePitch',
                         'qdd_LLHip1', 'qdd_LLHip2', 'qdd_LLHip3', 'qdd_LLKnee', 'qdd_LLAnkleRoll', 'qdd_LLAnklePitch',
                         'qdd_LRHip1', 'qdd_LRHip2', 'qdd_LRHip3', 'qdd_LRKnee', 'qdd_LRAnkleRoll', 'qdd_LRAnklePitch',
                         'Tau_LLHip1', 'Tau_LLHip2', 'Tau_LLHip3', 'Tau_LLKnee', 'Tau_LLAnkleRoll', 'Tau_LLAnklePitch',
                         'Tau_LRHip1', 'Tau_LRHip2', 'Tau_LRHip3', 'Tau_LRKnee', 'Tau_LRAnkleRoll', 'Tau_LRAnklePitch'])
        writer.writerows(sol)

    filename = logPath + 'logBaseStates.csv'
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

    filename = logPath + 'logCoMAndFeetPoses.csv'
    cs = []
    lfPoses = []
    rfPoses = []
    sol = np.zeros([len(time), 9])
    rdata = rmodel.createData()
    # Calculate CoM and foot poses for all states
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        cs.append(c)
        pinocchio.forwardKinematics(rmodel, rdata, q)
        pinocchio.updateFramePlacements(rmodel, rdata)
        lfId = rmodel.getFrameId('FL_SupportCenter')
        rfId = rmodel.getFrameId('FR_SupportCenter')
        lfPoses.append(rdata.oMf[lfId].translation) # TODO: Add rotation as seperate vector
        rfPoses.append(rdata.oMf[rfId].translation)
    for l in range(len(time)):
        sol[l] = [*cs[l], *lfPoses[l], *rfPoses[l]]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cx', 'Cy', 'Cz', 
                         'X_FL_SupportCenter', 'Y_FL_SupportCenter', 'Z_FL_SupportCenter',
                         'X_FR_SupportCenter', 'Y_FR_SupportCenter', 'Z_FR_SupportCenter'])
        writer.writerows(sol)


def setLimits(rmodel):
    # Add the free-flyer joint limits (floating base)
    ub = rmodel.upperPositionLimit
    ub[:7] = 1
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1
    rmodel.lowerPositionLimit = lb

    # If desired: Artificially reduce the torque limits
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
