import crocoddyl
import pinocchio
import numpy as np
import csv

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



def plotSolution(solver, fs, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(solver, list):
        rmodel = solver[0].problem.runningModels[0].state.pinocchio
        for s in solver:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            if bounds:
                models = s.problem.runningModels + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        xs, us = solver.xs, solver.us
        if bounds:
            models = s.problem.runningModels + [s.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu, nf = xs[0].shape[0], rmodel.nq, us[0].shape[0], fs[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    F = [0.] * 12
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
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

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['1', '2', '3', '4', '5', '6']
    # left foot
    plt.subplot(2, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 13))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title('joint torque [Nm]')
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
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 12))]
    plt.ylabel('RF')
    plt.xlabel('knots')
    plt.legend()


    # Plot floating base coordinates
    baseTranslationNames = ['x', 'y', 'z']
    plt.figure(figIndex + 1)
    plt.suptitle('Floating Base Coordinates')
    [plt.plot(X[k], label=baseTranslationNames[i]) for i, k in enumerate(range(0, 3))]
    plt.xlabel('Knots')
    plt.ylabel('Translation [m]')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()
        

    # Get 3 dim CoM
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    Cz = []
    for x in xs:
        q = x[:rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
        Cz.append(np.asscalar(c[2]))
    knots = list(range(0,len(Cz)))

    # Plotting the Center of Mass (x,y,z over knots)
    plt.figure(figIndex + 2)
    plt.suptitle('CoM')
    plt.subplot(1, 3, 1)
    plt.plot(knots, Cx)
    plt.xlabel('knots')
    plt.ylabel('x [m]')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(knots, Cy)
    plt.xlabel('knots')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(knots, Cz)
    plt.xlabel('knots')
    plt.ylabel('z [m]')
    plt.grid(True)
    if show:
        plt.show()

    # Plotting the Center of Mass (y,z over x)
    plt.figure(figIndex + 3)
    plt.suptitle('CoM')
    plt.subplot(1, 2, 1)
    plt.plot(Cx, Cy)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(Cx, Cz)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.grid(True)
    if show:
        plt.show()


    # Plotting the contact wrenches
    contactForceNames = ['fx','fy','fz'] 
    contactMomentNames = ['taux','tauy','tauz']
    plt.figure(figIndex + 4)

    plt.suptitle(figTitle)
    plt.subplot(2,2,1)
    [plt.plot(F[k], label=contactForceNames[i]) for i, k in enumerate(range(0, 3))]
    plt.title('Contact Forces [RF]')
    plt.xlabel('Knots')
    plt.ylabel('Force [N]')
    plt.legend()

    plt.suptitle(figTitle)
    plt.subplot(2,2,2)
    [plt.plot(F[k], label=contactMomentNames[i]) for i, k in enumerate(range(3, 6))]
    plt.plot()
    plt.title('Contact Moments [RF]')
    plt.xlabel('Knots')
    plt.ylabel('Moment [Nm]')
    plt.legend()

    plt.suptitle(figTitle)
    plt.subplot(2,2,3)
    [plt.plot(F[k], label=contactForceNames[i]) for i, k in enumerate(range(6, 9))]
    plt.title('Contact Forces [LF]')
    plt.xlabel('Knots')
    plt.ylabel('Force [N]')
    plt.legend()

    plt.suptitle(figTitle)
    plt.subplot(2,2,4)
    [plt.plot(F[k], label=contactMomentNames[i]) for i, k in enumerate(range(9, nf))]
    plt.plot()
    plt.title('Contact Moments [LF]')
    plt.xlabel('Knots')
    plt.ylabel('Moment [Nm]')
    plt.legend()
    if show:
        plt.show()



def logSolution(xs, us, fs, rmodel, GAITPHASES, ddp, timeStep):
    nx, nq, nu, nv = xs[0].shape[0], rmodel.nq, us[0].shape[0], rmodel.nv
    # print('nq: ' + str(nq) + '; nv: ' + str(nv) + '; nx: ' + str(nx) + '; nu: ' + str(nu))

    # Collect time steps
    time = []
    for i in range(len(GAITPHASES)):
        log = ddp[i].getCallbacks()[0]
        log.xs = log.xs[:-1] # Don't consider very last element since doubled (cmp. plotSolution)
        for t in range(len(log.xs)):
            time.append(round(timeStep*(i*len(log.xs)+t),2))

    filename = 'log/6Steps/logJointStatesAndEffort.csv'
    firstWrite = True
    rangeRelJoints = list(range(7,nq)) + list(range(nq + 6, nq + 18)) # Ignore floating base (fixed joints)
    X = [0.] * nx
    U = [0.] * nu
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        XRel = []
        #Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
        for j in range(nx):
            X[j] = [np.asscalar(x[j]) for x in log.xs] 
        for k in rangeRelJoints:
            XRel.append(X[k])
        sol = list(map(list, zip(*XRel))) # Transpose
        for j in range(nu):
            U[j] = [np.asscalar(u[j]) for u in log.us] 
        solU = list(map(list, zip(*U))) # Transpose
        # Include time and effort columns
        for l in range(len(sol)):
            sol[l] = [time[i*len(log.xs)+l]] + sol[l] + solU[l] 
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t[s]', 
                                 'q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch',
                                 'q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch',
                                 'qd_LRHip1', 'qd_LRHip2', 'qd_LRHip3', 'qd_LRKnee', 'qd_LRAnkleRoll', 'qd_LRAnklePitch',
                                 'qd_LLHip1', 'qd_LLHip2', 'qd_LLHip3', 'qd_LLKnee', 'qd_LLAnkleRoll', 'qd_LLAnklePitch',
                                 'Tau_LRHip1', 'Tau_LRHip2', 'Tau_LRHip3', 'Tau_LRKnee', 'Tau_LRAnkleRoll', 'Tau_LRAnklePitch',
                                 'Tau_LLHip1', 'Tau_LLHip2', 'Tau_LLHip3', 'Tau_LLKnee', 'Tau_LLAnkleRoll', 'Tau_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)
    
    filename = 'log/6Steps/logBaseStates.csv'
    firstWrite = True
    rangeRelJoints = list(range(0,7)) + list(range(nq, nq + 6)) # Ignore floating base (fixed joints)
    X = [0.] * nx
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        XRel = []
        sol = []
        #Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
        for j in range(nx):
            X[j] = [np.asscalar(x[j]) for x in log.xs] 
        for k in rangeRelJoints:
            XRel.append(X[k])
        sol = list(map(list, zip(*XRel))) #transpose
        # Include time column
        for l in range(len(sol)):
            sol[l] = [time[i*len(log.xs)+l]] + sol[l] 
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t[s]',
                                 'X', 'Y', 'Z', 'Qx', 'Qy', 'Qz', 'Qw',
                                 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)

    filename = 'log/6Steps/logContactWrenches.csv'
    # Include time column
    sol = np.zeros([len(time), 13])
    for l in range(len(time)):
        sol[l] = [*[time[l]],*fs[l]] 
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t[s]',
                         'Fx_FR_SupportCenter', 'Fy_FR_SupportCenter', 'Fz_FR_SupportCenter', 'Tx_FR_SupportCenter', 'Ty_FR_SupportCenter', 'Tz_FR_SupportCenter',
                         'Fx_FL_SupportCenter', 'Fy_FL_SupportCenter', 'Fz_FL_SupportCenter', 'Tx_FL_SupportCenter', 'Ty_FL_SupportCenter', 'Tz_FL_SupportCenter'])
        writer.writerows(sol)