import os
import sys

import numpy as np
import csv
import itertools

import crocoddyl
import example_robot_data
import pinocchio
from notebooks.biped_utils_rh5 import SimpleBipedGaitProblem, plotSolution
from pinocchio.robot_wrapper import RobotWrapper

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHLOG = 'log' in sys.argv

crocoddyl.switchToNumpyMatrix()

# Loading the RH5 Model
modelPath = "/home/dfki.uni-bremen.de/jesser/Dev/rh5-models"
#URDF_FILENAME = "RH5_PkgPath.urdf"
URDF_FILENAME = "RH5Legs_PkgPath_PtContact.urdf"
URDF_SUBPATH = "/abstract-urdf/urdf/" + URDF_FILENAME

rh5_legs = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer()) # Load URDF file
# Add the free-flyer joint limits (floating base)
rmodel = rh5_legs.model
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
# lims[17] = 70
# print(lims)
rmodel.effortLimit = lims

# Setting up the 3d walking problem
rightFoot = 'FR_SupportCenter'
leftFoot = 'FL_SupportCenter'
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)     

# Defining the initial state of the robot
q0 = gait.q0
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# Setting up all tasks
# Repetitive gait
GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}]
""" GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}] """
# Changing, advanced gait
""" GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 1.0, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.20,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.30,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}] """
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    if WITHDISPLAY and WITHPLOT:
        display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
        ddp[i].setCallbacks(
            [crocoddyl.CallbackLogger(),
             crocoddyl.CallbackVerbose(),
             crocoddyl.CallbackDisplay(display)])
    elif WITHDISPLAY:
        display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    elif WITHPLOT or WITHLOG:
        ddp[i].setCallbacks([
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
        ])
    else:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [rmodel.defaultState] * len(ddp[i].models())
    us = [m.quasiStatic(d, rmodel.defaultState) for m, d in list(zip(ddp[i].models(), ddp[i].datas()))[:-1]]
    ddp[i].solve(xs, us, 100, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]


nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
rangeRelJoints = list(range(7,nq)) + list(range(nq + 6, nq + 18)) # Ignore floating base (fixed joints)
X = [0.] * nx
print(nx)
print(nq)
print(nu)


# Export solution to .csv file
if WITHLOG:
    # headerQLR = ['q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch']
    # headerQLL = ['q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch']
    # headerVLR = ['v_LRHip1', 'v_LRHip2', 'v_LRHip3', 'v_LRKnee', 'v_LRAnkleRoll', 'v_LRAnklePitch']
    # headerVLL = ['v_LLHip1', 'v_LLHip2', 'v_LLHip3', 'v_LLKnee', 'v_LLAnkleRoll', 'v_LLAnklePitch']
    # headerULR = ['u_LRHip1', 'u_LRHip2', 'u_LRHip3', 'u_LRKnee', 'u_LRAnkleRoll', 'u_LRAnklePitch']
    # headerULL = ['u_LLHip1', 'u_LLHip2', 'u_LLHip3', 'u_LLKnee', 'u_LLAnkleRoll', 'u_LLAnklePitch']
    filename = 'logSolution.csv'
    firstWrite = True
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        XRel = []
        sol = []
        print(len(log.xs))
        print(len(log.us))

        if firstWrite: # Write ('w') headers
            firstWrite = False
            #Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
            for j in range(nx):
                X[j] = [np.asscalar(x[j]) for x in log.xs[:]] # Don't consider last element (doubled)
            for k in rangeRelJoints:
                XRel.append(X[k])
            XRel = list(map(list, zip(*XRel))) #transpose

            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch',
                                 'q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch',
                                'v_LRHip1', 'v_LRHip2', 'v_LRHip3', 'v_LRKnee', 'v_LRAnkleRoll', 'v_LRAnklePitch',
                                'v_LLHip1', 'v_LLHip2', 'v_LLHip3', 'v_LLKnee', 'v_LLAnkleRoll', 'v_LLAnklePitch',
                                'u_LRHip1', 'u_LRHip2', 'u_LRHip3', 'u_LRKnee', 'u_LRAnkleRoll', 'u_LRAnklePitch',
                                'u_LLHip1', 'u_LLHip2', 'u_LLHip3', 'u_LLKnee', 'u_LLAnkleRoll', 'u_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            #Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
            for j in range(nx):
                X[j] = [np.asscalar(x[j]) for x in log.xs[:]] # Don't consider first AND last element (doubled)
            for k in rangeRelJoints:
                sol.append(X[k])
            sol = list(map(list, zip(*sol))) #transpose
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)

    """ filename = 'logUs.csv'
    firstWrite = True
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        sol = log.us
        # filename = "uVals_Phase" + str(i) + ".csv"
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['u_LRHip1', 'u_LRHip2', 'u_LRHip3', 'u_LRKnee', 'u_LRAnkleRoll', 'u_LRAnklePitch',
                                 'u_LLHip1', 'u_LLHip2', 'u_LLHip3', 'u_LLKnee', 'u_LLAnkleRoll', 'u_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol) """

    filename = 'logFs.csv'
    firstWrite = True
    for i, phase in enumerate(GAITPHASES):
        # display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
        # fs = display.getForceTrajectoryFromSolver(ddp[i])
        log = ddp[i].getCallbacks()[0]
        sol = log.fs
        # filename = "uVals_Phase" + str(i) + ".csv"
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # writer.writerow(['u_LRHip1', 'u_LRHip2', 'u_LRHip3', 'u_LRKnee', 'u_LRAnkleRoll', 'u_LRAnklePitch',
                #                  'u_LLHip1', 'u_LLHip2', 'u_LLHip3', 'u_LLKnee', 'u_LLAnkleRoll', 'u_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                #writer.writerows(sol)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(rh5_legs, frameNames=[rightFoot, leftFoot])
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])

# Plotting the entire motion
if WITHPLOT:
    plotSolution(ddp, bounds=False, figIndex=1, show=False)

    for i, phase in enumerate(GAITPHASES):
        # title = phase.keys()[0] + " (phase " + str(i) + ")"
        title = list(phase.keys())[0] + " (phase " + str(i) + ")" #Fix python3 dict bug (TypeError: 'dict_keys' object does not support indexing) 
        log = ddp[i].getCallbacks()[0]
        crocoddyl.plotConvergence(log.costs,
                                  log.u_regs,
                                  log.x_regs,
                                  log.grads,
                                  log.stops,
                                  log.steps,
                                  figTitle=title,
                                  figIndex=i + 4,
                                  show=True if i == len(GAITPHASES) - 1 else False)
        


        # print(log.fs) SUCCESSFULL

        # Getting the contact wrenches
        # fs = log.fs
        # for f in fs[i]:
            
        # nf = fs[0].shape[0] # = 36
        # F = [0.] * nf
        # for i in range(nf):
        #     F[i] = [np.asscalar(f[i]) for f in fs]


        # Plotting the contact forces
        # forceDimName = ['x','y','z'] 
        # plt.figure(figIndex + 2)

        # plt.suptitle(figTitle)
        # plt.subplot(2,1,1)
        # [plt.plot(F[k], label=forceDimName[i]) for i, k in enumerate(range(0, len(forceDimName)))]
        # plt.title('Contact Forces [LF]')
        # plt.xlabel('Knots')
        # plt.ylabel('Force [Nm]')
        # plt.legend()

        # plt.suptitle(figTitle)
        # plt.subplot(2,1,2)
        # plt.plot()
        # plt.title('Contact Forces [RF]')
        # plt.xlabel('Knots')
        # plt.ylabel('Force [Nm]')
        # plt.legend()
        # if show:
        #     plt.show()
