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
# modelPath = "/home/dfki.uni-bremen.de/jesser/Dev/rh5-models"
modelPath = "/home/julian/Dev/rh5-models"
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
""" GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}] """
GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}]
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
    # xs = [rmodel.defaultState] * len(ddp[i].models())
    # us = [m.quasiStatic(d, rmodel.defaultState) for m, d in list(zip(ddp[i].models(), ddp[i].datas()))[:-1]]
    xs = [rmodel.defaultState] * (ddp[i].problem.T + 1)
    us = [
        m.quasiStatic(d, rmodel.defaultState)
        for m, d in list(zip(ddp[i].problem.runningModels, ddp[i].problem.runningDatas))
    ]
    ddp[i].solve(xs, us, 100, False, 0.1)
    
    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# BUG: Error caused by upgrade to v1.2.0 within 'display.getForceTrajectoryFromSolver'. 
# This Bug was fixed with v1.2.1 (not released yet); see https://github.com/loco-3d/crocoddyl/commit/d8fa7b4230f61e120d5cedd52e00de7a55c3454e
# Quick Fix: Manually modified function in /opt...;
# TODO: When robotpkg to v1.2.1 is released: Update to this version!
# Get contact wrenches f=[f,tau]
display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
# fsRel = np.zeros((len(GAITPHASES)*(len(ddp[i].problem.runningModels)-1),12)) # TODO: Erase -1?! e.g. for 3 gaitphases = [3*nKnots,12]
fsRel = np.zeros((len(GAITPHASES)*(len(ddp[i].problem.runningModels)),12)) # e.g. for 3 gaitphases = [3*nKnots,12]
for i, phase in enumerate(GAITPHASES):
    fs = display.getForceTrajectoryFromSolver(ddp[i])
    fs = fs[:-1] # Last element doubled
    #fsRel = np.zeros((len(fs),12))
    for j, x in enumerate(fs): # iter over all knots
        for f in fs[j]: # iter over all contacts (LF, RF)
            key = f["key"]
            wrench = f["f"]
            if key == "7": # right foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+3] = wrench.angular[k]
            elif key == "13": # left foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k+6] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+9] = wrench.angular[k]
            # print('Foot: ' + str(key), wrench) 
fs = fsRel

# Export solution to .csv file
if WITHLOG:
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    nv = rmodel.nv
    print('nq: ' + str(nq) + '; nv: ' + str(nv))
    filename = 'logSolutions/RH5Legs/logJointStates.csv'
    firstWrite = True
    rangeRelJoints = list(range(7,nq)) + list(range(nq + 6, nq + 18)) # Ignore floating base (fixed joints)
    X = [0.] * nx
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        log.xs = log.xs[:-1] # Don't consider last element (cmp. plotSolution; propably doubled)
        XRel = []
        sol = []
        #Get relevant joints states (x_LF, x_RF, v_LF, v_RF)
        for j in range(nx):
            X[j] = [np.asscalar(x[j]) for x in log.xs] 
        for k in rangeRelJoints:
            XRel.append(X[k])
        sol = list(map(list, zip(*XRel))) #transpose
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['q_LRHip1', 'q_LRHip2', 'q_LRHip3', 'q_LRKnee', 'q_LRAnkleRoll', 'q_LRAnklePitch',
                                 'q_LLHip1', 'q_LLHip2', 'q_LLHip3', 'q_LLKnee', 'q_LLAnkleRoll', 'q_LLAnklePitch',
                                 'qd_LRHip1', 'qd_LRHip2', 'qd_LRHip3', 'qd_LRKnee', 'qd_LRAnkleRoll', 'qd_LRAnklePitch',
                                 'qd_LLHip1', 'qd_LLHip2', 'qd_LLHip3', 'qd_LLKnee', 'qd_LLAnkleRoll', 'qd_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)

    filename = 'logSolutions/RH5Legs/logBaseStates.csv'
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
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y', 'Z', 'Qx', 'Qy', 'Qz', 'Qw',
                                 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)

    filename = 'logSolutions/RH5Legs/logEffort.csv'
    firstWrite = True
    U = [0.] * nu
    for i, phase in enumerate(GAITPHASES):
        log = ddp[i].getCallbacks()[0]
        sol = []
        for j in range(nu):
            U[j] = [np.asscalar(u[j]) for u in log.us] 
        sol = list(map(list, zip(*U))) #transpose
        if firstWrite: # Write ('w') headers
            firstWrite = False
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Tau_LRHip1', 'Tau_LRHip2', 'Tau_LRHip3', 'Tau_LRKnee', 'Tau_LRAnkleRoll', 'Tau_LRAnklePitch',
                                 'Tau_LLHip1', 'Tau_LLHip2', 'Tau_LLHip3', 'Tau_LLKnee', 'Tau_LLAnkleRoll', 'Tau_LLAnklePitch']) 
                writer.writerows(sol)
        else: # Append ('a') log of other phases (prevent overwriting)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(sol)

    filename = 'logSolutions/RH5Legs/logContactWrenches.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fx_FR_SupportCenter', 'Fy_FR_SupportCenter', 'Fz_FR_SupportCenter', 'Tx_FR_SupportCenter', 'Ty_FR_SupportCenter', 'Tz_FR_SupportCenter',
                         'Fx_FL_SupportCenter', 'Fy_FL_SupportCenter', 'Fz_FL_SupportCenter', 'Tx_FL_SupportCenter', 'Ty_FL_SupportCenter', 'Tz_FL_SupportCenter'])
        writer.writerows(fs)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(rh5_legs, frameNames=[rightFoot, leftFoot])
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])

# Plotting the entire motion
if WITHPLOT:
    plotSolution(ddp, fs, bounds=False, figIndex=1, show=False)

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
                                  figIndex=i + 6,
                                  show=True if i == len(GAITPHASES) - 1 else False)

def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]