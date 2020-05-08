import os
import sys
import numpy as np
import itertools

import crocoddyl
import example_robot_data
import pinocchio
from utils.jumpProblem import SimpleBipedGaitProblem
from utils.utils import setLimits, plotSolution, logSolution 
from pinocchio.robot_wrapper import RobotWrapper

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHLOG = 'log' in sys.argv

crocoddyl.switchToNumpyMatrix()

# Loading the RH5 Model
modelPath = os.path.join(os.environ.get('HOME'), "Dev/rh5-models")
URDF_FILENAME = "RH5Legs_PkgPath_PtContact.urdf"
URDF_SUBPATH = "/abstract-urdf/urdf/" + URDF_FILENAME

rh5_legs = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer()) # Load URDF file
rmodel = rh5_legs.model
setLimits(rmodel)

# Setting up the 3d walking problem
timeStep = 0.03
rightFoot = 'FR_SupportCenter'
leftFoot = 'FL_SupportCenter'
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)     

# Defining the initial state of the robot
q0 = gait.q0
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# Setting up all tasks
GAITPHASES = \
    [{'jumping': {'jumpHeight': 0.15, 'jumpLength': [0, 0.3, 0], 
                  'timeStep': timeStep, 'groundKnots': 5, 'flyingKnots': 10}}] # jumpLength is direction vector
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'jumping':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                          value['groundKnots'], value['flyingKnots']))
            ddp[i].th_stop = 1e-6                            

    # Add the callback functions
    print('*** SOLVE ' + key + ' ***')
    display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
    ddp[i].setCallbacks(
        [crocoddyl.CallbackLogger(),
         crocoddyl.CallbackVerbose(),
         crocoddyl.CallbackDisplay(display)])

    # Solving the problem with the DDP solver
    xs = [rmodel.defaultState] * (ddp[i].problem.T + 1)
    us = [
        m.quasiStatic(d, rmodel.defaultState)
        for m, d in list(zip(ddp[i].problem.runningModels, ddp[i].problem.runningDatas))
    ]
    ddp[i].solve(xs, us, 100, False, 0.1)
    
    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]


# Get contact wrenches f=[f,tau]
display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
fsRel = np.zeros((len(GAITPHASES)*(len(ddp[i].problem.runningModels)),12)) # e.g. for 3 gaitphases = [3*nKnots,12]
for i, phase in enumerate(GAITPHASES):
    fs = display.getForceTrajectoryFromSolver(ddp[i])
    fs = fs[:-1] # Last element doubled
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
    logSolution(xs, us, fs, rmodel, GAITPHASES, ddp, timeStep)

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