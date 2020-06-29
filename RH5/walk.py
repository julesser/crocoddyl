import os
import sys
import numpy as np
import itertools

import crocoddyl
import example_robot_data
import pinocchio
from utils.walkProblem import SimpleBipedGaitProblem
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
# stepKnots = 25
stepKnots = 50
supportKnots = 1
impulseKnots = 1
stepLength = 0.2
# stepHeight = 0.1
stepHeight = 0.05
rightFoot = 'FR_SupportCenter'
leftFoot = 'FL_SupportCenter'
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)     

# Defining the initial state of the robot
q0 = gait.q0
v0 = pinocchio.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# display = crocoddyl.GepettoDisplay(rh5_legs, 4, 4, frameNames=[rightFoot, leftFoot])
# display.display(xs=[x0])

# simName = 'results/' # Used when just testing
# simName = 'results/2Steps_10cmStride/'
# simName = 'results/2Steps_30cmStride/'
simName = 'results/LongGait/'
if not os.path.exists(simName):
    os.makedirs(simName)

# Perform 2 Steps
""" GAITPHASES = \
    [{'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}] """
# Perform 10 Steps
GAITPHASES = \
    [{'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
     {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
    {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
    {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
    {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
                  'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots'], value['isLastPhase']))
            # ddp[i].th_stop = 1e-8                                          

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
    # print(x0[:rmodel.nq]) # print last state of long gait as reference for two steps experiments


# Calc resulting CoM velocity (average) # TODO: Put in utils
logFirst = ddp[0].getCallbacks()[0]
logLast = ddp[-1].getCallbacks()[0]
first_com = pinocchio.centerOfMass(rmodel, rmodel.createData(), logFirst.xs[1][:rmodel.nq]) # calc CoM for init pose
final_com = pinocchio.centerOfMass(rmodel, rmodel.createData(), logLast.xs[-1][:rmodel.nq]) # calc CoM for final pose
# n_knots = 2*len(GAITPHASES)*(stepKnots + supportKnots + impulseKnots) 
n_knots = 2*len(GAITPHASES)*(stepKnots) # Don't consider: support knots -> pause; impulse knots -> dt=0
t_total = n_knots * timeStep # total time = f(knots, timeStep)
distance = final_com[0] - first_com[0]
v_com = distance / t_total
print('..................')
print('Simulation Results')
print('..................')
print('Step Time:    ' + str(stepKnots * timeStep) + ' s')
print('Step Length:  ' + str(distance / len(GAITPHASES)).strip('[]') + ' m')
print('CoM Velocity: ' + str(v_com).strip('[]') + ' m/s')

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
            if key == "7": # left foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+3] = wrench.angular[k]
            elif key == "13": # right foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k+6] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+9] = wrench.angular[k]
            # print('Foot: ' + str(key), wrench) # Check key-foot mapping
fs = fsRel

# Export solution to .csv files
if WITHLOG:
    logSolution(ddp, fs, timeStep, simName)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(rh5_legs, frameNames=[rightFoot, leftFoot])
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])

# Plotting the entire motion
if WITHPLOT:
    plotSolution(ddp, fs, simName, bounds=False, figIndex=1, show=False)

    # for i, phase in enumerate(GAITPHASES):
    #     # title = phase.keys()[0] + " (phase " + str(i) + ")"
    #     title = list(phase.keys())[0] + " (phase " + str(i) + ")" #Fix python3 dict bug (TypeError: 'dict_keys' object does not support indexing) 
    #     log = ddp[i].getCallbacks()[0]
    #     crocoddyl.plotConvergence(log.costs,
    #                               log.u_regs,
    #                               log.x_regs,
    #                               log.grads,
    #                               log.stops,
    #                               log.steps,
    #                               figTitle=title,
    #                               figIndex=i + 6,
    #                               show=True if i == len(GAITPHASES) - 1 else False)
