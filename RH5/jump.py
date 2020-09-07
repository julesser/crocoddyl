import os
import sys
import numpy as np
import itertools
import math 

import crocoddyl
import example_robot_data
import pinocchio
from utils.jumpProblem import HumanoidJumpProblem
from utils.utils import setLimits, calcAverageCoMVelocity, plotSolution, logSolution, addObstacleToViewer
from pinocchio.robot_wrapper import RobotWrapper

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHLOG = 'log' in sys.argv

# Loading the RH5 Model
modelPath = os.path.join(os.environ.get('HOME'), "Dev/rh5-models")
URDF_FILENAME = "RH5Humanoid_PkgPath.urdf"
URDF_SUBPATH = "/abstract-urdf/urdf/" + URDF_FILENAME

# Load the full model 
rh5_robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer()) # Load URDF file
# Create a list of joints to lock
# jointsToLock = ['ALShoulder1', 'ALShoulder2', 'ALShoulder3', 'ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
#                 'ARShoulder1', 'ARShoulder2', 'ARShoulder3', 'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
#                 'HeadPitch', 'HeadRoll', 'HeadYaw']
jointsToLock = ['ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
                'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
                'HeadPitch', 'HeadRoll', 'HeadYaw']
# Get the existing joint IDs
jointsToLockIDs = []
for jn in range(len(jointsToLock)):
    if rh5_robot.model.existJointName(jointsToLock[jn]):
        jointsToLockIDs.append(rh5_robot.model.getJointId(jointsToLock[jn]))
    else:
        print('Warning: joint ' + str(jointsToLock[jn]) + ' does not belong to the model!')
# Init CoM perfectly on feet center line
fixedJointConfig = np.array([0,0,0,0,0,0,0, # Floating Base
                        0,0,0,                 # Torso
                        -0.25,0.1,0,0,0,0,0, # Left Arm
                        0.25,-0.1,0,0,0,0,0,  # Right Arm
                        0,0,0,                 # Head
                        0,0,0,0,0,0,           # Left Leg     
                        0,0,0,0,0,0])        # Right Leg)
# Build the reduced model
rh5_robot.model, rh5_robot.visual_model = pinocchio.buildReducedModel(rh5_robot.model, rh5_robot.visual_model, jointsToLockIDs, fixedJointConfig)
rmodel = rh5_robot.model
setLimits(rmodel)

# Basics of jumping physics:
# 1. Calc falling time based on height: s=1/2*g*t^2 <-> t=sqrt(2h/g)
# >> 0.15s for 0.1m jump height
# 2. jumpingUpTime == fallingDownTime always: t_peak=0.5*t_total (velocity is a linear function v=a*t)
# >> 2*0.15s=0.3s
# 3. groundKnots = flyingKnots = 2*recoveryKnots
# Setting up the jumping problem
timeStep = 0.01
jumpHeight = 0.1
jumpLength = [0.6, 0, 0]
# jumpLength = [0.3, 0, 0]
jumpLength = [0, 0, 0]
groundKnots = 30
flyingKnots = round(2*math.sqrt(2*jumpHeight/9.81)/timeStep)
print(flyingKnots)
recoveryKnots = 30
impulseKnots = 1
knots = [groundKnots, flyingKnots]
rightFoot = 'FR_SupportCenter'
leftFoot = 'FL_SupportCenter'
gait = HumanoidJumpProblem(rmodel, rightFoot, leftFoot)

# Defining the initial state of the robot
x0 = gait.rmodel.defaultState

# Set camera perspective
cameraTF = [4., 5., 1.5, 0.2, 0.62, 0.72, 0.22] # isometric
# cameraTF = [6.4, 0, 2, 0.44, 0.44, 0.55, 0.55]  # front
# cameraTF = [0., 5.5, 1.2, 0., 0.67, 0.73, 0.] # side

display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
name = 'world/box'
# obsDim = [.2, 1, .1]
obsDim = [.25, 1, .2]
pos = [[jumpLength[0]+0.12, 0, obsDim[2]/2]]
# pos = [[0.4, 0, obsDim[2]/2], [1, 0, obsDim[2]/2], [1.6, 0, obsDim[2]/2]]
for i in range(len(pos)):
    addObstacleToViewer(display, name+str(i), obsDim, pos[i])
display.display(xs=[x0])

simName = 'results/Jump_Test/'
if not os.path.exists(simName):
    os.makedirs(simName)

# Perform one jump
GAITPHASES = \
    [{'jumping': {'jumpHeight': jumpHeight, 'jumpLength': jumpLength,
                  'timeStep': timeStep, 'groundKnots': groundKnots, 'flyingKnots': flyingKnots, 'recoveryKnots': recoveryKnots}}]
# GAITPHASES = \
#     [{'boxJumping': {'jumpHeight': jumpHeight, 'jumpLength': jumpLength, 'obstacleHeight': obsDim[2],
#                   'timeStep': timeStep, 'groundKnots': groundKnots, 'flyingKnots': flyingKnots, 'recoveryKnots': recoveryKnots}}]
    
# Perform multiple jumps
# GAITPHASES = \
#     [{'jumping': {'jumpHeight': jumpHeight, 'jumpLength': jumpLength,
#                   'timeStep': timeStep, 'groundKnots': groundKnots, 'flyingKnots': flyingKnots, 'recoveryKnots': recoveryKnots}},
#      {'jumping': {'jumpHeight': jumpHeight, 'jumpLength': jumpLength,
#                   'timeStep': timeStep, 'groundKnots': groundKnots, 'flyingKnots': flyingKnots, 'recoveryKnots': recoveryKnots}},
#      {'jumping': {'jumpHeight': jumpHeight, 'jumpLength': jumpLength,
#                   'timeStep': timeStep, 'groundKnots': groundKnots, 'flyingKnots': flyingKnots, 'recoveryKnots': recoveryKnots}}]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'jumping':
            # Creating a simple jumping problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                                  value['groundKnots'], value['flyingKnots'], value['recoveryKnots']))
        if key == 'boxJumping':
            # Creating a simple jumping problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createBoxJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['obstacleHeight'], 
                                            value['timeStep'], value['groundKnots'], value['flyingKnots'], value['recoveryKnots']))
        ddp[i].th_stop = 1e-7

    # Add the callback functions
    print('*** SOLVE ' + key + ' ***')
    ddp[i].setCallbacks(
        [crocoddyl.CallbackLogger(),
         crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [rmodel.defaultState] * (ddp[i].problem.T + 1)
    us = [
        m.quasiStatic(d, rmodel.defaultState)
        for m, d in list(zip(ddp[i].problem.runningModels, ddp[i].problem.runningDatas))
    ]
    print(ddp[i].solve(xs, us, 500, False, 0.1))

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    print('Displaying the motion in Gepetto..')
    display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])

# Export solution to .csv files
if WITHLOG:
    logPath = simName + '/logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logSolution(ddp, timeStep,logPath)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(ddp, simName, knots, bounds=True, figIndex=1, show=False)