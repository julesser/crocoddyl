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
# URDF_FILENAME = "RH5Torso_PkgPath.urdf"
# URDF_FILENAME = "RH5Humanoid_PkgPath_FixedArmsNHead.urdf"
URDF_FILENAME = "RH5Humanoid_PkgPath.urdf"
# URDF_SUBPATH = "/abstract-urdf/urdf/" + URDF_FILENAME
URDF_SUBPATH = "/abstract-urdf-deprecated/urdf/" + URDF_FILENAME # TODO: Take new one!!!! The new gives problem with the reduced model, fixedArmsNHead works fine

# Load the full model 
rh5_robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer()) # Load URDF file
print('standard model: dim=' + str(len(rh5_robot.model.joints)))
# print('standard model: names:')
# for jn in rh5_robot.model.names:
#     print(jn)
# Create a list of joints to lock
jointsToLock = ['ALShoulder1', 'ALShoulder2', 'ALShoulder3', 'ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
                'ARShoulder1', 'ARShoulder2', 'ARShoulder3', 'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
                'HeadPitch', 'HeadRoll', 'HeadYaw']
# jointsToLock = ['ALShoulder3', 'ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
#                 'ARShoulder3', 'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
#                 'HeadPitch', 'HeadRoll', 'HeadYaw']
# Get the existing joint IDs
jointsToLockIDs = []
for jn in range(len(jointsToLock)):
    if rh5_robot.model.existJointName(jointsToLock[jn]):
        jointsToLockIDs.append(rh5_robot.model.getJointId(jointsToLock[jn]))
    else:
        print('Warning: joint ' + str(jointsToLock[jn]) + ' does not belong to the model!')
# Set initial position of fixed joints
# Option 1: Like in smurf file
# initialJointConfig = np.matrix([0,0,0,0,0,0,0, # Floating Base
#                         0,0,0,                 # Torso
#                         -0.5,0.5,0,-0.3,0,0,0, # Left Arm
#                         0.5,-0.5,0,0.3,0,0,0,  # Right Arm
#                         0,0,0,                 # Head
#                         0,0,0,0,0,0,           # Left Leg     
#                         0,0,0,0,0,0]).T        # Right Leg)
# Option 2: Empirically from Shiveshs real robot observation
# initialJointConfig = np.matrix([0,0,0,0,0,0,0, # Floating Base
#                         0,0,0,                 # Torso
#                         -0.5,0.1,0,0,0,0,0, # Left Arm
#                         0.5,-0.1,0,0,0,0,0,  # Right Arm
#                         0,0,0,                 # Head
#                         0,0,0,0,0,0,           # Left Leg     
#                         0,0,0,0,0,0]).T        # Right Leg)
# Option 3: Theoretically CoM perfect at feet center line
initialJointConfig = np.matrix([0,0,0,0,0,0,0, # Floating Base
                        0,0,0,                 # Torso
                        -0.25,0.1,0,0,0,0,0, # Left Arm
                        0.25,-0.1,0,0,0,0,0,  # Right Arm
                        0,0,0,                 # Head
                        0,0,0,0,0,0,           # Left Leg     
                        0,0,0,0,0,0]).T        # Right Leg)
# Build the reduced model
# rh5_robot.model = pinocchio.buildReducedModel(rh5_robot.model, jointsToLockIDs, initialJointConfig) # If no displaying needed
rh5_robot.model, rh5_robot.visual_model = pinocchio.buildReducedModel(rh5_robot.model, rh5_robot.visual_model, jointsToLockIDs, initialJointConfig)
rmodel = rh5_robot.model
print('reduced model: dim=' + str(len(rh5_robot.model.joints)))
# print('standard model: names:')
# for jn in rmodel.names:
#     print(jn)
# for jn in rmodel.joints:
#     print(jn)
# Add joint limits
setLimits(rmodel)

# Setting up the 3d walking problem
timeStep = 0.03
# timeStep = 0.01 #TaskSpecific:Jumping
# stepKnots = 45
# supportKnots = 15
stepKnots = 90 #TaskSpecific:StaticWalking
supportKnots = 90
impulseKnots = 1
stepLength = 0.2
knots = [stepKnots, supportKnots, impulseKnots]
stepHeight = 0.05
rightFoot = 'FR_SupportCenter'
leftFoot = 'FL_SupportCenter'
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)     

# Defining the initial state of the robot
x0 = gait.rmodel.defaultState

# Set camera perspective
# cameraTF = [4., 5., 1.5, 0.2, 0.62, 0.72, 0.22] # isometric
cameraTF = [6.4, 0, 2, 0.44, 0.44, 0.55, 0.55] # front 
# cameraTF = [0., 5.5, 1.2, 0., 0.67, 0.73, 0.] # side 
display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
# display.display(xs=[x0])
# while True: # Get desired view params
#     print(rh5_robot.viewer.gui.getCameraTransform(rh5_robot.viz.windowID))

# simName = 'results/Test/' # Used when just testing
# simName = 'results/2Steps_10cmStride/'
# simName = 'results/HumanoidFixedArms/Jump_FootForward_50cm_CoP100/'
simName = 'results/HumanoidFixedArms/Test/'
if not os.path.exists(simName):
    os.makedirs(simName)

# Perform 2 Steps
# GAITPHASES = \
#     [{'walking': {'stepLength': stepLength, 'stepHeight': stepHeight, 'timeStep': timeStep, 
#                   'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
GAITPHASES = \
    [{'staticWalking': {'stepLength': stepLength, 'stepHeight': stepHeight, 'timeStep': timeStep, 
                        'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
# Perform 6 Steps
# GAITPHASES = \
#     [{'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
#                   'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
#      {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
#                   'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': False}},
#     {'walking': {'stepLength': stepLength, 'stepHeight': stepHeight,
#                   'timeStep': timeStep, 'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
# GAITPHASES = \
#     [{'squat': {'heightChange': 0.1, 'numKnots': 100, 'timeStep': timeStep}}]
# GAITPHASES = \
#     [{'squat': {'heightChange': 0.1, 'numKnots': 100, 'timeStep': timeStep}},
#      {'squat': {'heightChange': 0.1, 'numKnots': 100, 'timeStep': timeStep}},
#      {'squat': {'heightChange': 0.1, 'numKnots': 100, 'timeStep': timeStep}}]
# GAITPHASES = \
#     [{'balancing': {'supportKnots': 10, 'shiftKnots': 120, 'balanceKnots': 240, 'timeStep': timeStep}}]
# GAITPHASES = \
#     [{'jumping': {'jumpHeight': 0.15, 'jumpLength': [0.5, 0, 0], 
#                   'timeStep': timeStep, 'groundKnots': 50, 'flyingKnots': 15}}] # jumpLength is direction vector
    
ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots'], value['isLastPhase']))
        if key == 'staticWalking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createStaticWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                                value['stepKnots'], value['supportKnots'], value['isLastPhase']))
        if key == 'squat':
            # Creating a squat problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createSquatProblem(x0, value['heightChange'], value['numKnots'], value['timeStep']))
        if key == 'balancing':
            # Creating a balancing problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createBalancingProblem(x0, value['supportKnots'], value['shiftKnots'], value['balanceKnots'], 
                                            value['timeStep']))
        if key == 'jumping':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverBoxFDDP(
                gait.createFootTrajJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                          value['groundKnots'], value['flyingKnots']))
        ddp[i].th_stop = 1e-7                                      

    # Add the callback functions
    print('*** SOLVE ' + key + ' ***')
    display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
    ddp[i].setCallbacks(
        [crocoddyl.CallbackLogger(),
         crocoddyl.CallbackVerbose()])
        #  crocoddyl.CallbackDisplay(display)])

    # Solving the problem with the DDP solver
    xs = [rmodel.defaultState] * (ddp[i].problem.T + 1)
    us = [
        m.quasiStatic(d, rmodel.defaultState)
        for m, d in list(zip(ddp[i].problem.runningModels, ddp[i].problem.runningDatas))
    ]
    print(ddp[i].solve(xs, us, 500, False, 0.1))
    
    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]
    # print(x0[:rmodel.nq]) # print last state of long gait as reference for two steps experiments


# Calc resulting CoM velocity (average) # TODO: Put in utils
logFirst = ddp[0].getCallbacks()[0]
logLast = ddp[-1].getCallbacks()[0]
first_com = pinocchio.centerOfMass(rmodel, rmodel.createData(), logFirst.xs[1][:rmodel.nq]) # calc CoM for init pose
final_com = pinocchio.centerOfMass(rmodel, rmodel.createData(), logLast.xs[-1][:rmodel.nq]) # calc CoM for final pose
n_knots = 2*len(GAITPHASES)*(stepKnots+supportKnots) # Don't consider impulse knots (dt=0)
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
display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
fsRel = np.zeros((len(GAITPHASES)*(len(ddp[i].problem.runningModels)),12)) # e.g. for 3 gaitphases = [3*nKnots,12]
for i, phase in enumerate(GAITPHASES):
    fs = display.getForceTrajectoryFromSolver(ddp[i])
    fs = fs[:-1] # Last element doubled
    for j, x in enumerate(fs): # iter over all knots
        for f in fs[j]: # iter over all contacts (LF, RF)
            key = f["key"]
            wrench = f["f"]
            if key == "10": # left foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+3] = wrench.angular[k]
            elif key == "16": # right foot
                for k in range(3):
                    fsRel[i*len(fs)+j,k+6] = wrench.linear[k]
                    fsRel[i*len(fs)+j,k+9] = wrench.angular[k]
            # print('Foot: ' + str(key), wrench) # Check key-foot mapping
fs = fsRel

# Export solution to .csv files
if WITHLOG:
    logPath = simName+'/logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logSolution(ddp, fs, timeStep,logPath)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(rh5_robot, frameNames=[rightFoot, leftFoot])
    # rh5_robot.viewer.gui.startCapture(rh5_robot.viz.windowID, 'test', '.mp4') # TODO: Automate video recording (check params, nothing happens now)
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])
    # rh5_robot.viewer.gui.stopCapture(rh5_robot.viz.windowID)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(ddp, fs, simName, knots, bounds=False, figIndex=1, show=False)

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