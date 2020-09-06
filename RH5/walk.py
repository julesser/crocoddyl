import os
import sys
import numpy as np
import itertools

import crocoddyl
import example_robot_data
import pinocchio
from utils.walkProblem import SimpleBipedGaitProblem
from utils.utils import setLimits, calcAverageCoMVelocity, plotSolution, logSolution 
from pinocchio.robot_wrapper import RobotWrapper

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHLOG = 'log' in sys.argv

# Loading the RH5 Model
modelPath = os.path.join(os.environ.get('HOME'), "Dev/rh5-models")
# URDF_FILENAME = "RH5Torso_PkgPath.urdf"
# URDF_FILENAME = "RH5Humanoid_PkgPath_FixedArmsNHead.urdf"
URDF_FILENAME = "RH5Humanoid_PkgPath.urdf"
URDF_SUBPATH = "/abstract-urdf/urdf/" + URDF_FILENAME

# Load the full model 
rh5_robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer()) # Load URDF file
# print('standard model: dim=' + str(len(rh5_robot.model.joints)))
# print('standard model: names:')
# for jn in rh5_robot.model.names:
#     print(jn)
# Create a list of joints to lock
# jointsToLock = ['ALShoulder1', 'ALShoulder2', 'ALShoulder3', 'ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
#                 'ARShoulder1', 'ARShoulder2', 'ARShoulder3', 'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
#                 'HeadPitch', 'HeadRoll', 'HeadYaw']
# jointsToLock = ['ALShoulder3', 'ALElbow', 'ALWristRoll', 'ALWristYaw', 'ALWristPitch',
#                 'ARShoulder3', 'ARElbow', 'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
#                 'HeadPitch', 'HeadRoll', 'HeadYaw']
jointsToLock = ['ALWristRoll', 'ALWristYaw', 'ALWristPitch',
                'ARWristRoll', 'ARWristYaw', 'ARWristPitch',
                'HeadPitch', 'HeadRoll', 'HeadYaw']
# Get the existing joint IDs
jointsToLockIDs = []
for jn in range(len(jointsToLock)):
    if rh5_robot.model.existJointName(jointsToLock[jn]):
        jointsToLockIDs.append(rh5_robot.model.getJointId(jointsToLock[jn]))
    else:
        print('Warning: joint ' + str(jointsToLock[jn]) + ' does not belong to the model!')
# Init CoM perfectly on feet center line
fixedJointConfig = np.array([0,0,0,0,0,0,0,    # Floating Base
                        0,0,0,                 # Torso
                        -0.25,0.1,0,0,0,0,0,   # Left Arm
                        0.25,-0.1,0,0,0,0,0,   # Right Arm
                        0,0,0,                 # Head
                        0,0,0,0,0,0,           # Left Leg     
                        0,0,0,0,0,0])          # Right Leg)
# Build the reduced model
# rh5_robot.model = pinocchio.buildReducedModel(rh5_robot.model, jointsToLockIDs, fixedJointConfig) # If no displaying needed
rh5_robot.model, rh5_robot.visual_model = pinocchio.buildReducedModel(rh5_robot.model, rh5_robot.visual_model, jointsToLockIDs, fixedJointConfig)
rmodel = rh5_robot.model
# print('reduced model: dim=' + str(len(rh5_robot.model.joints)))
# print('standard model: names:')
# for jn in rmodel.names:
#     print(jn)
# for jn in rmodel.joints:
#     print(jn)
# for jn in rmodel.frames:
#     print(jn)
# Add joint limits
setLimits(rmodel)

gridSearchResults = []
baumgarteDGains = [35]
baumgartePGains = [0]
# baumgarteDGains = np.arange(20, 110, 10)
# baumgartePGains = np.arange(0, 1.2, 0.1)


for DGain in baumgarteDGains:
    for PGain in baumgartePGains:
        # Setting up the 3d walking problem
        timeStep = 0.03
        # timeStep = 0.01
        stepKnots = 45
        supportKnots = 15
        # stepKnots = 90  # TaskSpecific:StaticWalking
        # supportKnots = 90
        # stepKnots = 300  # TaskSpecific:StaticWalking_DT=0.01
        # supportKnots = 300
        impulseKnots = 1
        # stepLength = 0.2
        stepLength = 0.8 #TaskSpecific: DynamicWalking Large steps
        knots = [stepKnots, supportKnots]
        stepHeight = 0.05
        rightFoot = 'FR_SupportCenter'
        leftFoot = 'FL_SupportCenter'
        gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot, DGain, PGain)

        # Defining the initial state of the robot
        x0 = gait.rmodel.defaultState

        # Set camera perspective
        cameraTF = [4., 5., 1.5, 0.2, 0.62, 0.72, 0.22] # isometric
        # cameraTF = [6.4, 0, 2, 0.44, 0.44, 0.55, 0.55]  # front
        # cameraTF = [0., 5.5, 1.2, 0., 0.67, 0.73, 0.] # side
        # display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
        # display.display(xs=[x0])
        # while True: # Get desired view params
        #     print(rh5_robot.viewer.gui.getCameraTransform(rh5_robot.viz.windowID))

        # simName = 'results/HumanoidFixedArms/DynamicWalking_LargeSteps_CoP50_ArmsFreed/'
        simName = 'results/DynamicWalking_LargeSteps_CoP100_ArmsFreed/'
        # simName = 'results/HumanoidFixedArms/Analysis/GridSearchBaumgarteGains/DGain' + str(DGain) + '_PGain' + str(round(PGain,1)) + '/'
        if not os.path.exists(simName):
            os.makedirs(simName)

        # Perform 2 Steps
        GAITPHASES = \
            [{'walking': {'stepLength': stepLength, 'stepHeight': stepHeight, 'timeStep': timeStep,
                        'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
        # GAITPHASES = \
        #     [{'staticWalking': {'stepLength': stepLength, 'stepHeight': stepHeight, 'timeStep': timeStep,
        #                         'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
        # GAITPHASES = \
        #     [{'OneStepstaticWalking': {'stepLength': stepLength, 'stepHeight': stepHeight, 'timeStep': timeStep,
        #                         'stepKnots': stepKnots, 'supportKnots': supportKnots, 'isLastPhase': True}}]
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
        #     [{'squat': {'heightChange': 0.15, 'numKnots': 70, 'timeStep': timeStep}},
        #      {'squat': {'heightChange': 0.15, 'numKnots': 70, 'timeStep': timeStep}},
        #      {'squat': {'heightChange': 0.15, 'numKnots': 70, 'timeStep': timeStep}}]
        # GAITPHASES = \
        #     [{'balancing': {'supportKnots': 10, 'shiftKnots': 240, 'balanceKnots': 480, 'timeStep': timeStep}}]
                
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
                if key == 'OneStepstaticWalking':
                    # Creating a walking problem
                    ddp[i] = crocoddyl.SolverBoxFDDP(
                        gait.createOneStepStaticWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
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
                ddp[i].th_stop = 1e-7

            # Add the callback functions
            print('*** SOLVE ' + key + ' ***')
            # display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
            ddp[i].setCallbacks(
                # [crocoddyl.CallbackLogger()])
                [crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
            #  crocoddyl.CallbackDisplay(display)])

            # Solving the problem with the DDP solver
            xs = [rmodel.defaultState] * (ddp[i].problem.T + 1)
            us = [
                m.quasiStatic(d, rmodel.defaultState)
                for m, d in list(zip(ddp[i].problem.runningModels, ddp[i].problem.runningDatas))
            ]
            solved = ddp[i].solve(xs, us, 200, False, 0.1)
            print(solved)

            # Defining the final state as initial one for the next phase
            x0 = ddp[i].xs[-1]

        # Calc resulting CoM velocity (average)
        calcAverageCoMVelocity(ddp, rmodel , GAITPHASES, knots, timeStep)

        # Display the entire motion
        if WITHDISPLAY:
            print('Displaying the motion in Gepetto..')
            display = crocoddyl.GepettoDisplay(rh5_robot, cameraTF=cameraTF, frameNames=[rightFoot, leftFoot])
            # rh5_robot.viewer.gui.startCapture(rh5_robot.viz.windowID, 'test', '.mp4') # TODO: Automate video recording (check params, nothing happens now)
            for i, phase in enumerate(GAITPHASES):
                display.displayFromSolver(ddp[i])
                # rh5_robot.viewer.gui.stopCapture(rh5_robot.viz.windowID)

        # if WITHLOG or WITHDISPLAY:
        #     print('...............')
        #     print('Post-Processing')
        #     print('...............')

        # Export solution to .csv files
        if WITHLOG:
            logPath = simName + '/logs/'
            if not os.path.exists(logPath):
                os.makedirs(logPath)
            logSolution(ddp, timeStep,logPath)

        # Plotting the entire motion
        if WITHPLOT:
            minFeetError = plotSolution(ddp, simName, knots, bounds=False, figIndex=1, show=False)

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

    #     # Collect grid search data    
    #     log = ddp[0].getCallbacks()[0]
    #     sol = [solved, log.iters[-1], round(log.costs[-1]), round(minFeetError, 7), PGain, DGain]
    #     print(PGain, DGain)
    #     gridSearchResults.append(sol)
    # for test in gridSearchResults:
    #     print(test)
    # # Save results
    # import csv
    # filename = '/home/julian/Dev/crocoddyl/RH5/results/HumanoidFixedArms/Analysis/GridResults.csv'
    # with open(filename, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['solved', 'iterations', 'costs', 'minFeetError', 'PGain', 'DGain'])
    #     writer.writerows(gridSearchResults)
