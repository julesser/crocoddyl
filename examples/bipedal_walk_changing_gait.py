import os
import sys

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

crocoddyl.switchToNumpyMatrix()

# Creating the lower-body part of Talos
talos_legs = example_robot_data.loadTalosLegs()
""" lims = talos_legs.model.effortLimit
lims *= 0.5  # reduced artificially the torque limits defined in URDF
talos_legs.model.effortLimit = lims """

# Defining the initial state of the robot
q0 = talos_legs.model.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(talos_legs.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'
gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot) #Init problem; Class defined in python3.6/site-packages/utils/biped.py

# Setting up all tasks
# Define multiple phases for a full gait: One phase = One defined shooting problem, consisting of multiple DAMs
GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 1.0, 'stepHeight': 0.1,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.15,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.25,
                  'timeStep': 0.03, 'stepKnots': 25, 'supportKnots': 1}}]
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    if WITHDISPLAY and WITHPLOT:
        display = crocoddyl.GepettoDisplay(talos_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
        ddp[i].setCallbacks(
            [crocoddyl.CallbackLogger(),
             crocoddyl.CallbackVerbose(),
             crocoddyl.CallbackDisplay(display)])
    elif WITHDISPLAY:
        display = crocoddyl.GepettoDisplay(talos_legs, 4, 4, cameraTF, frameNames=[rightFoot, leftFoot])
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    elif WITHPLOT:
        ddp[i].setCallbacks([
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
        ])
    else:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [talos_legs.model.defaultState] * len(ddp[i].models())
    us = [m.quasiStatic(d, talos_legs.model.defaultState) for m, d in list(zip(ddp[i].models(), ddp[i].datas()))[:-1]]
    ddp[i].solve(xs, us, 100, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_legs, frameNames=[rightFoot, leftFoot])
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
                                  figIndex=i + 3,
                                  show=True if i == len(GAITPHASES) - 1 else False)