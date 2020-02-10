import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

crocoddyl.switchToNumpyMatrix()

# Load dynamic model of the arm
robot = example_robot_data.loadTalosArm()
robot_model = robot.model

DT = 1e-3
T = 100
targets = np.array([[.4, 0., .4, 0., 0., 0., 1.],
                    [.4, 0., .1, 0., 0., 0., 1.],
                    [.4, .4, .4, 0., 0., 0., 1.],
                    [.4, .4, .1, 0., 0., 0., 1.]])

display = crocoddyl.GepettoDisplay(robot)
#display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
display.robot.viewer.gui.addSphere('world/target1', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
display.robot.viewer.gui.addSphere('world/target2', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
display.robot.viewer.gui.addSphere('world/point2', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
display.robot.viewer.gui.addSphere('world/point3', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
#display.robot.viewer.gui.applyConfiguration('world/point', target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
display.robot.viewer.gui.applyConfigurations(['world/target1', 'world/target2', 'world/point2', 'world/point3'], targets.tolist())  # xyz+quaternion
display.robot.viewer.gui.refresh()

# Create the cost functions
state = crocoddyl.StateMultibody(robot.model)
xRegCost = crocoddyl.CostModelState(state)
uRegCost = crocoddyl.CostModelControl(state)

Mref = [0] * 4
Mref[0] = crocoddyl.FrameTranslation(robot_model.getFrameId("gripper_left_joint"), np.matrix(targets[0][0:3]).T)
Mref[1] = crocoddyl.FrameTranslation(robot_model.getFrameId("gripper_left_joint"), np.matrix(targets[1][0:3]).T)
Mref[2] = crocoddyl.FrameTranslation(robot_model.getFrameId("gripper_left_joint"), np.matrix(targets[2][0:3]).T)
Mref[3] = crocoddyl.FrameTranslation(robot_model.getFrameId("gripper_left_joint"), np.matrix(targets[3][0:3]).T)

goalTrackingCost = [0] * 4
goalTrackingCost[0] = crocoddyl.CostModelFrameTranslation(state, Mref[0])
goalTrackingCost[1] = crocoddyl.CostModelFrameTranslation(state, Mref[1])
goalTrackingCost[2] = crocoddyl.CostModelFrameTranslation(state, Mref[2])
goalTrackingCost[3] = crocoddyl.CostModelFrameTranslation(state, Mref[3])

# Create cost model per each action model
runningCostModels = [0] * 4
runningCostModels[0] = crocoddyl.CostModelSum(state)
runningCostModels[1] = crocoddyl.CostModelSum(state)
runningCostModels[2] = crocoddyl.CostModelSum(state)
runningCostModels[3] = crocoddyl.CostModelSum(state)

terminalCostModels = [0] * 4
terminalCostModels[0] = crocoddyl.CostModelSum(state)
terminalCostModels[1] = crocoddyl.CostModelSum(state)
terminalCostModels[2] = crocoddyl.CostModelSum(state)
terminalCostModels[3] = crocoddyl.CostModelSum(state)

# Then let's add the running and terminal cost functions
runningCostModels[0].addCost("gripperPose", goalTrackingCost[0], 1e1)
runningCostModels[0].addCost("stateReg", xRegCost, 1e-1)          
runningCostModels[0].addCost("ctrlReg", uRegCost, 1e-5)   
terminalCostModels[0].addCost("gripperPose", goalTrackingCost[0], 1e5)

runningCostModels[1].addCost("gripperPose", goalTrackingCost[1], 1e1) 
runningCostModels[1].addCost("stateReg", xRegCost, 1e-2)           
runningCostModels[1].addCost("ctrlReg", uRegCost, 1e-5)
terminalCostModels[1].addCost("gripperPose", goalTrackingCost[1], 1e5)

runningCostModels[2].addCost("gripperPose", goalTrackingCost[2], 1e1) 
runningCostModels[2].addCost("stateReg", xRegCost, 1e-1)           
runningCostModels[2].addCost("ctrlReg", uRegCost, 1e-5) 
terminalCostModels[2].addCost("gripperPose", goalTrackingCost[2], 1e5)

runningCostModels[3].addCost("gripperPose", goalTrackingCost[3], 1e1) 
runningCostModels[3].addCost("stateReg", xRegCost, 1e-1)           
runningCostModels[3].addCost("ctrlReg", uRegCost, 1e-5)
terminalCostModels[3].addCost("gripperPose", goalTrackingCost[3], 1e5)

# Create the actuation model
actuationModel = crocoddyl.ActuationModelFull(state)

# Create the action model
runningModels = [0] * 4
runningModels[0] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModels[0]), DT)
runningModels[1] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModels[1]), DT)
runningModels[2] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModels[2]), DT)
runningModels[3] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModels[3]), DT)

terminalModels = [0] * 4
terminalModels[0] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModels[0]))
terminalModels[1] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModels[1]))
terminalModels[2] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModels[2]))
terminalModels[3] = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModels[3]))

# Create the problem
#q0 = np.matrix([2., 1.5, -2., 0., 0., 0., 0.]).T
q0 = np.matrix([3., 3., 3., 0., 0., 0., 0.]).T
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])

seq0 = [runningModels[0]]*T + [terminalModels[0]]
seq1 = [runningModels[1]]*T + [terminalModels[1]]
seq2 = [runningModels[2]]*T + [terminalModels[2]]
seq3 = [runningModels[3]]*T 
#problem = crocoddyl.ShootingProblem(x0, [runningModels[0]] * T, terminalModels[0])
#problem = crocoddyl.ShootingProblem(x0, seq0+[runningModels[1]]*T, terminalModels[1])
#problem = crocoddyl.ShootingProblem(x0, seq0+seq1+[runningModels[2]]*T, terminalModels[2])
problem = crocoddyl.ShootingProblem(x0,seq0+seq1+seq2+seq3,terminalModels[3])

ddp = crocoddyl.SolverDDP(problem)
ddp.solve()

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
xT = ddp.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[:state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
print('Finally reached = ', robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T)