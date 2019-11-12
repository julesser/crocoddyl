import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from double_pendulum_utils import CostModelDoublePendulum, ActuationModelDoublePendulum

robot = example_robot_data.loadDoublePendulum()
robot_model = robot.model

robot.initViewer(loadModel=True)

q0 = np.array([3.14, 0])
robot.q0.flat = q0
robot.framesForwardKinematics(robot.q0)
robot.display(robot.q0)

robot.viewer.gui.refresh()

state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state, 1)
terminalCostModel = crocoddyl.CostModelSum(state, 1)

weights = np.array([1, 1, 1, 1] + [0.1] * 2)

xRegCost = crocoddyl.CostModelState(state, 1)
uRegCost = crocoddyl.CostModelControl(state, 1)
xPendCost = CostModelDoublePendulum(state,
                                    1,
                                    crocoddyl.ActivationModelWeightedQuad(np.matrix(weights).T))

runningCostModel.addCost("regx", xRegCost, 1e-6)
runningCostModel.addCost("regu", uRegCost, 1e-3)
runningCostModel.addCost("pend", xPendCost, 1)
terminalCostModel.addCost("ori2", xPendCost, 1e5)

dt = 1e-2

actModel = ActuationModelDoublePendulum(state, actLink=1)
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)

T = 100
x0 = [3.14, 0, 0., 0.]
problem = crocoddyl.ShootingProblem(np.matrix(x0).T, [runningModel] * T, terminalModel)

ddp = crocoddyl.SolverFDDP(problem)

ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

ddp.solve()

# crocoddyl.displayTrajectory(robot, ddp.xs, dt)