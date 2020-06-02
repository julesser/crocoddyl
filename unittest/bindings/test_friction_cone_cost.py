import crocoddyl
import pinocchio
import example_robot_data
import numpy as np

from test_utils import NUMDIFF_MODIFIER, assertNumDiff

crocoddyl.switchToNumpyMatrix()

# Create robot model and data
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()

# Create differential action model and data; link the contact data
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACT_3D = crocoddyl.ContactModel3D(
    ROBOT_STATE, crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('l_sole'),
                                            pinocchio.SE3.Random().translation), ACTUATION.nu, pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D)
CONTACTS.addContact("l_sole_contact", CONTACT_3D)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu)

frictionCone = crocoddyl.FrictionCone(np.matrix([0., 0., 1.]).T, 0.7, 4, False)
activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(frictionCone.lb, frictionCone.ub))
COSTS.addCost(
    "r_sole_friction_cone",
    crocoddyl.CostModelContactFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('r_sole'), frictionCone),
                                           ACTUATION.nu), 1.)
COSTS.addCost(
    "l_sole_friction_cone",
    crocoddyl.CostModelContactFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('l_sole'), frictionCone),
                                           ACTUATION.nu), 1.)
MODEL = crocoddyl.DifferentialActionModelContactFwdDynamics(ROBOT_STATE, ACTUATION, CONTACTS, COSTS, 0., True)
DATA = MODEL.createData()

# Created DAM numdiff
MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(MODEL)
MODEL_ND.disturbance *= 10
dnum = MODEL_ND.createData()

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(ACTUATION.nu)
MODEL.calc(DATA, x, u)
MODEL.calcDiff(DATA, x, u)
MODEL_ND.calc(dnum, x, u)
MODEL_ND.calcDiff(dnum, x, u)
assertNumDiff(DATA.Fx, dnum.Fx, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Fu, dnum.Fu, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lx, dnum.Lx, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lu, dnum.Lu, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
