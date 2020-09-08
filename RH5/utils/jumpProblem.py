import crocoddyl
import pinocchio
import numpy as np
import sys

class HumanoidJumpProblem:
    """ Defines a jumping problem for a full-size humanoid robot
    """
    def __init__(self, rmodel, rightFoot, leftFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        # self.state.lb[-self.state.nv:] *= 0.5 # Artificially reduce max joint velocity
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        # Defining default state
        # (1) Fixed arms
        # q0 = np.array([0,0,0.87955,0,0,0,1,         # Floating Base (quaternions) # Stable init pose from long-time gait
        #                 0.2,0,0,                    # Torso
        #                 0,0,-0.33,0.63,0,-0.30,     # Left Leg     
        #                 0,0,-0.33,0.63,0,-0.30])    # Right Leg
        # (2) Losen shoulder joints
        q0 = np.array([0,0,0.87955,0,0,0,1,         # Floating Base (quaternions) # Stable init pose from long-time gait
                        0.2,0,0,                    # Torso
                        -0.25,0.1,0,0.25,-0.1,0,    # Shoulders
                        0,0,-0.33,0.63,0,-0.30,     # Left Leg     
                        0,0,-0.33,0.63,0,-0.30])    # Right Leg
        self.q0 = q0
        self.comRefY = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.heightRef = self.rdata.oMf[self.rfId].translation[2] # height for RF and LF identical
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstJump = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.nsurf = np.array([0., 0., 1.])

    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots, recoveryKnots, final=False):
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY
        # print('comRef: ' + str(comRef))
        # print('lfPos0: ' + str(lfPos0))
        # print('rfPos0: ' + str(rfPos0))
        f0 = np.array(jumpLength)
        lfGoal = crocoddyl.FramePlacement(self.lfId, pinocchio.SE3(np.eye(3), lfPos0 + f0))
        rfGoal = crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), rfPos0 + f0))

        jumpingModels = []
        # Gain momentum
        takeOff = [self.createSwingFootModel(timeStep,[self.lfId, self.rfId]) for k in range(groundKnots)]
        # Fly up + down + impact
        flying = self.createSymmetricFlyingModels([lfPos0, rfPos0], jumpLength[0], jumpLength[2]+jumpHeight,
                                                    timeStep, flyingKnots, [], [self.lfId, self.rfId])

        # Recover to initial pose
        recovery = []
        for k in range(recoveryKnots):
            recovery += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], 
                              comTask=comRef+f0, swingFootTask=[lfGoal, rfGoal], poseRecovery=True)]
        
        jumpingModels += takeOff + flying + recovery
        # jumpingModels += takeOff + flying

        problem = crocoddyl.ShootingProblem(x0, jumpingModels, jumpingModels[-1])
        return problem

    def createSymmetricFlyingModels(self, feetPos0, jumpLength, jumpHeight, 
                                    timeStep, numKnots, supportFootIds, swingFootIds):
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # phKnots = numKnots / 2 # Problem: jumpHeight of last knot is greater than zero!
                phKnots = (numKnots / 2) - 0.5 # If even numKnots (not preferred): Two knots have maxHeight, target height for first and last knot is zero 
                
                if k < phKnots:
                    dp = np.array([jumpLength * (k + 1) / numKnots, 0., jumpHeight * k / phKnots]) 
                elif k == phKnots:
                    dp = np.array([jumpLength * (k + 1) / numKnots, 0., jumpHeight])
                else:
                    dp = np.array(
                        [jumpLength * (k + 1) / numKnots, 0., jumpHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp
                # print('p[' + str(k) + ']: ') 
                # print(p)
                # print('dp[' + str(k) + ']: ') 
                # print(dp)
                # print('tref[' + str(k) + ']: ') 
                # print(tref)
                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, swingFootTask=swingFootTask)
            ]
        # Action model for the foot switch
        footSwitchModel = self.createImpulseModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next jump
        for p in feetPos0:
            p += [jumpLength, 0., 0.]
        return footSwingModel + [footSwitchModel]

    def createBoxJumpingProblem(self, x0, jumpHeight, jumpLength, obstacleHeight, timeStep, groundKnots, flyingKnots, recoveryKnots, final=False):
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY
        # print('comRef: ' + str(comRef))
        # print('lfPos0: ' + str(lfPos0))
        # print('rfPos0: ' + str(rfPos0))
        f0 = jumpLength
        lfGoal = crocoddyl.FramePlacement(self.lfId, pinocchio.SE3(np.eye(3), lfPos0 + f0))
        rfGoal = crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), rfPos0 + f0))

        jumpingModels = []
        # Gain momentum
        takeOff = [self.createSwingFootModel(timeStep,[self.lfId, self.rfId]) for k in range(groundKnots)]
        # Fly up + down + impact
        jumpOnObs = self.createAsymmetricFlyingModels([lfPos0, rfPos0], jumpLength[0], jumpLength[2]+jumpHeight, obstacleHeight,
                                                    timeStep, flyingKnots, [], [self.lfId, self.rfId])
        jumpOffObs = self.createAsymmetricFlyingModels([lfPos0, rfPos0], jumpLength[0], jumpLength[2]+jumpHeight, obstacleHeight,
                                                    timeStep, flyingKnots, [], [self.lfId, self.rfId])

        # Recover to initial pose
        recovery = []
        # for k in range(recoveryKnots):
        #     recovery += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], 
        #                       comTask=comRef+f0+np.array([0,0,obstacleHeight]), swingFootTask=[lfGoal, rfGoal], poseRecovery=True)]
        
        # jumpingModels += takeOff + jumpOnObs + jumpOffObs + recovery
        jumpingModels += takeOff + jumpOnObs + jumpOffObs

        problem = crocoddyl.ShootingProblem(x0, jumpingModels, jumpingModels[-1])
        return problem

    def createAsymmetricFlyingModels(self, feetPos0, jumpLength, jumpHeight, obstacleHeight,
                                        timeStep, numKnots, supportFootIds, swingFootIds):
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                upDownRatio = jumpHeight / (jumpHeight + (jumpHeight - obstacleHeight))
                # upDownRatio = obstacleHeight/(jumpHeight+(jumpHeight-obstacleHeight))
                upKnots = (numKnots * upDownRatio) - 0.5
                # print(jumpHeight)
                # print(obstacleHeight)
                # print(upDownRatio)
                # print(upKnots)
                if k < upKnots:
                    dp = np.array([jumpLength * (k + 1) / numKnots, 0., jumpHeight * k / upKnots]) 
                elif k == upKnots:
                    dp = np.array([jumpLength * (k + 1) / numKnots, 0., jumpHeight])
                else:
                    dp = np.array(
                        [jumpLength * (k + 1) / numKnots, 0., jumpHeight * (1 - float(k - upKnots) / upKnots)])
                tref = p + dp
                    
                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, swingFootTask=swingFootTask)
            ]
            # print('p[' + str(k) + ']: ') 
            # print(p)
            # print('dp[' + str(k) + ']: ')
            # print(dp)
            print('tref[' + str(k) + ']: ')
            print(tref)
        # Action model for the foot switch
        footSwitchModel = self.createImpulseModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next jump
        for p in feetPos0:
            p += [jumpLength, 0., jumpHeight]
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, poseRecovery=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        baumgarteGains = np.array([0., 100.])
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            Mref = crocoddyl.FramePlacement(i, pinocchio.SE3.Identity())
            supportContactModel = \
                crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, baumgarteGains)
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            # friction cone cost
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e2)
            # center of pressure cost
            CoP = crocoddyl.CostModelContactCoPPosition(self.state, 
                crocoddyl.FrameCoPSupport(i, np.array([0.2, 0.08])), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_CoP", CoP, 1e3)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = crocoddyl.CostModelFramePlacement(self.state, i, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e6)
        
        # joint limits cost
        x_lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        x_ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        joint_limits = crocoddyl.CostModelState(self.state, activation_xbounds, 0 * self.rmodel.defaultState, self.actuation.nu)
        # costModel.addCost("jointLim", joint_limits, 1e3)
        
        if poseRecovery is True:
            poseWeights = np.array([0] * 6 + [1] * (self.state.nv - 6) + [0] * self.state.nv)
            stateRecovery = crocoddyl.CostModelState(self.state,
                                                    crocoddyl.ActivationModelWeightedQuad(poseWeights**2),
                                                    self.rmodel.defaultState, self.actuation.nu)
            costModel.addCost("stateRecovery", stateRecovery, 1e5)
        stateWeights = np.array([0] * 3 + [10.] * 3 + [50.] * 3 + [0.01] * (self.state.nv - 9) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)
        
        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                # Add foot track cost
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, 0)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e8)
        for i in supportFootIds:
            # Impulse center of pressure cost
            CoP = crocoddyl.CostModelImpulseCoPPosition(self.state, 
            crocoddyl.FrameCoPSupport(i, np.array([0.2, 0.08])))
            costModel.addCost(self.rmodel.frames[i].name + "_CoP", CoP, 1e3)
            # Impulse friction cone cost
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelImpulseFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone))
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e2)
    
        stateWeights = np.array([1.] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                            self.rmodel.defaultState, 0)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        return model
