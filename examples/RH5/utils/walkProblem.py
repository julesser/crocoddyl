import crocoddyl
import pinocchio
import numpy as np
import sys

class SimpleBipedGaitProblem:
    """ Defines a simple 3d locomotion problem
    """
    def __init__(self, rmodel, rightFoot, leftFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        # Defining default state
        # q0 = np.matrix([0,0,0.91,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).T # Init RH5 Full Body
        # q0 = np.matrix([0,0,0.91,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).T # Init Zero Configuration
        q0 = np.matrix([0,0,0.9163,0,0,0,1,      #q1-7:   Floating Base (quaternions) # Init pose between zero config and smurf
                        0,0,-0.1,0.2,0,-0.1,     #q8-13:  Left Leg     
                        0,0,-0.1,0.2,0,-0.1]).T  #q14-19: Right Leg
        """ q0 = np.matrix([0,0,0.88,0,0,0,1,          #q1-7:   Floating Base (quaternions) # Init like in smurf file
                        0,0,-0.353,0.642,0,-0.289,     #q8-13:  Left Leg     
                        0,0,-0.352,0.627,0,-0.275]).T  #q14-19: Right Leg """
        self.q0 = q0
        self.comRefY = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2])
        self.heightRef = self.rdata.oMf[self.rfId].translation[2] # height for RF and LF identical
        self.rmodel.defaultState = np.concatenate([q0, np.zeros((self.rmodel.nv, 1))])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.nsurf = np.matrix([0., 0., 1.]).T

    def createWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots, isLastPhase):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # correct reference height of feet
        comRef = (rfPos0 + lfPos0) / 2
        # comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])
        comRef[2] = self.comRefY
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId]) for k in range(supportKnots)]

        # Creating the action models for three steps
        # print(comRef)
        if self.firstStep is True:
            rStep = self.createFootstepModels(comRef, [rfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
            self.firstStep = False
        else:
            rStep = self.createFootstepModels(comRef, [rfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
        # print(comRef)
        if isLastPhase is True: 
            lStep = self.createFootstepModels(comRef, [lfPos0], 0.5* stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])
            # rearrangeKnots = 10
            # rearrangePose = [self.createRearrangeModel(timeStep, [self.rfId, self.lfId]) for k in range(rearrangeKnots)]
            # lStep += rearrangePose # TODO: Right now it has now effect on final joint pose -> right default state set?
        else: 
            lStep = self.createFootstepModels(comRef, [lfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])

        
        # We defined the problem as:
        loco3dModel += doubleSupport + rStep
        loco3dModel += doubleSupport + lStep
        # loco3dModel += rearrangePose

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length. The swing task
                # is decomposed on two phases: swing-up and swing-down. We decide
                # deliveratively to allocated the same number of nodes (i.e. phKnots)
                # in each phase. With this, we define a proper z-component for the
                # swing-leg motion.

                # phKnots = numKnots / 2 # Problem: stepHeight of last knot is greater than zero!
                if numKnots % 2 == 0: 
                    phKnots = (numKnots / 2) - 0.5 # If even numKnots (not preferred): Two knots have maxHeight, target height for first and last knot is zero 
                else: 
                    phKnots = (numKnots / 2) - 0.5 # If odd numKnots (preferred): One knot at stepHeight,target height for first and last knot is zero
                
                if k < phKnots:
                    dp = np.matrix([[stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots]]).T 
                elif k == phKnots:
                    dp = np.matrix([[stepLength * (k + 1) / numKnots, 0., stepHeight]]).T
                else:
                    dp = np.matrix(
                        [[stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)]]).T
                tref = np.asmatrix(p + dp)
                # print('p[' + str(k) + ']: ') 
                # print(p)
                # print('dp[' + str(k) + ']: ') 
                # print(dp)
                # print('tref[' + str(k) + ']: ') 
                # print(tref)
                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]
            comTask = np.matrix([stepLength * (k + 1) / numKnots, 0., 0.]).T * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask, pseudoImpulse=True) #TODO: Temporaily use PseudoImpulse because impulseModel does not contain acceleration information

        # Updating the current foot position for next step
        comPos0 += np.matrix([stepLength * comPercentage, 0., 0.]).T
        for p in feetPos0:
            p += np.matrix([[stepLength, 0., 0.]]).T
        return footSwingModel + [footSwitchModel]

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            Mref = crocoddyl.FramePlacement(i, pinocchio.SE3.Identity())
            supportContactModel = \
                crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, np.matrix([0., 0.]).T)
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = crocoddyl.CostModelFramePlacement(self.state, i, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e6)
        # Add cost for self-collision (joint limits) TODO: Right now no joint limits violated - Joint inperiodicity has other reason!
        # maxfloat = sys.float_info.max
        # xlb = np.vstack([
        #     -maxfloat * np.matrix(np.ones((6, 1))),  # dimension of the SE(3) manifold
        #     self.rmodel.lowerPositionLimit[7:],
        #     -maxfloat * np.matrix(np.ones((self.state.nv, 1)))
        # ])
        # xub = np.vstack([
        #     maxfloat * np.matrix(np.ones((6, 1))),  # dimension of the SE(3) manifold
        #     self.rmodel.upperPositionLimit[7:],
        #     maxfloat * np.matrix(np.ones((self.state.nv, 1)))
        # ])
        # print(xub)
        # bounds = crocoddyl.ActivationBounds(xlb, xub, 1.)
        # limitCost = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelQuadraticBarrier(bounds), self.rmodel.defaultState,
        #                                     self.actuation.nu)
        # costModel.addCost("limitCost", limitCost, 1e3)
        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        # costModel.addCost("stateReg", stateReg, 1e2)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createRearrangeModel(self, timeStep, supportFootIds):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            Mref = crocoddyl.FramePlacement(i, pinocchio.SE3.Identity())
            supportContactModel = \
                crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, np.matrix([0., 0.]).T)
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        # if isinstance(comTask, np.ndarray):
        #     comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
        #     costModel.addCost("comTrack", comTrack, 1e6)
        # for i in supportFootIds:
        #     cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
        #     frictionCone = crocoddyl.CostModelContactFrictionCone(
        #         self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
        #         crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
        #     costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e6)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """

        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            Mref = crocoddyl.FramePlacement(i, pinocchio.SE3.Identity())
            supportContactModel = crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, np.matrix([0., 0.]).T)
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = crocoddyl.CostModelFramePlacement(self.state, i, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e8)
                footVel = crocoddyl.FrameMotion(i.frame, pinocchio.Motion.Zero())
                impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, footVel, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_impulseVel", impulseFootVelCost, 1e6)

        stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (self.state.nv - 6) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
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
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, 0)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e8)
        stateWeights = np.array([1.] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(np.matrix(stateWeights**2).T),
                                            self.rmodel.defaultState, 0)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        return model