import crocoddyl
import pinocchio
import numpy as np

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
        # q0 = np.matrix([0,0,0.91,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).T # Init Zero Configuration
        """ q0 = np.matrix([0,0,0.90,0,0,0,1,        #q1-7:   Floating Base (quaternions) # Init pose between zero config and smurf
                        0,0,-0.1,0.2,0,-0.1,     #q8-13:  Left Leg     
                        0,0,-0.1,0.2,0,-0.1]).T  #q14-19: Right Leg """
        q0 = np.matrix([0,0,0.8793,0,0,0,1,      #q1-7:   Floating Base (quaternions) # Stable init pose from long-time gait
                        0,0,-0.33,0.63,0,-0.30,     #q8-13:  Left Leg     
                        0,0,-0.33,0.63,0,-0.30]).T  #q14-19: Right Leg
        self.q0 = q0
        self.rmodel.defaultState = np.concatenate([q0, np.zeros((self.rmodel.nv, 1))])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.nsurf = np.matrix([0., 0., 1.]).T

    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation
        lfFootPos0 = self.rdata.oMf[self.lfId].translation
        df = jumpLength[2] - rfFootPos0[2] # delta foot position
        print('df: ' + str(df))
        rfFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])
        print('comRef: ' + str(comRef))
        loco3dModel = []

        # Take off phase
        takeOff = [self.createSwingFootModel(timeStep,[self.lfId, self.rfId]) for k in range(groundKnots)] # 1. Gain momentum (flex knees) 
        
        # Fly up phase
        flyingUpPhase = []
        for k in range(flyingKnots):
            comTask = np.matrix([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]).T * (k + 1) / flyingKnots + comRef
            print(comTask)
            flyingUpPhase += [self.createSwingFootModel(timeStep, [], comTask=comTask)] # 2. ComTask towards jumpHeight
        
        # Fly down phase
        flyingDownPhase = []
        for k in range(flyingKnots): 
            comTask = np.matrix([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]).T * (flyingKnots - k -1) / flyingKnots + comRef
            print(comTask)
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])] # 3. Let gravity pull robot down
        
        # Impulse switch 
        f0 = np.matrix(jumpLength).T
        lfPlacement = crocoddyl.FramePlacement(self.lfId, pinocchio.SE3(np.eye(3), f0 + self.rdata.oMf[self.lfId].translation))
        rfPlacement = crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), f0 + self.rdata.oMf[self.rfId].translation))
        # footTask = [ //FIXED: The position argument (self.lfId) should be passed as a frame (e.g. 17) directly but the acoording POSITION of the frame
        #     crocoddyl.FramePlacement(self.lfId, pinocchio.SE3(np.eye(3), self.lfId + f0)),
        #     crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), self.rfId + f0))
        # ]
        footTask = [lfPlacement, rfPlacement]
        # landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, False)]
        landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, True)]
        
        # CoM correction phase 
        landed = []
        f0[2] = df
        for k in range(groundKnots*4): 
            comTask = comRef + f0
            print(comTask)
            landed += [self.createSwingFootModel(timeStep, [self.lfId, self.rfId], comTask=comTask)] # 3. Let gravity pull robot down

        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase 
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

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