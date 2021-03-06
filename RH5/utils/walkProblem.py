import crocoddyl
import pinocchio
import numpy as np
import sys

class SimpleBipedGaitProblem:
    """ Defines a simple 3d locomotion problem
    """
    def __init__(self, rmodel, rightFoot, leftFoot, DGain, PGain):
        self.baumgarteGains = np.array([PGain, DGain])
        print('baumgarteGains: ' + str(self.baumgarteGains))
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        # self.state.lb[-self.state.nv:] *= 0.5 # Artificially reduce max joint velocity
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        # Default state #TaskSepcific:StaticWalking/Squat
        # q0 = np.array([0,0,0.87955,0,0,0,1,         # Floating Base (quaternions) # Stable init pose from long-time gait
        #                 0.2,0,0,                    # Torso
        #                 0,0,-0.33,0.63,0,-0.30,     # Left Leg     
        #                 0,0,-0.33,0.63,0,-0.30])    # Right Leg
        # TaskSpecific:DynamicWalking
        q0 = np.array([0,0,0.87955,0,0,0,1,         # Floating Base (quaternions) # Stable init pose from long-time gait
                        0.2,0,0,                    # Torso
                        -0.25, 0.1, 0, 0,           # Left Arm
                        0.25, -0.1, 0, 0,           # Right Arm
                        0,0,-0.33,0.63,0,-0.30,     # Left Leg     
                        0,0,-0.33,0.63,0,-0.30])    # Right Leg
        # TaskSpecific:Static Walking from Mid of other step
        # q0 = np.array([0.0618,0.0020,0.8908,0.0003,0.0099,0.0114,0.9998,         # Floating Base (quaternions) # Stable init pose from long-time gait
        #                 0.1372,-0.0021,0.0001,                    # Torso
        #                 -0.0223,-0.0051,-0.2313,0.5316,0.0048,-0.3258,     # Left Leg     
        #                 -0.0215,-0.0055,-0.3402,0.5369,0.0049,-0.2199])  # Right Leg
        self.q0 = q0
        self.comRefY = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        # print(self.rdata.oMf[self.rfId].translation[2])
        self.heightRef = self.rdata.oMf[self.rfId].translation[2] # height for RF and LF identical
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.nsurf = np.array([0., 0., 1.])

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
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        # comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])
        comRef[2] = self.comRefY # Define global CoM target height 
        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId]) for k in range(supportKnots)]
        # Append 1s of recovery 
        stabilization = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], poseRecovery=True) for k in range(supportKnots)]
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
        else: 
            lStep = self.createFootstepModels(comRef, [lfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])

        # We define the problem as:
        loco3dModel += doubleSupport + rStep
        loco3dModel += doubleSupport + lStep
        if isLastPhase is True: 
            loco3dModel += stabilization

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createStaticWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots, isLastPhase):
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        # print('rfPos0, lfPos0')
        # print(rfPos0, lfPos0)
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY # Define global CoM target height 
        # Defining the action models along the time instances
        loco3dModel = []
        stabilization = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], poseRecovery=True) for k in range(60)]
        print('stabilization:')
        # stabilization = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comRef, poseRecovery=True) for k in range(120)]
        # Creating the action models for three steps
        # print('rfPos0, lfPos0')
        # print('comShiftToLF')
        # print(rfPos0, lfPos0)
        print('comShiftToLF:')
        comYHackyOffset = 0.0075
        comShiftToLF = self.createCoMShiftModels(supportKnots, comRef, lfPos0+[0,comYHackyOffset,0], timeStep)
        # print('rfPos0, lfPos0 after shift')
        # print(rfPos0, lfPos0)
        comTaskLF = np.hstack((lfPos0[0:2]+[0,comYHackyOffset], comRef[2]))
        print('comTaskLF: ' +  str(comTaskLF))
        if self.firstStep is True:
            # rStep = self.createFootstepModels(lfPos0, [rfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
            rStep = self.createFootstepModels(comTaskLF, [rfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
            self.firstStep = False
        else:
            rStep = self.createFootstepModels(comTaskLF, [rfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
        # print('comShiftToRF')
        # print('rfPos0, lfPos0')
        # print(rfPos0, lfPos0)
        print('comShiftToRF:')
        comYHackyOffset = -0.0125
        comShiftToRF = self.createCoMShiftModels(supportKnots, comTaskLF, rfPos0+[0,comYHackyOffset,0], timeStep)
        # print('rfPos0, lfPos0 after shift')
        # print(rfPos0, lfPos0)
        comTaskRF = np.hstack((rfPos0[0:2]+[0,comYHackyOffset], comRef[2]))
        print('comTaskRF: ' +  str(comTaskRF))
        if isLastPhase is True: 
            lStep = self.createFootstepModels(comTaskRF, [lfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])
        else: 
            lStep = self.createFootstepModels(comTaskRF, [lfPos0], stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])
        # print('End')
        # print('rfPos0, lfPos0')
        # print(rfPos0, lfPos0)
        # We define the problem as:
        loco3dModel += comShiftToLF + rStep
        loco3dModel += comShiftToRF + lStep
        if isLastPhase is True: 
            loco3dModel += stabilization

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createOneStepStaticWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots, isLastPhase):
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY # Define global CoM target height 
        # Defining the action models along the time instances
        loco3dModel = []
        stabilization = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], poseRecovery=True) for k in range(supportKnots)]
        
        # Perform only right step
        # comXHackyOffset = -0.02
        # comYHackyOffset = 0.01
        # comShiftToLF = self.createCoMShiftModels(supportKnots, comRef, lfPos0+[comXHackyOffset,comYHackyOffset,0], timeStep)
        # comTaskLF = np.hstack((lfPos0[0:2]+[comXHackyOffset,comYHackyOffset], comRef[2]))
        # rStep = self.createFootstepModels(comTaskLF, [rfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.lfId], [self.rfId])
        
        # Perform only left step
        comXHackyOffset = 0
        comYHackyOffset = -0.0165
        comShiftToRF = self.createCoMShiftModels(supportKnots, comRef, rfPos0+[comXHackyOffset,comYHackyOffset,0], timeStep)
        comTaskRF = np.hstack((rfPos0[0:2]+[comXHackyOffset,comYHackyOffset], comRef[2]))
        lStep = self.createFootstepModels(comTaskRF, [lfPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots, [self.rfId], [self.lfId])
        # We define the problem as:
        # loco3dModel += comShiftToLF + rStep
        loco3dModel += comShiftToRF + lStep
        if isLastPhase is True: 
            loco3dModel += stabilization

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createSquatProblem(self, x0, heightChange, numKnots, timeStep):
        """ Create a shooting problem for a simple squat in order to verify our framework
        """
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY
        # print('comRef:' + str(comRef[2]))
        squatModels = []
        comXHackyOffset = 0.005
        for k in range(numKnots):
            phKnots = (numKnots / 2)
            if k < phKnots:
                comTask = np.array([0., 0., -heightChange * (k + 1) / phKnots]) + comRef
            elif k == phKnots:
                comTask = np.array([0., 0., -heightChange]) + comRef
            else:
                comTask = np.array([0., 0., -heightChange * (1 - float(k - phKnots) / phKnots)]) + comRef
            # print(comTask[2])
            squatModels += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comTask)]
        squatModels += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comRef) for k in range(20)] # Enshure last CoM equals reference
        
        problem = crocoddyl.ShootingProblem(x0, squatModels, squatModels[-1])
        return problem

    def createBalancingProblem(self, x0, supportKnots, shiftKnots, balanceKnots, timeStep):
        """ Create a shooting problem for a simple squat in order to verify our framework
        """
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation
        rfPos0[2], lfPos0[2] = self.heightRef, self.heightRef # Set global target height of feet to initial height from q0
        comRef = (rfPos0 + lfPos0) / 2
        comRef[2] = self.comRefY
        
        balancingModels = []
        # Shift CoM over LF
        shiftingToLF = []
        print('shiftKnots: ' + str(shiftKnots))
        print('balanceKnots: ' + str(balanceKnots))
        print('rfPos0: ' + str(rfPos0))
        comYDiff = lfPos0[1] - comRef[1]
        comHackyYOffset = 0.0075
        for k in range(shiftKnots):
            comTask = np.array([0, (np.asscalar(comYDiff) * (k + 1) / shiftKnots) + comHackyYOffset, 0.]) + comRef
            shiftingToLF += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comTask)]
        # Foot task
        balancing = []
        relTargetRF = np.array([0., -0.05, 0.05])
        for k in range(balanceKnots):
            phKnots = (balanceKnots / 2)
            if k < phKnots:
                footPosTask = relTargetRF * ((k + 1) / phKnots) + rfPos0
            elif k == phKnots:
                footPosTask = relTargetRF + rfPos0
            else:
                print('go back to init:')
                footPosTask = relTargetRF * (1 - float(k - phKnots) / phKnots) + rfPos0
            print(footPosTask)
            swingFootTask = [crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), footPosTask))]
            # balancing += [self.createSwingFootModel(timeStep, [self.lfId], comTask=lfPos0, swingFootTask=swingFootTask)]
            comTask = np.hstack((lfPos0[0:2]+[0,comHackyYOffset], comRef[2]))
            print(comTask)
            # balancing += [self.createSwingFootModel(timeStep, [self.lfId], comTask=lfPos0, swingFootTask=swingFootTask)]
            balancing += [self.createSwingFootModel(timeStep, [self.lfId], comTask=comTask, swingFootTask=swingFootTask)]
        # Impact RF
        swingFootTask = [crocoddyl.FramePlacement(self.rfId, pinocchio.SE3(np.eye(3), rfPos0))]
        # impact = [self.createFootSwitchModel([self.rfId, self.lfId], swingFootTask, pseudoImpulse=True)]
        impact = [self.createFootSwitchModel([self.rfId, self.lfId], swingFootTask, pseudoImpulse=False)]
        # Shift CoM back to center
        shiftingToCenter = []
        for k in range(shiftKnots):
            comTask = np.array([0, np.asscalar(comYDiff) * (1 - (k / shiftKnots)), 0.]) + comRef
            shiftingToCenter += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comTask)]
        # Recover to initial pose
        stabilization = [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], poseRecovery=True) for k in range(60)]
        
        balancingModels += shiftingToLF + balancing + impact + shiftingToCenter + stabilization
        
        problem = crocoddyl.ShootingProblem(x0, balancingModels, balancingModels[-1])
        return problem

    def createCoMShiftModels(self, shiftKnots, comRef, comTarget, timeStep):
        shiftModels = []
        comXDiff = comTarget[0] - comRef[0]
        comYDiff = comTarget[1] - comRef[1] 
        # print('comXDiff, comYDiff')
        # print(comXDiff, comYDiff)
        for k in range(shiftKnots):
            comTask = np.array([np.asscalar(comXDiff) * (k + 1) / shiftKnots, 
                                 np.asscalar(comYDiff) * (k + 1) / shiftKnots, 
                                 0.]) + comRef
            # print(comTask)
            shiftModels += [self.createSwingFootModel(timeStep, [self.rfId, self.lfId], comTask=comTask)]
        return shiftModels

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
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots]) 
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp
                # print('p[' + str(k) + ']: ') 
                # print(p)
                # print('dp[' + str(k) + ']: ') 
                # print(dp)
                # print('tref[' + str(k) + ']: ') 
                # print(tref)
                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]
            comTask = comPos0 
            # comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * 1/2 + comPos0 #TaskSpecific:SonicWalk
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        # footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask, pseudoImpulse=True)
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask, pseudoImpulse=False)

        # Updating the current foot position for next step
        # comPos0 += [stepLength * 1/2, 0., 0.] #TaskSpecific:SonicWalk
        for p in feetPos0:
            p += [stepLength, 0., 0.]
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
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            Mref = crocoddyl.FramePlacement(i, pinocchio.SE3.Identity())
            supportContactModel = \
                crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, self.baumgarteGains) #TaskSpecific:DynamicWalking
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        # if isinstance(comTask, np.ndarray): # TaskSpecific:SonicWalk,Squatting
        #     comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
        #     costModel.addCost("comTrack", comTrack, 1e6)
        # if isinstance(comTask, np.ndarray): # TaskSpecific:Balancing&StaticWalking
        #     # com2DWeights = np.array([1, 1, 0]) # Neglect height of CoM
        #     com2DWeights = np.array([1, 1, 1]) # Neglect height of CoM
        #     com2DTrack = crocoddyl.CostModelCoMPosition(self.state, 
        #         crocoddyl.ActivationModelWeightedQuad(com2DWeights**2), comTask, self.actuation.nu)
        #     costModel.addCost("com2DTrack", com2DTrack, 1e6)
        for i in supportFootIds:
            # friction cone cost
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e2)
            # center of pressure cost
            CoP = crocoddyl.CostModelContactCoPPosition(self.state, 
                # crocoddyl.FrameCoPSupport(i, np.array([0.2, 0.08])), self.actuation.nu)
                crocoddyl.FrameCoPSupport(i, np.array([0.1, 0.04])), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_CoP", CoP, 1e3) # TaskSpecific:DynamicWalking
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = crocoddyl.CostModelFramePlacement(self.state, i, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e6)
        
        # joint limits cost
        x_lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        x_ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        joint_limits = crocoddyl.CostModelState(self.state, activation_xbounds, 0 * self.rmodel.defaultState, self.actuation.nu)
        costModel.addCost("jointLim", joint_limits, 1e6)
        
        if poseRecovery is True: # TODO: Invididual class for eliminating fight recovery vs Cop cost
            poseWeights = np.array([0] * 6 + [1] * (self.state.nv - 6) + [0] * self.state.nv)
            stateRecovery = crocoddyl.CostModelState(self.state,
                                                    crocoddyl.ActivationModelWeightedQuad(poseWeights**2),
                                                    self.rmodel.defaultState, self.actuation.nu)
            costModel.addCost("stateRecovery", stateRecovery, 1e4)
        # stateWeights = np.array([0] * 3 + [500.] * 3 + [10.] * 3 + [0.01] * (self.state.nv - 9) + [10] * self.state.nv)
        stateWeights = np.array([0] * 3 + [10.] * 3 + [10.] * 3 + [0.01] * (self.state.nv - 9) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
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
            supportContactModel = crocoddyl.ContactModel6D(self.state, Mref, self.actuation.nu, np.array([0., 40.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e2)
            # center of pressure
            CoP = crocoddyl.CostModelContactCoPPosition(self.state, 
                # crocoddyl.FrameCoPSupport(i, np.array([0.2, 0.08])), self.actuation.nu)
                crocoddyl.FrameCoPSupport(i, np.array([0.1, 0.04])), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_CoP", CoP, 1e3) # TaskSpecific:DynamicWalking
        if swingFootTask is not None:
            for i in swingFootTask:
                footTrack = crocoddyl.CostModelFramePlacement(self.state, i, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e8)
                footVel = crocoddyl.FrameMotion(i.frame, pinocchio.Motion.Zero())
                impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, footVel, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_impulseVel", impulseFootVelCost, 1e6)

        # stateWeights = np.array([0] * 3 + [500.] * 3 + [10.] * 3 + [0.01] * (self.state.nv - 9) + [10] * self.state.nv)
        stateWeights = np.array([0] * 3 + [10.] * 3 + [10.] * 3 + [0.01] * (self.state.nv - 9) + [10] * self.state.nv)
        stateReg = crocoddyl.CostModelState(self.state,
                                            crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
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
                # Add foot track cost
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, 0)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e8)
        for i in supportFootIds:
            # Impulse center of pressure cost
            CoP = crocoddyl.CostModelImpulseCoPPosition(self.state, 
            # crocoddyl.FrameCoPSupport(i, np.array([0.2, 0.08])))
            crocoddyl.FrameCoPSupport(i, np.array([0.1, 0.04])))
            costModel.addCost(self.rmodel.frames[i].name + "_CoP", CoP, 1e3) # TaskSpecific:DynamicWalking
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

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        return model