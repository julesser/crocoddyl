Note regarding the Dimensionality: 
n = 162 = 6*27 | 6 (robot) steps, 27 knots (time steps): 25 swing + 1 doubleSupport + 1 impulse

Description of the ---file--- Contents:

--- logBaseStates.csv ---
Pose of the Floating Base {Position, Orientation (Quaternions)},
Velocity of the Floating Base {Twist of Linear and Angular Velocities}

--- logJointStatesAndEffort.csv ---
q_*: Joint Positions 
dq_*: Joint Velocities 
Tau_*: Input Torques

--- logContactWrenches.csv ---
F*: Contact Forces 
T*: Contact Moments 

