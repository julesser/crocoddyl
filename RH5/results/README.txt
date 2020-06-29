
Description of the ---file--- Contents:

--- logBase.csv ---
Pose of the Floating Base {Position, Orientation (Quaternions)},
Velocity of the Floating Base {Twist of Linear and Angular Velocities}
Acceleration of the Floating Base {Twist of Linear and Angular Acceleration (body-fixed)}

--- logContactWrenches.csv ---
F*: Contact Forces 
T*: Contact Moments 

--- logJointSpace.csv ---
q_*: Joint Positions 
dq_*: Joint Velocities 
ddq_*: Joint Acceleration 
Tau_*: Input Torques

--- logTaskSpace.csv ---
C*: CoM position
Poses of the feet {Position, Orientation (Quaternions)}


Note regarding the Dimensionality (exemplary for 6 Steps) 
n = 162 = 6*27 | 6 (robot) steps, 27 knots (time steps): 25 swing + 1 doubleSupport + 1 impulse
