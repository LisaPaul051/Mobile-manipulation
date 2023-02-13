import modern_robotics as mr
import numpy as np
import pandas as pd


def FeedbackControl(robot_conf, Tse, Tse_d, Tse_d_next, Kp, Ki, Dt):

    # arm Jacobian
    Blist = [[0,0,0,0,0], [0,-1,-1,-1,0], [1,0,0,0,1], [0,-0.5076,-0.3526,-0.2176,0], [0.033,0,0,0,0], [0,0,0,0,0]]
    theta = robot_conf[3:8]
    Ja = mr.JacobianBody(Blist,theta)

    # base Jacobian
    r = 0.0475
    l = 0.235
    w = 0.15
    F = np.dot((r/4),(np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], [1, 1, 1, 1], [-1, 1, -1, 1]])))
    zeros = np.zeros((2,4))
    f = np.vstack((zeros,F))
    f = np.vstack((f,np.zeros((1,4))))
    Tbo = np.array([[1,0,0,0.1662], [0,1,0,0], [0,0,1,0.0026], [0,0,0,1]])
    Tbo_inv = mr.TransInv(Tbo)
    Moe = np.array([[1,0,0,0.033], [0,1,0,0], [0,0,1,0.6546], [0,0,0,1]])
    Toe = mr.FKinBody(Moe,Blist,theta)
    Toe_inv = mr.TransInv(Toe)
    Ad = mr.Adjoint(np.dot(Toe_inv,Tbo_inv))
    Jb = np.dot(Ad,f)

    # mobile manipulator Jacobian
    Je = np.column_stack((Jb,Ja))
    Je_pinv = np.linalg.pinv(Je)

    
    
    # inverses
    Xinv = np.linalg.inv(Tse)
    Xdinv = np.linalg.inv(Tse_d)

    # errors
    Xerr1 = mr.MatrixLog6(np.dot(Xinv,Tse_d))
    Xerr = mr.se3ToVec(Xerr1)
    
    Xerrd1 = mr.MatrixLog6(np.dot(Xdinv,Tse_d_next))
    Xerrd = mr.se3ToVec(Xerrd1)

    # feedfoward reference twist
    Vd_in_se3 = np.dot((1/Dt),Xerrd1)
    Vd = mr.se3ToVec(Vd_in_se3)
    
    Adx = np.dot(mr.Adjoint(Xinv), mr.Adjoint(Tse_d))
    AdxVd = np.dot(Adx, Vd)

    P_term = np.dot(Kp,Xerr)

    increment = np.dot(Xerr,Dt)
    I_term = np.dot(Ki,increment)

    # end-effector twist
    Vt = np.add(AdxVd, P_term, I_term)

    # velocities
    speeds = np.dot(Je_pinv,Vt)

    


# example
robot_conf = [0,0,0,0,0,0.2,-1.6,0]
Tse = np.array([[0.170, 0, 0.985, 0.387], [0,1,0,0], [-0.985, 0, 0.170, 0.570], [0,0,0,1]])
Tse_d = np.array([[0,0,1,0.5], [0,1,0,0], [-1,0,0,0.5], [0,0,0,1]])
Tse_d_next = np.array([[0,0,1,0.6], [0,1,0,0], [-1,0,0,0.3], [0,0,0,1]])
Kp = np.zeros((6,6))
Ki = np.zeros((6,6))
Dt = 0.01


FeedbackControl(robot_conf, Tse, Tse_d, Tse_d_next, Kp, Ki, Dt)

    
    
    
