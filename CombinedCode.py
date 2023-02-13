import modern_robotics as mr
import numpy as np
import pandas as pd
import math

def tse(initial_conf):
    Blist = [[0,0,0,0,0], [0,-1,-1,-1,0], [1,0,0,0,1], [0,-0.5076,-0.3526,-0.2176,0], [0.033,0,0,0,0], [0,0,0,0,0]]
    theta = initial_conf[3:8]
    phi = initial_conf[0]
    x = initial_conf[1]
    y = initial_conf[2]
    Tsb = np.array([[math.cos(phi),-math.sin(phi),0,x], [math.sin(phi),math.cos(phi),0,y], [0,0,1,0.0963], [0,0,0,1]])
    Tbo = np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])
    Moe = np.array([[1,0,0,0.033], [0,1,0,0], [0,0,1,0.6546], [0,0,0,1]])
    Toe = mr.FKinBody(Moe,Blist,theta)
    Tse = np.dot(np.dot(Tsb,Tbo),Toe)
    return Tse

def TrajectoryGeneration(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k):
    trajs = []
    Tf = 1
    N = (Tf*k)/0.01
    method = 3 #cubic time scaling

    def entries(traj, gripper_state):
        for v in traj:
            trajs.append((v, gripper_state))
        return trajs

    # move to initial standoff
    traj1 = mr.ScrewTrajectory(Tse_initial, np.dot(Tsc_initial,Tce_standoff), Tf, N, method)
    pos1 = entries(traj1, [0])
    

    # move griper down
    traj2 = mr.ScrewTrajectory(np.dot(Tsc_initial,Tce_standoff), np.dot(Tsc_initial,Tce_grasp), Tf, N, method)
    pos2 = entries(traj2, [0])

    # closing gripper
    traj3 = pos2[-1][0]
    for i in range(100):
        trajs.append((traj3, [1]))


    # move gripper back to standoff
    traj4 = mr.ScrewTrajectory(np.dot(Tsc_initial,Tce_grasp),np.dot(Tsc_initial,Tce_standoff) , Tf, N, method)
    entries(traj4, [1])

    # move to standoff above final configuration of object
    traj5 = mr.ScrewTrajectory(np.dot(Tsc_initial,Tce_standoff), np.dot(Tsc_final, Tce_standoff), Tf, N, method)
    entries(traj5, [1])

    # move gripper to final configuration of object
    traj6 = mr.ScrewTrajectory(np.dot(Tsc_final, Tce_standoff), np.dot(Tsc_final,Tce_grasp), Tf, N, method)
    pos6 = entries(traj6, [1])

    # opening gripper
    traj7 = pos6[-1][0]
    for i in range(100):
        trajs.append((traj7, [0]))


    # gripper back to standoff
    traj8 = mr.ScrewTrajectory(np.dot(Tsc_final,Tce_grasp), np.dot(Tsc_final,Tce_standoff), Tf, N, method)
    entries(traj8, [0])

    return trajs


def FeedbackControl(robot_conf, Tse, Tse_d, Tse_d_next, Kp, Ki, Dt):

    # arm Jacobian
    Blist = [[0,0,0,0,0], [0,-1,-1,-1,0], [1,0,0,0,1], [0,-0.5076,-0.3526,-0.2176,0], [0.033,0,0,0,0], [0,0,0,0,0]]
    theta = robot_conf[3:8]
    Ja = mr.JacobianBody(Blist,theta)

    # base Jacoboan
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

    return speeds


def NextStep(current_conf, speeds, Dt, max_speed, gripper_state):
    cfc = current_conf[:3]
    cfa = current_conf[3:8]
    cfw = current_conf[8:]
    r = 0.0475
    l = 0.235
    w = 0.15
    R = np.array([[1, 0, 0], [0, math.cos(cfc[0]), -math.sin(cfc[0])], [0, math.sin(cfc[0]), math.cos(cfc[0])]])
    
    # arm
    arm= []
    for x,y in zip(cfa, speeds[4:]):
        angle_a = x + y*Dt
        arm.append(angle_a)
    # wheel        
    wheel = []
    for x,y in zip(cfw, speeds[:4]):
        angle_w = x + y*Dt
        wheel.append(angle_w)

    # chassis
    chassis = []
    change_in_wheel_angle = np.array(wheel) - np.array(cfw)
    c = np.array(change_in_wheel_angle)
    F = (r/4)*(np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], [1, 1, 1, 1], [-1, 1, -1, 1]]))
    Vb = np.dot(F,c)
    wbz = Vb[0]
    vbx = Vb[1]
    vby = Vb[2]
    if wbz == 0:
        change_qb = np.array([[wbz],[vbx],[vby]])
    else:
        a = vbx*math.sin(wbz)
        b = math.cos(wbz)
        f = vby*math.sin(wbz) 
        change_qb = np.array([[wbz], [(a + vby*(b - 1))/wbz], [(f + vbx*(1 - b))/wbz]])
    change_q = np.dot(R, change_qb)
    ncfc = np.array([[cfc[0]], [cfc[1]], [cfc[2]]])
    q = np.add(ncfc, change_q)
    for v in [q[0][0], q[1][0], q[2][0]]:
        chassis.append(v)

    # all configurations
    conf = chassis + arm + wheel + gripper_state
      
    return conf
    

def mobile_manipulation(Tsc_initial, Tsc_final, Tse_initial, initial_conf, Kp, Ki):
    Tce_grasp = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    Tce_standoff = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])

    trajs = TrajectoryGeneration(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, 1)
    confs = []
    Dt = 0.01
    for i in range(len(trajs)-1):
        Tse = tse(initial_conf)
        speeds = FeedbackControl(initial_conf, Tse, trajs[i][0], trajs[i+1][0], Kp, Ki, Dt)
        conf = NextStep(initial_conf[:12], speeds, Dt, 15, trajs[i][1])
        confs.append(conf)
        initial_conf = conf
    df = pd.DataFrame(confs)
    df.to_csv("resultconfs.csv", header = False, index = False)
    




# example
Tse_initial = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])
Tsc_initial = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]])
Tsc_final = np.array([[0,1,0,0],[-1,0,0,-1],[0,0,1,0.025],[0,0,0,1]])
initial_conf = [0,0,0,0,0,0,0,0,0,0,0,0,0]
Kp = np.dot(5, np.identity(6))
Ki = np.dot(2, np.identity(6))

mobile_manipulation(Tsc_initial, Tsc_final, Tse_initial, initial_conf, Kp, Ki)



