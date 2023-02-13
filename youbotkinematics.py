import modern_robotics as mr
import numpy as np
import pandas as pd
import math


def NextStep(current_conf, speeds, Dt, max_speed):
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
    conf = chassis + arm + wheel +[0]
      
        
        
    return conf

# example
current_conf = [0,0,0,0,0,0,0,0,0,0,0,0]
speeds = [-10,10,10,-10,0,0,0,0,0]
Dt = 0.01
time = 0
max_speed = 3

all_conf = []
all_conf.append(current_conf + [0])

while time < 1:
    new = NextStep(current_conf, speeds, Dt, max_speed)
    all_conf.append(new)
    current_conf = new[:12]
    time += Dt

conf = pd.DataFrame(all_conf)
conf.to_csv("conf.csv", index = False, header = False)
    
    
        
