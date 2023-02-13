import modern_robotics as mr
import numpy as np
import pandas as pd

def TrajectoryGeneration(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k):
    Tf = 1
    N = (Tf*k)/0.01
    method = 3 #cubic time scaling
    allvalues = []

    def entries(traj, gripper_state):
        pos = []
        for v in traj:
            vals = []
            for val in [v[0,0],v[0,1],v[0,2],v[1,0],v[1,1],v[1,2],v[2,0],v[2,1],v[2,2],v[0,3],v[1,3],v[2,3]]:
                vals.append(val)
            allvalues.append(vals + gripper_state)
        return allvalues

    # move to initial standoff
    traj1 = mr.ScrewTrajectory(Tse_initial, np.dot(Tsc_initial,Tce_standoff), Tf, N, method)
    pos1 = entries(traj1, [0])
    

    # move griper down
    traj2 = mr.ScrewTrajectory(np.dot(Tsc_initial,Tce_standoff), np.dot(Tsc_initial,Tce_grasp), Tf, N, method)
    pos2 = entries(traj2, [0])

    # closing gripper
    traj3 = pos2[-1][:12]
    for i in range(100):
        allvalues.append(traj3 + [1])


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
    traj7 = pos6[-1][:12]
    for i in range(100):
        allvalues.append(traj7 + [0])


    # gripper back to standoff
    traj8 = mr.ScrewTrajectory(np.dot(Tsc_final,Tce_grasp), np.dot(Tsc_final,Tce_standoff), Tf, N, method)
    entries(traj8, [0])

    df = pd.DataFrame(allvalues)
    print(df)
    df.to_csv('trajectory.csv', header = False, index = False)



# example
# Tsb = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.0963],[0,0,0,1]])
# Tbo = np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])
# Toe = np.array([[1,0,0,0.033],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])
# Tse_initial = np.dot(np.dot(Tsb,Tbo),Toe)
# Tsc_initial = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]])
# Tsc_final = np.array([[0,1,0,0],[-1,0,0,-1],[0,0,1,0.025],[0,0,0,1]])
# Tce_grasp = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
# Tce_standoff = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])

# TrajectoryGeneration(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, 1)
