import pybullet as p
import pybullet_data
import pandas as pd
import math
import time
import os
print("CWD:", os.getcwd())
# CSV format: time, x, y, z, roll, pitch, yaw (in degrees)
data = pd.read_csv("positions_log.csv", header=None)
positions = data[[1, 2, 3]].values
orientations = data[[4, 5, 6]].values  # roll, pitch, yaw (degrees)

# Connect and set up simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # for plane.urdf
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load ground plane
p.loadURDF("plane.urdf")

# Set path to your unzipped kinova_gen3 folder
p.setAdditionalSearchPath("kinova_gen3_visual")
arm_id = p.loadURDF("kinova_gen3_visual/gen3.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# List joints and collect actuated revolute joints
actuated_joint_indices = []
print("== Kinova Gen3 Joints ==")
for i in range(p.getNumJoints(arm_id)):
    info = p.getJointInfo(arm_id, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    print(f"Index {i}: {joint_name}, Type: {joint_type}")
    if joint_type == p.JOINT_REVOLUTE:
        actuated_joint_indices.append(i)

# Set end-effector index (last revolute link: bracelet_link → joint_6)
end_effector_index = actuated_joint_indices[-1]

# Camera
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45,
                             cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

# Move to initial pose
init_pos = positions[0]
init_rpy = [math.radians(a) for a in orientations[0]]
init_orn = p.getQuaternionFromEuler(init_rpy)
init_joints = p.calculateInverseKinematics(arm_id, end_effector_index,
                                           init_pos, targetOrientation=init_orn)

for j, joint_index in enumerate(actuated_joint_indices):
    p.resetJointState(arm_id, joint_index, init_joints[j])
p.stepSimulation()
interval = 0.15
prev_pos = None

for i in range(len(positions)):
    pos = positions[i]
    rpy = [math.radians(a) for a in orientations[i]]
    target_orn = p.getQuaternionFromEuler(rpy)

    joint_targets = p.calculateInverseKinematics(arm_id, end_effector_index,
                                                 pos.tolist(), targetOrientation=target_orn)

    for j, joint_index in enumerate(actuated_joint_indices):
        p.setJointMotorControl2(arm_id, joint_index, p.POSITION_CONTROL,
                                targetPosition=joint_targets[j], force=200)

    if prev_pos is not None:
        p.addUserDebugLine(prev_pos, pos.tolist(), [1, 1, 0], 2, 0)
    prev_pos = pos.tolist()

    p.stepSimulation()
    time.sleep(interval)

print("Playback complete.")
time.sleep(2)
p.disconnect()
