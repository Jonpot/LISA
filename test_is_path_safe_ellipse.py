from utils.connect import RobotConnect
from utils.arm_mover import ArmMover
import time

# Set connection parameters
ip = "192.168.2.9"
port = 10000
credentials = ("jpotter2", "MSAScapstone")

# Create and open connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Instantiate ArmMover
mover = ArmMover(robot_connection)

# Define sample poses (only first 3 elements matter for forbidden area check)
# For example, start_pose is the current (or home) position and end_pose is near the forbidden area.
start_pose = [0.57, 0.00, 0.42, 90, 0, 90]
end_pose = [0.305, 0.206, 0.42, 90, 0, 90]

# Test the is_path_safe_ellipse function
if mover.is_path_safe_ellipse(start_pose, end_pose):
    print("Path from start_pose to end_pose is safe.")
else:
    print("Path from start_pose to end_pose is NOT safe (forbidden ellipse detected).")

#Test the path
time.sleep(2)
robot_connection.close_connection()
