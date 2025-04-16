import csv
import time
from utils.connect import RobotConnect
from utils.arm_mover import ArmMover

# Set connection parameters
ip = "192.168.2.9"
port = 10000
credentials = ("jpotter2", "MSAScapstone")

# Create and open connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Instantiate ArmMover
mover = ArmMover(robot_connection)

# Choose a test pose (modify as needed)
test_pose = [0.335, 0.266, 0.42, 90, 0, 90]

print("Starting movement with CSV logging (duration=5s, interval=0.5s)...")
# This call will log positions during the movement
# (move_to_pose calls a CSV logger internally)
if mover.move_to_pose(test_pose,duration=10):
    print("Movement executed successfully.")


# Close the connection
robot_connection.close_connection()
